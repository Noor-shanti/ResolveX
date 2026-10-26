"""A small, separate MiniLM + FAISS index for selected Ubuntu documentation."""

from __future__ import annotations

from datetime import datetime, timezone
from html.parser import HTMLParser
import hashlib
import json
from pathlib import Path
import re
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import faiss
import joblib
import numpy as np

from src.retrieval import MODEL_ID, MODEL_REVISION, chunk_spans


# Selected starting corpus, not a crawl of Ubuntu's documentation sites. Scope is
# deliberately explicit so products/releases are not silently treated as equal.
OFFICIAL_SOURCES = [
    {"url": "https://manpages.ubuntu.com/manpages/resolute/man1/nmcli.1.html", "title": "Ubuntu manpage: nmcli", "categories": ["Network & Connectivity"], "release_scope": "Ubuntu 26.04 LTS (Resolute); NetworkManager command reference", "source_type": "official_ubuntu_manpage"},
    {"url": "https://ubuntu.com/server/docs/how-to/networking/", "title": "Networking", "categories": ["Network & Connectivity"], "release_scope": "Ubuntu Server documentation; current documentation site"},
    {"url": "https://ubuntu.com/desktop/docs/en/26.04/tutorial/install-ubuntu-desktop/", "title": "Install Ubuntu Desktop", "categories": ["Boot & Startup", "Storage & Filesystems"], "release_scope": "Ubuntu Desktop 26.04; installation only"},
    {"url": "https://ubuntu.com/desktop/docs/en/26.04/how-to/software/", "title": "Install or remove software", "categories": ["Software & Packages"], "release_scope": "Ubuntu Desktop 26.04"},
    {"url": "https://ubuntu.com/server/docs/how-to/software/package-management/", "title": "Install and manage packages", "categories": ["Software & Packages"], "release_scope": "Current Ubuntu Server documentation; APT package-index, install and upgrade command reference"},
    {"url": "https://ubuntu.com/desktop/docs/en/latest/how-to/graphics/install-nvidia-drivers/", "title": "Install NVIDIA drivers", "categories": ["Hardware & Drivers"], "release_scope": "Latest Ubuntu Desktop documentation; check the current page against the ticket release"},
    {"url": "https://ubuntu.com/docs/steam/howto/dedicated-gpu/", "title": "Use a dedicated GPU with Steam", "categories": ["Hardware & Drivers"], "release_scope": "Ubuntu Steam documentation; the page does not state a specific Ubuntu release"},
    {"url": "https://documentation.ubuntu.com/release-notes/26.04/summary-for-lts-users/", "title": "Ubuntu 26.04 LTS NVIDIA driver guidance", "categories": ["Hardware & Drivers"], "release_scope": "Ubuntu 26.04 LTS release notes; recommended driver installation methods"},
    {"url": "https://documentation.ubuntu.com/desktop/en/26.04/how-to/troubleshoot/bluetooth/", "title": "Bluetooth", "categories": ["Hardware & Drivers"], "release_scope": "Ubuntu Desktop 26.04"},
    {"url": "https://manpages.ubuntu.com/manpages/resolute/man1/df.1.html", "title": "Ubuntu manpage: df", "categories": ["Storage & Filesystems"], "release_scope": "Ubuntu 26.04 LTS (Resolute); reports mounted filesystem space", "source_type": "official_ubuntu_manpage"},
    {"url": "https://manpages.ubuntu.com/manpages/resolute/man8/lsblk.8.html", "title": "Ubuntu manpage: lsblk", "categories": ["Storage & Filesystems"], "release_scope": "Ubuntu 26.04 LTS (Resolute); lists block devices", "source_type": "official_ubuntu_manpage"},
    {"url": "https://manpages.ubuntu.com/manpages/resolute/man1/ls.1.html", "title": "Ubuntu manpage: ls", "categories": ["Access & Permissions"], "release_scope": "Ubuntu 26.04 LTS (Resolute); long listing shows file metadata", "source_type": "official_ubuntu_manpage"},
]


class _DocumentText(HTMLParser):
    """Extract readable article text while skipping page navigation and scripts."""

    SKIP = {"script", "style", "nav", "header", "footer", "svg", "noscript"}
    BLOCK = {"p", "li", "h1", "h2", "h3", "h4", "pre", "blockquote", "tr"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.skip_depth = 0
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip_depth += 1
        elif not self.skip_depth and tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self.skip_depth:
            self.skip_depth -= 1
        elif not self.skip_depth and tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.skip_depth:
            self.parts.append(data)


def _plain_text(html):
    parser = _DocumentText()
    parser.feed(html)
    text = "\n".join(line.strip() for line in "".join(parser.parts).splitlines() if line.strip())
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _download(url, timeout=30, attempts=3):
    host = (urlparse(url).hostname or "").lower()
    if host not in {"ubuntu.com", "www.ubuntu.com", "documentation.ubuntu.com", "help.ubuntu.com", "manpages.ubuntu.com"}:
        raise ValueError(f"Refusing non-Ubuntu documentation host: {host}")
    request = Request(url, headers={
        "User-Agent": "Mozilla/5.0 (compatible; ResolveX-M5-doc-index/1.0)",
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "en-US,en;q=0.9",
    })
    for attempt in range(attempts):
        try:
            with urlopen(request, timeout=timeout) as response:
                content_type = response.headers.get_content_type()
                if content_type not in {"text/html", "application/xhtml+xml"}:
                    raise ValueError(f"Expected HTML at {url}; got {content_type}")
                return response.read().decode(response.headers.get_content_charset() or "utf-8", errors="replace")
        except HTTPError as error:
            if error.code not in {429, 500, 502, 503, 504} or attempt + 1 == attempts:
                raise
        except URLError:
            if attempt + 1 == attempts:
                raise
        time.sleep(0.5 * (2 ** attempt))


def _fingerprint(records):
    payload = json.dumps(records, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_official_docs(root, model, refresh=False):
    """Fetch the allowlisted pages and build a fingerprinted, separate FAISS index.

    The caller supplies the same pinned MiniLM instance used by the M4 retriever.
    Existing artifacts are loaded only when the allowlist and source text hashes match.
    """
    root = Path(root).resolve()
    output = root / "data/processed/retrieval/official_docs"
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    index_path, records_path = output / "official_docs.faiss", output / "chunks.joblib"

    if manifest_path.exists() and index_path.exists() and records_path.exists() and not refresh:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        records = joblib.load(records_path)
        file_hashes = {"official_docs.faiss": _file_hash(index_path), "chunks.joblib": _file_hash(records_path)}
        valid_cache = (
            manifest.get("source_hashes") == {r["url"]: r["content_hash"] for r in records if r["part"] == 0}
            and manifest.get("allowlist_fingerprint") == _fingerprint(OFFICIAL_SOURCES)
            and manifest.get("file_hashes") == file_hashes
        )
        if valid_cache and not manifest.get("failed_sources"):
            retriever = OfficialDocsRetriever(model, faiss.read_index(str(index_path)), records)
            retriever.manifest = manifest
            return retriever, manifest
        if manifest.get("allowlist_fingerprint") == _fingerprint(OFFICIAL_SOURCES):
            raise ValueError("Official-documentation cache files do not match their manifest; inspect them and pass refresh=True to rebuild")

    pages, failures = [], []
    retrieved_at = datetime.now(timezone.utc).isoformat()
    for source in OFFICIAL_SOURCES:
        try:
            body = _plain_text(_download(source["url"]))
            if len(body) < 200:
                raise ValueError("Downloaded page has too little readable content")
            pages.append({**source, "body": body, "content_hash": hashlib.sha256(body.encode("utf-8")).hexdigest()})
        except Exception as error:
            failures.append({"url": source["url"], "error": type(error).__name__,
                             "details": str(error)[:240]})

    chunks = []
    for page in pages:
        for part, (start, end, _large_code) in enumerate(chunk_spans(page["body"], max_words=100)):
            text = page["body"][start:end].strip()
            if text:
                chunks.append({"source_id": page["url"], "title": page["title"], "url": page["url"],
                               "categories": page["categories"], "release_scope": page["release_scope"],
                               "source_type": page.get("source_type", "official_ubuntu_documentation"), "part": part,
                               "content_hash": page["content_hash"], "text": text})
    if not chunks:
        raise RuntimeError("No official Ubuntu pages were fetched; no index was created")

    vectors = model.encode([r["title"] + "\n" + r["text"] for r in chunks], batch_size=32,
                           normalize_embeddings=True, show_progress_bar=True)
    vectors = np.asarray(vectors, dtype="float32")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(index_path))
    joblib.dump(chunks, records_path, compress=3)
    manifest = {
        "created_at": retrieved_at,
        "embedding_model": MODEL_ID,
        "embedding_revision": MODEL_REVISION,
        "allowlist_fingerprint": _fingerprint(OFFICIAL_SOURCES),
        "source_hashes": {p["url"]: p["content_hash"] for p in pages},
        "file_hashes": {"official_docs.faiss": _file_hash(index_path), "chunks.joblib": _file_hash(records_path)},
        "source_count": len(pages), "chunk_count": len(chunks), "failed_sources": failures,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    retriever = OfficialDocsRetriever(model, index, chunks)
    retriever.manifest = manifest
    return retriever, manifest


def load_official_docs(root, model):
    """Load the prepared official-docs index without network access or rebuilding."""
    root = Path(root).resolve()
    output = root / "data/processed/retrieval/official_docs"
    manifest_path = output / "manifest.json"
    index_path = output / "official_docs.faiss"
    records_path = output / "chunks.joblib"
    required = (manifest_path, index_path, records_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Prepared Ubuntu documentation index is missing; run notebook 08 preparation first"
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("allowlist_fingerprint") != _fingerprint(OFFICIAL_SOURCES):
        raise ValueError("Ubuntu documentation index is outdated; rebuild it from notebook 08")
    if manifest.get("embedding_model") != MODEL_ID or manifest.get("embedding_revision") != MODEL_REVISION:
        raise ValueError("Ubuntu documentation index uses a different embedding model")

    records = joblib.load(records_path)
    expected_sources = {
        record["url"]: record["content_hash"]
        for record in records if record.get("part") == 0
    }
    actual_hashes = {
        "official_docs.faiss": _file_hash(index_path),
        "chunks.joblib": _file_hash(records_path),
    }
    if manifest.get("source_hashes") != expected_sources or manifest.get("file_hashes") != actual_hashes:
        raise ValueError("Ubuntu documentation index does not match its manifest")
    if manifest.get("failed_sources"):
        raise ValueError("Ubuntu documentation index has failed sources; review it in notebook 08")

    index = faiss.read_index(str(index_path))
    if index.ntotal != len(records) or not records:
        raise ValueError("Ubuntu documentation index and chunk records do not match")
    retriever = OfficialDocsRetriever(model, index, records)
    retriever.manifest = manifest
    return retriever, manifest


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class OfficialDocsRetriever:
    """Exact cosine retrieval over the curated documentation chunks."""

    def __init__(self, model, index, chunks):
        self.model, self.index, self.chunks = model, index, chunks
        self.manifest = {}

    def search(self, query, k=3):
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Provide nonempty ticket text")
        if not 1 <= k <= len(self.chunks):
            raise ValueError("k is outside the official-documentation index")
        vector = self.model.encode([query], normalize_embeddings=True, show_progress_bar=False)
        scores, positions = self.index.search(np.asarray(vector, dtype="float32"), self.index.ntotal)
        best_by_url = {}
        for score, position in zip(scores[0], positions[0]):
            if position < 0:
                continue
            record = self.chunks[int(position)]
            if record["url"] not in best_by_url:
                best_by_url[record["url"]] = {**record, "score": float(score)}
        return sorted(best_by_url.values(), key=lambda item: (-item["score"], item["url"]))[:k]
