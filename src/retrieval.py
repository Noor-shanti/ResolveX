"""Evidence preparation and initial retrieval experiments for ResolveX M4."""

from collections import Counter, defaultdict
from contextlib import closing
import csv
import hashlib
import inspect
import json
from pathlib import Path
import random
import re
import sqlite3
import time
from tempfile import TemporaryDirectory
import xml.etree.ElementTree as ET

import joblib
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from src.data.preprocessing import extract_text


LEGACY_CODE = "6c2ac093f99b6505769e5536f17849d9f85b5c1f6b271c841d7eaf3ac3cadbfc"
# These hashes identify the unchanged preparation policies after the cache refactor.
LEGACY_COMPATIBLE = {
    "evidence": "1f94ec50ca3c250dbd7af24013ad9b92cc1171ea91e12a5212bdc7ed94c1ea00",
    "benchmark": "ef7688fed8db4ba3a93c5ab56b5be3429c7bc7b98fdba292be648892d3cdf4d5",
    "bm25": "178a4928a4540b3eab06761966497871276aec2c9d237dd9e94e0ec66474fb43",
    "pilot": "624a8a381e3ec649c0c5bc4c6e203e276463709f40222200919e852d4db9187e",
}
STAGE_OBJECTS = {
    "evidence": "select_answer chunk_spans xml_rows prepare_evidence extract_text",
    "benchmark": "duplicate_groups build_benchmark extract_text",
    "bm25": "technical_tokens BM25Index build_bm25",
    "pilot": "embedding_views DensePilot build_dense_pilot",
}


def _stage_code(stage):
    objects = [_stage_code, _signature_matches]
    objects += [globals()[name] for name in STAGE_OBJECTS[stage].split()]
    source = "\n".join(inspect.getsource(obj) for obj in objects)
    if stage == "benchmark":
        source += POST_LINK.pattern + DUPLICATE_NOTICE.pattern
    if stage == "bm25":
        source += TOKEN.pattern
    return hashlib.sha256(source.encode()).hexdigest()


def _signature_matches(saved, expected, stage):
    if saved == expected:
        return True
    # Accept the known old module only for the verified equivalent policy.
    legacy = {**expected, "code": LEGACY_CODE}
    return (expected["code"] == LEGACY_COMPATIBLE.get(stage) and saved == legacy)


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_answer(record):
    """Prefer a nonnegative valid accepted answer, else the best positive answer."""
    usable = [a for a in record["answers"] if a["body_text"].strip()]
    accepted = next((a for a in usable if a["answer_id"] == record["accepted_answer_id"]), None)
    if accepted and int(accepted["score"]) >= 0:
        return accepted, "accepted"
    positive = [a for a in usable if int(a["score"]) > 0]
    if positive:
        return min(positive, key=lambda a: (-int(a["score"]), int(a["answer_id"]))), "positive_fallback"
    return None, "no_supported_answer"


def chunk_spans(text, max_words=180):
    """Return contiguous character spans; fenced code is never split in storage."""
    if max_words < 1:
        raise ValueError("max_words must be positive")
    blocks, start, position, in_code = [], 0, 0, False
    for line in text.splitlines(keepends=True):
        end = position + len(line)
        if line.lstrip().startswith("```"):
            if not in_code and start < position:
                blocks.append((start, position, False))
                start = position
            in_code = not in_code
            if not in_code:
                blocks.append((start, end, True))
                start = end
        elif not in_code and not line.strip():
            blocks.append((start, end, False))
            start = end
        position = end
    if start < len(text):
        blocks.append((start, len(text), in_code))
    pieces = []
    for left, right, code in blocks:
        words = list(re.finditer(r"\S+", text[left:right]))
        if code or len(words) <= max_words:
            pieces.append((left, right, code and len(words) > max_words))
        else:
            points = [left] + [left + words[i].start() for i in range(max_words, len(words), max_words)] + [right]
            pieces.extend((a, b, False) for a, b in zip(points, points[1:]))
    chunks = []
    for left, right, oversized in pieces:
        if chunks and not oversized and not chunks[-1][2] and len(text[chunks[-1][0]:right].split()) <= max_words:
            chunks[-1] = (chunks[-1][0], right, False)
        else:
            chunks.append((left, right, oversized))
    if text:
        assert chunks[0][0] == 0 and chunks[-1][1] == len(text)
        assert all(a[1] == b[0] for a, b in zip(chunks, chunks[1:]))
    return chunks


def xml_rows(path):
    parser = ET.iterparse(path, events=("start", "end"))
    _, root = next(parser)
    for event, element in parser:
        if event == "end" and element.tag == "row":
            yield dict(element.attrib)
            element.clear()
            root.clear()


def prepare_evidence(root, max_words=180):
    """Build a disk-backed corpus, source attribution and lossless text spans."""
    root = Path(root)
    output = root / "data/processed/retrieval"
    output.mkdir(parents=True, exist_ok=True)
    db_path, manifest_path = output / "evidence.sqlite", output / "corpus_manifest.json"
    inputs = {name: file_hash(root / name) for name in [
        "data/processed/knowledge_base.jsonl", "data/raw/Users.xml", "data/raw/Posts.xml",
    ]}
    signature = {"inputs": inputs, "code": _stage_code("evidence"), "max_words": max_words}
    if db_path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not _signature_matches(manifest["signature"], signature, "evidence"):
            raise ValueError("Evidence cache differs from inputs or policy; inspect before rebuilding")
        return db_path, manifest
    if db_path.exists() or manifest_path.exists():
        raise FileExistsError("Incomplete evidence cache; inspect before rebuilding")
    counts = Counter()
    with TemporaryDirectory(prefix="evidence-", dir=output) as directory:
        staged = Path(directory) / "evidence.sqlite"
        connection = sqlite3.connect(staged)
        try:
            connection.executescript("""
                CREATE TABLE cases (
                    question_id TEXT PRIMARY KEY, answer_id TEXT UNIQUE NOT NULL,
                    title TEXT NOT NULL, question_text TEXT NOT NULL, answer_text TEXT NOT NULL,
                    selection TEXT NOT NULL, question_owner TEXT, answer_owner TEXT, metadata TEXT NOT NULL
                );
                CREATE TABLE chunks (
                    unit_id INTEGER PRIMARY KEY, question_id TEXT NOT NULL,
                    part TEXT NOT NULL, start INTEGER NOT NULL, end INTEGER NOT NULL,
                    oversized_code INTEGER NOT NULL
                );
                CREATE TABLE authors (owner_id TEXT PRIMARY KEY, name TEXT NOT NULL);
                CREATE TABLE post_authors (post_id TEXT PRIMARY KEY, name TEXT NOT NULL);
            """)
            with (root / "data/processed/knowledge_base.jsonl").open(encoding="utf-8") as handle:
                for line in handle:
                    record = json.loads(line)
                    counts["source_questions"] += 1
                    answer, policy = select_answer(record)
                    counts[policy] += 1
                    if answer is None:
                        continue
                    metadata = {
                        "question": {k: record[k] for k in ["question_id", "source_url", "creation_date", "score", "content_license", "owner_user_id", "tags"]},
                        "answer": {k: answer[k] for k in ["answer_id", "source_url", "creation_date", "score", "is_accepted", "content_license", "owner_user_id"]},
                        "selection": policy,
                    }
                    connection.execute("INSERT INTO cases VALUES (?,?,?,?,?,?,?,?,?)", (
                        record["question_id"], answer["answer_id"], record["title"], record["body_text"],
                        answer["body_text"], policy, record["owner_user_id"], answer["owner_user_id"],
                        json.dumps(metadata, ensure_ascii=False),
                    ))
                    for part, text in [("question", record["body_text"]), ("answer", answer["body_text"])]:
                        spans = chunk_spans(text, max_words)
                        connection.executemany("INSERT INTO chunks (question_id,part,start,end,oversized_code) VALUES (?,?,?,?,?)", [
                            (record["question_id"], part, a, b, int(large)) for a, b, large in spans
                        ])
                        counts["chunks"] += len(spans)
                        counts["oversized_code_chunks"] += sum(c for _, _, c in spans)
                    counts["cases"] += 1
                    if counts["cases"] % 50000 == 0:
                        connection.commit()
                        print("Prepared evidence cases:", counts["cases"], flush=True)
            connection.commit()
            needed = {r[0] for r in connection.execute("SELECT question_owner FROM cases UNION SELECT answer_owner FROM cases") if r[0]}
            for row in xml_rows(root / "data/raw/Users.xml"):
                if row["Id"] in needed and row.get("DisplayName", "").strip():
                    connection.execute("INSERT INTO authors VALUES (?,?)", (row["Id"], row["DisplayName"]))
            connection.commit()
            missing = {r[0] for r in connection.execute("""
                SELECT c.question_id FROM cases c LEFT JOIN authors a ON c.question_owner=a.owner_id WHERE a.owner_id IS NULL
                UNION SELECT c.answer_id FROM cases c LEFT JOIN authors a ON c.answer_owner=a.owner_id WHERE a.owner_id IS NULL
            """)}
            for row in xml_rows(root / "data/raw/Posts.xml"):
                if row["Id"] in missing:
                    if row.get("OwnerDisplayName", "").strip():
                        connection.execute("INSERT INTO post_authors VALUES (?,?)", (row["Id"], row["OwnerDisplayName"]))
                    missing.remove(row["Id"])
                    if not missing:
                        break
            connection.executescript("CREATE INDEX chunks_question ON chunks(question_id);")
            connection.commit()
            for side, post_column in [("question", "question_id"), ("answer", "answer_id")]:
                for status, condition in [
                    ("user_name", "a.owner_id IS NOT NULL"),
                    ("post_name", "a.owner_id IS NULL AND p.post_id IS NOT NULL"),
                    ("unavailable_name", "a.owner_id IS NULL AND p.post_id IS NULL"),
                ]:
                    counts[side + "_" + status] = connection.execute(f"""
                        SELECT count(*) FROM cases c LEFT JOIN authors a ON c.{side}_owner=a.owner_id
                        LEFT JOIN post_authors p ON c.{post_column}=p.post_id WHERE {condition}
                    """).fetchone()[0]
        finally:
            connection.close()
        staged.replace(db_path)
    manifest = {"signature": signature, "counts": dict(counts)}
    manifest["fingerprint"] = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return db_path, manifest


def duplicate_groups(path, seed=42):
    parent = {}
    def find(value):
        parent.setdefault(value, value)
        if parent[value] != value:
            parent[value] = find(parent[value])
        return parent[value]
    with open(path, encoding="utf-8", newline="") as handle:
        links = list(csv.DictReader(handle))
    for row in links:
        left, right = find(row["QuestionId"]), find(row["RelatedQuestionId"])
        parent[max(left, right)] = min(left, right)
    groups = {value: find(value) for value in parent}
    roots = sorted({groups[row["QuestionId"]] for row in links})
    random.Random(seed).shuffle(roots)
    cutoff = int(0.8 * len(roots))
    roles = {value: "development" if i < cutoff else "heldout" for i, value in enumerate(roots)}
    return links, groups, roles


def policy_review_sample(root, per_bucket=2, seed=42):
    """Small source review from development components, before retrieval scoring."""
    root = Path(root)
    _, groups, roles = duplicate_groups(root / "data/processed/duplicate_relationships.csv", seed)
    samples = defaultdict(list)
    with (root / "data/processed/knowledge_base.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if roles.get(groups.get(record["question_id"])) != "development":
                continue
            selected, policy = select_answer(record)
            accepted = next((a for a in record["answers"] if a["answer_id"] == record["accepted_answer_id"]), None)
            if accepted is not None:
                score = int(accepted["score"])
                bucket = "accepted_negative" if score < 0 else "accepted_zero" if score == 0 else "accepted_positive"
            else:
                bucket = policy
            inspected = selected or accepted or max(record["answers"], key=lambda a: int(a["score"]))
            key = hashlib.sha256(f"{seed}:{record['question_id']}".encode()).hexdigest()
            item = {"QuestionId": record["question_id"], "Bucket": bucket, "Title": record["title"],
                    "QuestionText": record["body_text"], "InspectedAnswerId": inspected["answer_id"],
                    "AnswerText": inspected["body_text"], "AnswerScore": int(inspected["score"]),
                    "Selection": policy, "SelectedAnswerId": selected["answer_id"] if selected else None}
            samples[bucket].append((key, item))
            samples[bucket].sort(key=lambda pair: pair[0])
            del samples[bucket][per_bucket:]
    return [item for bucket in sorted(samples) for _, item in samples[bucket]]


POST_LINK = re.compile(r"https?://(?:www\.)?askubuntu\.com/(questions|q|a)/(\d+)[^\s)]*", re.I)
DUPLICATE_NOTICE = re.compile(r"possible\s+duplicate|this question already has answers here|marked as duplicate", re.I)


def build_benchmark(root, db_path, corpus_fingerprint, seed=42):
    root = Path(root)
    output = Path(db_path).parent
    manifest_path = output / "benchmark_manifest.json"
    signature = {"corpus": corpus_fingerprint, "seed": seed, "code": _stage_code("benchmark"),
                 "links": file_hash(root / "data/processed/duplicate_relationships.csv"),
                 "questions": file_hash(root / "data/interim/questions.csv"),
                 "answers": file_hash(root / "data/interim/answers.csv")}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not _signature_matches(manifest["signature"], signature, "benchmark"):
            raise ValueError("Benchmark cache differs; inspect before rebuilding")
        return manifest
    links, groups, roles = duplicate_groups(root / "data/processed/duplicate_relationships.csv", seed)
    with open(root / "data/interim/answers.csv", encoding="utf-8", newline="") as handle:
        answer_parents = {row["Id"]: row["ParentId"] for row in csv.DictReader(handle)}
    with sqlite3.connect(db_path) as connection:
        eligible = {row[0] for row in connection.execute("SELECT question_id FROM cases")}
        positives = defaultdict(set)
        counts = Counter(source_pairs=len(links))
        for row in links:
            if row["RelatedQuestionId"] in eligible:
                positives[row["QuestionId"]].add(row["RelatedQuestionId"])
            else:
                counts["unavailable_target_pairs"] += 1
        counts["queries_with_available_targets"] = len(positives)
        connection.execute("""CREATE TABLE benchmark (
            query_id TEXT PRIMARY KEY, group_id TEXT NOT NULL, split TEXT NOT NULL,
            title TEXT NOT NULL, query_text TEXT NOT NULL, relevant_ids TEXT NOT NULL
        )""")
        with open(root / "data/interim/questions.csv", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                qid = row["Id"]
                if qid not in positives:
                    continue
                text = row["Title"] + "\n\n" + extract_text(row["Body"])
                targets = positives[qid] - {qid}
                linked_ids = {answer_parents.get(match.group(2)) if match.group(1).casefold() == "a"
                              else match.group(2) for match in POST_LINK.finditer(text)}
                if linked_ids & targets:
                    counts["direct_target_link_queries"] += 1
                    continue
                if DUPLICATE_NOTICE.search(text):
                    counts["duplicate_notice_queries"] += 1
                    continue
                text = POST_LINK.sub("", text).strip()
                if not text or not targets:
                    counts["empty_queries"] += 1
                    continue
                split = roles[groups[qid]]
                connection.execute("INSERT INTO benchmark VALUES (?,?,?,?,?,?)", (
                    qid, groups[qid], split, row["Title"], text, json.dumps(sorted(targets)),
                ))
                counts[split + "_queries"] += 1
                counts["eligible_pairs"] += len(targets)
        for role in ("development", "heldout"):
            counts[role + "_groups"] = connection.execute(
                "SELECT count(DISTINCT group_id) FROM benchmark WHERE split=?", (role,)
            ).fetchone()[0]
        overlap = connection.execute("SELECT group_id FROM benchmark GROUP BY group_id HAVING count(DISTINCT split)>1").fetchall()
        assert not overlap, "Duplicate components cross benchmark splits"
        connection.commit()
        with (output / "query_assignments.csv").open("x", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["QueryId", "GroupId", "Split"])
            writer.writerows(connection.execute("SELECT query_id,group_id,split FROM benchmark ORDER BY query_id"))
    manifest = {"signature": signature, "counts": dict(counts)}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def development_queries(db_path, limit=100, seed=42, candidate_ids=None):
    """Sample one query per development group; never score held-out queries."""
    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute("SELECT * FROM benchmark WHERE split='development' ORDER BY query_id").fetchall()
    by_group = defaultdict(list)
    candidate_ids = None if candidate_ids is None else set(candidate_ids)
    for row in rows:
        item = dict(row)
        item["relevant_ids"] = set(json.loads(item["relevant_ids"]))
        if candidate_ids is not None:
            item["relevant_ids"] &= candidate_ids
        if item["relevant_ids"]:
            by_group[item["group_id"]].append(item)
    rng = random.Random(seed)
    names = sorted(by_group)
    rng.shuffle(names)
    return [rng.choice(by_group[name]) for name in names[:limit]]


TOKEN = re.compile(r"(?u)\b\w[\w./+:-]*\b")


def technical_tokens(text):
    return TOKEN.findall(text.casefold())


class BM25Index:
    """Sparse BM25 with positive log IDF and fixed initial k1=1.2, b=0.75."""
    def __init__(self, question_ids, texts, k1=1.2, b=0.75):
        self.question_ids = np.asarray(question_ids)
        self.vectorizer = CountVectorizer(analyzer=technical_tokens, token_pattern=None,
                                          lowercase=False, dtype=np.float32)
        weights = self.vectorizer.fit_transform(texts).tocsc()
        lengths = np.asarray(weights.sum(axis=1)).ravel()
        document_frequency = np.diff(weights.indptr)
        idf = np.log1p((len(question_ids) - document_frequency + 0.5) / (document_frequency + 0.5))
        normalizer = k1 * (1 - b + b * lengths / lengths.mean())
        weights.data *= (k1 + 1) / (weights.data + normalizer[weights.indices])
        weights.data *= np.repeat(idf, document_frequency)
        self.weights = weights
        self.lookup = {qid: i for i, qid in enumerate(self.question_ids)}

    def search(self, query, exclude_question_id=None, k=10):
        terms = self.vectorizer.transform([query]).indices
        if not len(terms):
            return []
        scores = np.asarray(self.weights[:, terms].sum(axis=1)).ravel()
        if exclude_question_id in self.lookup:
            scores[self.lookup[exclude_question_id]] = -np.inf
        order = np.lexsort((np.arange(len(scores)), -scores))[:k]
        return [(str(self.question_ids[i]), float(scores[i])) for i in order if scores[i] > 0]


def build_bm25(db_path, corpus_fingerprint):
    path = Path(db_path).parent / "bm25.joblib"
    signature = {"corpus": corpus_fingerprint, "code": _stage_code("bm25"), "k1": 1.2, "b": 0.75}
    if path.exists():
        saved = joblib.load(path)
        if not _signature_matches(saved["signature"], signature, "bm25"):
            raise ValueError("BM25 cache differs; inspect before rebuilding")
        return saved["index"]
    with sqlite3.connect(db_path) as connection:
        question_ids = [row[0] for row in connection.execute("SELECT question_id FROM cases ORDER BY question_id")]
        texts = ("\n\n".join(row) for row in connection.execute("SELECT title,question_text,answer_text FROM cases ORDER BY question_id"))
        index = BM25Index(question_ids, texts)
    joblib.dump({"signature": signature, "index": index}, path, compress=3)
    return index


def evaluate_retrieval(queries, search):
    """Recall at question level and reciprocal rank, for development only here."""
    rows = []
    for query in queries:
        hits = search(query["query_text"], query["query_id"], 10)
        ranked = [qid for qid, _ in hits]
        assert query["query_id"] not in ranked
        assert len(ranked) == len(set(ranked))
        relevant = query["relevant_ids"]
        reciprocal = next((1 / rank for rank, qid in enumerate(ranked, 1) if qid in relevant), 0.0)
        row = {"QueryId": query["query_id"], "MRR@10": reciprocal}
        row.update({f"Recall@{k}": len(set(ranked[:k]) & relevant) / len(relevant) for k in (1, 5, 10)})
        rows.append(row)
    return rows


def embedding_views(title, text, tokenizer, max_tokens):
    """Window only embedding input; stored source and code blocks remain intact."""
    prefix = title + "\n\n" if title else ""
    budget = max_tokens - len(tokenizer(prefix, add_special_tokens=False)["input_ids"]) - 6
    if budget < 40:
        raise ValueError("Title leaves too little embedding context")
    offsets = tokenizer(text, add_special_tokens=False, truncation=False,
                        return_offsets_mapping=True, verbose=False)["offset_mapping"]
    views = []
    for start in range(0, len(offsets), budget - 32):
        end = min(start + budget, len(offsets))
        view = prefix + text[offsets[start][0]:offsets[end - 1][1]]
        length = len(tokenizer(view, add_special_tokens=True, truncation=False)["input_ids"])
        if length > max_tokens:
            raise ValueError("Embedding view would be truncated")
        views.append(view)
        if end == len(offsets):
            break
    return views


class DensePilot:
    """Exact cosine search over a fixed pilot corpus, collapsed by question ID."""
    def __init__(self, model, index, question_ids):
        self.model, self.index, self.question_ids = model, index, question_ids

    def search(self, query, exclude_question_id=None, k=10):
        views = embedding_views("", query, self.model.tokenizer, self.model.max_seq_length)
        if not views:
            return []
        vectors = self.model.encode(views, batch_size=16, normalize_embeddings=True, show_progress_bar=False)
        count = min(self.index.ntotal, max(64, k * 4))
        while True:
            scores, positions = self.index.search(vectors, count)
            combined = {}
            for score, position in zip(scores.ravel(), positions.ravel()):
                qid = self.question_ids[int(position)]
                if qid != exclude_question_id:
                    combined[qid] = max(combined.get(qid, -np.inf), float(score))
            ranked = sorted(combined.items(), key=lambda item: (-item[1], int(item[0])))
            if count == self.index.ntotal or (len(ranked) >= k and float(scores[:, -1].max()) < ranked[k - 1][1]):
                return ranked[:k]
            count = min(self.index.ntotal, count * 2)


def build_dense_pilot(db_path, corpus_fingerprint, model, model_id, revision, limit=2000, seed=42):
    """Random corpus pilot, chosen without relevance or retrieval-score feedback."""
    import faiss
    from importlib.metadata import version

    output = Path(db_path).parent
    manifest_path = output / "dense_pilot_manifest.json"
    index_path, ids_path = output / "dense_pilot.faiss", output / "dense_pilot_ids.joblib"
    signature = {"corpus": corpus_fingerprint, "model": model_id, "revision": revision,
                 "max_tokens": model.max_seq_length, "limit": limit, "seed": seed,
                 "code": _stage_code("pilot"), "sentence_transformers": version("sentence-transformers")}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not _signature_matches(manifest["signature"], signature, "pilot"):
            raise ValueError("Dense pilot cache differs; inspect before rebuilding")
        return DensePilot(model, faiss.read_index(str(index_path)), joblib.load(ids_path)), manifest
    with sqlite3.connect(db_path) as connection:
        ids = [row[0] for row in connection.execute("SELECT question_id FROM cases ORDER BY question_id")]
        candidate_ids = sorted(random.Random(seed).sample(ids, min(limit, len(ids))))
        placeholders = ",".join("?" for _ in candidate_ids)
        rows = connection.execute(f"""
            SELECT c.question_id,c.title,c.question_text,c.answer_text,u.part,u.start,u.end
            FROM chunks u JOIN cases c USING(question_id)
            WHERE c.question_id IN ({placeholders}) ORDER BY c.question_id,u.unit_id
        """, candidate_ids)
        texts, question_ids = [], []
        for qid, title, question, answer, part, start, end in rows:
            source = question if part == "question" else answer
            views = embedding_views(title, source[start:end], model.tokenizer, model.max_seq_length)
            texts.extend(views)
            question_ids.extend([qid] * len(views))
    batches = []
    for start in range(0, len(texts), 256):
        batches.append(model.encode(texts[start:start + 256], batch_size=16,
                                    normalize_embeddings=True, show_progress_bar=False))
        if start % 1024 == 0:
            print("Embedded pilot views:", min(start + 256, len(texts)), "/", len(texts), flush=True)
    vectors = np.concatenate(batches).astype(np.float32)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(index_path))
    joblib.dump(question_ids, ids_path)
    manifest = {"signature": signature, "candidate_ids": candidate_ids,
                "cases": len(candidate_ids), "embedding_views": len(question_ids),
                "dimension": vectors.shape[1], "index_bytes": index_path.stat().st_size}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return DensePilot(model, index, question_ids), manifest


# Shared-corpus comparison and the reusable retriever.
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
METHODS = ["bm25", "dense", "hybrid_0.25", "hybrid_0.50", "hybrid_0.75"]


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _save_json(path, value):
    path = Path(path)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    pending.replace(path)


def _cached_json(path, signature):
    if not path.exists():
        return None
    saved = json.loads(path.read_text(encoding="utf-8"))
    if saved["signature"] != signature:
        raise ValueError(f"{path.name} differs from the current experiment; do not reuse stale results")
    return saved


def _completion_code():
    names = ("prepare_comparison comparison_queries _sample_role _case_rows "
             "build_comparison_indexes fuse_rankings ranking_metrics compare_retrievers "
             "score_diagnostics freeze_retriever RetrievalEngine load_retriever "
             "evaluate_heldout retrieval_diagnostics")
    source = "\n".join(inspect.getsource(globals()[name]) for name in names.split())
    return _digest({"source": source, "bm25": _stage_code("bm25"),
                    "dense": _stage_code("pilot"), "methods": METHODS})


def _sample_role(db_path, role, candidate_ids, limit, seed):
    candidates = set(candidate_ids)
    groups = defaultdict(list)
    total = 0
    with closing(sqlite3.connect(Path(db_path).as_uri() + "?mode=ro", uri=True)) as connection:
        for qid, group, targets in connection.execute(
                "SELECT query_id,group_id,relevant_ids FROM benchmark WHERE split=? ORDER BY query_id", (role,)):
            if candidates.intersection(json.loads(targets)):
                groups[group].append(qid)
                total += 1
    rng = random.Random(seed)
    names = sorted(groups)
    rng.shuffle(names)
    selected = [rng.choice(groups[name]) for name in names[:limit]]
    return selected, {"eligible_queries": total, "eligible_groups": len(groups), "sampled_queries": len(selected)}


def prepare_comparison(db_path, corpus_fingerprint, case_limit=20000, query_limit=200, seed=42):
    """Fix a resource-limited corpus and group-sampled queries before scoring."""
    db_path = Path(db_path).resolve(strict=True)
    if case_limit < 2 or query_limit < 1:
        raise ValueError("Use at least two candidate cases and one query")
    output = db_path.parent / "comparison"
    output.mkdir(exist_ok=True)
    signature = {"corpus": corpus_fingerprint, "benchmark": file_hash(db_path.parent / "benchmark_manifest.json"),
                 "case_limit": case_limit, "query_limit": query_limit, "seed": seed,
                 "code": _completion_code()}
    saved = _cached_json(output / "scope.json", signature)
    if saved is not None:
        if saved["fingerprint"] != _digest({k: v for k, v in saved.items() if k != "fingerprint"}):
            raise ValueError("Comparison scope was edited")
        return saved
    if (output / "heldout_lock.json").exists():
        raise ValueError("Held-out evaluation has started; do not replace its scope")
    with closing(sqlite3.connect(db_path.as_uri() + "?mode=ro", uri=True)) as connection:
        ids = [r[0] for r in connection.execute("SELECT question_id FROM cases ORDER BY question_id")]
        total_queries = dict(connection.execute("SELECT split,count(*) FROM benchmark GROUP BY split"))
    selected = sorted(random.Random(seed).sample(ids, min(case_limit, len(ids))))
    scope = {"signature": signature, "candidate_ids": selected, "total_cases": len(ids),
             "coverage": {}, "query_ids": {}}
    for role in ("development", "heldout"):
        sampled, coverage = _sample_role(db_path, role, selected, query_limit, seed)
        if not sampled:
            raise ValueError(f"No eligible {role} queries in this candidate corpus")
        coverage["benchmark_queries"] = total_queries[role]
        scope["query_ids"][role] = sampled
        scope["coverage"][role] = coverage
    scope["fingerprint"] = _digest(scope)
    _save_json(output / "scope.json", scope)
    with (output / "query_assignments.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["QueryId", "Split"])
        for role, query_ids in scope["query_ids"].items():
            writer.writerows((qid, role) for qid in query_ids)
    return scope


def comparison_queries(db_path, scope, role="development"):
    if role not in ("development", "heldout"):
        raise ValueError("Unknown query role")
    selected = set(scope["query_ids"][role])
    candidates = set(scope["candidate_ids"])
    with closing(sqlite3.connect(Path(db_path).as_uri() + "?mode=ro", uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute("SELECT * FROM benchmark WHERE split=? ORDER BY query_id", (role,)).fetchall()
    queries = []
    for row in rows:
        if row["query_id"] in selected:
            item = dict(row)
            item["relevant_ids"] = (set(json.loads(item["relevant_ids"])) & candidates) - {item["query_id"]}
            if not item["relevant_ids"]:
                raise ValueError("A selected query no longer has an available target")
            queries.append(item)
    assert len(queries) == len(selected)
    assert len({q["group_id"] for q in queries}) == len(queries)
    return queries


def _case_rows(db_path, question_ids, chunks=False):
    ids = sorted(set(question_ids))
    with closing(sqlite3.connect(Path(db_path).as_uri() + "?mode=ro", uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        for start in range(0, len(ids), 500):
            batch = ids[start:start + 500]
            placeholders = ",".join("?" for _ in batch)
            columns, join, order = "c.*", "", "c.question_id"
            if chunks:
                columns += ",u.part,u.start,u.end"
                join = "JOIN chunks u USING(question_id)"
                order += ",u.unit_id"
            for row in connection.execute(
                    f"SELECT {columns} FROM cases c {join} WHERE c.question_id IN ({placeholders}) ORDER BY {order}", batch):
                yield dict(row)


def build_comparison_indexes(db_path, scope, model):
    """Build both methods over exactly the same cases; stream dense batches."""
    import faiss
    from importlib.metadata import version

    output = Path(db_path).parent / "comparison"
    signature = {"scope": scope["fingerprint"], "code": _completion_code(), "model": MODEL_ID,
                 "revision": MODEL_REVISION, "max_tokens": model.max_seq_length,
                 "sentence_transformers": version("sentence-transformers"), "sklearn": version("scikit-learn"),
                 "faiss": version("faiss-cpu")}
    manifest_path = output / "indexes.json"
    manifest = _cached_json(manifest_path, signature)
    if manifest is not None:
        for name, digest in manifest["files"].items():
            if file_hash(output / name) != digest:
                raise ValueError(f"Index file changed: {name}")
        bm25 = joblib.load(output / "bm25.joblib")
        dense = DensePilot(model, faiss.read_index(str(output / "dense.faiss")),
                           joblib.load(output / "dense_ids.joblib"))
        return {"bm25": bm25, "dense": dense, "manifest": manifest}
    if (output / "heldout_lock.json").exists():
        raise ValueError("Missing indexes after held-out evaluation; do not rebuild silently")
    started = time.perf_counter()
    rows = list(_case_rows(db_path, scope["candidate_ids"]))
    ids = [r["question_id"] for r in rows]
    assert ids == scope["candidate_ids"]
    bm25 = BM25Index(ids, ("\n\n".join([r["title"], r["question_text"], r["answer_text"]]) for r in rows))
    joblib.dump(bm25, output / "bm25.joblib", compress=3)
    del rows
    lexical_seconds = time.perf_counter() - started
    started = time.perf_counter()
    index = faiss.IndexFlatIP(model.get_sentence_embedding_dimension())
    texts, question_ids, pending_ids = [], [], []

    def add_batch():
        vectors = model.encode(texts, batch_size=16, normalize_embeddings=True, show_progress_bar=False)
        index.add(np.asarray(vectors, dtype=np.float32))
        question_ids.extend(pending_ids)
        texts.clear()
        pending_ids.clear()

    case_count, previous = 0, None
    for row in _case_rows(db_path, ids, chunks=True):
        if row["question_id"] != previous:
            case_count += 1
            previous = row["question_id"]
            if case_count % 1000 == 0:
                print(f"Dense cases: {case_count}/{len(ids)}", flush=True)
        source = row["question_text"] if row["part"] == "question" else row["answer_text"]
        for view in embedding_views(row["title"], source[row["start"]:row["end"]], model.tokenizer, model.max_seq_length):
            texts.append(view)
            pending_ids.append(row["question_id"])
            if len(texts) == 256:
                add_batch()
    if texts:
        add_batch()
    assert set(question_ids) == set(ids) and index.ntotal == len(question_ids)
    faiss.write_index(index, str(output / "dense.faiss"))
    joblib.dump(question_ids, output / "dense_ids.joblib")
    files = {name: file_hash(output / name) for name in ["bm25.joblib", "dense.faiss", "dense_ids.joblib"]}
    manifest = {"signature": signature, "cases": len(ids), "embedding_views": index.ntotal,
                "dimensions": index.d, "bm25_build_seconds": lexical_seconds,
                "dense_build_seconds": time.perf_counter() - started,
                "files": files, "index_bytes": sum((output / name).stat().st_size for name in files)}
    _save_json(manifest_path, manifest)
    return {"bm25": bm25, "dense": DensePilot(model, index, question_ids), "manifest": manifest}


def fuse_rankings(lexical, dense, dense_weight=0.5, rrf_k=60, k=10):
    if not 0 <= dense_weight <= 1 or rrf_k < 1 or k < 1:
        raise ValueError("Invalid rank-fusion settings")
    scores = defaultdict(float)
    for hits, weight in [(lexical, 1 - dense_weight), (dense, dense_weight)]:
        if weight == 0:
            continue
        for rank, (qid, _) in enumerate(hits, 1):
            scores[qid] += weight / (rrf_k + rank)
    return sorted(scores.items(), key=lambda pair: (-pair[1], int(pair[0])))[:k]


def ranking_metrics(ranked_ids, relevant):
    relevant = set(relevant)
    if not relevant or len(ranked_ids) != len(set(ranked_ids)):
        raise ValueError("Metrics require relevance labels and unique question IDs")
    result = {f"Recall@{k}": len(set(ranked_ids[:k]) & relevant) / len(relevant) for k in (1, 5, 10)}
    result["MRR@10"] = next((1 / rank for rank, qid in enumerate(ranked_ids[:10], 1) if qid in relevant), 0.0)
    return result


def compare_retrievers(db_path, scope, indexes, depth=50, rrf_k=60, max_query_ms=1000):
    """Development-only selection; reuse saved rankings on repeat execution."""
    output = Path(db_path).parent / "comparison"
    signature = {"scope": scope["fingerprint"], "indexes": _digest(indexes["manifest"]),
                 "depth": depth, "rrf_k": rrf_k, "max_query_ms": max_query_ms, "code": _completion_code()}
    cached = _cached_json(output / "development.json", signature)
    if cached is not None:
        return cached
    if (output / "heldout_lock.json").exists():
        raise ValueError("Development choices cannot change after held-out evaluation")
    if depth < 10:
        raise ValueError("Candidate depth must be at least 10")
    records = {name: [] for name in METHODS}
    for query in comparison_queries(db_path, scope):
        hits, milliseconds = {}, {}
        for name in ["bm25", "dense"]:
            started = time.perf_counter()
            hits[name] = indexes[name].search(query["query_text"], query["query_id"], depth)
            milliseconds[name] = 1000 * (time.perf_counter() - started)
        for name in METHODS:
            if name.startswith("hybrid_"):
                started = time.perf_counter()
                ranked = fuse_rankings(hits["bm25"], hits["dense"], float(name.split("_")[1]), rrf_k, depth)
                elapsed = sum(milliseconds.values()) + 1000 * (time.perf_counter() - started)
            else:
                ranked, elapsed = hits[name], milliseconds[name]
            assert query["query_id"] not in [qid for qid, _ in ranked]
            metrics = ranking_metrics([qid for qid, _ in ranked], query["relevant_ids"])
            records[name].append({"query_id": query["query_id"], "hits": ranked, "milliseconds": elapsed, **metrics})
    summary = []
    for name, rows in records.items():
        mean = {key: float(np.mean([r[key] for r in rows])) for key in ["Recall@1", "Recall@5", "Recall@10", "MRR@10"]}
        summary.append({"method": name, "queries": len(rows), "mean_query_ms": float(np.mean([r["milliseconds"] for r in rows])), **mean})
    eligible = [r for r in summary if r["mean_query_ms"] <= max_query_ms]
    if not eligible:
        raise ValueError("No method meets the declared latency budget; review development settings before testing")
    winner = max(eligible, key=lambda r: (r["MRR@10"], r["Recall@10"], -METHODS.index(r["method"])))
    report = {"signature": signature, "summary": summary, "selected_method": winner["method"], "records": records}
    _save_json(output / "development.json", report)
    return report


def score_diagnostics(hits):
    if not hits:
        return 0.0, 0.0
    score = float(hits[0][1])
    margin = float((score - hits[1][1]) / max(abs(score), 1e-12)) if len(hits) > 1 else 1.0
    return score, margin


def freeze_retriever(db_path, scope, indexes, development, review_quantile=0.10):
    """Freeze a development-derived score warning, not an answerability probability."""
    if not 0 <= review_quantile < 1:
        raise ValueError("review_quantile must be in [0, 1)")
    output = Path(db_path).parent / "comparison"
    signature = {"scope": scope["fingerprint"], "indexes": _digest(indexes["manifest"]),
                 "development": _digest(development), "review_quantile": review_quantile,
                 "code": _completion_code()}
    existing = _cached_json(output / "retriever.json", signature)
    if existing is not None:
        if existing["fingerprint"] != _digest({k: v for k, v in existing.items() if k != "fingerprint"}):
            raise ValueError("Frozen retriever was edited")
        return existing
    if (output / "heldout_lock.json").exists():
        raise ValueError("Cannot replace the frozen retriever after test evaluation")
    method = development["selected_method"]
    diagnostics = [score_diagnostics(r["hits"]) for r in development["records"][method] if r["hits"]]
    if not diagnostics:
        raise ValueError("No development results to derive a weak-evidence signal")
    scores, margins = np.asarray(diagnostics).T
    config = {"signature": signature, "method": method, "depth": development["signature"]["depth"],
              "rrf_k": development["signature"]["rrf_k"], "model": MODEL_ID, "revision": MODEL_REVISION,
              "weak_evidence": {"score_floor": float(np.quantile(scores, review_quantile)),
                                "margin_floor": float(np.quantile(margins, review_quantile))}}
    config["fingerprint"] = _digest(config)
    _save_json(output / "retriever.json", config)
    return config


class RetrievalEngine:
    def __init__(self, db_path, config, bm25=None, dense=None):
        self.db_path, self.config, self.bm25, self.dense = Path(db_path), config, bm25, dense

    def search(self, query, exclude_question_id=None, k=10):
        if not isinstance(query, str) or not query.strip() or not 1 <= k <= self.config["depth"]:
            raise ValueError("Provide nonempty text and k within the saved retrieval depth")
        method = self.config["method"]
        if method == "bm25":
            return self.bm25.search(query, exclude_question_id, k)
        if method == "dense":
            return self.dense.search(query, exclude_question_id, k)
        depth = self.config["depth"]
        return fuse_rankings(self.bm25.search(query, exclude_question_id, depth),
                             self.dense.search(query, exclude_question_id, depth),
                             float(method.split("_")[1]), self.config["rrf_k"], k)

    def evidence(self, question_ids):
        records = {r["question_id"]: r for r in _case_rows(self.db_path, question_ids)}
        with closing(sqlite3.connect(self.db_path.as_uri() + "?mode=ro", uri=True)) as connection:
            for record in records.values():
                metadata = json.loads(record.pop("metadata"))
                for side, id_key in [("question", "question_id"), ("answer", "answer_id")]:
                    owner = record.pop(side + "_owner")
                    row = connection.execute("SELECT name FROM authors WHERE owner_id=?", (owner,)).fetchone()
                    if row is None:
                        row = connection.execute("SELECT name FROM post_authors WHERE post_id=?", (record[id_key],)).fetchone()
                    metadata[side]["author_name"] = row[0] if row else None
                record["sources"] = metadata
        return records

    def retrieve(self, title, body="", k=5):
        if not isinstance(title, str) or not isinstance(body, str):
            raise TypeError("Title and body must be strings")
        query = (title.strip() + "\n\n" + body.strip()).strip()
        if not 1 <= k <= self.config["depth"]:
            raise ValueError("k is outside the saved retrieval depth")
        hits = self.search(query, k=max(2, k))
        score, margin = score_diagnostics(hits)
        policy = self.config["weak_evidence"]
        weak = not hits or score <= policy["score_floor"] or margin <= policy["margin_floor"]
        records = self.evidence([qid for qid, _ in hits[:k]])
        evidence = [{**records[qid], "rank": i, "score": float(value)} for i, (qid, value) in enumerate(hits[:k], 1)]
        return {"evidence": evidence, "weak_evidence": bool(weak),
                "top_score": score, "relative_margin": margin, "method": self.config["method"]}


def load_retriever(root, model=None):
    """Load frozen local artifacts for later integration without refitting."""
    import faiss
    from sentence_transformers import SentenceTransformer

    root = Path(root).resolve()
    db_path = root / "data/processed/retrieval/evidence.sqlite"
    output = db_path.parent / "comparison"
    config = json.loads((output / "retriever.json").read_text(encoding="utf-8"))
    if config["fingerprint"] != _digest({k: v for k, v in config.items() if k != "fingerprint"}):
        raise ValueError("Frozen retriever was edited")
    if config["signature"]["code"] != _completion_code():
        raise ValueError("Retriever code changed since development; review provenance before use")
    manifest = json.loads((output / "indexes.json").read_text(encoding="utf-8"))
    if _digest(manifest) != config["signature"]["indexes"]:
        raise ValueError("Index metadata differs from the frozen retriever")
    for name, digest in manifest["files"].items():
        if file_hash(output / name) != digest:
            raise ValueError(f"Index content differs: {name}")
    bm25 = dense = None
    method = config["method"]
    if method != "dense":
        bm25 = joblib.load(output / "bm25.joblib")
    if method != "bm25":
        if model is None:
            model = SentenceTransformer(config["model"], revision=config["revision"],
                cache_folder=str(root / "data/processed/embedding_cache"), device="cpu", local_files_only=True)
        dense = DensePilot(model, faiss.read_index(str(output / "dense.faiss")), joblib.load(output / "dense_ids.joblib"))
    return RetrievalEngine(db_path, config, bm25, dense)


def retrieval_diagnostics(engine, queries, ranked_rows, as_of="2026-10-23", limit=10):
    """Development examples, score warnings and source age; no test selection."""
    from datetime import datetime, timezone
    if any(q["split"] != "development" for q in queries):
        raise ValueError("Inspect development queries only before final evaluation")
    by_id = {q["query_id"]: q for q in queries}
    ids = {qid for row in ranked_rows for qid, _ in row["hits"][:5]}
    records = engine.evidence(ids)
    reference = datetime.fromisoformat(as_of).replace(tzinfo=timezone.utc)
    failures, ages, warnings = [], [], []
    policy = engine.config["weak_evidence"]
    for row in ranked_rows:
        query = by_id[row["query_id"]]
        hits = row["hits"]
        score, margin = score_diagnostics(hits)
        weak = not hits or score <= policy["score_floor"] or margin <= policy["margin_floor"]
        warnings.append({"QueryId": query["query_id"], "TopScore": score, "RelativeMargin": margin,
                         "WeakEvidence": bool(weak), "KnownTargetAt1": bool(hits and hits[0][0] in query["relevant_ids"])})
        for rank, (qid, _) in enumerate(hits[:5], 1):
            created = records[qid]["sources"]["answer"]["creation_date"]
            date = datetime.fromisoformat(created.replace("Z", "+00:00"))
            if date.tzinfo is None:
                date = date.replace(tzinfo=timezone.utc)
            ages.append({"QueryId": query["query_id"], "QuestionId": qid, "Rank": rank,
                         "AnswerDate": created, "AgeYears": (reference - date).days / 365.25})
        if hits and hits[0][0] not in query["relevant_ids"]:
            candidate = records[hits[0][0]]
            failures.append({"QueryId": query["query_id"], "QueryTitle": query["title"],
                             "QueryText": query["query_text"], "KnownTargets": sorted(query["relevant_ids"]),
                             "RetrievedId": candidate["question_id"], "RetrievedTitle": candidate["title"],
                             "AnswerText": candidate["answer_text"], "Sources": candidate["sources"],
                             "TopScore": score, "WeakEvidence": bool(weak)})
    failures.sort(key=lambda r: (-r["TopScore"], int(r["QueryId"])))
    return {"failures": failures[:limit], "source_ages": ages, "warnings": warnings}


def evaluate_heldout(engine, scope):
    """Evaluate the frozen retriever once; repeat notebook runs load the report."""
    output = engine.db_path.parent / "comparison"
    signature = {"retriever": engine.config["fingerprint"], "scope": scope["fingerprint"], "code": _completion_code()}
    if engine.config["signature"]["scope"] != scope["fingerprint"]:
        raise ValueError("Test scope differs from the frozen retriever")
    lock = _cached_json(output / "heldout_lock.json", signature)
    cached = _cached_json(output / "heldout_metrics.json", signature)
    if cached is not None:
        if lock is None:
            raise ValueError("Held-out report has no evaluation lock")
        return cached
    if lock is None:
        _save_json(output / "heldout_lock.json", {"signature": signature})
    rows = []
    for query in comparison_queries(engine.db_path, scope, "heldout"):
        started = time.perf_counter()
        hits = engine.search(query["query_text"], query["query_id"], 10)
        elapsed = 1000 * (time.perf_counter() - started)
        ids = [qid for qid, _ in hits]
        assert query["query_id"] not in ids
        score, margin = score_diagnostics(hits)
        policy = engine.config["weak_evidence"]
        rows.append({"QueryId": query["query_id"], "milliseconds": elapsed,
                     "WeakEvidence": bool(not hits or score <= policy["score_floor"] or margin <= policy["margin_floor"]),
                     **ranking_metrics(ids, query["relevant_ids"])})
    metrics = {name: float(np.mean([r[name] for r in rows])) for name in ["Recall@1", "Recall@5", "Recall@10", "MRR@10", "milliseconds", "WeakEvidence"]}
    report = {"signature": signature, "method": engine.config["method"], "queries": len(rows),
              "candidate_cases": len(scope["candidate_ids"]), "metrics": metrics, "coverage": scope["coverage"], "rows": rows}
    _save_json(output / "heldout_metrics.json", report)
    return report
