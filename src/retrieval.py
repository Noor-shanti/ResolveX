"""Evidence preparation and initial retrieval experiments for ResolveX M4."""

from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random
import re
import sqlite3
from tempfile import TemporaryDirectory
import xml.etree.ElementTree as ET

import joblib
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from src.data.preprocessing import extract_text


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
    signature = {"inputs": inputs, "code": file_hash(__file__), "max_words": max_words}
    if db_path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["signature"] != signature:
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
    signature = {"corpus": corpus_fingerprint, "seed": seed, "code": file_hash(__file__),
                 "links": file_hash(root / "data/processed/duplicate_relationships.csv"),
                 "questions": file_hash(root / "data/interim/questions.csv"),
                 "answers": file_hash(root / "data/interim/answers.csv")}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["signature"] != signature:
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
    signature = {"corpus": corpus_fingerprint, "code": file_hash(__file__), "k1": 1.2, "b": 0.75}
    if path.exists():
        saved = joblib.load(path)
        if saved["signature"] != signature:
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
                 "code": file_hash(__file__), "sentence_transformers": version("sentence-transformers")}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["signature"] != signature:
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
