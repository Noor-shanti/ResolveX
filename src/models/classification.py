"""Baseline classification and duplicate-aware splitting for ResolveX M3."""

import hashlib
import re
import warnings

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.exceptions import ConvergenceWarning
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax


def load_dataset(path):
    """Load candidates; tags remain metadata and never enter model text."""
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"Id", "Title", "BodyText", "Label"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing columns: {required - set(frame.columns)}")
    if frame.empty or frame["Id"].duplicated().any():
        raise ValueError("Expected nonempty data with unique question IDs")
    if frame["Label"].eq("").any():
        raise ValueError("Missing labels")
    frame["Text"] = frame["Title"] + "\n\n" + frame["BodyText"]
    if frame["Text"].str.strip().eq("").any():
        raise ValueError("Empty model input")
    return frame


def normalize_text(text):
    """Normalize whitespace and case only for duplicate detection."""
    return re.sub(r"\s+", " ", text).strip().casefold()


def make_splits(frame, duplicate_path, seed=42, reviewed_ids=(), excluded_ids=()):
    """Group linked questions and exact normalized text/title matches.

    Full duplicate links include intermediate IDs outside the ML dataset.
    Identical titles of at least 30 characters are conservatively grouped.
    GroupShuffleSplit targets 70/15/15 by groups, not by individual rows;
    actual row and class proportions must be inspected in the notebook.
    Reviewed components are reserved for training. Excluded review rows are
    marked after grouping and are absent from all active partitions.
    This does not detect every semantic near-duplicate.
    """
    reviewed_ids, excluded_ids = set(reviewed_ids), set(excluded_ids)
    if not excluded_ids.issubset(reviewed_ids):
        raise ValueError("Excluded review rows must be among reviewed IDs")
    if not reviewed_ids.issubset(set(frame["Id"])):
        raise ValueError("Unknown reviewed IDs")
    parent = {}

    def find(item):
        parent.setdefault(item, item)
        root = item
        while parent[root] != root:
            root = parent[root]
        while parent[item] != item:
            next_item = parent[item]
            parent[item] = root
            item = next_item
        return root

    def union(left, right):
        left, right = find(left), find(right)
        if left != right:
            parent[max(left, right)] = min(left, right)

    links = pd.read_csv(duplicate_path, dtype=str, keep_default_na=False)
    for left, right in links[["QuestionId", "RelatedQuestionId"]].itertuples(index=False, name=None):
        union(left, right)

    text_seen, title_seen = {}, {}
    for question_id, title, text in frame[["Id", "Title", "Text"]].itertuples(index=False, name=None):
        normalized = normalize_text(text)
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        if digest in text_seen:
            union(question_id, text_seen[digest])
        else:
            text_seen[digest] = question_id
        normalized_title = normalize_text(title)
        if len(normalized_title) >= 30:
            if normalized_title in title_seen:
                union(question_id, title_seen[normalized_title])
            else:
                title_seen[normalized_title] = question_id

    groups = frame["Id"].map(find)
    # Compute all components before excluding rows, preserving transitive links.
    protected_groups = set(groups.loc[frame["Id"].isin(reviewed_ids)])
    available = np.flatnonzero(~groups.isin(protected_groups))
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    _, remaining = next(splitter.split(frame.iloc[available], groups=groups.iloc[available]))
    remaining = available[remaining]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    validation, test = next(splitter.split(frame.iloc[remaining], groups=groups.iloc[remaining]))
    assignments = pd.DataFrame({"Id": frame["Id"], "GroupId": groups, "Split": "train"})
    assignments.loc[frame.index[remaining[validation]], "Split"] = "validation"
    assignments.loc[frame.index[remaining[test]], "Split"] = "test"
    assignments.loc[frame["Id"].isin(excluded_ids), "Split"] = "excluded"
    active = assignments.loc[assignments["Split"].ne("excluded")]
    if active.groupby("GroupId")["Split"].nunique().max() != 1:
        raise AssertionError("Duplicate group crosses active splits")
    if not assignments.loc[frame["Id"].isin(reviewed_ids - excluded_ids), "Split"].eq("train").all():
        raise AssertionError("Reviewed questions must remain in training")
    for split in ("train", "validation", "test"):
        labels = set(frame.loc[assignments["Split"].eq(split), "Label"])
        if labels != set(frame["Label"]):
            raise ValueError(f"Missing categories in {split}")
    return assignments


def build_pipeline(class_weight=None, C=1.0):
    """Initial settings are fixed; later selection uses validation only."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2), min_df=3, max_df=0.98,
            max_features=50000, sublinear_tf=True, dtype=np.float64,
        )),
        ("classifier", LogisticRegression(
            C=C, class_weight=class_weight, solver="lbfgs",
            max_iter=300, random_state=42,
        )),
    ])


def score_predictions(labels, predictions):
    return {
        "Accuracy": accuracy_score(labels, predictions),
        "Macro F1": f1_score(labels, predictions, average="macro", zero_division=0),
        "Weighted F1": f1_score(labels, predictions, average="weighted", zero_division=0),
    }


def sample_label_review(frame, per_category=5, seed=42):
    """Draw a repeatable review sample; reserve reviewed groups for training later."""
    return pd.concat([
        group.sample(n=min(per_category, len(group)), random_state=seed)
        for _, group in frame.groupby("Label", sort=True)
    ]).reset_index(drop=True)


def audit_near_duplicates(frame, assignments, sample_per_pair=100, seed=42):
    """Sample lexical similarity across each split pair without fitting features.

    Hashing is stateless: no training or test vocabulary is learned. Compare
    complete inputs, not labels. High cosine scores are candidates for review,
    not proof of duplication. This sample cannot guarantee semantic isolation.
    """
    vectorizer = HashingVectorizer(
        n_features=2**18, ngram_range=(1, 2), alternate_sign=False,
        norm="l2", dtype=np.float32,
    )
    active = assignments["Split"].isin(["train", "validation", "test"])
    data = frame.loc[active, ["Id", "Title", "Text"]].copy().reset_index(drop=True)
    data["Split"] = assignments.loc[active, "Split"].to_numpy()
    vectors = vectorizer.transform(data["Text"])
    matches = []
    for left, right in [("validation", "train"), ("test", "train"), ("test", "validation")]:
        queries = data.loc[data["Split"].eq(left)].sample(
            n=min(sample_per_pair, int(data["Split"].eq(left).sum())), random_state=seed,
        )
        targets = data.loc[data["Split"].eq(right)]
        target_vectors = vectors[targets.index]
        for start in range(0, len(queries), 10):
            batch = queries.iloc[start:start + 10]
            scores = linear_kernel(vectors[batch.index], target_vectors)
            for position, (_, query) in enumerate(batch.iterrows()):
                nearest = targets.iloc[int(scores[position].argmax())]
                matches.append({
                    "QueryId": query["Id"], "TargetId": nearest["Id"],
                    "QuerySplit": left, "TargetSplit": right,
                    "Similarity": float(scores[position].max()),
                    "QueryTitle": query["Title"], "TargetTitle": nearest["Title"],
                })
    return pd.DataFrame(matches).sort_values("Similarity", ascending=False).reset_index(drop=True)


def validation_roles(assignments, seed=42):
    """Reserve disjoint validation groups for selection/calibration/policy.

    Validation remains development data: its scores are not final evaluation.
    Target group shares are 40/30/30, with actual row counts reported separately.
    """
    positions = np.flatnonzero(assignments["Split"].eq("validation"))
    groups = assignments.iloc[positions]["GroupId"]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.60, random_state=seed)
    selection, rest = next(splitter.split(positions, groups=groups))
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    calibration, policy = next(splitter.split(rest, groups=groups.iloc[rest]))
    roles = pd.Series("", index=assignments.index, name="ValidationRole")
    roles.iloc[positions[selection]] = "selection"
    roles.iloc[positions[rest[calibration]]] = "calibration"
    roles.iloc[positions[rest[policy]]] = "policy"
    check = assignments.loc[roles.ne(""), ["GroupId"]].assign(Role=roles)
    if check.groupby("GroupId")["Role"].nunique().max() != 1:
        raise AssertionError("Validation groups cross development roles")
    return roles


def fit_checked(estimator, inputs, labels):
    """Fit without silently accepting iteration-limit failures."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        estimator.fit(inputs, labels)
    for warning in caught:
        if issubclass(warning.category, ConvergenceWarning):
            raise RuntimeError("Model did not converge: " + str(warning.message))
        warnings.warn(str(warning.message), warning.category, stacklevel=2)
    return estimator


def fit_cached(train, path, C=1.0, class_weight=None):
    """Reuse only a cache matching training rows, settings, code and version."""
    from pathlib import Path
    import joblib
    import sklearn

    rows = pd.util.hash_pandas_object(train[["Id", "Text", "Label"]], index=False)
    digest = hashlib.sha256(rows.to_numpy().tobytes())
    digest.update(Path(__file__).read_bytes())
    digest.update(f"{float(C)}:{class_weight}:{sklearn.__version__}".encode())
    signature = digest.hexdigest()
    path = Path(path)
    if path.exists():
        cached = joblib.load(path)
        if isinstance(cached, dict) and cached.get("training_signature") == signature:
            return cached["pipeline"], "loaded matching cache"
    pipeline = fit_checked(build_pipeline(class_weight=class_weight, C=C), train["Text"], train["Label"])
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "training_signature": signature}, path)
    return pipeline, "fitted on training rows"


def fit_temperature(logits, labels, classes):
    """Fit one positive temperature on calibration-only multiclass logits.

    Temperature scaling leaves the predicted category unchanged. It adjusts
    softmax concentration, not semantic label noise or out-of-domain detection.
    """
    targets = pd.Index(classes).get_indexer(labels)
    if (targets < 0).any() or len(targets) == 0:
        raise ValueError("Calibration labels must be known and nonempty")

    def objective(log_temperature):
        scaled = logits / np.exp(log_temperature)
        return float(np.mean(logsumexp(scaled, axis=1) - scaled[np.arange(len(targets)), targets]))

    result = minimize_scalar(objective, bounds=(-3.0, 3.0), method="bounded")
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError("Temperature optimization failed")
    if abs(result.x) > 2.99:
        raise RuntimeError("Temperature reached search boundary; investigate calibration")
    return float(np.exp(result.x))


def confidence_metrics(labels, probabilities, classes, bins=10):
    """NLL, multiclass Brier score and equal-width top-label ECE."""
    targets = pd.Index(classes).get_indexer(labels)
    if (targets < 0).any() or len(targets) == 0:
        raise ValueError("Expected nonempty known labels")
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == targets
    one_hot = np.eye(len(classes))[targets]
    bucket = np.minimum((confidence * bins).astype(int), bins - 1)
    reliability = []
    ece = 0.0
    for index in range(bins):
        mask = bucket == index
        if mask.any():
            agreement = float(correct[mask].mean())
            mean_confidence = float(confidence[mask].mean())
            ece += mask.mean() * abs(agreement - mean_confidence)
            reliability.append({"Bin": index, "Count": int(mask.sum()),
                                "Mean confidence": mean_confidence,
                                "Weak-label agreement": agreement})
    metrics = {
        "NLL": float(-np.log(np.clip(probabilities[np.arange(len(targets)), targets], 1e-15, 1)).mean()),
        "Brier": float(np.square(probabilities - one_hot).sum(axis=1).mean()),
        "ECE": float(ece),
    }
    return metrics, pd.DataFrame(reliability)


def choose_review_threshold(labels, probabilities, classes, target=0.95, min_accepted=200):
    """Maximize retained coverage meeting a policy-validation agreement target.

    The target is an explicit prototype operating choice against weak labels,
    not a safety guarantee or evidence of correctness. Thresholds come from
    observed confidence values; tied values are accepted or rejected together.
    """
    if not 0 < target <= 1 or min_accepted < 1:
        raise ValueError("Invalid policy target or minimum support")
    targets = pd.Index(classes).get_indexer(labels)
    if (targets < 0).any() or len(targets) == 0:
        raise ValueError("Expected nonempty known policy labels")
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == targets
    order = np.argsort(-confidence, kind="stable")
    sorted_confidence = confidence[order]
    ends = np.flatnonzero(np.r_[sorted_confidence[:-1] != sorted_confidence[1:], True])
    counts = ends + 1
    agreement = np.cumsum(correct[order])[ends] / counts
    eligible = (counts >= min_accepted) & (agreement >= target)
    if not eligible.any():
        return {"threshold": None, "target_agreement": target, "min_accepted": min_accepted,
                "accepted": 0, "coverage": 0.0, "agreement": None}
    best = np.flatnonzero(eligible)[-1]
    return {"threshold": float(sorted_confidence[ends[best]]), "target_agreement": target,
            "min_accepted": min_accepted, "accepted": int(counts[best]),
            "coverage": float(counts[best] / len(labels)), "agreement": float(agreement[best])}


def predict_ticket(artifact, title, body=""):
    """Return a candidate category, calibrated confidence and review flag.

    High confidence does not certify the domain or technical correctness.
    The engineer retains final authority, including for accepted predictions.
    """
    if not isinstance(title, str) or not isinstance(body, str):
        raise TypeError("Ticket title and body must be strings")
    if not (title.strip() or body.strip()):
        raise ValueError("Ticket text is empty")
    pipeline = artifact["pipeline"]
    probabilities = softmax(pipeline.decision_function([title + "\n\n" + body])
                            / artifact["temperature"], axis=1)[0]
    index = int(probabilities.argmax())
    confidence = float(probabilities[index])
    threshold = artifact["review_policy"]["threshold"]
    return {"category": str(pipeline.classes_[index]), "confidence": confidence,
            "human_review": bool(threshold is None or confidence < threshold)}


def sample_multicategory(questions_path, duplicate_path, single_ids, limit=100, seed=42):
    """Sample multi-category diagnostics outside known single-label components."""
    import csv
    import random
    from collections import defaultdict
    from src.data.preprocessing import extract_text
    from src.data.taxonomy import parse_tags, match_categories

    adjacency = defaultdict(set)
    links = pd.read_csv(duplicate_path, dtype=str)
    for left, right in links[["QuestionId", "RelatedQuestionId"]].itertuples(index=False, name=None):
        adjacency[left].add(right)
        adjacency[right].add(left)
    excluded = set(single_ids)
    pending = list(excluded & adjacency.keys())
    while pending:
        question = pending.pop()
        new_ids = adjacency[question] - excluded
        excluded.update(new_ids)
        pending.extend(new_ids)
    rng, sample, seen = random.Random(seed), [], 0
    with open(questions_path, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["Id"] in excluded:
                continue
            categories = match_categories(parse_tags(row.get("Tags", "")))
            if len(categories) < 2:
                continue
            seen += 1
            position = len(sample) if len(sample) < limit else rng.randrange(seen)
            if position < limit:
                record = {"Id": row["Id"], "Title": row["Title"],
                          "BodyText": extract_text(row["Body"]), "MatchedCategories": categories}
                if len(sample) < limit:
                    sample.append(record)
                else:
                    sample[position] = record
    return pd.DataFrame(sample)
