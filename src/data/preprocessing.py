"""Data preprocessing and dataset construction for ResolveX M2.

Reads from data/interim/ (questions.csv, answers.csv) and PostLinks.xml.
Writes final datasets to data/processed/.

Outputs
-------
ml_dataset.csv        Single-category questions for ML classification.
knowledge_base.jsonl  Questions and answers with preserved technical content for RAG.
retrieval_eval.csv    Duplicate-question pairs from PostLinks.xml for retrieval evaluation.
"""

import csv
from html.parser import HTMLParser
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from src.data.taxonomy import match_categories, parse_tags

# ---------------------------------------------------------------------------
# HTML cleaning
# ---------------------------------------------------------------------------

class _PostTextParser(HTMLParser):
    """Read markup without interpreting decoded command text as HTML."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.pre_depth = 0
        self.links = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "pre":
            self.parts.append("\n```\n")
            self.pre_depth += 1
        elif tag == "code" and not self.pre_depth:
            self.parts.append("`")
        elif tag == "a":
            self.links.append(attrs.get("href", ""))
        elif tag == "img":
            self.parts.append("[Image: " + attrs.get("alt", "") + "] (" + attrs.get("src", "") + ")")
        elif tag in {"p", "div", "br", "li", "blockquote", "h1", "h2", "h3", "tr"}:
            self.parts.append("\n")
        elif tag in {"td", "th"}:
            self.parts.append("\t")

    def handle_endtag(self, tag):
        if tag == "pre":
            self.pre_depth = max(0, self.pre_depth - 1)
            self.parts.append("\n```\n")
        elif tag == "code" and not self.pre_depth:
            self.parts.append("`")
        elif tag == "a" and self.links:
            url = self.links.pop()
            if url:
                self.parts.append(" (" + url + ")")
        elif tag in {"p", "div", "li", "blockquote", "h1", "h2", "h3", "tr"}:
            self.parts.append("\n")

    def handle_data(self, data):
        self.parts.append(data)


def extract_text(raw_html):
    """Preserve code whitespace, decoded entities, links and image descriptions.

    Raw HTML remains available in interim data and knowledge-base records.
    This text is a derived representation, not a replacement for the source.
    """
    parser = _PostTextParser()
    parser.feed(raw_html or "")
    parser.close()
    return "".join(parser.parts).strip()


# ---------------------------------------------------------------------------
# ML dataset
# ---------------------------------------------------------------------------

def build_ml_dataset(questions_path, output_path):
    """Build the single-category ML classification dataset.

    Each row contains the question Id, Title, cleaned Body text, Tags, and
    the assigned Label. Only single-category questions are included.

    Ask Ubuntu Tags are preserved as metadata but must not be used as
    classifier input features because they are the source of the label.

    Parameters
    ----------
    questions_path : str or Path
        Path to data/interim/questions.csv.
    output_path : str or Path
        Destination path for the ML dataset CSV.

    Returns
    -------
    int
        Number of rows written.
    """
    questions_path = Path(questions_path)
    output_path = Path(output_path)
    if not questions_path.is_file():
        raise FileNotFoundError(questions_path)
    if output_path.exists():
        raise FileExistsError(f"Output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fields = ["Id", "Title", "BodyText", "Tags", "Label", "CreationDate"]
    count = 0

    with (
        questions_path.open(encoding="utf-8", newline="") as qf,
        output_path.open("x", encoding="utf-8", newline="") as out,
    ):
        reader = csv.DictReader(qf)
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()

        for row in reader:
            tags = parse_tags(row.get("Tags", "") or "")
            categories = match_categories(tags)
            if len(categories) != 1:
                continue  # Keep only single-category questions.
            writer.writerow({
                "Id": row["Id"],
                "Title": (row.get("Title") or "").strip(),
                "BodyText": extract_text(row.get("Body") or ""),
                "Tags": row.get("Tags", ""),
                "Label": categories[0],
                "CreationDate": row.get("CreationDate", ""),
            })
            count += 1

    return count


# ---------------------------------------------------------------------------
# Knowledge base
# ---------------------------------------------------------------------------

def _is_valid_accepted_answer(question, answer_index):
    """Return True if the question's AcceptedAnswerId resolves to a real answer.

    A valid reference must exist and belong to the referring question.
    """
    accepted_id = (question.get("AcceptedAnswerId") or "").strip()
    if not accepted_id:
        return False
    answer = answer_index.get(accepted_id)
    if answer is None:
        return False  # Missing accepted answer.
    if answer.get("ParentId", "").strip() != question["Id"].strip():
        return False  # Mismatched ParentId.
    return True


def build_knowledge_base(questions_path, answers_path, output_path):
    """Build the technical knowledge base for RAG retrieval.

    Each JSON Lines record contains a question and its linked answers with
    cleaned body text, preserved technical metadata, and available attribution identifiers. Author display names require a
    later Users.xml join; these fields alone do not establish attribution compliance.

    Only questions with at least one linked answer are included.
    Invalid accepted-answer references are excluded and documented.

    Parameters
    ----------
    questions_path : str or Path
        Path to data/interim/questions.csv.
    answers_path : str or Path
        Path to data/interim/answers.csv.
    output_path : str or Path
        Destination path for the knowledge base JSONL file.

    Returns
    -------
    dict
        Summary counts: questions written, answers written, invalid accepted
        answers excluded.
    """
    questions_path = Path(questions_path)
    answers_path = Path(answers_path)
    output_path = Path(output_path)

    if not questions_path.is_file():
        raise FileNotFoundError(questions_path)
    if not answers_path.is_file():
        raise FileNotFoundError(answers_path)
    if output_path.exists():
        raise FileExistsError(f"Output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load all answers indexed by Id, grouped by ParentId.
    answer_index = {}   # Id -> answer row
    answers_by_parent = {}  # ParentId -> [answer rows]

    with answers_path.open(encoding="utf-8", newline="") as af:
        for row in csv.DictReader(af):
            answer_id = row["Id"].strip()
            parent_id = row.get("ParentId", "").strip()
            answer_index[answer_id] = row
            answers_by_parent.setdefault(parent_id, []).append(row)

    counts = {"questions": 0, "answers": 0, "invalid_accepted": 0,
              "missing_accepted": 0, "mismatched_accepted": 0,
              "answer_count_mismatches": 0, "questions_without_answers": 0}

    with (
        questions_path.open(encoding="utf-8", newline="") as qf,
        output_path.open("x", encoding="utf-8") as out,
    ):
        for question in csv.DictReader(qf):
            q_id = question["Id"].strip()
            linked_answers = answers_by_parent.get(q_id, [])
            accepted_id = (question.get("AcceptedAnswerId") or "").strip()
            accepted_valid = _is_valid_accepted_answer(question, answer_index)
            if accepted_id and not accepted_valid:
                counts["invalid_accepted"] += 1
                key = "missing_accepted" if accepted_id not in answer_index else "mismatched_accepted"
                counts[key] += 1
            recorded_count = int(question.get("AnswerCount") or 0)
            counts["answer_count_mismatches"] += recorded_count != len(linked_answers)
            if not linked_answers:
                counts["questions_without_answers"] += 1
                continue

            record = {
                "question_id": q_id,
                "source_url": f"https://askubuntu.com/questions/{q_id}",
                "body_html": question.get("Body", ""),
                "recorded_answer_count": recorded_count,
                "actual_answer_count": len(linked_answers),
                "recorded_accepted_answer_id": accepted_id or None,
                "title": (question.get("Title") or "").strip(),
                "body_text": extract_text(question.get("Body") or ""),
                "tags": parse_tags(question.get("Tags", "") or ""),
                "creation_date": question.get("CreationDate", ""),
                "score": question.get("Score", ""),
                "accepted_answer_id": accepted_id if accepted_valid else None,
                "content_license": question.get("ContentLicense", ""),
                "owner_user_id": question.get("OwnerUserId", ""),
                "answers": [],
            }

            for ans in linked_answers:
                record["answers"].append({
                    "answer_id": ans["Id"].strip(),
                    "source_url": f"https://askubuntu.com/a/{ans['Id'].strip()}",
                    "body_html": ans.get("Body", ""),
                    "body_text": extract_text(ans.get("Body") or ""),
                    "score": ans.get("Score", ""),
                    "creation_date": ans.get("CreationDate", ""),
                    "is_accepted": ans["Id"].strip() == accepted_id and accepted_valid,
                    "content_license": ans.get("ContentLicense", ""),
                    "owner_user_id": ans.get("OwnerUserId", ""),
                })
                counts["answers"] += 1

            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            counts["questions"] += 1

    return counts


# ---------------------------------------------------------------------------
# Retrieval evaluation data
# ---------------------------------------------------------------------------

def build_retrieval_eval(postlinks_path, questions_path, output_path, knowledge_base_path=None):
    """Build duplicate-question pairs for retrieval evaluation.

    Extracts LinkTypeId == 3 (Duplicate) links from PostLinks.xml and
    retains known queries and known targets (or KB targets when supplied).
    Without a KB path, returns full duplicate relationships for split grouping.
    Retrieval evaluation must exclude the query itself from ranked results.

    Parameters
    ----------
    postlinks_path : str or Path
        Path to data/raw/PostLinks.xml.
    questions_path : str or Path
        Path to data/interim/questions.csv (used to validate question IDs).
    output_path : str or Path
        Destination path for the retrieval evaluation CSV.

    Returns
    -------
    int
        Number of valid duplicate pairs written.
    """
    postlinks_path = Path(postlinks_path)
    questions_path = Path(questions_path)
    output_path = Path(output_path)

    if not postlinks_path.is_file():
        raise FileNotFoundError(postlinks_path)
    if not questions_path.is_file():
        raise FileNotFoundError(questions_path)
    if output_path.exists():
        raise FileExistsError(f"Output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load question IDs for validation.
    known_ids = set()
    with questions_path.open(encoding="utf-8", newline="") as qf:
        for row in csv.DictReader(qf):
            known_ids.add(row["Id"].strip())

    target_ids = known_ids
    if knowledge_base_path is not None:
        with Path(knowledge_base_path).open(encoding="utf-8") as handle:
            target_ids = {json.loads(line)["question_id"] for line in handle}
    seen = set()
    fields = ["QuestionId", "RelatedQuestionId"]
    count = 0

    with output_path.open("x", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()

        for _, element in ET.iterparse(postlinks_path, events=("end",)):
            if element.tag != "row":
                element.clear()
                continue
            # LinkTypeId 3 = Duplicate
            if element.attrib.get("LinkTypeId") != "3":
                element.clear()
                continue
            q_id = element.attrib.get("PostId", "").strip()
            related_id = element.attrib.get("RelatedPostId", "").strip()
            element.clear()
            if (q_id in known_ids and related_id in target_ids
                    and q_id != related_id and (q_id, related_id) not in seen):
                seen.add((q_id, related_id))
                writer.writerow({"QuestionId": q_id, "RelatedQuestionId": related_id})
                count += 1

    return count


def prepare_datasets(root, overwrite=False):
    """Build a consistent set in staging, then publish after successful builds.

    Existing outputs require explicit overwrite=True. Raw and interim files
    are never modified. The summary records observed counts, not targets.
    """
    from tempfile import TemporaryDirectory

    root = Path(root)
    output = root / "data" / "processed"
    output.mkdir(parents=True, exist_ok=True)
    names = ["ml_dataset.csv", "knowledge_base.jsonl", "retrieval_eval.csv",
             "duplicate_relationships.csv", "preparation_summary.json"]
    if not overwrite and any((output / name).exists() for name in names):
        raise FileExistsError("Prepared files exist; use overwrite=True to rebuild deliberately")
    questions = root / "data" / "interim" / "questions.csv"
    answers = root / "data" / "interim" / "answers.csv"
    links = root / "data" / "raw" / "PostLinks.xml"
    with TemporaryDirectory(prefix="preparation-", dir=output) as directory:
        staging = Path(directory)
        summary = {"ml_rows": build_ml_dataset(questions, staging / names[0])}
        summary["knowledge_base"] = build_knowledge_base(questions, answers, staging / names[1])
        summary["duplicate_relationships"] = build_retrieval_eval(links, questions, staging / names[3])
        summary["retrieval_pairs"] = build_retrieval_eval(links, questions, staging / names[2], staging / names[1])
        (staging / names[4]).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        for name in names:
            (staging / name).replace(output / name)
    return summary
