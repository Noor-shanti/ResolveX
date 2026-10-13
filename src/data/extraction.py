"""Streaming extraction of Ask Ubuntu questions and answers from Posts.xml."""

import csv
from pathlib import Path
import xml.etree.ElementTree as ET

QUESTION_FIELDS = (
    "Id", "PostTypeId", "AcceptedAnswerId", "CreationDate", "Score",
    "ViewCount", "Body", "OwnerUserId", "LastActivityDate", "Title",
    "Tags", "AnswerCount", "CommentCount", "ContentLicense",
)
ANSWER_FIELDS = (
    "Id", "PostTypeId", "ParentId", "CreationDate", "Score", "Body",
    "OwnerUserId", "LastActivityDate", "CommentCount", "ContentLicense",
)


def preview_posts(xml_path, limit=5):
    """Preview post attributes without loading the entire XML dump."""
    if limit < 1:
        return []
    preview = []
    for _, element in ET.iterparse(xml_path, events=("end",)):
        if element.tag == "row":
            preview.append(dict(element.attrib))
            element.clear()
            if len(preview) >= limit:
                break
    return preview


def extract_posts(xml_path, questions_path, answers_path):
    """Export raw post bodies and metadata without lossy HTML cleaning.

    Output is written once; existing outputs are never silently overwritten.
    """
    xml_path = Path(xml_path)
    questions_path = Path(questions_path)
    answers_path = Path(answers_path)
    if not xml_path.is_file():
        raise FileNotFoundError(xml_path)
    if questions_path.resolve() == answers_path.resolve():
        raise ValueError("Question and answer output paths must differ")
    if questions_path.exists() or answers_path.exists():
        raise FileExistsError("Output already exists; review it before re-extracting")
    questions_path.parent.mkdir(parents=True, exist_ok=True)
    answers_path.parent.mkdir(parents=True, exist_ok=True)
    counts = {"questions": 0, "answers": 0, "other_post_types": 0}
    try:
        with questions_path.open("x", newline="", encoding="utf-8") as qf, \
             answers_path.open("x", newline="", encoding="utf-8") as af:
            q_writer = csv.DictWriter(qf, fieldnames=QUESTION_FIELDS)
            a_writer = csv.DictWriter(af, fieldnames=ANSWER_FIELDS)
            q_writer.writeheader()
            a_writer.writeheader()
            for _, element in ET.iterparse(xml_path, events=("end",)):
                if element.tag != "row":
                    continue
                attrs = element.attrib
                kind = attrs.get("PostTypeId")
                if kind == "1":
                    q_writer.writerow({key: attrs.get(key, "") for key in QUESTION_FIELDS})
                    counts["questions"] += 1
                elif kind == "2":
                    a_writer.writerow({key: attrs.get(key, "") for key in ANSWER_FIELDS})
                    counts["answers"] += 1
                else:
                    counts["other_post_types"] += 1
                element.clear()
    except BaseException:
        questions_path.unlink(missing_ok=True)
        answers_path.unlink(missing_ok=True)
        raise
    return counts
