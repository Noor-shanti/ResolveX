import unittest

from src.retrieval import BM25Index, RetrievalEngine


class RetrievalExclusionTests(unittest.TestCase):
    def setUp(self):
        self.index = BM25Index(
            ["100", "200", "300", "400"],
            [
                "apt package repair dependencies",
                "apt package install dependencies",
                "apt package update dependencies",
                "apt package remove dependencies",
            ],
        )
        self.engine = RetrievalEngine(
            "unused.sqlite",
            {"method": "bm25", "depth": 4,
             "weak_evidence": {"score_floor": 0.0, "margin_floor": 0.0}},
            bm25=self.index,
        )
        self.engine.evidence = lambda ids: {
            qid: {"question_id": qid, "answer_text": "Evidence for " + qid}
            for qid in ids
        }

    def test_exclusion_is_applied_before_top_k_and_retrieval_backfills(self):
        result = self.engine.retrieve(
            "apt package dependencies", k=3, exclude_question_ids=["100"]
        )

        self.assertEqual([item["question_id"] for item in result["evidence"]], ["200", "300", "400"])
        self.assertEqual(len(result["evidence"]), 3)

    def test_multiple_exclusions_are_applied_before_top_k(self):
        result = self.engine.retrieve(
            "apt package dependencies", k=2, exclude_question_ids=["100", "200"]
        )

        self.assertEqual([item["question_id"] for item in result["evidence"]], ["300", "400"])


if __name__ == "__main__":
    unittest.main()
