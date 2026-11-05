import unittest

import numpy as np

from src.official_docs import OfficialDocsRetriever


class _Model:
    def encode(self, texts, **_kwargs):
        return np.asarray([[1.0, 0.0]], dtype="float32")


class _Index:
    ntotal = 3

    def search(self, _vector, _count):
        return (
            np.asarray([[0.92, 0.81, 0.72]], dtype="float32"),
            np.asarray([[0, 1, 2]], dtype="int64"),
        )


class OfficialDocsLaneTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            {
                "source_id": "desktop-2604", "source_type": "official_ubuntu_documentation",
                "title": "Ubuntu Desktop network settings", "url": "https://ubuntu.test/desktop",
                "text": "Desktop network settings passage", "categories": ["Network & Connectivity"],
                "release_scope": "Ubuntu Desktop 26.04",
            },
            {
                "source_id": "desktop-2604", "source_type": "official_ubuntu_documentation",
                "title": "Ubuntu Desktop network settings", "url": "https://ubuntu.test/desktop",
                "text": "Second chunk of same page", "categories": ["Network & Connectivity"],
                "release_scope": "Ubuntu Desktop 26.04",
            },
            {
                "source_id": "server-current", "source_type": "official_ubuntu_documentation",
                "title": "Ubuntu Server networking", "url": "https://ubuntu.test/server",
                "text": "Current Ubuntu Server documentation passage", "categories": ["Network & Connectivity"],
                "release_scope": "Ubuntu Server documentation; current documentation site",
            },
        ]
        self.retriever = OfficialDocsRetriever(_Model(), _Index(), self.chunks)

    def test_results_keep_official_identity_rank_score_and_page_deduplication(self):
        results = self.retriever.search("network", k=3)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["lane"], "official_ubuntu_docs")
        self.assertEqual(results[0]["source_type"], "official_ubuntu_documentation")
        self.assertEqual(results[0]["url"], "https://ubuntu.test/desktop")
        self.assertEqual(results[0]["text"], "Desktop network settings passage")
        self.assertEqual(results[0]["rank"], 1)
        self.assertAlmostEqual(results[0]["score"], 0.92)
        self.assertEqual(results[0]["retrieval_method"], "official_docs_minilm_faiss_cosine")
        self.assertEqual(results[1]["rank"], 2)

    def test_agent_mode_can_return_multiple_relevant_passages_from_one_page(self):
        results = self.retriever.search("network", k=3, deduplicate_pages=False)
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]["url"], results[1]["url"])
        self.assertEqual(results[1]["text"], "Second chunk of same page")

    def test_explicit_scope_is_matched_and_conflicting_release_is_reported(self):
        results = self.retriever.search(
            "network", k=3, product_type="ubuntu_desktop", ubuntu_release="Ubuntu 26.04 LTS"
        )
        desktop, server = results
        self.assertEqual(desktop["supported_releases"], ["26.04"])
        self.assertEqual(desktop["scope_status"], "match")
        self.assertEqual(server["scope_status"], "mismatch")

    def test_current_wording_does_not_claim_release_compatibility(self):
        result = self.retriever.search(
            "network", k=3, product_type="ubuntu_server", ubuntu_release="24.04"
        )[1]
        self.assertEqual(result["product_scope_status"], "match")
        self.assertEqual(result["release_scope_status"], "unknown")
        self.assertEqual(result["scope_status"], "unknown")


if __name__ == "__main__":
    unittest.main()
