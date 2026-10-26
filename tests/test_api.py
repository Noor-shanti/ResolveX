import unittest

import numpy as np
from fastapi.testclient import TestClient

from src.api import ApiServices, create_app


class _ClassifierPipeline:
    classes_ = np.array(["Network & Connectivity", "Software & Packages"])

    def decision_function(self, rows):
        return np.array([[2.0, 0.2]])


class _Agent:
    def run(self, title, body):
        return {
            "category": "Network & Connectivity",
            "classification": {
                "category": "Network & Connectivity",
                "confidence": 0.86,
                "human_review": False,
            },
            "status": "needs_review",
            "summary": "Review the evidence before troubleshooting.",
            "engineer_review_required": True,
            "steps": [{
                "instruction": "Check the current connection state.",
                "citations": [{"source_id": "OD1", "quote": "nmcli device status"}],
            }],
            "sources": [{
                "source_id": "OD1",
                "source_type": "official_ubuntu_manpage",
                "title": "Ubuntu manpage: nmcli",
                "url": "https://manpages.ubuntu.com/manpages/resolute/man1/nmcli.1.html",
                "release_scope": "Ubuntu 26.04 LTS",
                "answer_excerpt": "nmcli device status",
            }],
            "missing_information": [],
            "review_reasons": ["Review the result before taking further action."],
            "citation_check": "passed",
        }


def _services(agent=True):
    return ApiServices(
        classifier={
            "pipeline": _ClassifierPipeline(),
            "temperature": 1.0,
            "review_policy": {"threshold": 0.5},
        },
        retriever=object(),
        agent=_Agent() if agent else None,
    )


class ApiTests(unittest.TestCase):
    def test_components_load_once_and_classification_returns_contract(self):
        calls = []
        app = create_app(lambda root: (calls.append(root) or _services()))
        with TestClient(app) as client:
            first = client.post("/api/v1/classify", json={"title": "Wi-Fi unavailable"})
            second = client.post("/api/v1/classify", json={"title": "Wi-Fi unavailable"})
        self.assertEqual(first.status_code, 200)
        self.assertEqual(first.json()["category"], "Network & Connectivity")
        self.assertGreater(first.json()["confidence"], 0.5)
        self.assertFalse(first.json()["human_review"])
        self.assertEqual(second.status_code, 200)
        self.assertEqual(len(calls), 1)

    def test_analysis_preserves_review_flag_and_source_metadata(self):
        app = create_app(lambda root: _services())
        with TestClient(app) as client:
            response = client.post("/api/v1/analyze", json={
                "title": "Wi-Fi unavailable", "description": "Ubuntu Desktop 26.04"
            })
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertTrue(result["human_review"])
        self.assertEqual(result["status"], "needs_review")
        self.assertEqual(result["sources"][0]["source_id"], "OD1")
        self.assertEqual(result["sources"][0]["release_scope"], "Ubuntu 26.04 LTS")
        self.assertEqual(result["steps"][0]["citations"][0]["quote"], "nmcli device status")

    def test_validation_errors_do_not_echo_ticket_text(self):
        app = create_app(lambda root: _services())
        with TestClient(app) as client:
            response = client.post("/api/v1/classify", json={"title": "PRIVATE TICKET TEXT " * 80})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"]["code"], "invalid_request")
        self.assertNotIn("PRIVATE TICKET TEXT", response.text)

    def test_analysis_without_agent_is_a_sanitized_service_error(self):
        app = create_app(lambda root: _services(agent=False))
        with TestClient(app) as client:
            live = client.get("/health/live")
            ready = client.get("/health/ready")
            result = client.post("/api/v1/analyze", json={"title": "Ticket"})
        self.assertEqual(live.status_code, 200)
        self.assertEqual(ready.status_code, 503)
        self.assertEqual(result.status_code, 503)
        self.assertEqual(result.json()["error"]["code"], "analysis_unavailable")


if __name__ == "__main__":
    unittest.main()
