import unittest
from contextlib import contextmanager
from pathlib import Path
import tempfile

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


@contextmanager
def _client(agent=True):
    with tempfile.TemporaryDirectory() as temp_dir:
        app = create_app(lambda root: _services(agent=agent), root=Path(temp_dir))
        with TestClient(app) as client:
            yield client


class ApiTests(unittest.TestCase):
    def test_components_load_once_and_classification_returns_contract(self):
        calls = []
        with tempfile.TemporaryDirectory() as temp_dir:
            app = create_app(lambda root: (calls.append(root) or _services()), root=Path(temp_dir))
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
        with _client() as client:
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
        with _client() as client:
            response = client.post("/api/v1/classify", json={"title": "PRIVATE TICKET TEXT " * 80})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"]["code"], "invalid_request")
        self.assertNotIn("PRIVATE TICKET TEXT", response.text)

    def test_analysis_without_agent_is_a_sanitized_service_error(self):
        with _client(agent=False) as client:
            live = client.get("/health/live")
            ready = client.get("/health/ready")
            result = client.post("/api/v1/analyze", json={"title": "Ticket"})
        self.assertEqual(live.status_code, 200)
        self.assertEqual(ready.status_code, 503)
        self.assertEqual(result.status_code, 503)
        self.assertEqual(result.json()["error"]["code"], "analysis_unavailable")

    def test_requester_ticket_submission_tracking_and_privacy(self):
        with _client() as client:
            created = client.post("/api/v1/tickets", json={
                "title": "Wi-Fi drops", "description": "Private ticket description"
            })
            self.assertEqual(created.status_code, 201)
            ticket = created.json()
            self.assertTrue(ticket["tracking_token"])

            status = client.get(
                "/api/v1/requester/status",
                headers={"X-Ticket-Token": ticket["tracking_token"]},
            )
            bad_token = client.get(
                "/api/v1/requester/status",
                headers={"X-Ticket-Token": "wrong-token"},
            )

        self.assertEqual(status.status_code, 200)
        self.assertEqual(status.json()["status"], "received")
        self.assertNotIn("description", status.json())
        self.assertEqual(bad_token.status_code, 404)

    def test_engineer_queue_persists_analysis_and_status_in_local_prototype(self):
        with _client() as client:
            created = client.post("/api/v1/tickets", json={"title": "Wi-Fi issue", "description": "Ubuntu 26.04"}).json()
            queue = client.get("/api/v1/engineer/tickets")
            analyzed = client.post(f"/api/v1/engineer/tickets/{created['ticket_id']}/analyze")
            updated = client.patch(
                f"/api/v1/engineer/tickets/{created['ticket_id']}/status",
                json={"status": "resolved", "status_note": "Issue reviewed by support."},
            )
            tracked = client.get(
                "/api/v1/requester/status",
                headers={"X-Ticket-Token": created["tracking_token"]},
            )

        self.assertEqual(queue.status_code, 200)
        self.assertEqual(queue.json()[0]["ticket_id"], created["ticket_id"])
        self.assertEqual(analyzed.status_code, 200)
        self.assertEqual(analyzed.json()["status"], "in_review")
        self.assertEqual(analyzed.json()["analysis"]["status"], "needs_review")
        self.assertEqual(updated.status_code, 200)
        self.assertEqual(tracked.json()["status"], "resolved")
        self.assertNotIn("analysis", tracked.json())

    def test_invalid_ticket_status_transition_is_rejected(self):
        with _client() as client:
            created = client.post("/api/v1/tickets", json={"title": "Ticket"}).json()
            response = client.patch(
                f"/api/v1/engineer/tickets/{created['ticket_id']}/status",
                json={"status": "resolved"},
            )
        self.assertEqual(response.status_code, 409)

    def test_requester_reply_status_is_not_part_of_current_ticket_lifecycle(self):
        with _client() as client:
            created = client.post("/api/v1/tickets", json={"title": "Ticket"}).json()
            response = client.patch(
                f"/api/v1/engineer/tickets/{created['ticket_id']}/status",
                json={"status": "awaiting_requester"},
            )
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
