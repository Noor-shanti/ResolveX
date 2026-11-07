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
    def __init__(self):
        self.calls = []
        self.engineer_calls = []

    def run(self, title, body, **context):
        self.calls.append((title, body, context))
        return {
            "category": "Network & Connectivity",
            "classification": {
                "category": "Network & Connectivity",
                "confidence": 0.86,
                "human_review": False,
            },
            "status": "needs_review",
            "decision_outcome": "limited_diagnostic_guidance",
            "summary": "Review the evidence before troubleshooting.",
            "answer": "The resolver query can check whether the hostname resolves.",
            "answer_citations": [{
                "source_id": "OD1", "quote": "resolvectl query HOSTNAME",
            }],
            "engineer_review_required": True,
            "steps": [{
                "instruction": "Check the current connection state.",
                "command": "nmcli device status",
                "citations": [{"source_id": "OD1", "quote": "nmcli device status"}],
            }],
            "sources": [{
                "source_id": "OD1",
                "lane": "official_ubuntu_docs",
                "source_type": "official_ubuntu_manpage",
                "title": "Ubuntu manpage: nmcli",
                "url": "https://manpages.ubuntu.com/manpages/resolute/man1/nmcli.1.html",
                "release_scope": "Ubuntu 26.04 LTS",
                "product_scope": None,
                "supported_releases": ["26.04"],
                "scope_status": "unknown",
                "retrieval_method": "official_docs_minilm_faiss_cosine",
                "rank": 1,
                "score": 0.81,
                "source_date": None,
                "date_status": "unknown",
                "answer_excerpt": "nmcli device status",
            }],
            "missing_information": [],
            "review_reasons": ["Review the result before taking further action."],
            "citation_check": "passed",
            "answer_validation": {"status": "passed", "issues": []},
            "step_validation": {"status": "passed", "issues": []},
            "safety_assessment": {
                "decision": "allow",
                "apt_transaction": {"status": "absent", "kernel_removal_planned": False},
                "recommendations": [],
            },
            "withheld_steps": [],
        }

    def answer_engineer_question(
        self, title, body, question, conversation_history=None, **context
    ):
        self.engineer_calls.append((title, body, question, conversation_history, context))
        return self.run(title, body, **context)


class _FailingEngineerQuestionAgent(_Agent):
    def answer_engineer_question(self, *args, **kwargs):
        raise RuntimeError("provider unavailable; do not expose internal details")


def _services(agent=True, agent_instance=None):
    return ApiServices(
        classifier={
            "pipeline": _ClassifierPipeline(),
            "temperature": 1.0,
            "review_policy": {"threshold": 0.5},
        },
        retriever=object(),
        agent=(agent_instance or _Agent()) if agent else None,
    )


@contextmanager
def _client(agent=True, agent_instance=None):
    with tempfile.TemporaryDirectory() as temp_dir:
        app = create_app(lambda root: _services(agent=agent, agent_instance=agent_instance), root=Path(temp_dir))
        with TestClient(app) as client:
            yield client


class ApiTests(unittest.TestCase):
    def test_app_can_use_an_isolated_ticket_database(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            isolated_database = root / "benchmark" / "tickets.sqlite3"
            app = create_app(
                lambda _root: _services(),
                root=root,
                ticket_database_path=isolated_database,
            )
            with TestClient(app) as client:
                response = client.post("/api/v1/tickets", json={"title": "Isolated benchmark ticket"})

            self.assertEqual(response.status_code, 201)
            self.assertTrue(isolated_database.is_file())
            self.assertFalse((root / "data/processed/support_tickets.sqlite3").exists())

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
        self.assertNotIn("decision_outcome", result)
        self.assertEqual(result["sources"][0]["source_id"], "OD1")
        self.assertEqual(result["sources"][0]["release_scope"], "Ubuntu 26.04 LTS")
        self.assertEqual(result["sources"][0]["lane"], "official_ubuntu_docs")
        self.assertEqual(result["sources"][0]["scope_status"], "unknown")
        self.assertEqual(result["sources"][0]["retrieval_method"], "official_docs_minilm_faiss_cosine")
        self.assertEqual(result["answer"], "The resolver query can check whether the hostname resolves.")
        self.assertEqual(result["answer_citations"][0]["source_id"], "OD1")
        self.assertEqual(result["answer_citations"][0]["quote"], "resolvectl query HOSTNAME")
        self.assertEqual(result["steps"][0]["command"], "nmcli device status")
        self.assertEqual(result["steps"][0]["citations"][0]["quote"], "nmcli device status")
        self.assertEqual(result["answer_validation"]["status"], "passed")
        self.assertEqual(result["safety_assessment"]["decision"], "allow")

    def test_direct_analysis_passes_optional_product_and_release_context(self):
        agent = _Agent()
        with _client(agent_instance=agent) as client:
            response = client.post("/api/v1/analyze", json={
                "title": "DNS lookup fails",
                "description": "Ubuntu Server ticket",
                "product_type": "Ubuntu Server",
                "ubuntu_release": "22.04",
            })

        self.assertEqual(response.status_code, 200)
        self.assertEqual(agent.calls[0][2], {
            "product_type": "ubuntu_server",
            "ubuntu_release": "22.04",
        })

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

    def test_ticket_environment_fields_are_optional_and_persist_separately(self):
        with _client() as client:
            legacy_shape = client.post("/api/v1/tickets", json={"title": "Older client ticket"})
            structured = client.post("/api/v1/tickets", json={
                "title": "Server package issue",
                "description": "Package installation fails.",
                "product_type": "ubuntu_server",
                "ubuntu_release": "24.04",
            })
            queue = client.get("/api/v1/engineer/tickets").json()

        self.assertEqual(legacy_shape.status_code, 201)
        self.assertIsNone(legacy_shape.json()["product_type"])
        self.assertIsNone(legacy_shape.json()["ubuntu_release"])
        self.assertEqual(structured.status_code, 201)
        self.assertEqual(structured.json()["product_type"], "ubuntu_server")
        self.assertEqual(structured.json()["ubuntu_release"], "24.04")
        by_id = {ticket["ticket_id"]: ticket for ticket in queue}
        self.assertIsNone(by_id[legacy_shape.json()["ticket_id"]]["product_type"])
        saved = by_id[structured.json()["ticket_id"]]
        self.assertEqual(saved["product_type"], "ubuntu_server")
        self.assertEqual(saved["ubuntu_release"], "24.04")

    def test_product_type_accepts_missing_null_blank_and_supported_labels(self):
        cases = [
            ({}, None),
            ({"product_type": None}, None),
            ({"product_type": ""}, None),
            ({"product_type": "   "}, None),
            ({"product_type": "Ubuntu Desktop"}, "ubuntu_desktop"),
            ({"product_type": "Ubuntu Server"}, "ubuntu_server"),
        ]
        with _client() as client:
            for index, (fields, expected) in enumerate(cases):
                with self.subTest(fields=fields):
                    response = client.post("/api/v1/tickets", json={
                        "title": f"Optional product case {index}",
                        **fields,
                    })
                    self.assertEqual(response.status_code, 201)
                    self.assertEqual(response.json()["product_type"], expected)

    def test_ticket_rejects_unsupported_product_type(self):
        with _client() as client:
            response = client.post("/api/v1/tickets", json={
                "title": "Unsupported distro",
                "product_type": "linux_mint",
            })
        self.assertEqual(response.status_code, 422)

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

    def test_engineer_analysis_passes_saved_environment_fields_to_agent(self):
        agent = _Agent()
        with _client(agent_instance=agent) as client:
            created = client.post("/api/v1/tickets", json={
                "title": "Server network issue", "description": "DNS lookup fails.",
                "product_type": "ubuntu_server", "ubuntu_release": "24.04",
            }).json()
            response = client.post(
                f"/api/v1/engineer/tickets/{created['ticket_id']}/analyze"
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(agent.calls), 1)
        self.assertEqual(agent.calls[0][2], {
            "product_type": "ubuntu_server", "ubuntu_release": "24.04",
        })

    def test_engineer_can_question_agent_about_ticket_and_reopen_saved_thread(self):
        agent = _Agent()
        with _client(agent_instance=agent) as client:
            created = client.post("/api/v1/tickets", json={
                "title": "DNS lookup fails", "description": "Ubuntu Server cannot resolve example.com.",
                "product_type": "ubuntu_server", "ubuntu_release": "24.04",
            }).json()
            ticket_id = created["ticket_id"]
            first = client.post(
                f"/api/v1/engineer/tickets/{ticket_id}/agent-conversation",
                json={"question": "Does the cited source say this fixes DNS?"},
            )
            second = client.post(
                f"/api/v1/engineer/tickets/{ticket_id}/agent-conversation",
                json={"question": "What should I verify before sharing the command?"},
            )
            reopened = client.get(f"/api/v1/engineer/tickets/{ticket_id}/agent-conversation")

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(reopened.status_code, 200)
        self.assertEqual([m["role"] for m in reopened.json()["messages"]], [
            "engineer", "agent", "engineer", "agent",
        ])
        self.assertEqual(reopened.json()["messages"][1]["analysis"]["sources"][0]["source_id"], "OD1")
        self.assertEqual(agent.engineer_calls[0][:3], (
            "DNS lookup fails", "Ubuntu Server cannot resolve example.com.",
            "Does the cited source say this fixes DNS?",
        ))
        self.assertEqual(agent.engineer_calls[0][4], {
            "product_type": "ubuntu_server", "ubuntu_release": "24.04",
        })
        self.assertEqual(agent.engineer_calls[1][3], [
            {"role": "engineer", "content": "Does the cited source say this fixes DNS?"},
            {"role": "agent", "content": (
                "Review the evidence before troubleshooting.\n\n"
                "The resolver query can check whether the hostname resolves."
            )},
        ])

    def test_engineer_agent_conversation_rejects_missing_ticket_and_blank_question(self):
        with _client() as client:
            missing = client.post(
                "/api/v1/engineer/tickets/RX-MISSING/agent-conversation",
                json={"question": "Why?"},
            )
            created = client.post("/api/v1/tickets", json={"title": "DNS issue"}).json()
            blank = client.post(
                f"/api/v1/engineer/tickets/{created['ticket_id']}/agent-conversation",
                json={"question": "   "},
            )
            empty_thread = client.get(
                f"/api/v1/engineer/tickets/{created['ticket_id']}/agent-conversation"
            )
        self.assertEqual(missing.status_code, 404)
        self.assertEqual(blank.status_code, 422)
        self.assertEqual(empty_thread.json()["messages"], [])

    def test_agent_failure_is_saved_as_a_safe_internal_reply(self):
        with _client(agent_instance=_FailingEngineerQuestionAgent()) as client:
            created = client.post("/api/v1/tickets", json={"title": "DNS issue"}).json()
            response = client.post(
                f"/api/v1/engineer/tickets/{created['ticket_id']}/agent-conversation",
                json={"question": "What should I check?"},
            )

        self.assertEqual(response.status_code, 200)
        messages = response.json()["messages"]
        self.assertEqual([message["role"] for message in messages], ["engineer", "agent"])
        self.assertIn("temporarily unavailable", messages[-1]["content"])
        self.assertNotIn("provider unavailable", response.text)
        self.assertEqual(messages[-1]["analysis"]["status"], "needs_review")

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
