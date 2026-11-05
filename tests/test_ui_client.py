import unittest
from types import SimpleNamespace
import sys
from unittest.mock import MagicMock, Mock, patch

import requests

from src.ui_client import (
    ApiRequestError,
    ask_engineer_agent,
    analyze_ticket,
    create_ticket,
    get_engineer_agent_conversation,
    get_engineer_queue,
    get_requester_ticket,
    update_engineer_ticket,
)


class UiAnalysisRenderingTests(unittest.TestCase):
    def test_engineer_agent_thread_renders_internal_speakers_and_linked_sources(self):
        streamlit = MagicMock()
        with patch.dict(sys.modules, {"streamlit": streamlit}):
            import app as app_module

        message = {
            "role": "agent",
            "content": "The source documents the query syntax.",
            "analysis": {
                "status": "needs_review",
                "answer_citations": [{"source_id": "OD1", "quote": "query HOSTNAME"}],
                "steps": [],
                "sources": [{
                    "source_id": "OD1",
                    "title": "Ubuntu 24.04 resolvectl manpage",
                    "url": "https://manpages.ubuntu.com/resolvectl",
                }],
                "review_reasons": ["Verify release scope."],
            },
        }
        with patch.object(app_module, "st", streamlit):
            app_module._render_agent_message(message)

        streamlit.chat_message.assert_called_once_with("assistant")
        self.assertIn("ResolveX agent", [call.args[0] for call in streamlit.caption.call_args_list])
        self.assertTrue(any("Ubuntu 24.04 resolvectl manpage" in call.args[0]
                            for call in streamlit.markdown.call_args_list))
        self.assertTrue(any("query HOSTNAME" in str(call.args) for call in streamlit.caption.call_args_list))

    def test_render_result_displays_informational_answer_and_its_citation(self):
        streamlit = MagicMock()
        with patch.dict(sys.modules, {"streamlit": streamlit}):
            import app as app_module

        columns = [MagicMock() for _ in range(3)]
        streamlit.columns.return_value = columns
        result = {
            "status": "needs_review",
            "summary": "Review the sourced explanation.",
            "answer": "The query checks whether the hostname resolves.",
            "answer_citations": [{
                "source_id": "OD1", "quote": "query HOSTNAME resolves domain names",
            }],
            "steps": [{
                "instruction": "Query the hostname www.google.com to check if it resolves.",
                "command": "resolvectl query www.google.com",
                "citations": [{
                    "source_id": "OD1",
                    "quote": "query HOSTNAME resolves domain names",
                }],
            }], "sources": [{
                "source_id": "OD1", "title": "Ubuntu manpage: resolvectl",
                "source_type": "official_ubuntu_manpage",
                "url": "https://manpages.ubuntu.com/resolvectl",
            }],
            "missing_information": [], "review_reasons": [],
            "answer_validation": {"status": "needs_review", "issues": ["Check lexical support."]},
            "step_validation": {"status": "passed", "issues": []},
            "safety_assessment": {
                "decision": "allow",
                "apt_transaction": {"status": "absent", "kernel_removal_planned": False},
                "recommendations": [],
            },
            "withheld_steps": [],
        }

        with patch.object(app_module, "st", streamlit):
            app_module._render_result(result)

        rendered = [call.args[0] for call in streamlit.markdown.call_args_list]
        self.assertIn("The query checks whether the hostname resolves.", rendered)
        streamlit.code.assert_called_once_with("resolvectl query www.google.com", language="bash")
        self.assertTrue(any("query HOSTNAME resolves domain names" in item for item in rendered))
        self.assertTrue(any("Draft answer" in item for item in rendered))
        self.assertFalse(any("Answer validation:" in item for item in rendered))
        self.assertFalse(any("Step validation:" in item for item in rendered))
        self.assertFalse(any("Safety decision:" in item for item in rendered))

    def test_agent_generation_error_shows_friendly_unavailable_message(self):
        streamlit = MagicMock()
        with patch.dict(sys.modules, {"streamlit": streamlit}):
            import app as app_module

        message = {
            "role": "agent",
            "content": "The recommendation could not be validated; no steps are provided.",
            "analysis": {
                "error": "GoogleAPIError: 503 UNAVAILABLE with provider details",
                "review_reasons": ["Generation failed: GoogleAPIError: 503 UNAVAILABLE"],
            },
        }
        with patch.object(app_module, "st", streamlit):
            app_module._render_agent_message(message)

        streamlit.warning.assert_called_once_with(
            "The AI service is temporarily unavailable. Please try again in a little while."
        )
        streamlit.write.assert_not_called()
        streamlit.expander.assert_not_called()


class UiApiClientTests(unittest.TestCase):
    def test_engineer_agent_conversation_reads_and_posts_ticket_scoped_questions(self):
        session = Mock()
        thread = {"ticket_id": "RX-123", "messages": []}
        session.request.side_effect = [
            SimpleNamespace(ok=True, status_code=200, json=lambda: thread),
            SimpleNamespace(ok=True, status_code=200, json=lambda: thread),
        ]

        self.assertEqual(
            get_engineer_agent_conversation("http://localhost:8000", "RX-123", session=session),
            thread,
        )
        self.assertEqual(
            ask_engineer_agent(
                "http://localhost:8000", "RX-123", "Explain the evidence", session=session
            ),
            thread,
        )

        self.assertEqual(session.request.call_args_list[0].args[:2], (
            "GET", "http://localhost:8000/api/v1/engineer/tickets/RX-123/agent-conversation",
        ))
        self.assertEqual(session.request.call_args_list[1].kwargs["json"], {
            "question": "Explain the evidence",
        })

    def test_posts_ticket_to_analysis_endpoint(self):
        session = Mock()
        session.request.return_value = SimpleNamespace(
            ok=True,
            status_code=200,
            json=lambda: {"status": "needs_review", "human_review": True},
        )

        result = analyze_ticket("http://localhost:8000/", "Wi-Fi issue", "Ubuntu 24.04", session=session)

        self.assertEqual(result["status"], "needs_review")
        session.request.assert_called_once_with(
            "POST",
            "http://localhost:8000/api/v1/analyze",
            json={"title": "Wi-Fi issue", "description": "Ubuntu 24.04"},
            headers=None,
            timeout=180,
        )

    def test_direct_analysis_can_send_environment_context(self):
        session = Mock()
        session.request.return_value = SimpleNamespace(
            ok=True,
            status_code=200,
            json=lambda: {"status": "needs_review", "human_review": True},
        )

        analyze_ticket(
            "http://localhost:8000", "DNS lookup fails", "Details",
            product_type="ubuntu_server", ubuntu_release="22.04", session=session,
        )

        self.assertEqual(session.request.call_args.kwargs["json"], {
            "title": "DNS lookup fails",
            "description": "Details",
            "product_type": "ubuntu_server",
            "ubuntu_release": "22.04",
        })

    def test_api_validation_message_is_presented_safely(self):
        session = Mock()
        session.request.return_value = SimpleNamespace(
            ok=False,
            status_code=422,
            json=lambda: {"error": {"message": "Request validation failed."}},
        )

        with self.assertRaisesRegex(ApiRequestError, "Request validation failed"):
            analyze_ticket("http://localhost:8000", "", "", session=session)

    def test_unavailable_service_is_explained(self):
        session = Mock()
        session.request.return_value = SimpleNamespace(
            ok=False,
            status_code=503,
            json=lambda: {"error": {"message": "Ticket analysis is not configured."}},
        )

        with self.assertRaisesRegex(ApiRequestError, "not configured"):
            analyze_ticket("http://localhost:8000", "Ticket", "Details", session=session)

    def test_connection_and_timeout_failures_are_readable(self):
        for exception, expected in (
            (requests.Timeout(), "too long"),
            (requests.ConnectionError(), "Could not reach"),
        ):
            session = Mock()
            session.request.side_effect = exception
            with self.subTest(exception=type(exception).__name__):
                with self.assertRaisesRegex(ApiRequestError, expected):
                    analyze_ticket("http://localhost:8000", "Ticket", "Details", session=session)

    def test_unexpected_success_payload_is_rejected(self):
        session = Mock()
        session.request.return_value = SimpleNamespace(ok=True, status_code=200, json=lambda: {"ok": True})

        with self.assertRaisesRegex(ApiRequestError, "expected analysis format"):
            analyze_ticket("http://localhost:8000", "Ticket", "Details", session=session)

    def test_requester_submission_and_tracking_use_private_token_header(self):
        session = Mock()
        session.request.side_effect = [
            SimpleNamespace(ok=True, status_code=201, json=lambda: {"ticket_id": "RX-123", "tracking_token": "private-token"}),
            SimpleNamespace(ok=True, status_code=200, json=lambda: {"ticket_id": "RX-123", "status": "received"}),
        ]

        created = create_ticket("http://localhost:8000", "Wi-Fi issue", "Ubuntu 24.04", session=session)
        status = get_requester_ticket("http://localhost:8000", created["tracking_token"], session=session)

        self.assertEqual(status["status"], "received")
        self.assertEqual(session.request.call_args_list[1].args[1], "http://localhost:8000/api/v1/requester/status")
        self.assertEqual(session.request.call_args_list[1].kwargs["headers"], {"X-Ticket-Token": "private-token"})

    def test_requester_submission_sends_environment_as_separate_optional_fields(self):
        session = Mock()
        session.request.return_value = SimpleNamespace(
            ok=True,
            status_code=201,
            json=lambda: {"ticket_id": "RX-123", "tracking_token": "private-token"},
        )

        create_ticket(
            "http://localhost:8000",
            "Server package issue",
            "Package installation fails.",
            product_type="ubuntu_server",
            ubuntu_release="24.04",
            session=session,
        )

        self.assertEqual(session.request.call_args.kwargs["json"], {
            "title": "Server package issue",
            "description": "Package installation fails.",
            "product_type": "ubuntu_server",
            "ubuntu_release": "24.04",
        })

    def test_requester_tracking_rejects_empty_private_code(self):
        session = Mock()

        with self.assertRaisesRegex(ApiRequestError, "private tracking code"):
            get_requester_ticket("http://localhost:8000", "", session=session)

        session.request.assert_not_called()

    def test_engineer_queue_and_status_update_need_no_local_access_code(self):
        session = Mock()
        session.request.side_effect = [
            SimpleNamespace(ok=True, status_code=200, json=lambda: []),
            SimpleNamespace(ok=True, status_code=200, json=lambda: {"ticket_id": "RX-123", "status": "resolved"}),
        ]

        self.assertEqual(get_engineer_queue("http://localhost:8000", session=session), [])
        updated = update_engineer_ticket("http://localhost:8000", "RX-123", "resolved", "Done", session=session)

        self.assertEqual(updated["status"], "resolved")
        self.assertIsNone(session.request.call_args_list[0].kwargs["headers"])
        self.assertEqual(session.request.call_args_list[1].kwargs["json"], {"status": "resolved", "status_note": "Done"})


if __name__ == "__main__":
    unittest.main()
