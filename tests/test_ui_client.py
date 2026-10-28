import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import requests

from src.ui_client import (
    ApiRequestError,
    analyze_ticket,
    create_ticket,
    get_engineer_queue,
    get_requester_ticket,
    update_engineer_ticket,
)


class UiApiClientTests(unittest.TestCase):
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
