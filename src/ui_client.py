"""Small HTTP client used by the Streamlit interface."""

from __future__ import annotations

from typing import Any

import requests


class ApiRequestError(RuntimeError):
    """A safe, user-facing error returned while calling the ResolveX API."""


def _request(
    api_base: str,
    method: str,
    path: str,
    *,
    json: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 180,
    session: Any = requests,
) -> Any:
    try:
        response = session.request(
            method,
            f"{api_base.rstrip('/')}{path}",
            json=json,
            headers=headers,
            timeout=timeout,
        )
    except requests.Timeout as exc:
        raise ApiRequestError("The analysis took too long. Check the API and try again.") from exc
    except requests.ConnectionError as exc:
        raise ApiRequestError("Could not reach the FastAPI service. Start it and check its address in the sidebar.") from exc
    except requests.RequestException as exc:
        raise ApiRequestError("The API request failed. Check the service and try again.") from exc

    try:
        payload = response.json()
    except ValueError as exc:
        raise ApiRequestError("The API returned an unreadable response. Check the backend logs.") from exc

    if not response.ok:
        error = payload.get("error", {}) if isinstance(payload, dict) else {}
        message = error.get("message") if isinstance(error, dict) else None
        if response.status_code == 422:
            raise ApiRequestError(message or "The ticket could not be validated. Check the title and description.")
        if response.status_code == 503:
            raise ApiRequestError(message or "The analysis service is unavailable. Check backend readiness and model credentials.")
        raise ApiRequestError(message or f"Analysis failed (HTTP {response.status_code}). Check the backend logs.")

    return payload


def _requester_headers(tracking_token: str) -> dict[str, str]:
    return {"X-Ticket-Token": tracking_token}


def create_ticket(api_base: str, title: str, description: str, *, session: Any = requests) -> dict[str, Any]:
    payload = _request(api_base, "POST", "/api/v1/tickets", json={"title": title, "description": description}, session=session)
    if not isinstance(payload, dict) or not payload.get("tracking_token"):
        raise ApiRequestError("The service did not return a ticket tracking credential.")
    return payload


def get_requester_ticket(api_base: str, tracking_token: str, *, session: Any = requests) -> dict[str, Any]:
    if not tracking_token.strip():
        raise ApiRequestError("Enter the private tracking code you received when submitting the ticket.")
    payload = _request(
        api_base,
        "GET",
        "/api/v1/requester/status",
        headers=_requester_headers(tracking_token.strip()),
        session=session,
    )
    if not isinstance(payload, dict) or "ticket_id" not in payload:
        raise ApiRequestError("The service returned an invalid ticket status response.")
    return payload


def get_engineer_queue(api_base: str, *, session: Any = requests) -> list[dict[str, Any]]:
    payload = _request(api_base, "GET", "/api/v1/engineer/tickets", session=session)
    if not isinstance(payload, list):
        raise ApiRequestError("The service returned an invalid engineer queue response.")
    return payload


def analyze_engineer_ticket(api_base: str, ticket_id: str, *, session: Any = requests) -> dict[str, Any]:
    payload = _request(api_base, "POST", f"/api/v1/engineer/tickets/{ticket_id}/analyze", session=session)
    if not isinstance(payload, dict) or "ticket_id" not in payload:
        raise ApiRequestError("The service returned an invalid analysis response.")
    return payload


def update_engineer_ticket(
    api_base: str,
    ticket_id: str,
    status: str,
    status_note: str = "",
    *,
    session: Any = requests,
) -> dict[str, Any]:
    payload = _request(
        api_base,
        "PATCH",
        f"/api/v1/engineer/tickets/{ticket_id}/status",
        json={"status": status, "status_note": status_note},
        session=session,
    )
    if not isinstance(payload, dict) or "ticket_id" not in payload:
        raise ApiRequestError("The service returned an invalid status update response.")
    return payload


def analyze_ticket(
    api_base: str,
    title: str,
    description: str,
    *,
    timeout: float = 180,
    session: Any = requests,
) -> dict[str, Any]:
    """Submit a local-prototype analysis request to FastAPI."""
    payload = _request(
        api_base,
        "POST",
        "/api/v1/analyze",
        json={"title": title, "description": description},
        timeout=timeout,
        session=session,
    )
    if not isinstance(payload, dict) or "status" not in payload:
        raise ApiRequestError("The API response did not match the expected analysis format.")
    return payload
