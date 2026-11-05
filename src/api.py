"""FastAPI application for the ResolveX local prototype."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
import logging
import os
from pathlib import Path
from typing import Any, Callable, Literal

import joblib
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.models.classification import predict_ticket
from src.ticket_store import TicketStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger("resolvex.api")


@dataclass
class ApiServices:
    classifier: dict[str, Any]
    retriever: Any
    agent: Any | None


class TicketTextRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(max_length=1000)
    description: str = Field(default="", max_length=12000)

    @model_validator(mode="after")
    def validate_ticket_text(self):
        self.title = self.title.strip()
        self.description = self.description.strip()
        if not self.title and not self.description:
            raise ValueError("Provide a ticket title or description")
        if len(self.title) + len(self.description) > 12000:
            raise ValueError("Ticket title and description must total 12,000 characters or fewer")
        return self


class TicketRequest(TicketTextRequest):
    product_type: Literal["ubuntu_desktop", "ubuntu_server"] | None = None
    ubuntu_release: str | None = Field(default=None, max_length=32)

    @field_validator("product_type", mode="before")
    @classmethod
    def normalize_product_type(cls, value):
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return None
            return {
                "Ubuntu Desktop": "ubuntu_desktop",
                "Ubuntu Server": "ubuntu_server",
            }.get(value, value)
        return value

    @model_validator(mode="after")
    def normalize_ubuntu_release(self):
        self.ubuntu_release = self.ubuntu_release.strip() or None if self.ubuntu_release else None
        return self


class ClassificationResponse(BaseModel):
    category: str
    confidence: float = Field(ge=0, le=1)
    human_review: bool


class CitationResponse(BaseModel):
    source_id: str
    quote: str


class StepResponse(BaseModel):
    instruction: str
    command: str | None = None
    citations: list[CitationResponse]


class EvidenceSourceResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    source_id: str
    source_type: str
    title: str
    url: str
    release_scope: str | None = None


class ValidationSummary(BaseModel):
    status: str
    issues: list[str] = Field(default_factory=list)


class AnalysisResponse(BaseModel):
    category: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    human_review: bool
    status: Literal["recommendation", "needs_review", "unsupported"]
    summary: str
    answer: str = ""
    answer_citations: list[CitationResponse] = Field(default_factory=list)
    steps: list[StepResponse]
    sources: list[EvidenceSourceResponse]
    missing_information: list[str]
    review_reasons: list[str]
    citation_check: str | None = None
    answer_validation: ValidationSummary | None = None
    step_validation: ValidationSummary | None = None
    safety_assessment: dict[str, Any] | None = None
    withheld_steps: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None


class ReadinessResponse(BaseModel):
    status: Literal["ready", "degraded"]
    components: dict[str, bool]


class TicketCreatedResponse(BaseModel):
    ticket_id: str
    title: str
    tracking_token: str
    status: str
    created_at: str
    updated_at: str
    product_type: Literal["ubuntu_desktop", "ubuntu_server"] | None = None
    ubuntu_release: str | None = None


class RequesterTicketResponse(BaseModel):
    ticket_id: str
    title: str
    status: str
    status_note: str
    created_at: str
    updated_at: str


class TicketStatusUpdate(BaseModel):
    status: Literal["received", "in_review", "resolved", "escalated"]
    status_note: str = Field(default="", max_length=2000)


class EngineerTicketResponse(RequesterTicketResponse):
    description: str
    product_type: Literal["ubuntu_desktop", "ubuntu_server"] | None = None
    ubuntu_release: str | None = None
    analysis: AnalysisResponse | None = None


class EngineerAgentQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=4000)

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("Enter a question for the support agent")
        return value


class EngineerAgentMessageResponse(BaseModel):
    message_id: int
    ticket_id: str
    role: Literal["engineer", "agent"]
    content: str
    analysis: AnalysisResponse | None = None
    created_at: str


class EngineerAgentConversationResponse(BaseModel):
    ticket_id: str
    messages: list[EngineerAgentMessageResponse]


def _load_api_services(root: Path) -> ApiServices:
    """Load local artifacts once. API startup never downloads or rebuilds indexes."""
    load_dotenv(root / ".env")
    classifier_path = root / "data/processed/classifier.joblib"
    classifier = joblib.load(classifier_path)

    from src.retrieval import MODEL_ID, MODEL_REVISION, load_retriever

    retriever = load_retriever(root)
    agent = None
    api_key = os.getenv("GEMINI_API_KEY")
    from src.llm_provider import configured_model_name, configured_provider
    provider = configured_provider()
    if api_key:
        from sentence_transformers import SentenceTransformer
        from src.official_docs import load_official_docs
        from src.support_agent import SupportAgent, load_full_corpus_bm25, make_m5_chains

        if retriever.dense is not None:
            embedding_model = retriever.dense.model
        else:
            embedding_model = SentenceTransformer(
                MODEL_ID,
                revision=MODEL_REVISION,
                cache_folder=str(root / "data/processed/embedding_cache"),
                device="cpu",
                local_files_only=True,
            )
        official_retriever, _ = load_official_docs(root, embedding_model)
        chains = make_m5_chains(api_key, provider=provider)
        pipeline_mode = os.getenv("AGENT_PIPELINE_MODE", "single").strip().lower()
        use_multi_stage = pipeline_mode in {"multi", "multi_stage", "4stage"}
        agent = SupportAgent(
            classifier, retriever, chains["chain"], official_retriever,
            load_full_corpus_bm25(root),
            action_extractor=chains["action_extractor"] if use_multi_stage else None,
            evidence_assessor=chains["evidence_assessor"] if use_multi_stage else None,
            conflict_assessor=chains["conflict_assessor"] if use_multi_stage else None,
            model_name=configured_model_name(provider),
        )
    return ApiServices(classifier=classifier, retriever=retriever, agent=agent)


def _error(status_code: int, code: str, message: str):
    raise HTTPException(status_code=status_code, detail={"code": code, "message": message})


def _adapt_agent_response(raw: dict[str, Any]) -> AnalysisResponse:
    """Keep the public API contract independent of the support-agent internals."""
    classification = raw.get("classification") or {}
    status = raw.get("status", "needs_review")
    if status not in {"recommendation", "needs_review", "unsupported"}:
        status = "needs_review"
    return AnalysisResponse(
        category=raw.get("category") or classification.get("category"),
        confidence=classification.get("confidence"),
        human_review=(
            bool(raw.get("engineer_review_required", False))
            or bool(classification.get("human_review", False))
            or status != "recommendation"
        ),
        status=status,
        summary=raw.get("summary", "Engineer review is required."),
        answer=raw.get("answer") or "",
        answer_citations=raw.get("answer_citations") or [],
        steps=raw.get("steps") or [],
        sources=raw.get("sources") or [],
        missing_information=raw.get("missing_information") or [],
        review_reasons=raw.get("review_reasons") or [],
        citation_check=raw.get("citation_check"),
        answer_validation=raw.get("answer_validation"),
        step_validation=raw.get("step_validation"),
        safety_assessment=raw.get("safety_assessment"),
        withheld_steps=raw.get("withheld_steps") or [],
        error=raw.get("error"),
    )


def create_app(
    services_loader: Callable[[Path], ApiServices] = _load_api_services,
    root: Path = PROJECT_ROOT,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        resolved_root = Path(root).resolve()
        app.state.services = services_loader(resolved_root)
        app.state.ticket_store = TicketStore(resolved_root / "data/processed/support_tickets.sqlite3")
        yield
        app.state.services = None
        app.state.ticket_store = None

    app = FastAPI(
        title="ResolveX API",
        version="1.0.0",
        description="Local API for the ResolveX Ubuntu/Linux support prototype.",
        lifespan=lifespan,
    )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        # Exclude submitted input values so ticket text is not echoed in error responses.
        errors = [
            {"location": error.get("loc", []), "message": error.get("msg", "Invalid value")}
            for error in exc.errors()
        ]
        return JSONResponse(
            status_code=422,
            content={"error": {"code": "invalid_request", "message": "Request validation failed.", "details": errors}},
        )

    @app.exception_handler(HTTPException)
    async def http_error_handler(request: Request, exc: HTTPException):
        detail = exc.detail if isinstance(exc.detail, dict) else {"code": "request_failed", "message": str(exc.detail)}
        return JSONResponse(status_code=exc.status_code, content={"error": detail}, headers=exc.headers)

    @app.get("/health/live", tags=["health"])
    def liveness():
        return {"status": "alive"}

    @app.get("/health/ready", response_model=ReadinessResponse, tags=["health"])
    def readiness(request: Request):
        services: ApiServices = request.app.state.services
        components = {
            "classifier": services.classifier is not None,
            "retriever": services.retriever is not None,
            "support_agent": services.agent is not None,
        }
        status = "ready" if all(components.values()) else "degraded"
        result = ReadinessResponse(status=status, components=components)
        if status != "ready":
            return JSONResponse(status_code=503, content=result.model_dump())
        return result

    @app.post("/api/v1/classify", response_model=ClassificationResponse, tags=["tickets"])
    def classify_ticket(payload: TicketTextRequest, request: Request):
        services: ApiServices = request.app.state.services
        try:
            return predict_ticket(services.classifier, payload.title, payload.description)
        except ValueError as exc:
            _error(422, "invalid_ticket", str(exc))
        except Exception as exc:
            logger.error("Classification failed (%s)", type(exc).__name__)
            _error(503, "classification_unavailable", "Ticket classification is temporarily unavailable.")

    @app.post("/api/v1/analyze", response_model=AnalysisResponse, tags=["tickets"])
    def analyze_ticket(payload: TicketRequest, request: Request):
        services: ApiServices = request.app.state.services
        if services.agent is None:
            _error(503, "analysis_unavailable", "Ticket analysis is not configured; check the server's model credentials and readiness.")
        try:
            raw = services.agent.run(
                payload.title,
                payload.description,
                product_type=payload.product_type,
                ubuntu_release=payload.ubuntu_release,
            )
            return _adapt_agent_response(raw)
        except ValueError as exc:
            _error(422, "invalid_ticket", str(exc))
        except Exception as exc:
            logger.error("Ticket analysis failed (%s)", type(exc).__name__)
            _error(503, "analysis_unavailable", "Ticket analysis is temporarily unavailable.")

    @app.post("/api/v1/tickets", response_model=TicketCreatedResponse, status_code=201, tags=["tickets"])
    def submit_ticket(payload: TicketRequest, request: Request):
        record, token = request.app.state.ticket_store.create(
            payload.title,
            payload.description,
            product_type=payload.product_type,
            ubuntu_release=payload.ubuntu_release,
        )
        return {**record, "tracking_token": token}

    @app.get("/api/v1/requester/status", response_model=RequesterTicketResponse, tags=["tickets"])
    def requester_ticket_status(
        request: Request,
        tracking_token: str | None = Header(default=None, alias="X-Ticket-Token"),
    ):
        if not tracking_token:
            _error(401, "ticket_token_required", "A ticket tracking credential is required.")
        record = request.app.state.ticket_store.get_for_requester(tracking_token)
        if record is None:
            _error(404, "ticket_not_found", "Ticket not found or tracking code is invalid.")
        return record

    @app.get("/api/v1/engineer/tickets", response_model=list[EngineerTicketResponse], tags=["tickets"])
    def engineer_ticket_queue(request: Request):
        return request.app.state.ticket_store.list_for_engineer()

    @app.post("/api/v1/engineer/tickets/{ticket_id}/analyze", response_model=EngineerTicketResponse, tags=["tickets"])
    def analyze_queued_ticket(ticket_id: str, request: Request):
        services: ApiServices = request.app.state.services
        if services.agent is None:
            _error(503, "analysis_unavailable", "Ticket analysis is not configured; check the server's model credentials and readiness.")
        ticket = request.app.state.ticket_store.get_for_engineer(ticket_id)
        if ticket is None:
            _error(404, "ticket_not_found", "Ticket was not found.")
        try:
            result = _adapt_agent_response(services.agent.run(
                ticket["title"], ticket["description"],
                product_type=ticket.get("product_type"),
                ubuntu_release=ticket.get("ubuntu_release"),
            ))
        except ValueError as exc:
            _error(422, "invalid_ticket", str(exc))
        except Exception as exc:
            logger.error("Ticket analysis failed (%s)", type(exc).__name__)
            _error(503, "analysis_unavailable", "Ticket analysis is temporarily unavailable.")
        updated = request.app.state.ticket_store.save_analysis(ticket_id, result.model_dump())
        return updated

    @app.get(
        "/api/v1/engineer/tickets/{ticket_id}/agent-conversation",
        response_model=EngineerAgentConversationResponse,
        tags=["conversations"],
    )
    def engineer_agent_conversation(ticket_id: str, request: Request):
        messages = request.app.state.ticket_store.get_agent_conversation(ticket_id)
        if messages is None:
            _error(404, "ticket_not_found", "Ticket was not found.")
        return {"ticket_id": ticket_id.strip().upper(), "messages": messages}

    @app.post(
        "/api/v1/engineer/tickets/{ticket_id}/agent-conversation",
        response_model=EngineerAgentConversationResponse,
        tags=["conversations"],
    )
    def ask_ticket_agent(
        ticket_id: str,
        payload: EngineerAgentQuestion,
        request: Request,
    ):
        services: ApiServices = request.app.state.services
        if services.agent is None:
            _error(503, "analysis_unavailable", "Ticket analysis is not configured; check the server's model credentials and readiness.")
        store: TicketStore = request.app.state.ticket_store
        ticket = store.get_for_engineer(ticket_id)
        if ticket is None:
            _error(404, "ticket_not_found", "Ticket was not found.")
        history = store.get_agent_conversation(ticket_id) or []
        store.add_agent_message(ticket_id, "engineer", payload.question)
        prior_turns = []
        remaining_context_chars = 8000
        for message in reversed(history[-8:]):
            if remaining_context_chars <= 0:
                break
            content = message["content"][:remaining_context_chars]
            prior_turns.append({"role": message["role"], "content": content})
            remaining_context_chars -= len(content)
        prior_turns.reverse()
        try:
            raw = services.agent.answer_engineer_question(
                ticket["title"],
                ticket["description"],
                payload.question,
                prior_turns,
                product_type=ticket.get("product_type"),
                ubuntu_release=ticket.get("ubuntu_release"),
            )
            result = _adapt_agent_response(raw)
        except ValueError as exc:
            _error(422, "invalid_question", str(exc))
        except Exception as exc:
            logger.error("Engineer-agent turn failed (%s)", type(exc).__name__)
            result = AnalysisResponse(
                status="needs_review",
                summary="The support agent is temporarily unavailable. The engineer question was saved.",
                steps=[],
                sources=[],
                missing_information=[],
                review_reasons=["Agent generation failed; retry the question after checking model availability."],
                error="Agent generation failed",
                human_review=True,
            )
        content_parts = [result.summary]
        if result.answer:
            content_parts.append(result.answer)
        if result.error:
            content_parts.append("The agent could not complete this response; review the error details.")
        store.add_agent_message(
            ticket_id, "agent", "\n\n".join(part for part in content_parts if part), result.model_dump()
        )
        return {
            "ticket_id": ticket_id.strip().upper(),
            "messages": store.get_agent_conversation(ticket_id),
        }

    @app.patch("/api/v1/engineer/tickets/{ticket_id}/status", response_model=EngineerTicketResponse, tags=["tickets"])
    def update_ticket_status(
        ticket_id: str,
        payload: TicketStatusUpdate,
        request: Request,
    ):
        try:
            updated = request.app.state.ticket_store.update_status(ticket_id, payload.status, payload.status_note)
        except ValueError as exc:
            _error(409, "invalid_status_transition", str(exc))
        if updated is None:
            _error(404, "ticket_not_found", "Ticket was not found.")
        return updated

    return app


app = create_app()
