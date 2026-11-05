"""Streamlit interface for the ResolveX support engineer workflow."""

from __future__ import annotations

import html
import os
from datetime import datetime
from textwrap import dedent

import streamlit as st

from src.ui_client import (
    ApiRequestError,
    analyze_engineer_ticket,
    ask_engineer_agent,
    create_ticket,
    get_engineer_agent_conversation,
    get_engineer_queue,
    get_requester_ticket,
    update_engineer_ticket,
)


st.set_page_config(
    page_title="ResolveX | Ubuntu/Linux Technical Support",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root {
  --ink: #211a31; --muted: #777186; --line: #e9e5ef; --paper: #fbfaff;
  --purple: #7044c7; --purple-dark: #4c2c91; --orange: #f06a43;
  --green: #18794e; --red: #b4233c; --amber: #936200;
}
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; color: var(--ink); }
.stApp { background: radial-gradient(ellipse at 80% -15%, #eee6ff 0, transparent 35%), var(--paper); }
header[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { background: #211a31; border-right: 1px solid #332a44; }
[data-testid="stSidebar"] * { color: #f5f1fb !important; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #c8c0d6 !important; }
[data-testid="stSidebar"] hr { border-color: #443a55; }
[data-testid="stSidebar"] .stTextInput input { background: #30273f; border-color: #514662; }
.block-container { max-width: 1380px; padding-top: 2.3rem; padding-bottom: 4rem; }
h1, h2, h3 { font-family: 'Manrope', sans-serif !important; letter-spacing: -0.035em; }
h1 { font-size: clamp(2.1rem, 4vw, 3.25rem) !important; line-height: 1.08 !important; }
h2 { font-size: 1.45rem !important; }
.eyebrow { text-transform: uppercase; letter-spacing: .16em; color: var(--purple); font-size: .72rem; font-weight: 800; }
.hero-copy { color: var(--muted); font-size: 1.04rem; max-width: 670px; line-height: 1.65; }
.panel { background:#fff; border:1px solid var(--line); border-radius:20px; padding:24px 26px; box-shadow:0 10px 30px #281c4310; }
.panel-title { font:700 1.08rem 'Manrope',sans-serif; margin:0 0 5px; }
.panel-subtitle { color:var(--muted); font-size:.9rem; margin:0 0 17px; }
.result-banner { border-radius:18px; padding:20px 24px; margin:12px 0 18px; border:1px solid; }
.result-banner.review { background:#fff8e8; border-color:#f1dda9; }
.result-banner.ready { background:#edfaf3; border-color:#bce8cd; }
.result-banner.unsupported { background:#fff0f1; border-color:#f2c4ca; }
.banner-label { font-size:.72rem; text-transform:uppercase; letter-spacing:.13em; font-weight:800; margin-bottom:6px; }
.banner-title { font:700 1.15rem 'Manrope',sans-serif; margin-bottom:4px; }
.banner-copy { color:#61596e; line-height:1.55; }
.metric { background:#fff; border:1px solid var(--line); padding:17px 19px; border-radius:16px; min-height:92px; }
.metric.category-metric { border-top:3px solid #4263a8; }
.metric.confidence-metric { border-top:3px solid #7044c7; }
.metric.review-needed { border-top:3px solid #b77912; background:#fffaf0; }
.metric.review-clear { border-top:3px solid #21805a; background:#f4fbf7; }
.metric.agent-assessment { border-top:3px solid #7044c7; background:#f8f5ff; }
.metric-label { color:var(--muted); font-size:.78rem; margin-bottom:8px; }
.metric-value { color:var(--ink); font:700 1.02rem 'Manrope',sans-serif; }
.support-section-heading { display:flex; align-items:center; gap:11px; padding:3px 0 13px; margin-bottom:10px; border-bottom:1px solid var(--line); }
.support-section-heading .section-dot { width:11px; height:11px; flex:0 0 11px; border-radius:50%; background:var(--section-accent); box-shadow:0 0 0 5px var(--section-wash); }
.support-section-heading .section-copy { display:flex; flex-direction:column; gap:2px; }
.support-section-heading small { color:var(--section-accent); font-size:.67rem; font-weight:800; letter-spacing:.12em; }
.support-section-heading strong { color:var(--ink); font:700 1.08rem 'Manrope',sans-serif; }
.support-section-heading.request { --section-accent:#4263a8; --section-wash:#edf3ff; }
.support-section-heading.analysis { --section-accent:#7044c7; --section-wash:#f3efff; }
.support-section-heading.update { --section-accent:#21805a; --section-wash:#eff9f3; }
.ticket-meta { display:flex; align-items:center; gap:9px; flex-wrap:wrap; margin:7px 0 14px; color:var(--muted); font-size:.82rem; }
.ticket-state { display:inline-flex; align-items:center; border-radius:999px; padding:4px 10px; font-weight:700; }
.ticket-state.received { background:#edf3ff; color:#36588f; }
.ticket-state.in-review { background:#f3efff; color:#6036ae; }
.ticket-state.resolved { background:#edfaf3; color:#176b46; }
.ticket-state.escalated { background:#fff1e9; color:#a3471e; }
.request-status { --status:#7044c7; --status-bg:#f3efff; background:linear-gradient(105deg,var(--status-bg) 0,#fff 48%); border:1px solid var(--line); border-left:5px solid var(--status); border-radius:18px; padding:22px 24px; margin:20px 0; box-shadow:0 10px 30px #281c430c; }
.request-status.received { --status:#4263a8; --status-bg:#edf3ff; }
.request-status.in-review { --status:#7044c7; --status-bg:#f3efff; }
.request-status.resolved { --status:#18794e; --status-bg:#edfaf3; }
.request-status.escalated { --status:#b65425; --status-bg:#fff1e9; }
.request-status-row { display:grid; grid-template-columns:150px minmax(0,1fr); gap:14px; padding:12px 0; border-bottom:1px solid var(--line); line-height:1.55; }
.request-status-row:first-child { padding-top:0; }
.request-status-row:last-child { padding-bottom:0; border-bottom:0; }
.request-status-label { color:var(--muted); font-weight:700; }
.request-status-value { color:var(--ink); }
.request-status-chip { display:inline-flex; align-items:center; gap:8px; border-radius:999px; padding:5px 11px; color:var(--status); background:var(--status-bg); border:1px solid var(--status); font-weight:700; }
.request-status-chip::before { content:""; width:8px; height:8px; flex:0 0 8px; border-radius:50%; background:var(--status); }
@media (max-width: 760px) { .request-status-row { grid-template-columns:1fr; gap:3px; } }
.step-card { background:#fff; border:1px solid var(--line); border-radius:15px; padding:17px 19px; margin:10px 0; }
.step-number { display:inline-flex; width:27px; height:27px; border-radius:9px; align-items:center; justify-content:center; background:#f0eaff; color:var(--purple-dark); font-weight:800; margin-right:9px; }
.step-copy { line-height:1.65; }
.source-card { border:1px solid var(--line); border-radius:14px; background:#fff; padding:15px 17px; margin:9px 0; }
.source-top { display:flex; gap:8px; align-items:center; flex-wrap:wrap; margin-bottom:7px; }
.source-type { font-size:.69rem; text-transform:uppercase; letter-spacing:.08em; font-weight:800; padding:4px 8px; border-radius:99px; }
.source-type.official { background:#e9f6ee; color:#17643e; }
.source-type.community { background:#f2edff; color:#5b3aa2; }
.source-type.other { background:#f1eff5; color:#514b5d; }
.source-id { color:var(--muted); font-size:.75rem; }
.source-title { font-weight:700; margin-bottom:4px; }
.source-link { color:var(--purple-dark); font-size:.86rem; text-decoration:none; }
.quote { color:#686173; font-size:.88rem; border-left:3px solid #d5c8f5; padding-left:11px; margin-top:11px; line-height:1.6; }
.footer-note { color:#898397; font-size:.78rem; border-top:1px solid var(--line); padding-top:18px; margin-top:30px; }
.stButton > button[kind="primary"] { border:0; border-radius:12px; min-height:3rem; font-weight:700; background:linear-gradient(110deg,#7044c7,#5830a5); box-shadow:0 8px 20px #7044c733; }
.stButton > button[kind="primary"]:hover { border:0; background:linear-gradient(110deg,#6135b5,#482584); }
.stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] { border-radius:11px !important; }
div[data-testid="stForm"] { border:0; padding:0; }
div[class*="st-key-floating-agent-launcher"] { position:fixed; right:26px; bottom:24px; z-index:10000; width:72px; }
div[class*="st-key-floating-agent-launcher"] button { width:70px; height:70px; min-height:70px; border-radius:50%; border:1px solid #d8cbed; background:linear-gradient(145deg,#fff,#f1eaff); color:#6036ae; font-size:2.25rem; box-shadow:0 8px 24px #211a3130; }
div[class*="st-key-floating-agent-launcher"] button p { font-size:2.25rem !important; line-height:1 !important; margin:0 !important; }
div[class*="st-key-floating-agent-launcher"] button:hover { border-color:#7044c7; color:#4c2c91; transform:translateY(-2px); }
div[class*="st-key-floating-agent-panel"] { position:fixed; right:26px; bottom:100px; z-index:9999; width:min(420px,calc(100vw - 32px)); max-height:72vh; overflow:auto; background:#fff; border:1px solid #ded5ee; border-radius:18px; padding:18px; box-shadow:0 16px 48px #211a3133; }
div[class*="st-key-floating-agent-panel"] > div { background:#fff; }
.floating-agent-title { font:700 1.05rem 'Manrope',sans-serif; color:var(--ink); margin-bottom:2px; }
.floating-agent-eyebrow { color:var(--purple); font-size:.66rem; font-weight:800; letter-spacing:.12em; }
@media (max-width: 760px) { .block-container { padding:1.25rem 1rem 3rem; } .panel { padding:19px; } }
@media (max-width: 760px) { div[class*="st-key-floating-agent-launcher"] { right:16px; bottom:16px; } div[class*="st-key-floating-agent-panel"] { right:12px; bottom:88px; width:calc(100vw - 24px); max-height:72vh; } }
</style>
"""


def _safe(value: object) -> str:
    return html.escape(str(value or ""), quote=True)


def _format_submitted_at(value: str) -> str:
    try:
        submitted_at = datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone()
    except (AttributeError, ValueError):
        return "Submission date unavailable"
    return submitted_at.strftime("%b %d, %Y at %I:%M %p").replace(" 0", " ")


def _source_type(source: dict) -> tuple[str, str]:
    kind = str(source.get("source_type", "")).lower()
    if "official" in kind or "ubuntu_doc" in kind:
        return "Official Ubuntu documentation", "official"
    if "askubuntu" in kind or "community" in kind:
        return "Ask Ubuntu community answer", "community"
    return str(source.get("source_type") or "Technical source"), "other"


def _render_result(result: dict) -> None:
    status = result.get("status", "needs_review")
    if status == "recommendation":
        banner_class, label, heading = "ready", "Suggested next steps", "Review before sharing or taking action"
    elif status == "unsupported":
        banner_class, label, heading = "unsupported", "No reliable next step found", "The available information was not enough to suggest what to do"
    else:
        banner_class, label, heading = "review", "Your review is needed", "Check the information and sources before deciding what to do"

    st.markdown(
        f"""<div class="result-banner {banner_class}">
        <div class="banner-label">{label}</div><div class="banner-title">{heading}</div>
        <div class="banner-copy">{_safe(result.get('summary'))}</div></div>""",
        unsafe_allow_html=True,
    )

    left, right, third = st.columns([1.2, 1, 1])
    category = result.get("category") or "Not assigned"
    confidence = result.get("confidence")
    confidence_text = f"{confidence:.1%}" if isinstance(confidence, (int, float)) else "Not available"
    with left:
        st.markdown(f'<div class="metric category-metric"><div class="metric-label">Predicted category</div><div class="metric-value">{_safe(category)}</div></div>', unsafe_allow_html=True)
    with right:
        st.markdown(f'<div class="metric confidence-metric"><div class="metric-label">Category confidence</div><div class="metric-value">{confidence_text}</div></div>', unsafe_allow_html=True)
    with third:
        assessment, _ = _agent_assessment(result)
        st.markdown(f'<div class="metric agent-assessment"><div class="metric-label">Agent assessment</div><div class="metric-value">{_safe(assessment)}</div></div>', unsafe_allow_html=True)

    steps = result.get("steps") or []
    sources = result.get("sources") or []
    source_by_id = {str(source.get("source_id")): source for source in sources}
    answer = result.get("answer") or ""
    if answer:
        st.markdown("### Draft answer")
        st.markdown(_safe(answer))
        for citation in result.get("answer_citations") or []:
            src = source_by_id.get(str(citation.get("source_id")), {})
            source_name = _safe(src.get("title") or citation.get("source_id"))
            quote = _safe(citation.get("quote"))
            st.caption(f"Answer evidence · {source_name}")
            if quote:
                st.markdown(f'<div class="quote">“{quote}”</div>', unsafe_allow_html=True)

    st.markdown("### Suggested next steps")
    if steps:
        for index, step in enumerate(steps, 1):
            instruction = _safe(step.get("instruction", ""))
            st.markdown(f'<div class="step-card"><span class="step-number">{index}</span><span class="step-copy">{instruction}</span></div>', unsafe_allow_html=True)
            command = step.get("command")
            if command:
                st.code(command, language="bash")
            for citation in step.get("citations", []):
                src = source_by_id.get(str(citation.get("source_id")), {})
                source_name = _safe(src.get("title") or citation.get("source_id"))
                quote = _safe(citation.get("quote"))
                st.caption(f"Evidence · {source_name}")
                if quote:
                    st.markdown(f'<div class="quote">“{quote}”</div>', unsafe_allow_html=True)
    else:
        st.info("No next steps are suggested yet. Check the available information and details that may be missing.")

    missing = result.get("missing_information") or []
    reasons = result.get("review_reasons") or []
    if missing or reasons:
        st.markdown("### What needs checking")
        note_cols = st.columns(2)
        if missing:
            with note_cols[0]:
                st.markdown("**Details still needed**")
                for item in missing:
                    st.markdown(f"- {_safe(item)}")
        if reasons:
            with note_cols[1]:
                st.markdown("**Why this needs review**")
                for item in reasons:
                    st.markdown(f"- {_safe(item)}")

    safety = result.get("safety_assessment") or {}
    withheld_steps = result.get("withheld_steps") or []
    transaction = safety.get("apt_transaction") or {}
    if safety.get("decision") == "escalate":
        if transaction.get("kernel_removal_planned"):
            st.warning("Safety review needed: the proposed package operation would remove a Linux kernel.")
        else:
            safety_reasons = [
                detail
                for recommendation in (safety.get("recommendations") or [])
                for detail in recommendation.get("reasons", [])
            ]
            st.warning("Safety review needed: " + (safety_reasons[0] if safety_reasons else "the agent flagged a potentially risky action."))
    if withheld_steps:
        st.markdown("### Suggestions not shown")
        for item in withheld_steps:
            st.markdown(f"**{_safe(item.get('instruction') or 'A suggested action was withheld')}**")
            for issue in item.get("issues") or []:
                st.caption(_safe(issue))

    st.markdown("### Information used")
    if not sources:
        st.caption("No supporting links were available for this review.")
    for source in sources:
        label, css = _source_type(source)
        title = _safe(source.get("title") or "Untitled source")
        source_id = _safe(source.get("source_id") or "")
        url = str(source.get("url") or "")
        safe_url = _safe(url) if url.startswith(("https://", "http://")) else ""
        release = source.get("release_scope")
        release_line = f'<div class="source-id">Release scope: {_safe(release)}</div>' if release else ""
        link = f'<a class="source-link" href="{safe_url}" target="_blank" rel="noopener noreferrer">Open source ↗</a>' if safe_url else "<span class=\"source-id\">Source link unavailable</span>"
        st.markdown(
            f'<div class="source-card"><div class="source-top"><span class="source-type {css}">{_safe(label)}</span><span class="source-id">{source_id}</span></div><div class="source-title">{title}</div>{release_line}{link}</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="footer-note">Suggestions should be checked by the support team before they are shared or acted on.</div>', unsafe_allow_html=True)


def _status_label(status: str) -> str:
    labels = {
        "received": "Request received",
        "in_review": "Being reviewed by support",
        "resolved": "Resolved",
        "escalated": "Specialist review needed",
    }
    return labels.get(status, status.replace("_", " ").title())


def _status_description(status: str) -> str:
    descriptions = {
        "received": "Your request is in the support team's queue and is waiting to be reviewed.",
        "in_review": "A support team member is reviewing your request.",
        "resolved": "The support team has marked this request as resolved.",
        "escalated": "The support team marked this request for specialist review. This status does not mean it was automatically transferred.",
    }
    return descriptions.get(status, "Check the latest update from the support team.")


def _agent_assessment(result: dict) -> tuple[str, str]:
    """Return a useful, case-specific summary instead of the always-on review flag."""
    status = result.get("status")
    safety = result.get("safety_assessment") or {}
    if status == "unsupported":
        return "Outside support scope", "unsupported"
    if safety.get("decision") == "escalate":
        return "Escalation advised", "escalate"
    if result.get("missing_information"):
        return "More details needed", "review"
    if status == "recommendation":
        return "Recommendation ready", "ready"
    return "Needs review", "review"


def _render_request_status(ticket: dict) -> None:
    status = ticket.get("status", "received")
    style = {
        "received": "received",
        "in_review": "in-review",
        "resolved": "resolved",
        "escalated": "escalated",
    }.get(status, "received")
    note = ticket.get("status_note")
    rows = [
        f'<div class="request-status-row"><span class="request-status-label">Status:</span><span class="request-status-value"><span class="request-status-chip">{_safe(_status_label(status))}</span></span></div>',
        f'<div class="request-status-row"><span class="request-status-label">What this means:</span><span class="request-status-value">{_safe(_status_description(status))}</span></div>',
        f'<div class="request-status-row"><span class="request-status-label">Your request:</span><span class="request-status-value">{_safe(ticket.get("title"))}</span></div>',
    ]
    if note:
        rows.append(
            f'<div class="request-status-row"><span class="request-status-label">Message from support:</span><span class="request-status-value">{_safe(note)}</span></div>'
        )
    card_html = dedent(
        f"""
        <section class="request-status {style}">
          {''.join(rows)}
        </section>
        """
    ).strip()
    st.markdown(card_html, unsafe_allow_html=True)


def _requester_portal(api_base: str) -> None:
    st.markdown("## Tell us what you need help with")
    st.markdown("Describe the problem and include any error message. The support team will review your request.")
    with st.form("requester_ticket", clear_on_submit=True):
        title = st.text_input("Briefly describe the problem", max_chars=1000, placeholder="For example: Wi-Fi disconnects after the computer wakes up")
        product_label = st.selectbox("Ubuntu system", ["Not sure", "Ubuntu Desktop", "Ubuntu Server"])
        ubuntu_release = st.text_input("Ubuntu release (optional)", max_chars=32, placeholder="For example: 24.04")
        description = st.text_area("What happened?", height=170, max_chars=12000, placeholder="Tell us what you expected, what happened instead, and any recent changes.")
        submitted = st.form_submit_button("Send request", type="primary", use_container_width=True)
    if submitted:
        if not title.strip() and not description.strip():
            st.error("Enter an issue summary or description.")
            return
        product_type = {
            "Ubuntu Desktop": "ubuntu_desktop",
            "Ubuntu Server": "ubuntu_server",
        }.get(product_label)
        try:
            with st.spinner("Submitting your ticket…"):
                ticket = create_ticket(
                    api_base,
                    title.strip(),
                    description.strip(),
                    product_type=product_type,
                    ubuntu_release=ubuntu_release.strip() or None,
                )
            st.session_state["new_ticket_credentials"] = {
                "tracking_token": ticket["tracking_token"],
            }
        except ApiRequestError as exc:
            st.error(str(exc))

    credentials = st.session_state.get("new_ticket_credentials")
    if credentials:
        if "tracking_token" not in credentials:
            old_request_code = credentials.get("request_code", "")
            credentials["tracking_token"] = credentials.get("tracking_token") or (
                old_request_code.partition(".")[2] or ""
            )
        with st.container(border=True):
            st.success("Your request was sent. Save this private code to check for updates later.")
            st.markdown("**Private tracking code**")
            st.code(credentials["tracking_token"], language=None)
            st.caption("Enter this code in the “Private tracking code” box on the status page. Keep it private.")


def _requester_tracking(api_base: str) -> None:
    st.markdown("## Check your request status")
    st.markdown("Enter the private tracking code you received when you sent your request.")
    existing = st.session_state.get("new_ticket_credentials", {})
    with st.form("track_ticket"):
        tracking_token = st.text_input("Private tracking code", value=existing.get("tracking_token", ""), type="password", placeholder="Paste your private code")
        submitted = st.form_submit_button("Check for updates", type="primary")
    if submitted:
        try:
            with st.spinner("Checking for updates…"):
                st.session_state["tracked_ticket"] = get_requester_ticket(api_base, tracking_token.strip())
        except ApiRequestError as exc:
            st.session_state.pop("tracked_ticket", None)
            st.error(str(exc))
    ticket = st.session_state.get("tracked_ticket")
    if ticket:
        _render_request_status(ticket)


def _render_agent_message(message: dict) -> None:
    is_engineer = message.get("role") == "engineer"
    with st.chat_message("user" if is_engineer else "assistant"):
        st.caption("IT Support" if is_engineer else "ResolveX agent")
        analysis = message.get("analysis") or {}
        if not is_engineer and analysis.get("error"):
            st.warning("The AI service is temporarily unavailable. Please try again in a little while.")
            return

        st.write(message.get("content") or "")
        if not is_engineer and analysis:
            sources = {
                source.get("source_id"): source
                for source in analysis.get("sources", [])
                if source.get("source_id")
            }
            citations = list(analysis.get("answer_citations", []))
            for step in analysis.get("steps", []):
                citations.extend(step.get("citations", []))
            if citations or analysis.get("error"):
                with st.expander("Evidence and response checks"):
                    if analysis.get("error"):
                        st.warning(analysis["error"])
                    for citation in citations:
                        source = sources.get(citation.get("source_id"), {})
                        if source.get("url"):
                            st.markdown(f"[{_safe(source.get('title') or citation['source_id'])}]({source['url']})")
                        else:
                            st.markdown(f"**{_safe(citation.get('source_id'))}**")
                        st.caption(citation.get("quote") or "")
                    st.caption(f"Agent status: {analysis.get('status', 'not available')}")
                    if analysis.get("review_reasons"):
                        st.write("Review notes:", analysis["review_reasons"])


def _engineer_agent_panel(api_base: str, ticket_id: str) -> None:
    """Render the ticket-scoped agent conversation in a floating assistant widget."""
    state_key = "floating_agent_open_ticket"
    is_open = st.session_state.get(state_key) == ticket_id
    with st.container(key="floating-agent-launcher"):
        if st.button("🤖", key=f"floating-agent-toggle-{ticket_id}", help="Open the ResolveX assistant"):
            st.session_state[state_key] = None if is_open else ticket_id
            st.rerun()

    if not is_open:
        return

    with st.container(key="floating-agent-panel"):
        top = st.columns([1, 0.15])
        with top[0]:
            st.markdown('<div class="floating-agent-eyebrow">INTERNAL TICKET ASSISTANT</div><div class="floating-agent-title">ResolveX support agent</div>', unsafe_allow_html=True)
        with top[1]:
            if st.button("×", key=f"floating-agent-close-{ticket_id}", help="Close assistant"):
                st.session_state[state_key] = None
                st.rerun()
        st.caption("Private to IT Support. Review suggestions before acting.")
        try:
            conversation = get_engineer_agent_conversation(api_base, ticket_id)
        except ApiRequestError as exc:
            st.error(str(exc))
            conversation = {"messages": []}

        with st.container(height=300, border=True):
            if not conversation["messages"]:
                with st.chat_message("assistant"):
                    st.caption("ResolveX agent")
                    st.write("Hi! I can explain this ticket’s analysis, review its sources, or suggest a safe next check. Type your question below and select **Send**.")
            for message in conversation["messages"]:
                _render_agent_message(message)

        with st.form(f"agent-question-{ticket_id}", clear_on_submit=True):
            question = st.text_input(
                "Message the assistant",
                placeholder="Type a question about this ticket…",
                label_visibility="collapsed",
            )
            submitted = st.form_submit_button("Send", type="primary", use_container_width=True)
        if submitted and question.strip():
            try:
                with st.spinner("ResolveX is reviewing the ticket and evidence…"):
                    ask_engineer_agent(api_base, ticket_id, question.strip())
                st.rerun()
            except ApiRequestError as exc:
                st.error(str(exc))


def _engineer_workspace(api_base: str) -> None:
    st.markdown("## Support team inbox")
    try:
        queue = get_engineer_queue(api_base)
        st.session_state["engineer_queue"] = queue
    except ApiRequestError as exc:
        st.error(str(exc))
        return

    if not queue:
        st.info("No tickets have been submitted yet.")
        return

    by_id = {ticket["ticket_id"]: ticket for ticket in queue}
    selected_id = st.selectbox(
        f"Requests ({len(queue)})",
        list(by_id),
        format_func=lambda ticket_id: f"{by_id[ticket_id]['title']} · {_status_label(by_id[ticket_id]['status'])}",
    )
    ticket = by_id[selected_id]
    status_style = {
        "received": "received",
        "in_review": "in-review",
        "resolved": "resolved",
        "escalated": "escalated",
    }.get(ticket["status"], "received")
    with st.container(border=True):
        st.markdown('<div class="support-section-heading request"><span class="section-dot"></span><span class="section-copy"><small>EMPLOYEE REQUEST</small><strong>Ticket details</strong></span></div>', unsafe_allow_html=True)
        st.markdown(f"**{_safe(ticket['title'])}**")
        st.markdown(
            f'<div class="ticket-meta"><span>{_safe(ticket["ticket_id"])}</span><span>·</span><span>Submitted {_format_submitted_at(ticket["created_at"])}</span><span class="ticket-state {status_style}">{_safe(_status_label(ticket["status"]))}</span></div>',
            unsafe_allow_html=True,
        )
        with st.expander("Read the request details", expanded=True):
            product_label = {
                "ubuntu_desktop": "Ubuntu Desktop",
                "ubuntu_server": "Ubuntu Server",
            }.get(ticket.get("product_type"), "Not specified")
            release = ticket.get("ubuntu_release") or "Not specified"
            st.caption(f"System: {product_label} · Ubuntu release: {release}")
            st.write(ticket.get("description") or "No additional details were provided.")

    with st.container(border=True):
        st.markdown('<div class="support-section-heading analysis"><span class="section-dot"></span><span class="section-copy"><small>SUPPORTING EVIDENCE</small><strong>ResolveX analysis</strong></span></div>', unsafe_allow_html=True)
        if st.button("Analyze request", type="primary", use_container_width=True):
            try:
                with st.spinner("Checking the request and its sources…"):
                    analyze_engineer_ticket(api_base, selected_id)
                st.rerun()
            except ApiRequestError as exc:
                st.error(str(exc))

        if ticket.get("analysis"):
            st.markdown("---")
            _render_result(ticket["analysis"])

    _engineer_agent_panel(api_base, selected_id)

    with st.container(border=True):
        st.markdown('<div class="support-section-heading update"><span class="section-dot"></span><span class="section-copy"><small>SUPPORT TEAM ACTION</small><strong>Update request</strong></span></div>', unsafe_allow_html=True)
        status_choices = ["received", "in_review", "resolved", "escalated"]
        with st.form(f"ticket_status_update_{selected_id}"):
            new_status = st.selectbox("Request status", status_choices, index=status_choices.index(ticket["status"]), format_func=_status_label)
            status_note = st.text_area("Message for the requester", value=ticket.get("status_note", ""), max_chars=2000, placeholder="Add a brief update visible on the request status page.")
            save_status = st.form_submit_button("Save update", type="primary")
    if save_status:
        try:
            update_engineer_ticket(api_base, selected_id, new_status, status_note)
            st.success("Request updated.")
            st.rerun()
        except ApiRequestError as exc:
            st.error(str(exc))


st.markdown(STYLE, unsafe_allow_html=True)

st.markdown('<div class="eyebrow">RESOLVEX</div>', unsafe_allow_html=True)
st.title("Ubuntu/Linux Technical Support")

def requester_submit_page() -> None:
    _requester_portal(api_base)


def requester_track_page() -> None:
    _requester_tracking(api_base)


def engineer_queue_page() -> None:
    _engineer_workspace(api_base)


submit_page = st.Page(requester_submit_page, title="Get help", icon="📝", default=True)
track_page = st.Page(requester_track_page, title="Check request status", icon="🔎")
engineer_page = st.Page(engineer_queue_page, title="Support team inbox", icon="🛠️")
navigation = st.navigation([submit_page, track_page, engineer_page], position="hidden")

with st.sidebar:
    st.markdown('<div style="font:800 1.55rem Manrope,sans-serif; letter-spacing:-.05em">◈ Resolve<span style="color:#ff9b76">X</span></div>', unsafe_allow_html=True)
    st.caption("AI-POWERED TECHNICAL SUPPORT & RESOLUTION")
    st.caption("UBUNTU / LINUX PROTOTYPE")
    st.divider()
    st.page_link(submit_page, label="Get help", icon="📝")
    st.page_link(track_page, label="Check request status", icon="🔎")
    st.page_link(engineer_page, label="Support team inbox", icon="🛠️")

api_base = os.getenv("RESOLVEX_API_URL", "http://127.0.0.1:8000").strip().rstrip("/")
navigation.run()
