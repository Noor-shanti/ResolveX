# ResolveX

**AI-Powered Technical Support & Resolution Platform**

ResolveX is a decision-support platform designed to help IT Support Engineers analyze technical issues, retrieve relevant technical knowledge, and reach more reliable resolutions.

The initial prototype focuses on **Ubuntu/Linux Technical Support**.

## Overview

Technical support engineers often need to classify incoming issues, search for similar cases, investigate possible solutions, and verify technical information before deciding how to handle a ticket.

ResolveX brings these steps into one workflow by combining:

- Technical issue classification with confidence scoring
- Similar-case and technical knowledge retrieval
- Grounded troubleshooting recommendations with supporting sources
- Human review and escalation when confidence or evidence is insufficient

The IT Support Engineer remains responsible for the final decision.

## How It Works

```text
Employee submits Ticket
        ↓
Ticket appears in the Support Team Inbox
        ↓
IT Support Engineer requests Technical Analysis
        ↓
Issue Classification + Confidence
        ↓
Similar Cases + Knowledge Retrieval
        ↓
Grounded Recommendation (when evidence supports one) + Sources + Review Signals
        ↓
IT Support Engineer reviews, resolves or escalates
        ↓
Employee checks request status
```

## Current Status

**Current focus: Complete M5 - Technical Support Agent**

- **M1–M4 — Completed**
- **M5 — In Progress:** Re-run the 12 evaluation cases and verify at least one useful recommendation against its sources and Ubuntu release.
- **M6 — Completed**
- **M7 — Completed**

## Run the Backend Locally

Install the project requirements, then start the API from the repository root:

```bash
python -m pip install -r requirements.txt
uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` for the interactive API reference. The service expects the saved classifier, retrieval and Ubuntu documentation artifacts under `data/processed/`. Set `GEMINI_API_KEY` in the local `.env` file to enable ticket analysis; classification remains available without it.

The API supports health checks, classification, ticket analysis and ticket management. Tickets and their analysis are stored in `data/processed/support_tickets.sqlite3`; requesters use a private tracking code to check status.

## Run the Streamlit Interface

With the API running as described above, start the interface from the repository root in a second terminal:

```bash
streamlit run app.py
```

The interface connects to `http://127.0.0.1:8000` by default. Requesters can submit tickets and check their status; the support team can review the analysis and update each ticket. The support-team inbox has no sign-in, so keep this prototype local and do not use it with real employee data.

## Architecture

```text
Streamlit requester and support-team pages
                    ↓
                  FastAPI ───── SQLite ticket store
                    ↓
        Technical Support Agent
        ├── Ticket Classification
        ├── Similar-case retrieval
        └── Knowledge retrieval
             ├── Ask Ubuntu
             └── Official Ubuntu documentation
                    ↓
                  LLM
                    ↓
          Grounded Recommendation (when evidence supports one) + Sources + Review Signals
                    ↓
          IT Support Engineer review
```

The initial design uses one primary Technical Support Agent that coordinates specialized system capabilities rather than treating every component as a separate agent. Recommendations are for engineer review; citations do not by themselves establish technical correctness.

## Repository Structure

```text
ResolveX/
├── docs/          # Project documentation and milestones
├── data/          # Raw, intermediate, and processed data
├── notebooks/     # Data analysis and experimentation
├── src/           # Reusable application and AI/ML source code
├── tests/         # API and interface-client tests
├── app.py         # Streamlit interface
├── README.md
├── requirements.txt
└── .gitignore
```

The repository structure will evolve as new product components are implemented.

## Data

The prototype uses the **Ask Ubuntu Stack Exchange Data Dump** as its primary technical data source.

The retrieval agent also uses a separate, limited collection of official Ubuntu documentation. These sources are labeled separately from community answers.

The dataset is used to support:

- Technical issue classification
- Technical knowledge retrieval
- Similar-case retrieval
- Retrieval and system evaluation

Raw source data is kept unchanged, while derived datasets are produced through reproducible data-processing workflows.

## Technical Issue Categories

The Ubuntu/Linux prototype currently supports six technical issue categories:

1. Network & Connectivity
2. Boot & Startup
3. Software & Packages
4. Hardware & Drivers
5. Storage & Filesystems
6. Access & Permissions

These categories represent the internal technical taxonomy of the ResolveX prototype.

## Technology

ResolveX is being developed using:

- **Python** — data processing and AI/ML development
- **Machine Learning (scikit-learn)** — technical issue classification and confidence scoring
- **RAG (Retrieval-Augmented Generation)** — retrieval of relevant technical knowledge for grounded troubleshooting
- **LLM** — generation of troubleshooting recommendations using retrieved evidence
- **LangChain** — support-agent workflow orchestration
- **FastAPI** — backend APIs and system integration
- **Streamlit** — application interface
- **SQLite** — local ticket storage
- **Langfuse** — AI observability and evaluation


## Documentation

Detailed project documentation is available under `docs/`:

- `PROJECT_STUDY.md` — product scope, feasibility, technical approach, architecture, risks, and evaluation strategy
- `MILESTONES.md` — implementation milestones, task progress, and links to the corresponding project work

## Data Attribution

ResolveX uses data from the **Ask Ubuntu Stack Exchange Data Dump**.

Dataset source: [Stack Exchange Data Dump](https://archive.org/details/stackexchange)

Ask Ubuntu content is used in accordance with the applicable Stack Exchange licensing and attribution requirements.
