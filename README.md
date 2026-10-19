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
Technical Issue Classification + Confidence
        ↓
IT Support Engineer Reviews
        ↓
Technical Support Agent
        ↓
Similar Cases + Knowledge Retrieval
        ↓
Grounded Recommendation + Sources
        ↓
Engineer Reviews
        ↓
Resolve or Escalate
```

## Current Status

**M4 — RAG & Retrieval — In Progress**

The classifier is available and retrieval development is in progress. Evidence preparation, the benchmark and initial retrieval experiments are documented in `notebooks/07_retrieval_preparation.ipynb`.

Next: Complete retrieval comparisons and held-out evaluation.

## Architecture

```text
Streamlit
    ↓
FastAPI
    ↓
Technical Support Agent
    ├── Ticket Classification
    ├── Similar-Issue Search
    └── Knowledge Retrieval
              ↓
      Ask Ubuntu Knowledge Base
              ↓
             LLM
              ↓
Grounded Recommendation + Sources
              ↓
      IT Support Engineer
```

The initial design uses one primary Technical Support Agent that coordinates specialized system capabilities rather than treating every component as a separate agent.

## Repository Structure

```text
ResolveX/
├── docs/          # Project documentation and milestones
├── data/          # Raw, intermediate, and processed data
├── notebooks/     # Data analysis and experimentation
├── src/           # Reusable application and AI/ML source code
├── README.md
├── requirements.txt
└── .gitignore
```

The repository structure will evolve as new product components are implemented.

## Data

The prototype uses the **Ask Ubuntu Stack Exchange Data Dump** as its primary technical data source.

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
- **FastAPI** — backend APIs and system integration
- **Streamlit** — application interface
- **Langfuse** — AI observability and evaluation


## Documentation

Detailed project documentation is available under `docs/`:

- `PROJECT_STUDY.md` — product scope, feasibility, technical approach, architecture, risks, and evaluation strategy
- `MILESTONES.md` — implementation milestones, task progress, and links to the corresponding project work

## Data Attribution

ResolveX uses data from the **Ask Ubuntu Stack Exchange Data Dump**.

Dataset source: [Stack Exchange Data Dump](https://archive.org/details/stackexchange)

Ask Ubuntu content is used in accordance with the applicable Stack Exchange licensing and attribution requirements.