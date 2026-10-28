# ResolveX — Project Milestones

## M1 — Project Research & Feasibility
**Status:** Completed<br>
**Summary:** Validate the project idea, scope, dataset, and technical feasibility before implementation.

### Definition of Done (DoD)
- Project problem and target users are clearly defined.
- Prototype scope and data feasibility are validated.
- Initial system architecture is defined.
- Project Study is reviewed and approved by supervisors.

### Tasks
- [x] Define the problem and target users. → `docs/PROJECT_STUDY.md`
- [x] Research and compare project directions. → `docs/PROJECT_STUDY.md`
- [x] Select Ubuntu/Linux as the prototype domain. → `docs/PROJECT_STUDY.md`
- [x] Select and analyze the Ask Ubuntu dataset. → `notebooks/01_askubuntu_exploration.ipynb`
- [x] Define and validate the six technical issue categories. → `notebooks/02_taxonomy_feasibility.ipynb`
- [x] Analyze dataset coverage and category ambiguity. → `notebooks/02_taxonomy_feasibility.ipynb`
- [x] Define the product workflow and initial architecture. → `docs/PROJECT_STUDY.md`
- [x] Identify key risks and limitations. → `docs/PROJECT_STUDY.md`
- [x] Prepare the Project Study. → `docs/PROJECT_STUDY.md`
- [x] Review the proposal with supervisors.
- [x] Apply feedback and finalize the project scope.


### Outcomes
- ResolveX selected as the final project direction.
- Ubuntu/Linux selected as the initial prototype domain.
- Ask Ubuntu validated as the primary data source.
- Six technical issue categories defined for the prototype.
- **177,194 single-category cases** identified for ML development.
- **36,568 multi-category cases** identified for ambiguity evaluation.
- Initial end-to-end product workflow and system architecture defined.
- Project Study prepared for supervisor review.

## M2 — Data Preparation & EDA
**Status:** Completed<br>
**Summary:** Prepare final datasets and perform EDA.

### Definition of Done (DoD)
- Final datasets for ML and knowledge retrieval are prepared.
- Data preprocessing is completed and documented.
- Exploratory Data Analysis is completed.
- Data is ready for model and retrieval development.

### Tasks
- [x] Perform data cleaning and preprocessing. → `src/data/preprocessing.py`
- [x] Finalize the ML dataset. → `src/data/preprocessing.py`, `notebooks/03_data_preparation.ipynb`
- [x] Prepare the technical knowledge base for retrieval. → `src/data/preprocessing.py`, `notebooks/03_data_preparation.ipynb`
- [x] Preserve code blocks, commands, and required metadata. → `src/data/preprocessing.py`
- [x] Prepare duplicate-question relationships for retrieval evaluation. → `notebooks/03_data_preparation.ipynb`
- [x] Analyze class distribution and text characteristics. → `notebooks/04_eda.ipynb`
- [x] Analyze relevant temporal and answer-related characteristics. → `notebooks/04_eda.ipynb`
- [x] Create EDA visualizations. → `notebooks/04_eda.ipynb`
- [x] Document preprocessing decisions and final datasets. → `notebooks/03_data_preparation.ipynb`

### Outcomes
- **177,194 single-category questions** prepared as ML classification candidates.
- **318,537 questions and 523,728 answers** prepared for knowledge retrieval.
- **41,931 duplicate relationships** preserved, including **41,411 pairs** suitable for retrieval evaluation.
- Preprocessing and validation completed and documented in `notebooks/03_data_preparation.ipynb`.
- EDA completed with visualizations and findings in `notebooks/04_eda.ipynb`.

### Limitations
- Category labels are tag-derived and require semantic review before model evaluation.
- Community answers may be outdated and are not verified solutions.

## M3 — ML Classification
**Status:** Completed<br>
**Summary:** Train and evaluate a baseline technical issue classifier using question titles and bodies.

### Definition of Done (DoD)
- Label quality is reviewed on a sample, with decisions and limitations documented.
- Reproducible train, validation and test splits prevent known duplicate leakage.
- A TF-IDF + Logistic Regression classifier is trained and evaluated.
- Classification errors and confidence behavior are analyzed.
- The fitted pipeline is saved and can predict a category, confidence and human-review flag for new text.

### Tasks
- [x] Review a sample from each category and document incorrect or ambiguous labels.
- [x] Create train/validation/test splits using full duplicate relationships; check identical and near-duplicate text across splits.
- [x] Build a TF-IDF + Logistic Regression pipeline using Title and BodyText only; fit text features on training data only.
- [x] Compare the baseline with a majority-class reference and evaluate class weighting on validation data.
- [x] Select model settings using validation data.
- [x] Analyze mistakes, confidence calibration and multi-category cases; select a human-review threshold using validation data.
- [x] Inspect predictions on ticket-like examples and document the limits of transferring forum results to support tickets.
- [x] Freeze development choices, then evaluate once on the held-out test set; report Accuracy, Macro F1, Weighted F1, per-class Precision/Recall/F1 and a Confusion Matrix.
- [x] Save the fitted pipeline and confidence policy, and provide a reusable prediction function.
- [x] Run a post-evaluation diagnostic comparing the classifier with a literal tag-keyword rule on the held-out test set (no tuning or model changes). → `notebooks/06_label_circularity_check.ipynb`

### Outcomes
- A TF-IDF + Logistic Regression classifier selected using validation data.
- Label review, duplicate-aware splits, confidence calibration and evaluation documented in `notebooks/05_ml_classification.ipynb`.
- Held-out Accuracy **89.68%**, Macro F1 **0.8856**, and Weighted F1 **0.8967** against tag-derived labels.
- A saved classifier with calibrated confidence and a validation-based human-review flag, with reusable prediction logic in `src/models/classification.py`.
- Diagnostic on the test set: classifier agreement with weak labels 89.68% versus 66.13% for a literal tag-keyword rule; on the 3,430 questions with no literal tag match to their own label, agreement is 68.63% and 45.42% receive an additional review flag.

### Limitations
- Scores measure agreement with weak labels on forum data, not verified enterprise-ticket diagnoses.
- Confidence-based review does not detect every ambiguous or unsupported request; engineers retain final review.
- The keyword comparison is one literal baseline and does not establish label-noise rates or that the classifier understands the text; a small qualitative review of ten no-match examples (AI-assisted) found three weakly supported labels.

## M4 — RAG & Retrieval
**Status:** Completed<br>
**Summary:** Build and evaluate retrieval of relevant Ask Ubuntu cases and technical evidence.

### Definition of Done (DoD)
- Evidence is prepared with technical content and source attribution preserved.
- A duplicate-aware benchmark separates development and held-out query groups and excludes self matches.
- BM25, dense and hybrid retrieval are compared using development queries.
- Frozen retrieval settings are evaluated on held-out queries with Recall@K and MRR.
- Reusable retrieval returns ranked evidence and sources, with findings and limitations documented.

### Tasks
- [x] Define answer selection, fallback and chunking policies while preserving commands and code. → `src/retrieval.py`, `notebooks/07_retrieval_preparation.ipynb`
- [x] Preserve source metadata and resolve available author attribution. → `src/retrieval.py`
- [x] Build a benchmark using recorded duplicate targets and full-graph query groups. Exclude self matches and evaluate by question ID. → `src/retrieval.py`, `notebooks/07_retrieval_preparation.ipynb`
- [x] Implement a BM25 baseline. → `src/retrieval.py`
- [x] Evaluate a pretrained dense embedding model with a local vector index on a development pilot. → `notebooks/07_retrieval_preparation.ipynb`
- [x] Compare BM25, dense and hybrid retrieval on development queries; select settings using retrieval quality and runtime. → `src/retrieval.py`, `notebooks/07_retrieval_preparation.ipynb`
- [x] Inspect failure cases, source age and weak-evidence behavior on development and ticket-like examples. → `src/retrieval.py`, `notebooks/07_retrieval_preparation.ipynb`
- [x] Freeze development choices, then report Recall@1/5/10 and MRR@10 once on held-out queries. → `src/retrieval.py`, `notebooks/07_retrieval_preparation.ipynb`
- [x] Save reproducible indexes and benchmark assignments, and provide a reusable retrieval function. → `src/retrieval.py`, `notebooks/07_retrieval_preparation.ipynb`

### Outcomes
- **240,555 cases** prepared with **618,330 chunks**, preserving technical content and source attribution.
- BM25, dense and hybrid retrieval compared on the same **5,000 cases** and **145 development queries**; MiniLM + FAISS selected.
- On **44 held-out queries**: Recall@1 **0.6364**, Recall@5 **0.8409**, Recall@10 **0.8864** and MRR@10 **0.7188**.
- Source age, retrieval mistakes and ticket-like examples reviewed in `notebooks/07_retrieval_preparation.ipynb`.
- Saved indexes and a reusable retriever in `src/retrieval.py` return ranked evidence, sources and a development-derived weak-evidence flag; **20.45%** of held-out queries are flagged.

### Limitations
- Final evaluation covers a sampled corpus and 44 eligible held-out queries, not the full knowledge base or real support tickets.
- Recorded duplicate targets are incomplete relevance labels; community answers require technical and version review.
- The weak-evidence flag is a heuristic and can miss unsuitable advice.

## M5 — Technical Support Agent
**Status:** In Progress<br>
**Summary:** Use classification, retrieval and Ubuntu documentation to draft support recommendations for engineer review.

### Definition of Done (DoD)
- The agent combines the saved classifier and retriever with official Ubuntu documentation and an LLM.
- Recommendations include their sources; the agent requests review when evidence is weak or incompatible.
- A 12-case qualitative evaluation covers all six categories and reviews source relevance, technical correctness, release fit and abstention.
- At least one useful recommendation is checked against its source, and findings and limitations are documented.

### Tasks
- [x] Connect the saved classifier and retriever to an LLM and add Ubuntu documentation as a separate source. -> `src/support_agent.py`, `src/official_docs.py`
- [x] Return the category, evidence, suggested steps and human-review reasons with source links. -> `src/support_agent.py`
- [x] Add checks for missing details, weak or unrelated evidence, release compatibility and unsafe steps. -> `src/support_agent.py`
- [x] Prepare 12 evaluation cases across the six categories; exclude source questions and known duplicates where applicable. -> `notebooks/08_technical_support_agent.ipynb`
- [x] Run the qualitative evaluation and document the results. -> `notebooks/08_technical_support_agent.ipynb`
- [x] Save the reusable agent workflow for the API stage. -> `src/support_agent.py`
- [ ] Re-run the cases after the latest changes and review each response against its expected behavior and sources.

### Expected Outcomes
- An agent that uses the existing classifier, retriever and Ubuntu documentation.
- Source-linked recommendations or a clear request for engineer review.
- A small evaluation and reusable workflow for the API stage.

### Outcomes
- The latest 12-case run completed without notebook errors; the Ubuntu documentation index contained 12 pages and 304 chunks.
- Ten cases abstained. S1 (APT) and H1 (Steam/NVIDIA) returned drafts, but neither met its expected behavior or passed technical review. Both were marked for human review while still showing steps.
- The run also exposed unrelated retrieval results and a missed low-risk Wi-Fi diagnostic. The H1 expected behavior assumes Kubuntu 21.10, which is absent from the loaded Ask Ubuntu question text.
- Leakage checks for H1, S1 and T1 found no matching source questions or known duplicates in the retrieved results.

### Limitations
- Twelve cases are a small qualitative review and do not measure performance on company tickets.
- The documentation set is incomplete, and source links do not by themselves prove that a recommendation is correct. Engineers must review the advice.

## M6 — Backend & API
**Status:** Completed<br>
**Summary:** Integrate components through FastAPI.

### Definition of Done (DoD)
- FastAPI loads the saved classifier, M4 retriever and current M5 support agent for API use.
- Health, classification and ticket-analysis endpoints accept validated requests and return consistent responses.
- A stable API response preserves category, confidence, suggested steps, human-review status, evidence sources, missing information and review reasons.
- Invalid requests and component failures return clear errors without exposing credentials.
- Local endpoint examples cover classification, a support-agent response, a review-required response and an invalid request.

### Tasks
- [x] Define request and response schemas for health, classification and ticket analysis. -> `src/api.py`
- [x] Build a FastAPI app that loads the classifier, retriever and support agent once at startup. -> `src/api.py`
- [x] Add an adapter between the API and the M5 agent so internal agent changes do not require changes to the API response contract. -> `src/api.py`
- [x] Preserve the agent's steps, sources and human-review status in the API response; recommendations remain experimental while M5 is In Progress. -> `src/api.py`
- [x] Validate inputs and handle model, retrieval, LLM and API failures with clear error responses. -> `src/api.py`
- [x] Verify classification and readiness with local artifacts; test support-agent, review-required and invalid-request responses through the API. -> `tests/test_api.py`

### Outcomes
- FastAPI exposes liveness, readiness, classification and ticket-analysis endpoints; model artifacts load once during application startup.
- A stable response adapter preserves category, confidence, steps, citations, source metadata and human-review status.
- Local artifact smoke checks and four API tests passed. No live Gemini generation was used for verification.

## M7 — Streamlit Application
**Status:** Completed<br>
**Summary:** Build the end-to-end product interface.

### Definition of Done (DoD)
- Requesters can submit and track tickets; IT Support Engineers can review the queue, inspect analysis and update ticket status.
- Ticket data persists through the API, and requester status lookup requires a private tracking code.
- Analysis presents category, confidence, suggested steps, sources and review signals, distinguishing official Ubuntu documentation from Ask Ubuntu answers.
- Recommendation, review, unsupported and API-error states display clearly, with the engineer responsible for decisions.
- The local prototype's visual workflow and authentication limitation are documented.

### Tasks
- [x] Define the requester and IT Support Engineer workflow in one Streamlit app.
- [x] Implement persistent ticket submission, private status tracking, support queue, analysis and status updates. -> `src/api.py`, `src/ticket_store.py`, `app.py`
- [x] Present analysis steps, sources, release scope and review signals; distinguish official Ubuntu documentation from community answers. -> `app.py`
- [x] Test ticket privacy, persistence, status transitions, analysis states and API errors. -> `tests/test_api.py`, `tests/test_ui_client.py`
- [x] Run the requester and support-team pages and verify recommendation, review, unsupported and API-error displays.
- [x] Document local setup and the prototype's limitations. -> `README.md`

### Outcomes
- The Streamlit app supports ticket submission and private tracking for requesters, and queue review, analysis and status updates for the support team. -> `app.py`
- FastAPI and SQLite persist tickets and analysis; the requester uses one private tracking code. -> `src/api.py`, `src/ticket_store.py`
- All 16 API and API-client tests passed. The three pages and sample recommendation, review, unsupported and API-error displays rendered without app exceptions. -> `tests/test_api.py`, `tests/test_ui_client.py`
- M7 is complete; M5 remains in progress because the interface checks do not establish the technical correctness of recommendations.

### Limitations
- M5 recommendations still require technical validation; the interface does not establish their correctness.
- Requester tracking is read-only. “Specialist review needed” records a status and does not automatically route the ticket.
- The local support-team workspace has no authentication and is not ready for shared or production use.

## M8 — AI Evaluation
**Status:** Not Started<br>
**Summary:** Evaluate and monitor AI behavior.

## M9 — Testing & Refinement
**Status:** Not Started<br>
**Summary:** Test the complete system and improve failures.

## M10 — Deployment & Documentation
**Status:** Not Started<br>
**Summary:** Deploy and prepare final documentation.
