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
**Status:** In Progress<br>
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
- [ ] Compare BM25, dense and hybrid retrieval on development queries; select settings using retrieval quality and runtime.
- [ ] Inspect failure cases, source age and weak-evidence behavior on development and ticket-like examples.
- [ ] Freeze development choices, then report Recall@1/5/10 and MRR@10 once on held-out queries.
- [ ] Save reproducible indexes and benchmark assignments, and provide a reusable retrieval function.

### Current Outcomes
- **240,555 cases** prepared with **618,330 chunks**, preserving technical content and source metadata.
- A duplicate-aware benchmark with **28,721 development** and **2,924 held-out queries**.
- BM25 evaluated on **100 development queries**: Recall@10 **0.2400** and MRR@10 **0.1198**.
- A MiniLM + FAISS pilot built on **2,000 cases** and evaluated on **56 development queries**; its results are not directly comparable with full-corpus BM25.

### Expected Outcomes
- Reusable evidence preparation, indexing and retrieval logic.
- A notebook documenting the benchmark, comparisons, findings and limitations.
- A selected retriever returning ranked evidence and sources for M5.
- Reproducible local artifacts and held-out retrieval metrics.

## M5 — Technical Support Agent
**Status:** Not Started<br>
**Summary:** Build the ticket analysis and resolution workflow.

## M6 — Backend & API
**Status:** Not Started<br>
**Summary:** Integrate components through FastAPI.

## M7 — Streamlit Application
**Status:** Not Started<br>
**Summary:** Build the end-to-end product interface.

## M8 — AI Evaluation
**Status:** Not Started<br>
**Summary:** Evaluate and monitor AI behavior.

## M9 — Testing & Refinement
**Status:** Not Started<br>
**Summary:** Test the complete system and improve failures.

## M10 — Deployment & Documentation
**Status:** Not Started<br>
**Summary:** Deploy and prepare final documentation.
