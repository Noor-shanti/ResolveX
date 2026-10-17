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

### Outcomes
- A TF-IDF + Logistic Regression classifier selected using validation data.
- Label review, duplicate-aware splits, confidence calibration and evaluation documented in `notebooks/05_ml_classification.ipynb`.
- Held-out Accuracy **89.68%**, Macro F1 **0.8856**, and Weighted F1 **0.8967** against tag-derived labels.
- A saved classifier with calibrated confidence and a validation-based human-review flag, with reusable prediction logic in `src/models/classification.py`.

### Limitations
- Scores measure agreement with weak labels on forum data, not verified enterprise-ticket diagnoses.
- Confidence-based review does not detect every ambiguous or unsupported request; engineers retain final review.

## M4 — RAG & Retrieval
**Status:** Not Started<br>
**Summary:** Build and evaluate technical knowledge retrieval.

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
