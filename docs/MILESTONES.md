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
**Status:** Not Started<br>
**Summary:** Train and evaluate a baseline technical issue classifier using question titles and bodies.

### Definition of Done (DoD)
- Label quality is reviewed on a sample, with decisions and limitations documented.
- Reproducible train, validation and test splits prevent known duplicate leakage.
- A TF-IDF + Logistic Regression classifier is trained and evaluated.
- Classification errors and confidence behavior are analyzed.
- The fitted pipeline is saved and can predict a category, confidence and human-review flag for new text.

### Tasks
- [ ] Review a sample from each category and document incorrect or ambiguous labels.
- [ ] Create train/validation/test splits using full duplicate relationships; check identical and near-duplicate text across splits.
- [ ] Build a TF-IDF + Logistic Regression pipeline using Title and BodyText only; fit text features on training data only.
- [ ] Compare the baseline with a majority-class reference and evaluate class weighting on validation data.
- [ ] Select model settings using validation data, then evaluate the selected model on the held-out test set.
- [ ] Report Accuracy, Macro F1, Weighted F1, per-class Precision/Recall/F1 and a Confusion Matrix.
- [ ] Analyze mistakes, confidence calibration and multi-category cases; select a human-review threshold using validation data.
- [ ] Inspect predictions on ticket-like examples and document the limits of transferring forum results to support tickets.
- [ ] Save the fitted pipeline and provide a reusable prediction function.

### Expected Outcomes
- Reusable data splitting, training, and prediction logic.
- A notebook documenting label review, experiments, evaluation, and findings.
- A saved model pipeline and reproducible split assignments.
- Category predictions with confidence scores and a validation-based human-review flag.

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
