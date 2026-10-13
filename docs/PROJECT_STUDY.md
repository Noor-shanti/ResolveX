# ResolveX — Project Study

**Project Status:** Proposal & Implementation Plan  
**Prototype Domain:** Ubuntu/Linux Technical Support

# 1. Project Overview

**ResolveX** is an AI-powered technical support and resolution platform designed to assist IT support teams in handling technical issues more efficiently.

The platform supports the workflow from the moment a technical support ticket is submitted until the support engineer reaches a resolution. Instead of requiring the engineer to manually classify the issue, search through previous cases, and review multiple sources for possible solutions, ResolveX brings these steps together within one support workflow.

When a ticket is submitted, the system analyzes its title and description and predicts the most relevant technical issue category together with a confidence score. The support engineer can then request further analysis, where ResolveX searches a technical knowledge base for similar previously solved problems and relevant troubleshooting information.

The retrieved evidence is used to generate a grounded troubleshooting recommendation with its supporting sources. The support engineer reviews the recommendation and remains responsible for the final decision.

If the system has low confidence or cannot retrieve sufficient evidence, the case is flagged for human review or escalation rather than presenting an unsupported solution.

The initial prototype focuses on **Ubuntu/Linux Technical Support**, providing a well-defined domain in which the complete workflow can be implemented and evaluated.

---

# 2. Problem Statement

Technical support engineers receive issues that vary significantly in type, complexity, and available information.

For every incoming request, an engineer may need to understand the problem, identify its technical category, search for similar cases, investigate possible solutions, verify relevant technical information, and finally decide how the issue should be handled.

A significant part of this process involves repetitive classification and information retrieval. Searching manually can also become inefficient when the available technical knowledge is distributed across a large number of previously solved problems.

At the same time, directly generating answers using a language model without reliable supporting information introduces another problem: the generated recommendation may sound convincing even when it is unsupported or inappropriate for the current issue.

ResolveX addresses these problems by combining automatic issue classification, retrieval of relevant technical cases and evidence, and grounded troubleshooting assistance within one workflow while keeping the IT Support Engineer in control of the final decision.

---

# 3. Target Users

The primary user of ResolveX is the **IT Support Engineer**.

The engineer receives technical support tickets and uses the platform to understand and process them. ResolveX assists the engineer by showing the predicted technical issue category and confidence, retrieving related solved cases, presenting relevant technical evidence, and generating a troubleshooting recommendation supported by sources.

The engineer can review, accept, modify, or reject the recommendation and decide whether the ticket should be resolved or escalated.

The secondary user is the **Employee / Requester**, who submits a technical support ticket containing a title and description of the problem and can follow the status of the request.

ResolveX is therefore designed as a **decision-support system for technical support teams**, rather than as an autonomous replacement for support engineers.

---

# 4. Product Scenario

Consider an employee using Ubuntu who installs a system update and later discovers that Wi-Fi is no longer working.

The employee opens ResolveX and submits a support ticket:

**Title:**

Wi-Fi stopped working after Ubuntu update

**Description:**

The employee describes what happened, the symptoms, and any available technical information.

ResolveX analyzes the ticket and may produce:

**Technical Issue Category:** Network & Connectivity

**Confidence:** 91%

The IT Support Engineer receives the ticket together with this recommendation.

When the engineer requests technical analysis, the system searches its Ubuntu/Linux technical knowledge base for similar problems and relevant previously documented solutions.

The most relevant information is retrieved and provided as context for generating a troubleshooting recommendation. The recommendation is therefore based on retrieved technical evidence rather than being generated only from the language model's internal knowledge.

The engineer can inspect both the recommendation and its supporting sources before deciding what action to take.

If the available evidence is sufficient, the engineer can proceed with the recommended troubleshooting process and eventually resolve the ticket.

If the classification confidence is low, the problem appears ambiguous, or reliable supporting evidence cannot be found, ResolveX flags the ticket for **Human Review / Escalation**.

### Product Flow

**Employee submits Ticket → Issue Classification + Confidence → IT Support Engineer reviews Ticket → Technical Analysis → Similar Cases & Relevant Evidence Retrieved → Grounded Troubleshooting Recommendation + Sources → Engineer Reviews → Resolve or Escalate**

---

# 5. Prototype Scope

The long-term concept of ResolveX is not limited to one operating system. However, attempting to support every technical domain in the initial implementation would make development and evaluation difficult to control.

For this reason, the first prototype will focus specifically on:

**Ubuntu/Linux Technical Support**

Six technical issue categories have been defined for the prototype:

1. **Network & Connectivity**
2. **Boot & Startup**
3. **Software & Packages**
4. **Hardware & Drivers**
5. **Storage & Filesystems**
6. **Access & Permissions**

These categories represent the internal technical taxonomy of the ResolveX prototype. They are not intended to represent official Ask Ubuntu departments or the organizational structure of a specific company.

Limiting the prototype to a clear technical domain also allows the classification, retrieval, recommendation, and escalation behavior to be evaluated more consistently.

---

# 6. Data Source and Feasibility Study

The primary data source selected for the prototype is the **Ask Ubuntu Stack Exchange Data Dump**.

Ask Ubuntu provides a large collection of real technical questions and community answers related specifically to Ubuntu/Linux problems, making it suitable for both issue classification and technical knowledge retrieval.

An initial feasibility analysis was performed before implementation.

The extracted data contains:

- **414,451 questions**
- **523,728 answers**
- **318,557 questions with AnswerCount > 0**
- **136,162 questions with a recorded AcceptedAnswerId**
- approximately **3,162 unique tags**

The available tags were analyzed to determine whether the proposed six technical categories could be represented with sufficient data.

The taxonomy audit produced:

| Classification Result | Questions |
| --- | --- |
| Exactly one technical category | 177,194 |
| Multiple technical categories | 36,568 |
| No selected category | 200,689 |

The **177,194 questions associated with exactly one category** provide an initial **single-category dataset** for supervised classification.

Their distribution is:

| Technical Issue Category | Samples |
| --- | --- |
| Hardware & Drivers | 43,777 |
| Software & Packages | 43,585 |
| Network & Connectivity | 30,107 |
| Boot & Startup | 27,048 |
| Storage & Filesystems | 17,841 |
| Access & Permissions | 14,836 |
| **Total** | **177,194** |

The **36,568 questions associated with multiple categories** are particularly useful because they represent more ambiguous technical issues. Instead of forcing them into one artificial category during initial training, they can later be used to study model confidence and human-review behavior.

The unmatched questions will not automatically be labeled as `Other`, since the absence of one of the selected tags does not necessarily mean that the actual technical issue lies outside the six categories.

The initial study therefore indicates that the dataset is sufficiently large and structured to proceed with the proposed prototype.

---

# 7. Technical Approach

ResolveX will process a support ticket through several connected stages.

First, the ticket title and description will be analyzed by a text classification model to estimate the most relevant technical issue category and provide a confidence score.

The initial classification baseline will use **TF-IDF with Logistic Regression**. This provides a strong and interpretable baseline for text classification while also allowing confidence estimates to be analyzed. More complex approaches will only be considered if experiments show that they provide meaningful improvement.

The Ask Ubuntu tags used to construct the category labels will **not** be included as model input features, since doing so would introduce target leakage. The model will receive only the textual information corresponding to the support issue.

After classification, the ticket can be analyzed further by the **Technical Support Agent**, which coordinates the technical investigation.

Rather than representing every system component as a separate agent, the initial design uses one agent with access to specialized capabilities, including ticket classification, similar-issue search, and technical knowledge retrieval.

When additional technical evidence is required, the system performs semantic retrieval against a knowledge base constructed from relevant Ask Ubuntu questions and their associated solutions.

Retrieved cases provide context for the language model, which generates a troubleshooting recommendation grounded in those sources.

The resulting output presented to the support engineer can therefore include:

**Technical Category + Confidence + Similar Cases + Relevant Evidence + Troubleshooting Recommendation + Sources**

If confidence or supporting evidence is insufficient, the workflow moves toward human review rather than forcing a recommendation.

---

# 8. Knowledge Base and Retrieval

The technical knowledge base will be constructed from relevant Ask Ubuntu questions and their associated answers, with particular attention to questions containing Accepted Answers.

Technical preprocessing must preserve information that may be necessary for solving a problem, including terminal commands, code blocks, and inline code.

Relevant metadata such as question and answer identifiers, timestamps, scores, source information, and required attribution information will also be retained where appropriate.

The retrieval process will use semantic similarity to identify technical cases relevant to the current support ticket.

A simplified flow is:

**Current Technical Issue → Semantic Search → Relevant Similar Cases → Technical Evidence → Grounded Recommendation**

Ask Ubuntu also provides relationships between duplicate or related questions through `PostLinks`. These relationships create an opportunity to evaluate whether the retrieval system actually finds known related questions instead of evaluating retrieval only through subjective inspection.

An Accepted Answer will be treated as **community-supported technical evidence**, not as guaranteed or officially verified technical guidance.

---

# 9. System Architecture

ResolveX will follow a layered architecture separating the user interface, backend services, intelligent components, and data layer.

The **Streamlit interface** will provide the user-facing prototype for submitting and reviewing tickets.

A **FastAPI backend** will expose the core system functionality through structured API endpoints and connect the interface to the classification, retrieval, and AI components.

At the intelligence layer, the **Technical Support Agent** will coordinate the ticket analysis workflow and interact with the classification and retrieval capabilities when required.

The retrieval layer will search the indexed Ubuntu/Linux knowledge base and return relevant evidence to the language model.

The language model will then formulate the troubleshooting recommendation based on the current ticket and retrieved context.

The overall architecture can be represented as:

**Streamlit Interface → FastAPI Backend → Technical Support Agent → Classification / Similar-Issue Search / Knowledge Retrieval → Ask Ubuntu Knowledge Base → LLM → Grounded Recommendation + Sources → IT Support Engineer**

Evaluation and observability will operate across this workflow to capture system behavior and identify failure cases.

---

# 10. Evaluation Strategy

ResolveX will not be evaluated only as one final output. Each intelligent component will be evaluated according to its own responsibility.

For technical issue classification, evaluation will include **Accuracy, Macro F1-score, Weighted F1-score, per-category Precision/Recall/F1, Confusion Matrix, and confidence analysis**.

Special attention will be given to data splitting because Ask Ubuntu contains duplicate and closely related questions. Duplicate-connected questions should not be allowed to create unrealistic train/test leakage.

The multi-category cases will also be useful for investigating how classification confidence behaves when a technical issue naturally overlaps several categories.

For semantic retrieval, known duplicate-question relationships can provide a retrieval benchmark. Metrics such as **Recall@K** and **Mean Reciprocal Rank (MRR)** can measure whether relevant known cases appear among the highest-ranked retrieved results.

The final troubleshooting recommendations will be evaluated for relevance, grounding in retrieved evidence, unsupported claims, source usage, and appropriate abstention when evidence is insufficient.

The complete workflow will also be tested using predefined support scenarios to evaluate whether the system uses the appropriate capabilities, handles uncertainty correctly, and escalates cases when necessary.

**Langfuse** will be integrated into the AI workflow to support tracing and observability of elements such as prompts, retrieved context, tool calls, model responses, latency, and failure cases.

---

# 11. Human-in-the-Loop and Escalation

ResolveX is intentionally designed with a human-in-the-loop workflow.

The system assists the engineer but does not independently determine that a technical problem has been solved.

Human review becomes particularly important when:

- classification confidence is low;
- the issue spans several technical areas;
- retrieved evidence is weak;
- available sources conflict;
- the recommendation cannot be sufficiently grounded.

Tickets that cannot be confidently assigned to one of the six supported technical issue categories will **not** be forced into an artificial `Other` category. When classification confidence is insufficient or the issue appears to fall outside the supported prototype scope, the ticket will instead be **flagged for human review**.

The IT Support Engineer retains the ability to override the predicted category, reject or modify a recommendation, and escalate the ticket when necessary.

This design makes uncertainty an explicit part of the workflow instead of hiding it behind an apparently confident generated answer.

---

# 12. Technology Stack

The initial implementation is expected to use:

**Data & Machine Learning:** Python, Pandas, NumPy, scikit-learn and Jupyter Notebook.

**Backend:** FastAPI with REST APIs.

**Application Interface:** Streamlit.

**Retrieval & AI:** an evaluated embedding model, a vector store such as FAISS or Chroma, an LLM API, and LangChain where it provides useful orchestration rather than unnecessary abstraction.

**Evaluation & Observability:** Langfuse together with custom evaluation scripts and benchmarks.

**Development:** Git and GitHub.

Some implementation choices, particularly the embedding model and vector-store configuration, will be finalized experimentally rather than fixed before evaluation.

---

# 13. Risks and Limitations

The current prototype has several known limitations that must be considered during implementation and evaluation.

**Domain Shift:** Ask Ubuntu questions are community forum posts rather than real corporate support tickets. Strong performance on Ask Ubuntu data therefore cannot automatically be interpreted as equivalent performance on enterprise ticket data.

**Historical Knowledge:** The available data contains historical Ubuntu/Linux content, and some solutions may become outdated as operating systems, packages, and hardware change.

**Community Answers:** Accepted Answers are valuable signals of community-supported solutions but are not guaranteed to be technically correct or appropriate in every environment.

**Prototype Taxonomy:** The six categories were designed specifically for ResolveX and should not be presented as official Ask Ubuntu categories or real organizational departments.

**Licensing and Attribution:** Source information and appropriate metadata must be retained so that the final knowledge base and application can comply with the applicable Stack Exchange content licensing and attribution requirements.

These limitations will be documented explicitly rather than hidden by the prototype.

---

# 14. Scope Boundaries

The initial ResolveX prototype will focus on demonstrating a complete, measurable technical support workflow.

It will **not** attempt to support every possible IT domain, automatically execute troubleshooting commands on user devices, replace IT Support Engineers, guarantee that community answers are correct, or train a foundation language model.

The architecture will also avoid adding complexity without a product requirement. Multiple agents will only be introduced if later experiments identify genuinely separate autonomous responsibilities that improve the system. Similarly, additional integration protocols will only be introduced when they solve an actual integration requirement.

The priority is a coherent and evaluable product rather than maximizing the number of technologies used.

---

# 15. Expected Final Product

The final ResolveX prototype is expected to provide an end-to-end technical support experience in which an employee can submit a technical issue and an IT Support Engineer can process it through one integrated system.

The expected workflow is:

**Submit Technical Ticket → Classify Technical Issue + Confidence → Review Ticket → Analyze Issue → Retrieve Similar Solved Cases and Technical Evidence → Generate Grounded Troubleshooting Recommendation + Sources → Engineer Reviews → Resolve or Escalate**

The final application should demonstrate that the individual intelligent components operate as part of one coherent product and that their behavior can be evaluated rather than demonstrated only through selected examples.

The intended result is a functional prototype combining **technical ticket management, machine learning classification, evidence retrieval, grounded AI assistance, API-based system integration, human decision-making, and measurable AI evaluation** within a single technical support platform.