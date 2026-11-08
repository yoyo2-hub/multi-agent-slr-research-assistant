### Multi-Agent SLR-Based Research Idea Assessment and Improvement System

<p align="center">
  An intelligent Multi-Agent RAG system designed to analyze research ideas, explore existing scientific literature, identify potential research gaps, and suggest evidence-grounded improvements.
</p>

---

## 📌 Overview

ResearchForge is an AI-powered research assistant that helps researchers evaluate and improve their research ideas using a combination of **Systematic Literature Review (SLR), Retrieval-Augmented Generation (RAG), and Multi-Agent Systems**.

Given an initial research idea, ResearchForge explores relevant scientific publications in the field of Computer Science, analyzes existing approaches, compares the proposed idea with previous studies, identifies potential research gaps, and generates recommendations to enhance the original research proposal.

Unlike a traditional literature review tool, ResearchForge aims to go beyond information retrieval and summarization by providing an interactive, collaborative, and evidence-based research idea assessment process.

## 🎯 Objectives

The main objectives of ResearchForge are:

- Analyze and structure user-provided research ideas.
- Retrieve relevant scientific publications from academic sources.
- Analyze existing research methods, contributions, and limitations.
- Compare proposed research ideas with existing studies.
- Identify potential research gaps and research opportunities.
- Critically evaluate the strengths and weaknesses of research ideas.
- Generate evidence-grounded suggestions for research improvement.
- Help researchers formulate stronger research questions and contributions.

## 🏗️ System Architecture

ResearchForge follows a Multi-Agent architecture orchestrated using **LangGraph**, with a shared Scientific RAG system that provides relevant scientific evidence to the agents.

### High-Level Workflow

```text
              Research Idea
                    |
                    v
          +--------------------+
          |  Agent 1           |
          |  Idea Analyzer     |
          +--------------------+
                    |
                    v
          +--------------------+
          |  Agent 2           |
          |  Literature        |
          |  Explorer          |
          +--------------------+
                    |
                    v
          +--------------------+
          |  Agent 3           |
          |  Scientific Paper  |
          |  Analyzer          |
          +--------------------+
                    |
                    v
          +--------------------+
          |  Agent 4           |
          |  Research          |
          |  Comparison        |
          +--------------------+
                    |
                    v
          +--------------------+
          |  Agent 5           |
          |  Research Gap      |
          |  Discovery         |
          +--------------------+
                    |
                    v
          +--------------------+
          |  Agent 6           |
          |  Critical Reviewer |
          +--------------------+
                    |
                    v
          +--------------------+
          |  Agent 7           |
          |  Research          |
          |  Improvement       |
          +--------------------+
                    |
                    v
             Final Report
```

The agents interact with a shared Scientific RAG system to retrieve relevant publications and supporting evidence throughout the workflow.

## 🤖 The Seven Specialized Agents

| Agent | Role |
|---|---|
| 1. Research Idea Analyzer | Structures the initial research idea into problems, objectives, methods, and research questions. |
| 2. Literature Explorer | Searches academic databases, retrieves publications, and performs literature screening. |
| 3. Scientific Paper Analyzer | Extracts methodologies, contributions, results, limitations, and future work from scientific papers. |
| 4. Research Comparison Agent | Compares the proposed idea with existing scientific studies. |
| 5. Research Gap Discovery Agent | Identifies potential research gaps and opportunities based on literature evidence. |
| 6. Critical Reviewer | Evaluates the idea's strengths, weaknesses, feasibility, and potential novelty risks. |
| 7. Research Improvement Agent | Generates recommendations and proposes an improved research idea. |

## 📚 Scientific RAG Component

The Scientific RAG system acts as a shared knowledge retrieval layer for the seven agents.

Its main responsibilities include:

1. **Document Collection:** Collect scientific publications and accessible PDF documents.
2. **Text Extraction:** Extract content and metadata from scientific papers.
3. **Chunking:** Divide documents into meaningful scientific sections and passages.
4. **Embedding Generation:** Convert text passages into vector representations.
5. **Indexing:** Store embeddings and metadata in a vector database.
6. **Hybrid Retrieval:** Combine semantic vector search with keyword-based retrieval (BM25).
7. **Re-ranking:** Improve retrieval relevance using a Cross-Encoder.
8. **Evidence Retrieval:** Provide relevant passages with source information, paper identifiers, and page references.

The RAG component helps ground agent-generated analyses and recommendations in scientific publications rather than relying exclusively on the language model's internal knowledge.

## ⚙️ Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| LLM Framework | LangChain |
| Multi-Agent Orchestration | LangGraph |
| Language Models | OpenAI API or compatible LLMs |
| Academic Data Sources | OpenAlex, Semantic Scholar |
| PDF Processing | PyMuPDF |
| Embeddings | Sentence Transformers |
| Vector Database | Qdrant |
| Keyword Retrieval | BM25 |
| Re-ranking | Cross-Encoder |
| Interface | Streamlit |
| Data Processing | Pandas, Scikit-learn |
| Metadata Storage | SQLite (MVP) |

## 📂 Project Structure

```text
ResearchForge/
│
├── app.py
├── requirements.txt
├── .env.example
├── README.md
│
├── agents/
│   ├── idea_analyzer.py
│   ├── literature_explorer.py
│   ├── paper_analyzer.py
│   ├── comparison_agent.py
│   ├── gap_discovery_agent.py
│   ├── critical_reviewer.py
│   └── improvement_agent.py
│
├── rag/
│   ├── ingestion.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── indexing.py
│   ├── retriever.py
│   ├── reranker.py
│   └── evidence.py
│
├── slr/
│   ├── search.py
│   ├── deduplication.py
│   ├── screening.py
│   └── extraction.py
│
├── graph/
│   ├── state.py
│   ├── nodes.py
│   └── workflow.py
│
├── evaluation/
│   ├── retrieval_eval.py
│   ├── gap_eval.py
│   └── improvement_eval.py
│
├── database/
├── data/
├── reports/
└── tests/
```

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/ResearchForge.git
cd ResearchForge
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**Windows:**

```bash
venv\Scripts\activate
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file based on `.env.example`.

```env
OPENAI_API_KEY=your_api_key
OPENALEX_API_KEY=your_api_key
SEMANTIC_SCHOLAR_API_KEY=your_api_key
```

Only configure the API keys required by the providers you actually use.

### 5. Run the application

```bash
streamlit run app.py
```

## 🔄 Workflow

A typical ResearchForge execution follows these steps:

**Step 1 — Research Idea Submission**

The user submits an initial research idea in Computer Science.

**Step 2 — Idea Analysis**

The first agent extracts the research problem, objectives, methods, and keywords.

**Step 3 — Literature Exploration**

Relevant scientific publications are retrieved, deduplicated, and screened.

**Step 4 — Scientific Paper Analysis**

The system analyzes selected publications and extracts their contributions, methodologies, results, and limitations.

**Step 5 — Research Comparison**

The proposed idea is compared with existing studies.

**Step 6 — Research Gap Discovery**

Potential research gaps and opportunities are identified based
