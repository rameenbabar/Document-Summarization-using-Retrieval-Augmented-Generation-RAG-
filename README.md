# Document Summarization using Retrieval Augmented Generation(RAG)

## Overview

This project implements a **Retrieval-Augmented Generation (RAG)** system using **LangChain**, **OpenAI embeddings**, and **Qdrant vector database**. It supports ingestion and semantic search over both **structured data (CSV)** and **unstructured documents (PDFs)**.

We ingest:
- News articles from the `cnn_dailymail` dataset (CSV-based)
- A corporate Annual Report (PDF-based)

The vector store is managed using **Qdrant**, and semantic search uses **hybrid retrieval** (dense + sparse). A **LangGraph-powered pipeline** orchestrates the RAG flow from user query → transformation → retrieval → generation.
The final answers are generated using **Google Gemini** (via `ChatGoogleGenerativeAI`).

---

## Data Sources

### 1. CSV Data (CNN DailyMail)

- Hugging Face Dataset: [cnn_dailymail](https://huggingface.co/datasets/cnn_dailymail)
- Each record represents a news article and its highlights summary

### 2. PDF Data (Annual Report)

- A structured annual report ingested via UnstructuredPDFLoader
- Preprocessed with semantic chunking using `SemanticChunker`

---

## ⚙️ Setup Instructions

### Prerequisites

- Python 3.10+
- Access to:
  - [Qdrant Cloud](https://cloud.qdrant.io/)
  - [OpenAI Platform](https://platform.openai.com/)
  - [Google Generative AI](https://ai.google.dev)

---

### Clone the Repository

```bash
git clone https://github.com/your-username/Document-Summarization-using-Retrieval-Augmented-Generation-RAG-.git
cd Document-Summarization-using-Retrieval-Augmented-Generation-RAG-
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Environment Configuration

All sensitive credentials and configuration parameters are stored in a dedicated `configs.py`

#### configs.py Template

```bash
OPENAI_API_KEY = "your-openai-key"
GOOGLE_API_KEY = "your-gemini-key"
QDRANT_API_KEY = "your-qdrant-api-key"
QDRANT_URL = "https://your-qdrant-instance"
```

Note: Never commit this file to version control. Add it to your .gitignore to protect your credentials.

---

## 🧩 Data Ingestion Pipeline

### CSV Ingestion Flow (`ingestion.ipynb`)

Load subset of CNN/DailyMail dataset.

Save and load articles via CSVLoader.

Embed documents using OpenAI Embeddings.

Store vectors in Qdrant with sparse + dense representations.

### PDF Ingestion Flow (`pdf_ingestion.ipynb.py`)

Load PDF using UnstructuredPDFLoader.

Extract and semantically chunk text with SemanticChunker.

Generate embeddings and store in a separate Qdrant collection.

---

## 🔄 Retrieval-Augmented Generation (RAG)

### Implemented using LangGraph:

#### Query Transformation: 
Reformulates user questions using Gemini

#### Document Retrieval: 
Hybrid semantic search with Qdrant

#### Answer Generation: 
Final answer generated using Gemini

### Two RAG pipelines exist:

- One for CSV (`chat.py`)

- One for PDF (`chat_pdf.py`)

---

## 📊 Vector DB: Qdrant

**Qdrant is used for storing high-dimensional embeddings and supports:**

Fast hybrid retrieval

Sparse + dense vector search

Cloud endpoint secured via API key

---

## 📌 Streamlit Interface (`app.py`)

Run the following command to launch the interactive web interface:

```bash
streamlit run app.py
```

### Features of the Interface:

**User Input:** Type natural language queries.

**Summary:** Extracted answer from the document.

**Context:** Snippets retrieved from the vector store.

**Transformed Query:** Reformulated version of the input query via Gemini.

**Similarity Scores:** BM25-based match score of the retrieved chunks.

**Latency:** Total time taken for the entire RAG pipeline.

---

## 💡 Technologies Used

**LangChain:**	RAG orchestration + vector pipelines

**OpenAI Embeddings:**	Semantic understanding (text-embedding-3-large)

**Qdrant Cloud:**	Vector store with hybrid retrieval

**HuggingFace Datasets:**	Loading large-scale news articles

**Unstructured Loader:**	Parsing and segmenting PDFs

**LangGraph:**	Declarative multi-step RAG pipelines

**Gemini:** (Google)	LLM for answer generation

---
