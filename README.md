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
