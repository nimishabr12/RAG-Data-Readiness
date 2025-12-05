# RAG Data-Readiness Tool

A Retrieval-Augmented Generation (RAG) system built with Cohere for evaluating and improving data readiness in AI applications.

## Overview

This tool helps assess the quality and readiness of document data for RAG applications by providing:
- Document ingestion and preprocessing pipelines
- Multiple chunking strategies for optimal text segmentation
- Embedding generation and vector storage using ChromaDB
- Query and retrieval capabilities powered by Cohere
- Interactive Streamlit interface for testing and evaluation
- Evaluation framework for measuring RAG performance

## Project Structure

```
├── data/                    # Document storage (place PDF files here)
├── pipeline/
│   ├── ingest.py           # Document loading and preprocessing
│   └── chunk_baseline.py   # Text chunking strategies
├── rag/
│   ├── index_baseline.py   # Embedding generation and indexing
│   └── query.py            # Retrieval and response generation
├── eval/
│   └── questions.json      # Evaluation questions and metrics
├── app/
│   └── streamlit_app.py    # Interactive web interface
├── requirements.txt        # Python dependencies
└── .env.example           # Environment configuration template
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/nimishabr12/RAG-Data-Readiness.git
cd RAG-Data-Readiness
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
```bash
cp .env.example .env
# Edit .env and add your Cohere API key
```

## Usage

### Running the Streamlit App
```bash
streamlit run app/streamlit_app.py
```

### Pipeline Execution
```bash
# Ingest documents
python pipeline/ingest.py

# Generate chunks
python pipeline/chunk_baseline.py

# Create embeddings and index
python rag/index_baseline.py

# Query the system
python rag/query.py
```

## NVIDIA 10-K Case Study

### Document
This implementation uses NVIDIA's 10-K annual report (130 pages, ~500K characters) as a real-world corpus for evaluating RAG data-readiness strategies.

### Experiment Setup
We compared two RAG approaches over the same NVIDIA 10-K corpus using Cohere's API suite:

**Baseline:**
- Simple word-based chunking (500 tokens, 10% overlap)
- Direct vector retrieval (top-5)
- Cohere `embed-english-v3.0` + `command-r-plus-08-2024`

**Improved:**
- Section-aware chunking with quality filtering
- Two-stage retrieval: retrieve 20 candidates, rerank to top-3
- Cohere `embed-english-v3.0` + `rerank-english-v3.0` + `command-r-plus-08-2024`

### Results (10 Questions, LLM-as-a-Judge)

| Metric | Baseline | Improved | Delta |
|--------|----------|----------|-------|
| **Accuracy** | 70% | 70% | 0% |
| **Faithfulness** | 90% | 70% | -20% |
| **Avg Context** | 21,753 chars | 14,806 chars | **-32%** ✓ |
| **Avg Latency** | 9.24s | 7.76s | **-16%** ✓ |

**Key Findings:**
- **Cost-Performance Tradeoff**: Improved mode maintains identical accuracy while reducing context size by 32% and latency by 16%
- **Faithfulness Impact**: 20% reduction in faithfulness suggests the model occasionally makes inferences beyond retrieved context
- **Production Recommendation**: Improved mode is suitable for cost-sensitive deployments where minor faithfulness degradation is acceptable

**Evaluation Framework:**
- 10 factual questions about NVIDIA financials and business segments
- LLM-as-a-judge scoring (correctness: 1.0/0.5/0.0, faithfulness: 1.0/0.0)
- Side-by-side comparison via Streamlit dashboard (`streamlit run app/streamlit_app.py`)

## Technologies

- **Cohere**: LLM and embedding generation
- **ChromaDB**: Vector database for semantic search
- **Streamlit**: Interactive web interface
- **pypdf/pdfplumber**: PDF document processing
- **pandas**: Data analysis and evaluation

## License

MIT License
