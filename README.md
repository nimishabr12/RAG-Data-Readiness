# RAG Data Readiness

Evaluate how data preprocessing affects RAG answer quality, cost, and latency on real-world documents.

## Overview

This project compares baseline and improved RAG pipelines over the same corpus to demonstrate how data preparation influences retrieval quality, context efficiency, and system performance. Built for ML engineers, data engineers, and product teams exploring RAG evaluation and optimization.

**What it does:**  
Runs identical document corpora through two retrieval pipelines—baseline (naive chunking) and improved (section-aware chunking + quality filtering)—then measures answer correctness, faithfulness, context size, and latency.

**Why it exists:**  
Enterprises often plug LLMs into messy documents, resulting in noisy context, wasted tokens, and unreliable answers. This repo demonstrates, with concrete metrics, how structured preprocessing changes those outcomes.

**Who it's for:**  
Engineers and product teams who want to reason about RAG pipelines through experiments, not assumptions.

## Key Features

- Baseline and improved RAG pipelines over the same document set
- Section-aware chunking and quality filtering modules
- Evaluation harness tracking answer correctness, faithfulness, context size, and latency
- Streamlit interface for side-by-side answer comparison and metrics visualization
- Case study using a real NVIDIA 10-K financial filing (130 pages, ~500K characters)

## Architecture

```
PDF(s) → Ingest →
    ├─ Baseline:  fixed chunks → embed → vector DB → top-k → LLM
    └─ Improved:  smart chunks + quality filter → embed → vector DB
                      └→ many candidates → rerank → top-m → LLM

Eval: fixed question set → run through both pipelines → results.csv → summarize
```

**Components:**
- **Ingest & chunking**: Extract text and split into semantic units (baseline: fixed-size, improved: section-aware)
- **Quality filtering**: Score and drop low-quality chunks (boilerplate, duplicates, too short/long)
- **Indexing**: Embed chunks and store in vector database (ChromaDB)
- **Retrieval**: Vector similarity search, optionally reranked for relevance
- **Evaluation**: Run standardized questions, log answers and metrics, compute summary statistics

## Project Structure

```
├── data/                    # Document storage (place PDF files here)
├── pipeline/
│   ├── ingest.py           # Document loading and preprocessing
│   ├── chunk_baseline.py   # Fixed-size text chunking
│   ├── chunk_smart.py      # Section-aware chunking
│   └── quality.py          # Chunk quality scoring and filtering
├── rag/
│   ├── index_baseline.py   # Baseline embedding and indexing
│   ├── index_improved.py   # Improved embedding and indexing
│   └── query.py            # Retrieval and answer generation
├── eval/
│   ├── questions.json      # Evaluation questions and reference hints
│   ├── run_eval.py         # Execute evaluation over both pipelines
│   └── summarize_results.py # Aggregate metrics and generate reports
├── app/
│   └── streamlit_app.py    # Interactive web interface
├── requirements.txt        # Python dependencies
└── .env.example           # Environment configuration template
```

## Setup

1. **Clone the repository:**

```bash
git clone https://github.com/nimishabr12/RAG-Data-Readiness.git
cd RAG-Data-Readiness
```

2. **Create and activate a virtual environment:**

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

4. **Configure environment variables:**

```bash
cp .env.example .env
# Edit .env and add your LLM provider API key (default implementation uses Cohere)
```

## Usage

### Running the Streamlit App

```bash
streamlit run app/streamlit_app.py
```

The app provides:
- **Chat mode**: Ask questions and see baseline vs improved answers side-by-side with context sizes
- **Metrics mode**: View evaluation summary tables and performance deltas

### Pipeline Execution

```bash
# Ingest documents
python pipeline/ingest.py

# Generate chunks (baseline)
python pipeline/chunk_baseline.py

# Create embeddings and index (baseline)
python rag/index_baseline.py

# Create embeddings and index (improved)
python rag/index_improved.py

# Query the system
python rag/query.py
```

### Running Evaluation

```bash
# Execute evaluation over question set
python eval/run_eval.py

# Summarize results
python eval/summarize_results.py
```

## NVIDIA 10-K Case Study

### Document

This implementation uses NVIDIA's 10-K annual report (130 pages, ~500K characters) as a real-world corpus for evaluating RAG data-readiness strategies.

### Experiment Setup

We compared two RAG approaches over the same NVIDIA 10-K corpus:

**Baseline:**
- Simple word-based chunking (500 tokens, 10% overlap)
- Direct vector retrieval (top-5)
- Cohere embed-english-v3.0 + command-r-plus-08-2024

**Improved:**
- Section-aware chunking with quality filtering
- Two-stage retrieval: retrieve 20 candidates, rerank to top-3
- Cohere embed-english-v3.0 + rerank-english-v3.0 + command-r-plus-08-2024

### Results (10 Questions, LLM-as-a-Judge)

| Metric          | Baseline   | Improved   | Delta     |
|-----------------|------------|------------|----------|
| Accuracy        | 70%        | 70%        | 0%       |
| Faithfulness    | 90%        | 70%        | -20%     |
| Avg Context     | 21,753 chars | 14,806 chars | -32%   |
| Avg Latency     | 9.24s      | 7.76s      | -16%     |

### Key Findings

- **Cost-Performance Tradeoff**: Improved mode maintains identical accuracy while reducing context size by 32% and latency by 16%
- **Faithfulness Impact**: 20% reduction in faithfulness suggests the model occasionally makes inferences beyond retrieved context
- **Production Recommendation**: Improved mode is suitable for cost-sensitive deployments where minor faithfulness degradation is acceptable

### Evaluation Framework

- 10 factual questions about NVIDIA financials and business segments
- LLM-as-a-judge scoring (correctness: 1.0/0.5/0.0, faithfulness: 1.0/0.0)
- Side-by-side comparison via Streamlit dashboard (`streamlit run app/streamlit_app.py`)

## Technologies

**Current implementation uses:**
- **Cohere**: LLM, embedding generation, and reranking (can be swapped for other providers)
- **ChromaDB**: Vector database for semantic search
- **Streamlit**: Interactive web interface
- **pypdf/pdfplumber**: PDF document processing
- **pandas**: Data analysis and evaluation

## Configuration & Extensibility

### Changing Models or Providers

The reference implementation uses Cohere for embeddings, reranking, and generation. To use alternative providers:

1. Modify the client initialization in `rag/query.py`
2. Update embedding calls in `rag/index_baseline.py` and `rag/index_improved.py`
3. Adjust reranking logic in `rag/query.py` or remove if not needed

### Adjusting Retrieval Parameters

Edit these knobs in `rag/query.py`:
- `initial_candidates`: Number of chunks retrieved before reranking (improved mode)
- `max_final_chunks`: Number of chunks sent to LLM after reranking
- `chunk_overlap`: Overlap percentage in baseline chunking

### Using Different Documents

1. Place new PDFs in `data/` directory
2. Update `eval/questions.json` with relevant questions for your corpus
3. Rebuild indexes: `python rag/index_baseline.py` and `python rag/index_improved.py`
4. Run evaluation: `python eval/run_eval.py`

## Limitations & Future Work

**Current limitations:**
- Single-document focus (NVIDIA 10-K); multi-document and multi-tenant scenarios not yet covered
- Small evaluation set (10-20 questions); production systems typically require larger benchmarks
- LLM-as-judge evaluation; consider human labeling or automatic metrics (RAGAS, context recall, precision)

**Ideas for extension:**
- Multi-document corpora with cross-document retrieval
- Additional domains: legal contracts, support tickets, medical records
- Richer metrics: context recall, context precision, answer relevancy
- Automated judge pipelines (RAGAS integration, GPT-4 eval)
- Multi-language support

## License

MIT License
