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

## Technologies

- **Cohere**: LLM and embedding generation
- **ChromaDB**: Vector database for semantic search
- **Streamlit**: Interactive web interface
- **pypdf/pdfplumber**: PDF document processing
- **pandas**: Data analysis and evaluation

## License

MIT License
