"""
Indexing module for RAG system.
Handles embedding generation and vector storage using ChromaDB.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cohere
import chromadb
from dotenv import load_dotenv

from pipeline.ingest import load_pdf
from pipeline.chunk_baseline import chunk_text_baseline


def create_embeddings(chunks: list[str], cohere_api_key: str) -> list:
    """
    Generate embeddings for text chunks using Cohere embed-4 model.

    Args:
        chunks: List of text chunks
        cohere_api_key: Cohere API key

    Returns:
        List of embeddings (vectors)
    """
    co = cohere.Client(cohere_api_key)

    print(f"Creating embeddings for {len(chunks)} chunks using Cohere embed-4...")

    # Cohere embed API - using embed-english-v3.0 (embed-4)
    response = co.embed(
        texts=chunks,
        model="embed-english-v3.0",
        input_type="search_document"
    )

    embeddings = response.embeddings

    print(f"Successfully created {len(embeddings)} embeddings")
    print(f"Embedding dimension: {len(embeddings[0])}")

    return embeddings


def index_documents(chunks: list[str], embeddings: list, collection_name: str = "nvidia_report") -> None:
    """
    Index document chunks in ChromaDB with embeddings and metadata.

    Args:
        chunks: List of text chunks
        embeddings: Corresponding embeddings
        collection_name: Name of the ChromaDB collection
    """
    # Initialize ChromaDB client (persistent storage)
    client = chromadb.PersistentClient(path="./chroma_db")

    # Delete collection if it exists (for clean runs)
    try:
        client.delete_collection(name=collection_name)
        print(f"Deleted existing collection: {collection_name}")
    except:
        pass

    # Create collection
    collection = client.create_collection(
        name=collection_name,
        metadata={"description": "Nvidia Report embeddings with Cohere embed-4"}
    )

    # Prepare data for ChromaDB
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "chunk_index": i,
            "chunk_length": len(chunks[i]),
            "word_count": len(chunks[i].split())
        }
        for i in range(len(chunks))
    ]

    # Add to collection
    print(f"Adding {len(chunks)} chunks to ChromaDB collection '{collection_name}'...")
    collection.add(
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
        ids=ids
    )

    print(f"Successfully indexed {len(chunks)} chunks in ChromaDB")
    print(f"Collection '{collection_name}' now has {collection.count()} documents")


def main():
    """Main indexing workflow."""
    # Load environment variables
    load_dotenv()
    cohere_api_key = os.getenv("COHERE_API_KEY")

    if not cohere_api_key:
        print("Error: COHERE_API_KEY not found in .env file")
        return

    pdf_path = "data/Nvidia Report.pdf"

    if not os.path.exists(pdf_path):
        print(f"Error: {pdf_path} not found")
        return

    # Step 1: Load PDF
    print("=" * 60)
    print("STEP 1: Loading PDF")
    print("=" * 60)
    full_text, pages = load_pdf(pdf_path)
    print(f"Loaded {len(pages)} pages with {len(full_text)} characters\n")

    # Step 2: Create chunks
    print("=" * 60)
    print("STEP 2: Creating baseline chunks")
    print("=" * 60)
    chunks = chunk_text_baseline(full_text, max_tokens=500)
    print(f"Created {len(chunks)} chunks")
    print(f"Average words per chunk: {sum(len(c.split()) for c in chunks) // len(chunks)}\n")

    # Step 3: Create embeddings
    print("=" * 60)
    print("STEP 3: Creating embeddings with Cohere embed-4")
    print("=" * 60)
    embeddings = create_embeddings(chunks, cohere_api_key)
    print()

    # Step 4: Index in ChromaDB
    print("=" * 60)
    print("STEP 4: Indexing in ChromaDB")
    print("=" * 60)
    index_documents(chunks, embeddings)
    print()

    print("=" * 60)
    print("INDEXING COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
