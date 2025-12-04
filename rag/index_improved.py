"""
Improved indexing module for RAG system.
Uses smart chunking with quality filtering for better retrieval performance.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cohere
import chromadb
from dotenv import load_dotenv

from pipeline.ingest import load_pdf
from pipeline.chunk_smart import extract_sections, chunk_text_smart
from pipeline.quality import filter_chunks, analyze_chunk_quality


def create_embeddings(chunks: list[dict], cohere_api_key: str) -> list:
    """
    Generate embeddings for text chunks using Cohere embed-4 model.

    Args:
        chunks: List of chunk dicts with 'text' field
        cohere_api_key: Cohere API key

    Returns:
        List of embeddings (vectors)
    """
    co = cohere.Client(cohere_api_key)

    # Extract text from chunk dicts
    texts = [chunk['text'] for chunk in chunks]

    print(f"Creating embeddings for {len(texts)} chunks using Cohere embed-english-v3.0...")

    # Cohere embed API - using embed-english-v3.0 (embed-4)
    response = co.embed(
        texts=texts,
        model="embed-english-v3.0",
        input_type="search_document"
    )

    embeddings = response.embeddings

    print(f"Successfully created {len(embeddings)} embeddings")
    print(f"Embedding dimension: {len(embeddings[0])}")

    return embeddings


def index_documents_improved(chunks: list[dict], embeddings: list, collection_name: str = "nvidia_improved") -> None:
    """
    Index document chunks in ChromaDB with embeddings and rich metadata.

    Args:
        chunks: List of chunk dicts with metadata
        embeddings: Corresponding embeddings
        collection_name: Name of the ChromaDB collection (default: nvidia_improved)
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
        metadata={"description": "Nvidia Report with smart chunking and quality filtering"}
    )

    # Prepare data for ChromaDB
    ids = [chunk['id'] for chunk in chunks]
    documents = [chunk['text'] for chunk in chunks]

    # Rich metadata including section info and quality scores
    metadatas = [
        {
            "section_id": chunk.get('section_id', ''),
            "section_title": chunk.get('section_title', ''),
            "chunk_length": len(chunk['text']),
            "word_count": len(chunk['text'].split()),
            "quality_score": chunk.get('quality_score', 0.0),
            "start_char": chunk.get('start_char', 0),
            "end_char": chunk.get('end_char', 0)
        }
        for chunk in chunks
    ]

    # Add to collection
    print(f"Adding {len(chunks)} chunks to ChromaDB collection '{collection_name}'...")
    collection.add(
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

    print(f"Successfully indexed {len(chunks)} chunks in ChromaDB")
    print(f"Collection '{collection_name}' now has {collection.count()} documents")


def main():
    """Main improved indexing workflow."""
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

    print("=" * 70)
    print("IMPROVED INDEXING PIPELINE")
    print("=" * 70)
    print()

    # Step 1: Load PDF
    print("=" * 70)
    print("STEP 1: Loading PDF")
    print("=" * 70)
    full_text, pages = load_pdf(pdf_path)
    print(f"Loaded {len(pages)} pages with {len(full_text)} characters\n")

    # Step 2: Extract sections
    print("=" * 70)
    print("STEP 2: Extracting sections with smart chunking")
    print("=" * 70)
    sections = extract_sections(full_text)
    print(f"Extracted {len(sections)} sections\n")

    # Step 3: Create smart chunks
    print("=" * 70)
    print("STEP 3: Creating smart chunks")
    print("=" * 70)
    candidate_chunks = chunk_text_smart(sections, max_tokens=600)
    print(f"Created {len(candidate_chunks)} candidate chunks")
    print(f"Average words per chunk: {sum(len(c['text'].split()) for c in candidate_chunks) // len(candidate_chunks)}\n")

    # Step 4: Analyze quality
    print("=" * 70)
    print("STEP 4: Analyzing chunk quality")
    print("=" * 70)
    quality_stats = analyze_chunk_quality(candidate_chunks)
    print(f"Total chunks: {quality_stats['total_chunks']}")
    print(f"Mean quality score: {quality_stats['mean_score']:.3f}")
    print(f"Score range: [{quality_stats['min_score']:.3f}, {quality_stats['max_score']:.3f}]")
    print(f"Low quality (< 0.4): {quality_stats['low_quality_count']}")
    print(f"Medium quality (0.4-0.7): {quality_stats['medium_quality_count']}")
    print(f"High quality (>= 0.7): {quality_stats['high_quality_count']}\n")

    # Step 5: Filter chunks
    print("=" * 70)
    print("STEP 5: Filtering low-quality chunks")
    print("=" * 70)
    filtered_chunks = filter_chunks(candidate_chunks, min_score=0.4)
    removed_count = len(candidate_chunks) - len(filtered_chunks)
    print(f"Kept {len(filtered_chunks)} chunks (removed {removed_count})")
    print(f"Retention rate: {len(filtered_chunks)/len(candidate_chunks)*100:.1f}%\n")

    # Show section distribution
    section_counts = {}
    for chunk in filtered_chunks:
        section_id = chunk.get('section_id', 'unknown')
        section_counts[section_id] = section_counts.get(section_id, 0) + 1

    print("Top sections by chunk count:")
    for section_id, count in sorted(section_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  ITEM {section_id}: {count} chunks")
    print()

    # Step 6: Create embeddings
    print("=" * 70)
    print("STEP 6: Creating embeddings with Cohere embed-4")
    print("=" * 70)
    embeddings = create_embeddings(filtered_chunks, cohere_api_key)
    print()

    # Step 7: Index in ChromaDB (separate collection)
    print("=" * 70)
    print("STEP 7: Indexing in ChromaDB (improved collection)")
    print("=" * 70)
    index_documents_improved(filtered_chunks, embeddings, collection_name="nvidia_improved")
    print()

    # Summary comparison
    print("=" * 70)
    print("INDEXING COMPLETE - SUMMARY")
    print("=" * 70)
    print(f"Collection name: nvidia_improved")
    print(f"Total chunks indexed: {len(filtered_chunks)}")
    print(f"Chunks removed by quality filter: {removed_count}")
    print(f"Average quality score: {quality_stats['mean_score']:.3f}")
    print(f"Sections covered: {len(section_counts)}")
    print()
    print("Improvements over baseline:")
    print("  * Smart chunking respects document structure")
    print("  * Section metadata preserved for better context")
    print("  * Quality filtering removes low-value chunks")
    print("  * Rich metadata (section titles, quality scores, positions)")
    print()
    print("Note: Baseline collection 'nvidia_report' is preserved.")
    print("=" * 70)


if __name__ == "__main__":
    main()
