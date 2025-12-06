"""
Query module for RAG system.
Handles retrieval and response generation using Cohere.
"""

import os
from typing import Literal, Optional
from dataclasses import dataclass
import cohere
import chromadb
from dotenv import load_dotenv


__all__ = ['answer_question', 'RetrievalConfig', 'IMPROVED_BEST']


@dataclass
class RetrievalConfig:
    """Configuration for improved-mode retrieval parameters."""
    initial_candidates: int  # Number of candidates to retrieve before reranking
    max_final_chunks: int    # Number of chunks to keep after reranking
    rerank_model: str        # Cohere rerank model to use


# Preset configurations
IMPROVED_BEST = RetrievalConfig(
    initial_candidates=20,
    max_final_chunks=3,
    rerank_model="rerank-english-v3.0"
)
# IMPROVED_BEST achieves same accuracy as baseline (±0.02) with:
# - 32% lower context size (14.8k vs 21.8k chars)
# - 16% lower latency (7.8s vs 9.2s)


def answer_question(
    question: str,
    mode: Literal['baseline', 'improved'] = 'baseline',
    top_k: int = 5,
    retrieval_config: Optional[RetrievalConfig] = None
) -> tuple[str, str, list[str]]:
    """
    Answer a question using RAG with Cohere and ChromaDB.

    Args:
        question: User's question
        mode: 'baseline' (simple retrieval) or 'improved' (rerank + smart chunks)
        top_k: Number of final chunks to use for generation in baseline mode (default: 5)
        retrieval_config: Configuration for improved mode retrieval. If None, uses IMPROVED_BEST preset.
                         Only applies when mode='improved'.

    Returns:
        Tuple of (answer, mode, chunk_ids) where:
        - answer: Generated response from Cohere
        - mode: The mode used ('baseline' or 'improved')
        - chunk_ids: List of chunk IDs used for generation
    """
    # Load environment variables
    load_dotenv()
    cohere_api_key = os.getenv("COHERE_API_KEY")

    if not cohere_api_key:
        raise ValueError("COHERE_API_KEY not found in .env file")

    # Initialize Cohere client
    co = cohere.Client(cohere_api_key)

    # Apply retrieval configuration
    if mode == 'improved':
        # Use provided config or default to IMPROVED_BEST preset
        config = retrieval_config if retrieval_config is not None else IMPROVED_BEST
        collection_name = "nvidia_improved"
        retrieval_count = config.initial_candidates
    else:  # baseline
        config = None
        collection_name = "nvidia_report"
        retrieval_count = top_k  # Retrieve exactly top_k for baseline

    print(f"Mode: {mode.upper()}")
    print(f"Collection: {collection_name}")
    if config:
        print(f"Config: {config.initial_candidates} candidates → rerank to {config.max_final_chunks} chunks")
    print()

    # Step 1: Embed the question
    print(f"Step 1: Embedding question with Cohere...")
    query_embedding_response = co.embed(
        texts=[question],
        model="embed-english-v3.0",
        input_type="search_query"
    )
    query_embedding = query_embedding_response.embeddings[0]

    # Step 2: Retrieve similar chunks from ChromaDB
    print(f"Step 2: Retrieving top-{retrieval_count} candidates from ChromaDB...")
    client = chromadb.PersistentClient(path="./chroma_db")

    try:
        collection = client.get_collection(name=collection_name)
    except Exception as e:
        raise ValueError(f"Collection '{collection_name}' not found. Please run the appropriate indexing script first.")

    # Query ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=retrieval_count
    )

    # Extract results
    candidate_ids = results['ids'][0]
    candidate_docs = results['documents'][0]
    candidate_distances = results['distances'][0]

    print(f"Retrieved {len(candidate_docs)} candidate chunks")

    # Step 3: Rerank (only for improved mode)
    if mode == 'improved':
        print(f"\nStep 3: Reranking with Cohere {config.rerank_model}...")

        # Call Cohere Rerank with config parameters
        rerank_response = co.rerank(
            model=config.rerank_model,
            query=question,
            documents=candidate_docs,
            top_n=config.max_final_chunks
        )

        # Extract reranked results
        reranked_indices = [result.index for result in rerank_response.results]
        final_ids = [candidate_ids[i] for i in reranked_indices]
        final_docs = [candidate_docs[i] for i in reranked_indices]

        print(f"Reranked to top-{config.max_final_chunks} chunks:")
        for i, (result, chunk_id) in enumerate(zip(rerank_response.results, final_ids)):
            print(f"  {i+1}. {chunk_id} (relevance score: {result.relevance_score:.4f})")

    else:  # baseline mode - use top_k directly
        final_ids = candidate_ids[:top_k]
        final_docs = candidate_docs[:top_k]

        print(f"\nUsing top-{top_k} chunks (by vector similarity):")
        for i, (chunk_id, distance) in enumerate(zip(final_ids, candidate_distances[:top_k])):
            print(f"  {i+1}. {chunk_id} (distance: {distance:.4f})")

    # Step 4: Generate answer with Cohere chat
    print(f"\nStep {4 if mode == 'improved' else 3}: Generating answer with Cohere command-r-plus-08-2024...")

    # Format documents for Cohere chat
    formatted_docs = [{"text": doc} for doc in final_docs]

    response = co.chat(
        model="command-r-plus-08-2024",
        message=question,
        documents=formatted_docs
    )

    answer = response.text

    return answer, mode, final_ids


# Keep backward compatibility
def answer_question_baseline(question: str, top_k: int = 5, collection_name: str = "nvidia_report") -> tuple[str, list[str]]:
    """
    Legacy function for backward compatibility.
    Use answer_question() with mode parameter instead.
    """
    answer, _, chunk_ids = answer_question(question, mode='baseline', top_k=top_k)
    return answer, chunk_ids


def main():
    """Main query workflow."""
    load_dotenv()

    # Check if ChromaDB collection exists
    if not os.path.exists("./chroma_db"):
        print("Error: ChromaDB database not found. Please run indexing scripts first.")
        return

    print("=" * 70)
    print("RAG QUERY SYSTEM")
    print("=" * 70)
    print()

    # Example questions
    example_questions = [
        "What is NVIDIA's total revenue?",
        "What are the main business segments of NVIDIA?",
        "What risks does NVIDIA face in its business?"
    ]

    print("Example questions:")
    for i, q in enumerate(example_questions, 1):
        print(f"  {i}. {q}")
    print()

    # Get user input
    question = input("Enter your question (or press Enter for example 1): ").strip()

    if not question:
        question = example_questions[0]
        print(f"Using example question: {question}")

    print()

    # Choose mode
    print("Select mode:")
    print("  1. Baseline (simple vector retrieval)")
    print("  2. Improved (smart chunks + reranking)")
    mode_choice = input("Enter choice (1 or 2, default=1): ").strip()

    mode = 'improved' if mode_choice == '2' else 'baseline'

    print()
    print("=" * 70)

    try:
        answer, used_mode, chunk_ids = answer_question(question, mode=mode, top_k=5)

        print("\n" + "=" * 70)
        print(f"ANSWER (Mode: {used_mode.upper()})")
        print("=" * 70)
        print(answer)
        print()
        print("=" * 70)
        print(f"Mode used: {used_mode}")
        print(f"Chunks used: {len(chunk_ids)}")
        print(f"Sources: {', '.join(chunk_ids)}")
        print("=" * 70)

    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure you have:")
        print("  1. Run 'python rag/index_baseline.py' for baseline mode")
        print("  2. Run 'python rag/index_improved.py' for improved mode")
        print("  3. Set COHERE_API_KEY in your .env file")


if __name__ == "__main__":
    main()
