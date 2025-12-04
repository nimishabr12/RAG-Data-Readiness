"""
Query module for RAG system.
Handles retrieval and response generation using Cohere.
"""

import os
import cohere
import chromadb
from dotenv import load_dotenv


def answer_question_baseline(question: str, top_k: int = 5, collection_name: str = "nvidia_report") -> tuple[str, list[str]]:
    """
    Answer a question using RAG with Cohere and ChromaDB.

    Args:
        question: User's question
        top_k: Number of chunks to retrieve (default: 5)
        collection_name: ChromaDB collection name

    Returns:
        Tuple of (answer, chunk_ids) where:
        - answer: Generated response from Cohere
        - chunk_ids: List of chunk IDs used for retrieval
    """
    # Load environment variables
    load_dotenv()
    cohere_api_key = os.getenv("COHERE_API_KEY")

    if not cohere_api_key:
        raise ValueError("COHERE_API_KEY not found in .env file")

    # Initialize Cohere client
    co = cohere.Client(cohere_api_key)

    # Step 1: Embed the question
    print(f"Embedding question with Cohere...")
    query_embedding_response = co.embed(
        texts=[question],
        model="embed-english-v3.0",
        input_type="search_query"
    )
    query_embedding = query_embedding_response.embeddings[0]

    # Step 2: Retrieve similar chunks from ChromaDB
    print(f"Retrieving top-{top_k} similar chunks from ChromaDB...")
    client = chromadb.PersistentClient(path="./chroma_db")

    try:
        collection = client.get_collection(name=collection_name)
    except Exception as e:
        raise ValueError(f"Collection '{collection_name}' not found. Please run index_baseline.py first.")

    # Query ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    # Extract results
    chunk_ids = results['ids'][0]
    documents = results['documents'][0]
    distances = results['distances'][0]

    print(f"Retrieved {len(documents)} chunks")
    for i, (chunk_id, distance) in enumerate(zip(chunk_ids, distances)):
        print(f"  {i+1}. {chunk_id} (distance: {distance:.4f})")

    # Step 3: Call Cohere chat with RAG
    print(f"\nGenerating answer with Cohere command-r-plus-08-2024...")

    # Format documents for Cohere chat
    formatted_docs = [{"text": doc} for doc in documents]

    response = co.chat(
        model="command-r-plus-08-2024",
        message=question,
        documents=formatted_docs
    )

    answer = response.text

    return answer, chunk_ids


def main():
    """Main query workflow."""
    load_dotenv()

    # Check if ChromaDB collection exists
    if not os.path.exists("./chroma_db"):
        print("Error: ChromaDB database not found. Please run 'python rag/index_baseline.py' first.")
        return

    print("=" * 60)
    print("RAG QUERY SYSTEM - Baseline")
    print("=" * 60)
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
        print(f"Using example question: {question}\n")

    print("=" * 60)

    try:
        answer, chunk_ids = answer_question_baseline(question, top_k=5)

        print("\n" + "=" * 60)
        print("ANSWER")
        print("=" * 60)
        print(answer)
        print()
        print("=" * 60)
        print(f"Sources: {', '.join(chunk_ids)}")
        print("=" * 60)

    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure you have:")
        print("  1. Run 'python rag/index_baseline.py' to create the index")
        print("  2. Set COHERE_API_KEY in your .env file")


if __name__ == "__main__":
    main()
