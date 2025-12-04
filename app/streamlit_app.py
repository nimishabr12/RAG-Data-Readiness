"""
Streamlit application for RAG data-readiness tool.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import chromadb
from dotenv import load_dotenv

from rag.query import answer_question_baseline

load_dotenv()


def get_chunk_texts(chunk_ids: list[str], collection_name: str = "nvidia_report") -> list[str]:
    """
    Retrieve the full text of chunks by their IDs.

    Args:
        chunk_ids: List of chunk IDs to retrieve
        collection_name: ChromaDB collection name

    Returns:
        List of chunk texts
    """
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection(name=collection_name)

    result = collection.get(ids=chunk_ids)
    return result['documents']


def main():
    st.set_page_config(
        page_title="RAG Data-Readiness Tool",
        page_icon="🤖",
        layout="wide"
    )

    st.title("🤖 RAG Data-Readiness Tool")
    st.markdown("**Powered by Cohere & ChromaDB**")
    st.markdown("---")

    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    top_k = st.sidebar.slider("Number of chunks to retrieve", min_value=1, max_value=10, value=5)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "This tool uses Retrieval-Augmented Generation (RAG) to answer questions "
        "about the NVIDIA report using Cohere's embed-4 and command-r models."
    )

    # Check if database exists
    if not os.path.exists("./chroma_db"):
        st.error("⚠️ ChromaDB database not found!")
        st.info("Please run `python rag/index_baseline.py` first to build the index.")
        return

    # Check API key
    if not os.getenv("COHERE_API_KEY"):
        st.error("⚠️ COHERE_API_KEY not found!")
        st.info("Please add your Cohere API key to the `.env` file.")
        return

    # Example questions
    st.markdown("### 💡 Example Questions")
    example_questions = [
        "What is NVIDIA's total revenue?",
        "What are the main business segments of NVIDIA?",
        "What risks does NVIDIA face in its business?",
        "What is NVIDIA's strategy for AI and data centers?"
    ]

    cols = st.columns(2)
    for i, question in enumerate(example_questions):
        col = cols[i % 2]
        if col.button(question, key=f"example_{i}"):
            st.session_state.question = question

    st.markdown("---")

    # Main query interface
    st.markdown("### 🔍 Ask a Question")

    # Text input with session state
    question = st.text_area(
        "Enter your question about the NVIDIA report:",
        value=st.session_state.get('question', ''),
        height=100,
        placeholder="Type your question here..."
    )

    # Submit button
    if st.button("🚀 Get Answer", type="primary"):
        if question:
            with st.spinner("🔄 Processing your question..."):
                try:
                    # Call RAG system
                    answer, chunk_ids = answer_question_baseline(question, top_k=top_k)

                    # Display answer
                    st.markdown("---")
                    st.markdown("### ✅ Answer")
                    st.success(answer)

                    # Display retrieved chunks
                    st.markdown("---")
                    st.markdown(f"### 📚 Retrieved Chunks ({len(chunk_ids)} sources)")

                    # Get full chunk texts
                    chunk_texts = get_chunk_texts(chunk_ids)

                    # Display each chunk in an expander
                    for i, (chunk_id, chunk_text) in enumerate(zip(chunk_ids, chunk_texts)):
                        with st.expander(f"📄 Source {i+1}: {chunk_id}"):
                            st.text_area(
                                "Chunk Text",
                                value=chunk_text,
                                height=200,
                                key=f"chunk_{i}",
                                disabled=True
                            )

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.info(
                        "Make sure you have:\n"
                        "1. Run `python rag/index_baseline.py` to create the index\n"
                        "2. Set `COHERE_API_KEY` in your `.env` file"
                    )
        else:
            st.warning("⚠️ Please enter a question.")

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Built with Streamlit, Cohere, and ChromaDB"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
