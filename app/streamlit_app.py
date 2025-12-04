"""
Streamlit application for RAG data-readiness tool.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import chromadb
from dotenv import load_dotenv

from rag.query import answer_question

load_dotenv()


def get_chunk_data(chunk_ids: list[str], collection_name: str = "nvidia_report") -> list[dict]:
    """
    Retrieve the full text and metadata of chunks by their IDs.

    Args:
        chunk_ids: List of chunk IDs to retrieve
        collection_name: ChromaDB collection name

    Returns:
        List of dicts with text and metadata
    """
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection(name=collection_name)

    result = collection.get(ids=chunk_ids, include=['documents', 'metadatas'])

    chunks = []
    for i, chunk_id in enumerate(chunk_ids):
        chunks.append({
            'id': chunk_id,
            'text': result['documents'][i],
            'metadata': result['metadatas'][i] if result['metadatas'] else {}
        })

    return chunks


def estimate_tokens(text: str) -> int:
    """
    Estimate token count from text length.
    Rule of thumb: ~4 characters per token for English text.

    Args:
        text: Text to estimate tokens for

    Returns:
        Estimated token count
    """
    return len(text) // 4


def main():
    st.set_page_config(
        page_title="RAG Data-Readiness Tool",
        page_icon="🤖",
        layout="wide"
    )

    st.title("🤖 RAG Data-Readiness Comparison Tool")
    st.markdown("**Side-by-Side Baseline vs Improved RAG Performance**")
    st.markdown("---")

    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    top_k = st.sidebar.slider("Number of chunks for answer", min_value=1, max_value=10, value=5)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "**Baseline:**\n"
        "- Simple word-based chunking\n"
        "- Direct vector retrieval\n"
        "\n"
        "**Improved:**\n"
        "- Smart section-aware chunking\n"
        "- Quality filtering\n"
        "- Cohere Rerank"
    )

    # Check if database exists
    if not os.path.exists("./chroma_db"):
        st.error("⚠️ ChromaDB database not found!")
        st.info("Please run `python rag/index_baseline.py` and `python rag/index_improved.py` first.")
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

    cols = st.columns(4)
    for i, question in enumerate(example_questions):
        if cols[i].button(question, key=f"example_{i}", use_container_width=True):
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
    if st.button("🚀 Compare Both Modes", type="primary", use_container_width=True):
        if question:
            st.markdown("---")

            # Create two columns for side-by-side comparison
            col1, col2 = st.columns(2)

            # BASELINE MODE
            with col1:
                st.markdown("### 📊 BASELINE MODE")
                with st.spinner("Processing baseline..."):
                    try:
                        answer_baseline, mode_baseline, chunk_ids_baseline = answer_question(
                            question, mode='baseline', top_k=top_k
                        )

                        # Display answer
                        st.markdown("#### ✅ Answer")
                        st.success(answer_baseline)

                        # Estimate tokens
                        chunks_baseline = get_chunk_data(chunk_ids_baseline, collection_name="nvidia_report")
                        total_context = question + " " + " ".join([c['text'] for c in chunks_baseline])
                        estimated_tokens = estimate_tokens(total_context)

                        st.markdown(f"**Estimated prompt tokens:** ~{estimated_tokens:,}")
                        st.markdown(f"**Chunks used:** {len(chunk_ids_baseline)}")

                        # Display chunks
                        st.markdown("#### 📚 Supporting Chunks")
                        for i, chunk in enumerate(chunks_baseline):
                            section_title = chunk['metadata'].get('section_title', 'N/A')
                            with st.expander(f"Chunk {i+1}: {chunk['id']}", expanded=False):
                                st.markdown(f"**Section:** {section_title}")
                                st.text_area(
                                    "Text",
                                    value=chunk['text'][:500] + "..." if len(chunk['text']) > 500 else chunk['text'],
                                    height=150,
                                    key=f"baseline_chunk_{i}",
                                    disabled=True
                                )

                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")

            # IMPROVED MODE
            with col2:
                st.markdown("### 🚀 IMPROVED MODE")
                with st.spinner("Processing improved..."):
                    try:
                        answer_improved, mode_improved, chunk_ids_improved = answer_question(
                            question, mode='improved', top_k=top_k
                        )

                        # Display answer
                        st.markdown("#### ✅ Answer")
                        st.success(answer_improved)

                        # Estimate tokens
                        chunks_improved = get_chunk_data(chunk_ids_improved, collection_name="nvidia_improved")
                        total_context = question + " " + " ".join([c['text'] for c in chunks_improved])
                        estimated_tokens = estimate_tokens(total_context)

                        st.markdown(f"**Estimated prompt tokens:** ~{estimated_tokens:,}")
                        st.markdown(f"**Chunks used:** {len(chunk_ids_improved)}")

                        # Display chunks
                        st.markdown("#### 📚 Supporting Chunks")
                        for i, chunk in enumerate(chunks_improved):
                            section_title = chunk['metadata'].get('section_title', 'N/A')
                            with st.expander(f"Chunk {i+1}: {chunk['id']}", expanded=False):
                                st.markdown(f"**Section:** {section_title}")
                                st.markdown(f"**Quality Score:** {chunk['metadata'].get('quality_score', 'N/A'):.3f}")
                                st.text_area(
                                    "Text",
                                    value=chunk['text'][:500] + "..." if len(chunk['text']) > 500 else chunk['text'],
                                    height=150,
                                    key=f"improved_chunk_{i}",
                                    disabled=True
                                )

                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")

        else:
            st.warning("⚠️ Please enter a question.")

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Built with Streamlit, Cohere, and ChromaDB | Compare Baseline vs Improved RAG"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
