"""
Streamlit application for RAG data-readiness tool.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import chromadb
import pandas as pd
from dotenv import load_dotenv

from rag.query import answer_question

# Import get_summary from eval module
try:
    from eval.summarize_results import get_summary
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False

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


def render_chat_mode(top_k: int):
    """Render the chat comparison interface."""
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


def render_metrics_mode():
    """Render the metrics dashboard."""
    st.markdown("### 📊 Evaluation Metrics Dashboard")
    st.markdown("Compare baseline vs improved RAG performance based on evaluation results.")
    st.markdown("---")

    # Check if evaluation results exist
    if not os.path.exists("eval/results.csv") or not os.path.exists("eval/gold_labels.csv"):
        st.warning("⚠️ No evaluation results found!")
        st.info(
            "To generate evaluation metrics:\n\n"
            "1. Run evaluation: `python eval/run_eval.py --evaluate-quality --max-questions 10`\n"
            "2. Extract gold labels: `python eval/extract_gold_labels.py`\n"
            "3. Refresh this page"
        )
        return

    # Load metrics
    try:
        with st.spinner("Loading evaluation metrics..."):
            metrics = get_summary("eval/results.csv", "eval/gold_labels.csv")

        if 'message' in metrics:
            st.warning(metrics['message'])
            return

        # Summary statistics
        st.markdown(f"**Total Evaluated Queries:** {metrics['total_evaluated']}")
        st.markdown("---")

        # Main metrics table
        st.markdown("### 📈 Performance Metrics")

        if 'baseline' in metrics['by_mode'] and 'improved' in metrics['by_mode']:
            baseline = metrics['by_mode']['baseline']
            improved = metrics['by_mode']['improved']

            # Create metrics dataframe
            metrics_data = {
                'Mode': ['Baseline', 'Improved'],
                'Accuracy': [
                    baseline['correctness']['avg_score'],
                    improved['correctness']['avg_score']
                ],
                'Faithfulness': [
                    baseline['faithfulness']['avg_score'],
                    improved['faithfulness']['avg_score']
                ],
                'Avg Context (chars)': [
                    f"{baseline['efficiency']['avg_context_chars']:,.0f}",
                    f"{improved['efficiency']['avg_context_chars']:,.0f}"
                ],
                'Avg Elapsed Time (sec)': [
                    f"{baseline['efficiency']['avg_elapsed_time']:.2f}",
                    f"{improved['efficiency']['avg_elapsed_time']:.2f}"
                ]
            }

            df = pd.DataFrame(metrics_data)
            st.dataframe(df, use_container_width=True, hide_index=True)

            st.markdown("---")

            # Delta comparison
            st.markdown("### 📊 Delta Analysis (Improved vs Baseline)")

            col1, col2, col3, col4 = st.columns(4)

            # Accuracy delta
            accuracy_delta = improved['correctness']['avg_score'] - baseline['correctness']['avg_score']
            with col1:
                st.metric(
                    "Accuracy",
                    f"{improved['correctness']['avg_score']:.3f}",
                    f"{accuracy_delta:+.3f}",
                    delta_color="normal"
                )

            # Faithfulness delta
            faith_delta = improved['faithfulness']['avg_score'] - baseline['faithfulness']['avg_score']
            with col2:
                st.metric(
                    "Faithfulness",
                    f"{improved['faithfulness']['avg_score']:.3f}",
                    f"{faith_delta:+.3f}",
                    delta_color="normal"
                )

            # Context size delta
            ctx_delta = improved['efficiency']['avg_context_chars'] - baseline['efficiency']['avg_context_chars']
            ctx_delta_pct = (ctx_delta / baseline['efficiency']['avg_context_chars'] * 100) if baseline['efficiency']['avg_context_chars'] > 0 else 0
            with col3:
                st.metric(
                    "Avg Context",
                    f"{improved['efficiency']['avg_context_chars']:,.0f} chars",
                    f"{ctx_delta_pct:+.1f}%",
                    delta_color="inverse"  # Lower is better for context
                )

            # Time delta
            time_delta = improved['efficiency']['avg_elapsed_time'] - baseline['efficiency']['avg_elapsed_time']
            time_delta_pct = (time_delta / baseline['efficiency']['avg_elapsed_time'] * 100) if baseline['efficiency']['avg_elapsed_time'] > 0 else 0
            with col4:
                st.metric(
                    "Avg Time",
                    f"{improved['efficiency']['avg_elapsed_time']:.2f}s",
                    f"{time_delta_pct:+.1f}%",
                    delta_color="inverse"  # Lower is better for time
                )

            st.markdown("---")

            # Detailed breakdown
            st.markdown("### 📋 Detailed Breakdown")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Baseline Mode")
                st.markdown(f"**Correctness:**")
                st.markdown(f"- Fully correct: {baseline['correctness']['fully_correct']} ({baseline['correctness']['accuracy']*100:.1f}%)")
                st.markdown(f"- Partially correct: {baseline['correctness']['partially_correct']}")
                st.markdown(f"- Incorrect: {baseline['correctness']['incorrect']}")
                st.markdown(f"\n**Faithfulness:**")
                st.markdown(f"- Faithful: {baseline['faithfulness']['faithful']} ({baseline['faithfulness']['faithfulness_rate']*100:.1f}%)")
                st.markdown(f"- Unfaithful: {baseline['faithfulness']['unfaithful']}")

            with col2:
                st.markdown("#### Improved Mode")
                st.markdown(f"**Correctness:**")
                st.markdown(f"- Fully correct: {improved['correctness']['fully_correct']} ({improved['correctness']['accuracy']*100:.1f}%)")
                st.markdown(f"- Partially correct: {improved['correctness']['partially_correct']}")
                st.markdown(f"- Incorrect: {improved['correctness']['incorrect']}")
                st.markdown(f"\n**Faithfulness:**")
                st.markdown(f"- Faithful: {improved['faithfulness']['faithful']} ({improved['faithfulness']['faithfulness_rate']*100:.1f}%)")
                st.markdown(f"- Unfaithful: {improved['faithfulness']['unfaithful']}")

            # Cost savings analysis
            st.markdown("---")
            st.markdown("### 💰 Cost Savings Analysis")

            if ctx_delta_pct < 0:
                st.success(
                    f"**Improved mode reduces context by {abs(ctx_delta_pct):.1f}%**, "
                    f"resulting in significant API cost savings while maintaining {improved['correctness']['avg_score']:.1%} accuracy."
                )
            else:
                st.info(
                    f"Improved mode uses {ctx_delta_pct:.1f}% more context "
                    f"with {improved['correctness']['avg_score']:.1%} accuracy."
                )

    except Exception as e:
        st.error(f"❌ Error loading metrics: {str(e)}")
        st.exception(e)


def main():
    st.set_page_config(
        page_title="RAG Data-Readiness Tool",
        page_icon="🤖",
        layout="wide"
    )

    st.title("🤖 RAG Data-Readiness Tool")
    st.markdown("**Compare Baseline vs Improved RAG Performance**")
    st.markdown("---")

    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")

    # Mode selector
    mode = st.sidebar.radio(
        "Select Mode",
        ["Chat", "Metrics"],
        index=0,
        help="Chat: Interactive Q&A comparison | Metrics: View evaluation results"
    )

    st.sidebar.markdown("---")

    # Chat mode configuration
    if mode == "Chat":
        top_k = st.sidebar.slider("Number of chunks for answer", min_value=1, max_value=10, value=5)
    else:
        top_k = 5  # Default for metrics mode

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

    # Check if database exists (for chat mode)
    if mode == "Chat":
        if not os.path.exists("./chroma_db"):
            st.error("⚠️ ChromaDB database not found!")
            st.info("Please run `python rag/index_baseline.py` and `python rag/index_improved.py` first.")
            return

        # Check API key
        if not os.getenv("COHERE_API_KEY"):
            st.error("⚠️ COHERE_API_KEY not found!")
            st.info("Please add your Cohere API key to the `.env` file.")
            return

    # Render appropriate mode
    if mode == "Chat":
        render_chat_mode(top_k)
    else:
        if not METRICS_AVAILABLE:
            st.error("⚠️ Metrics module not available. Please check eval/summarize_results.py exists.")
            return
        render_metrics_mode()

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Built with Streamlit, Cohere, and ChromaDB | RAG Data-Readiness Tool"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
