"""
Example: Using custom retrieval configurations with improved mode.

This demonstrates how to use the RetrievalConfig class to customize
improved-mode retrieval parameters beyond the IMPROVED_BEST preset.
"""

from rag.query import answer_question, RetrievalConfig, IMPROVED_BEST


def example_default_preset():
    """Use the default IMPROVED_BEST preset."""
    print("=" * 70)
    print("Example 1: Default IMPROVED_BEST preset")
    print("=" * 70)
    print(f"Config: {IMPROVED_BEST.initial_candidates} candidates → "
          f"{IMPROVED_BEST.max_final_chunks} chunks, model={IMPROVED_BEST.rerank_model}")
    print()

    # This uses IMPROVED_BEST automatically
    answer, mode, chunks = answer_question(
        "What is NVIDIA's total revenue?",
        mode='improved'
    )

    print(f"\nAnswer: {answer[:200]}...")
    print(f"Chunks used: {len(chunks)}")
    print()


def example_custom_config():
    """Use a custom retrieval configuration."""
    print("=" * 70)
    print("Example 2: Custom retrieval configuration")
    print("=" * 70)

    # Create custom config with different parameters
    custom_config = RetrievalConfig(
        initial_candidates=15,  # Fewer initial candidates
        max_final_chunks=5,     # More final chunks
        rerank_model="rerank-english-v3.0"
    )

    print(f"Custom config: {custom_config.initial_candidates} candidates → "
          f"{custom_config.max_final_chunks} chunks")
    print()

    answer, mode, chunks = answer_question(
        "What are NVIDIA's main business segments?",
        mode='improved',
        retrieval_config=custom_config
    )

    print(f"\nAnswer: {answer[:200]}...")
    print(f"Chunks used: {len(chunks)}")
    print()


def example_aggressive_config():
    """Use an aggressive config for maximum efficiency."""
    print("=" * 70)
    print("Example 3: Aggressive efficiency configuration")
    print("=" * 70)

    # More aggressive: fewer candidates, fewer final chunks
    aggressive_config = RetrievalConfig(
        initial_candidates=10,
        max_final_chunks=2,
        rerank_model="rerank-english-v3.0"
    )

    print(f"Aggressive config: {aggressive_config.initial_candidates} candidates → "
          f"{aggressive_config.max_final_chunks} chunks")
    print("(Lower latency, lower cost, but may sacrifice some accuracy)")
    print()

    answer, mode, chunks = answer_question(
        "What risks does NVIDIA face?",
        mode='improved',
        retrieval_config=aggressive_config
    )

    print(f"\nAnswer: {answer[:200]}...")
    print(f"Chunks used: {len(chunks)}")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("CUSTOM RETRIEVAL CONFIGURATION EXAMPLES")
    print("=" * 70)
    print("\nIMPROVED_BEST preset achieves:")
    print("  • Same accuracy as baseline (±0.02)")
    print("  • 32% lower context size (14.8k vs 21.8k chars)")
    print("  • 16% lower latency (7.8s vs 9.2s)")
    print("\n")

    # Run examples
    example_default_preset()
    example_custom_config()
    example_aggressive_config()

    print("=" * 70)
    print("All examples completed!")
    print("=" * 70)
