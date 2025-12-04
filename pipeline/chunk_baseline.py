"""
Text chunking module for RAG pipeline.
Implements baseline chunking strategies.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.ingest import load_pdf


def chunk_text_baseline(text: str, max_tokens: int = 500) -> list[str]:
    """
    Split text into overlapping chunks using word-based token approximation.

    Uses a simple heuristic: ~1.3 words per token (common English approximation).
    Chunks overlap by 10% to maintain context between chunks.

    Args:
        text: Input text to chunk
        max_tokens: Maximum number of tokens per chunk (default: 500)

    Returns:
        List of text chunks
    """
    # Token approximation: ~1.3 words per token
    words_per_token = 1.3
    max_words = int(max_tokens * words_per_token)

    # 10% overlap between chunks
    overlap_words = int(max_words * 0.1)

    # Split text into words
    words = text.split()

    chunks = []
    start_idx = 0

    while start_idx < len(words):
        # Get chunk of max_words
        end_idx = min(start_idx + max_words, len(words))
        chunk_words = words[start_idx:end_idx]

        # Join words back into text
        chunk = " ".join(chunk_words)
        chunks.append(chunk)

        # Move start index forward, accounting for overlap
        start_idx += (max_words - overlap_words)

        # Break if we've processed all words
        if end_idx == len(words):
            break

    return chunks


def main():
    """Main chunking workflow."""
    pdf_path = "data/Nvidia Report.pdf"

    if not os.path.exists(pdf_path):
        print(f"Error: {pdf_path} not found")
        return

    print("Loading PDF...")
    full_text, pages = load_pdf(pdf_path)

    print(f"Loaded {len(pages)} pages with {len(full_text)} characters")
    print(f"\nChunking text with baseline strategy...")

    chunks = chunk_text_baseline(full_text, max_tokens=500)

    print(f"\nChunking complete!")
    print(f"Total chunks created: {len(chunks)}")
    print(f"Average chunk length: {sum(len(c) for c in chunks) // len(chunks)} characters")
    print(f"Average words per chunk: {sum(len(c.split()) for c in chunks) // len(chunks)} words")

    # Show first chunk as example
    print(f"\n--- First Chunk Preview ---")
    preview = chunks[0][:300].encode('ascii', errors='ignore').decode('ascii')
    print(preview + "...")


if __name__ == "__main__":
    main()
