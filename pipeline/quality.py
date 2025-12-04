"""
Quality scoring and filtering module for RAG pipeline.
Identifies and filters out low-quality chunks.
"""

import sys
import os
import re
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.ingest import load_pdf
from pipeline.chunk_baseline import chunk_text_baseline
from pipeline.chunk_smart import extract_sections, chunk_text_smart


# Boilerplate phrases commonly found in financial documents
BOILERPLATE_PHRASES = [
    'forward-looking statements',
    'forward looking statements',
    'see accompanying notes',
    'see note',
    'see notes',
    'table of contents',
    'this page intentionally left blank',
    'continued on next page',
    'end of table',
    'see item',
    'refer to item',
    'incorporated by reference',
    'filed herewith',
    'furnished herewith',
    'exhibits index',
    'signature page',
    'power of attorney',
    'certify that',
    'certification pursuant to',
    'sarbanes-oxley act',
]


def score_chunk(chunk: dict) -> float:
    """
    Score a chunk based on quality metrics.

    Scoring criteria:
    - Penalizes very short chunks (< 200 chars)
    - Penalizes very long chunks (> 2000-2500 chars)
    - Penalizes chunks with boilerplate phrases
    - Penalizes chunks with low information density

    Args:
        chunk: Chunk dict with 'text' field (and optionally other metadata)

    Returns:
        Quality score between 0.0 and 1.0 (higher is better)
    """
    text = chunk.get('text', '')
    if not text:
        return 0.0

    score = 1.0
    char_count = len(text)
    word_count = len(text.split())

    # Penalty 1: Very short chunks (< 200 chars)
    if char_count < 200:
        # Aggressive penalty for very short chunks
        penalty = (200 - char_count) / 200 * 0.6
        score -= penalty

    # Penalty 2: Very long chunks (> 2000 chars)
    elif char_count > 2000:
        # Penalty increases as chunk gets longer
        if char_count > 2500:
            penalty = 0.4
        else:
            penalty = (char_count - 2000) / 500 * 0.3
        score -= penalty

    # Penalty 3: Boilerplate phrases
    text_lower = text.lower()
    boilerplate_count = 0
    for phrase in BOILERPLATE_PHRASES:
        if phrase in text_lower:
            boilerplate_count += 1

    if boilerplate_count > 0:
        # Penalty based on number of boilerplate phrases found
        penalty = min(boilerplate_count * 0.15, 0.5)
        score -= penalty

    # Penalty 4: Low information density
    # Check for repetitive patterns, excessive whitespace, etc.
    if word_count > 0:
        # Too many short words (< 3 chars) suggests low information
        short_words = sum(1 for word in text.split() if len(word) < 3)
        short_word_ratio = short_words / word_count
        if short_word_ratio > 0.5:
            score -= 0.2

        # Very low average word length
        avg_word_length = sum(len(word) for word in text.split()) / word_count
        if avg_word_length < 3.5:
            score -= 0.15

    # Penalty 5: Excessive punctuation (might indicate formatting issues)
    punct_count = sum(1 for c in text if c in '.,;:!?')
    if word_count > 0:
        punct_ratio = punct_count / word_count
        if punct_ratio > 0.3:
            score -= 0.2

    # Penalty 6: Mostly uppercase (likely headers or boilerplate)
    if char_count > 0:
        upper_count = sum(1 for c in text if c.isupper())
        upper_ratio = upper_count / char_count
        if upper_ratio > 0.5:
            score -= 0.3

    # Ensure score stays in [0, 1] range
    score = max(0.0, min(1.0, score))

    return score


def filter_chunks(chunks: list[dict], min_score: float = 0.4) -> list[dict]:
    """
    Filter chunks based on quality scores.

    Args:
        chunks: List of chunk dicts
        min_score: Minimum quality score to keep (default: 0.4)

    Returns:
        List of chunks with score >= min_score, with 'quality_score' added
    """
    filtered_chunks = []

    for chunk in chunks:
        score = score_chunk(chunk)
        chunk['quality_score'] = score

        if score >= min_score:
            filtered_chunks.append(chunk)

    return filtered_chunks


def analyze_chunk_quality(chunks: list[dict]) -> dict:
    """
    Analyze quality distribution of chunks.

    Args:
        chunks: List of chunk dicts

    Returns:
        Dict with quality statistics
    """
    scores = [score_chunk(chunk) for chunk in chunks]

    if not scores:
        return {
            'total_chunks': 0,
            'mean_score': 0.0,
            'min_score': 0.0,
            'max_score': 0.0,
            'low_quality_count': 0,
            'high_quality_count': 0
        }

    return {
        'total_chunks': len(scores),
        'mean_score': sum(scores) / len(scores),
        'min_score': min(scores),
        'max_score': max(scores),
        'low_quality_count': sum(1 for s in scores if s < 0.4),
        'medium_quality_count': sum(1 for s in scores if 0.4 <= s < 0.7),
        'high_quality_count': sum(1 for s in scores if s >= 0.7)
    }


def main():
    """Main quality analysis workflow."""
    pdf_path = "data/Nvidia Report.pdf"

    if not os.path.exists(pdf_path):
        print(f"Error: {pdf_path} not found")
        return

    print("=" * 70)
    print("CHUNK QUALITY ANALYSIS")
    print("=" * 70)
    print()

    # Load PDF
    print("Loading PDF...")
    full_text, pages = load_pdf(pdf_path)
    print(f"Loaded {len(pages)} pages\n")

    # Compare baseline vs smart chunking quality
    print("=" * 70)
    print("BASELINE CHUNKING QUALITY")
    print("=" * 70)
    baseline_chunks = chunk_text_baseline(full_text, max_tokens=500)
    baseline_chunks_dict = [{'text': chunk} for chunk in baseline_chunks]

    baseline_stats = analyze_chunk_quality(baseline_chunks_dict)
    print(f"Total chunks: {baseline_stats['total_chunks']}")
    print(f"Mean quality score: {baseline_stats['mean_score']:.3f}")
    print(f"Score range: [{baseline_stats['min_score']:.3f}, {baseline_stats['max_score']:.3f}]")
    print(f"Low quality (< 0.4): {baseline_stats['low_quality_count']}")
    print(f"Medium quality (0.4-0.7): {baseline_stats['medium_quality_count']}")
    print(f"High quality (>= 0.7): {baseline_stats['high_quality_count']}")

    # Filter baseline chunks
    filtered_baseline = filter_chunks(baseline_chunks_dict, min_score=0.4)
    print(f"\nAfter filtering (min_score=0.4): {len(filtered_baseline)} chunks retained "
          f"({len(filtered_baseline)/len(baseline_chunks_dict)*100:.1f}%)")
    print()

    # Smart chunking quality
    print("=" * 70)
    print("SMART CHUNKING QUALITY")
    print("=" * 70)
    sections = extract_sections(full_text)
    smart_chunks = chunk_text_smart(sections, max_tokens=600)

    smart_stats = analyze_chunk_quality(smart_chunks)
    print(f"Total chunks: {smart_stats['total_chunks']}")
    print(f"Mean quality score: {smart_stats['mean_score']:.3f}")
    print(f"Score range: [{smart_stats['min_score']:.3f}, {smart_stats['max_score']:.3f}]")
    print(f"Low quality (< 0.4): {smart_stats['low_quality_count']}")
    print(f"Medium quality (0.4-0.7): {smart_stats['medium_quality_count']}")
    print(f"High quality (>= 0.7): {smart_stats['high_quality_count']}")

    # Filter smart chunks
    filtered_smart = filter_chunks(smart_chunks, min_score=0.4)
    print(f"\nAfter filtering (min_score=0.4): {len(filtered_smart)} chunks retained "
          f"({len(filtered_smart)/len(smart_chunks)*100:.1f}%)")
    print()

    # Show examples of filtered out chunks
    print("=" * 70)
    print("EXAMPLES OF LOW-QUALITY CHUNKS (Filtered Out)")
    print("=" * 70)

    low_quality_examples = [chunk for chunk in smart_chunks if score_chunk(chunk) < 0.4]
    for i, chunk in enumerate(low_quality_examples[:3]):
        score = score_chunk(chunk)
        preview = chunk['text'][:150].replace('\n', ' ')
        print(f"\n[Example {i+1}] Score: {score:.3f}")
        print(f"Section: {chunk.get('section_title', 'N/A')}")
        print(f"Length: {len(chunk['text'])} chars")
        print(f"Text: {preview}...")

    print("\n" + "=" * 70)
    print("EXAMPLES OF HIGH-QUALITY CHUNKS (Retained)")
    print("=" * 70)

    high_quality_examples = [chunk for chunk in smart_chunks if score_chunk(chunk) >= 0.7]
    for i, chunk in enumerate(high_quality_examples[:3]):
        score = score_chunk(chunk)
        preview = chunk['text'][:150].replace('\n', ' ')
        print(f"\n[Example {i+1}] Score: {score:.3f}")
        print(f"Section: {chunk.get('section_title', 'N/A')}")
        print(f"Length: {len(chunk['text'])} chars")
        print(f"Text: {preview}...")

    print("\n" + "=" * 70)
    print("QUALITY ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
