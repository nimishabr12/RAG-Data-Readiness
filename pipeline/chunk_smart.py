"""
Smart chunking module for RAG pipeline.
Implements section-aware, structure-preserving chunking strategies.
"""

import sys
import os
import re
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.ingest import load_pdf


def extract_sections(text: str) -> list[dict]:
    """
    Split the Nvidia 10-K text into sections based on ITEM headings.

    Args:
        text: Full document text

    Returns:
        List of dicts with {section_id, title, text, start_char, end_char}
    """
    # Pattern to match ITEM headings (e.g., "ITEM 1.", "ITEM 1A.", "ITEM 1B.")
    # Handles various formats like "ITEM 1.", "ITEM 1A.", "Item 1.", etc.
    item_pattern = r'(?:^|\n)\s*ITEM\s+(\d+[A-Z]?\.?)\s+([^\n]+)'

    matches = list(re.finditer(item_pattern, text, re.IGNORECASE | re.MULTILINE))

    sections = []

    for i, match in enumerate(matches):
        section_id = match.group(1).rstrip('.')
        title = match.group(2).strip()
        start_char = match.start()

        # End is either the start of next section or end of document
        end_char = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        section_text = text[start_char:end_char].strip()

        sections.append({
            'section_id': section_id,
            'title': title,
            'text': section_text,
            'start_char': start_char,
            'end_char': end_char
        })

    print(f"Extracted {len(sections)} sections from document")

    return sections


def is_table_line(line: str, digit_threshold: float = 0.15) -> bool:
    """
    Detect if a line is likely part of a table based on digit density.

    Args:
        line: Text line to check
        digit_threshold: Minimum ratio of digits to characters

    Returns:
        True if line appears to be part of a table
    """
    if not line.strip():
        return False

    # Count digits and total non-whitespace characters
    digits = sum(c.isdigit() for c in line)
    non_space = sum(not c.isspace() for c in line)

    if non_space == 0:
        return False

    digit_ratio = digits / non_space

    # Also check for common table indicators
    has_multiple_spaces = '  ' in line or '\t' in line
    has_separators = any(sep in line for sep in ['|', '─', '━', '—'])

    return digit_ratio >= digit_threshold or (has_multiple_spaces and digits > 0) or has_separators


def extract_text_blocks(text: str) -> list[dict]:
    """
    Extract structured text blocks (paragraphs, bullet lists, tables).

    Args:
        text: Text to parse

    Returns:
        List of dicts with {type, text, lines}
    """
    lines = text.split('\n')
    blocks = []
    current_block = []
    current_type = None

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Detect table blocks
        if is_table_line(line):
            # Flush current block
            if current_block and current_type:
                blocks.append({
                    'type': current_type,
                    'text': '\n'.join(current_block),
                    'lines': len(current_block)
                })
                current_block = []

            # Collect table lines
            table_lines = [line]
            i += 1
            while i < len(lines) and (is_table_line(lines[i]) or not lines[i].strip()):
                table_lines.append(lines[i])
                i += 1

            blocks.append({
                'type': 'table',
                'text': '\n'.join(table_lines).strip(),
                'lines': len(table_lines)
            })
            current_type = None
            continue

        # Detect bullet points
        elif re.match(r'^\s*[•\-\*\·]\s+', line) or re.match(r'^\s*\d+[\.\)]\s+', line):
            if current_type != 'bullet':
                # Flush previous block
                if current_block and current_type:
                    blocks.append({
                        'type': current_type,
                        'text': '\n'.join(current_block),
                        'lines': len(current_block)
                    })
                current_block = []
                current_type = 'bullet'

            current_block.append(line)

        # Empty line - potential paragraph boundary
        elif not stripped:
            if current_block and current_type:
                blocks.append({
                    'type': current_type,
                    'text': '\n'.join(current_block),
                    'lines': len(current_block)
                })
                current_block = []
                current_type = None

        # Regular paragraph text
        else:
            if current_type != 'paragraph':
                if current_block and current_type:
                    blocks.append({
                        'type': current_type,
                        'text': '\n'.join(current_block),
                        'lines': len(current_block)
                    })
                current_block = []
                current_type = 'paragraph'

            current_block.append(line)

        i += 1

    # Flush remaining block
    if current_block and current_type:
        blocks.append({
            'type': current_type,
            'text': '\n'.join(current_block),
            'lines': len(current_block)
        })

    return blocks


def chunk_text_smart(sections: list[dict], max_tokens: int = 600) -> list[dict]:
    """
    Create smart chunks that respect document structure.

    Args:
        sections: List of section dicts from extract_sections()
        max_tokens: Maximum tokens per chunk (using word approximation)

    Returns:
        List of chunk dicts with {id, text, section_title, start_char, end_char}
    """
    # Token approximation: ~1.3 words per token
    words_per_token = 1.3
    max_words = int(max_tokens * words_per_token)

    all_chunks = []
    chunk_id = 0

    for section in sections:
        section_id = section['section_id']
        section_title = f"ITEM {section_id}. {section['title']}"
        section_text = section['text']
        section_start = section['start_char']

        # Extract text blocks (paragraphs, bullets, tables)
        blocks = extract_text_blocks(section_text)

        current_chunk_blocks = []
        current_word_count = 0

        for block in blocks:
            block_text = block['text']
            block_words = len(block_text.split())

            # If single block exceeds max, split it (but try to keep tables together)
            if block_words > max_words:
                # Flush current chunk if any
                if current_chunk_blocks:
                    chunk_text = '\n\n'.join(current_chunk_blocks)
                    all_chunks.append({
                        'id': f'chunk_{chunk_id}',
                        'text': chunk_text,
                        'section_title': section_title,
                        'section_id': section_id,
                        'start_char': section_start,
                        'end_char': section_start + len(chunk_text)
                    })
                    chunk_id += 1
                    current_chunk_blocks = []
                    current_word_count = 0

                # For tables, keep them as one chunk even if large
                if block['type'] == 'table':
                    all_chunks.append({
                        'id': f'chunk_{chunk_id}',
                        'text': block_text,
                        'section_title': section_title,
                        'section_id': section_id,
                        'start_char': section_start,
                        'end_char': section_start + len(block_text)
                    })
                    chunk_id += 1
                else:
                    # Split large non-table blocks by sentences
                    sentences = re.split(r'(?<=[.!?])\s+', block_text)
                    temp_chunk = []
                    temp_words = 0

                    for sentence in sentences:
                        sent_words = len(sentence.split())
                        if temp_words + sent_words > max_words and temp_chunk:
                            chunk_text = ' '.join(temp_chunk)
                            all_chunks.append({
                                'id': f'chunk_{chunk_id}',
                                'text': chunk_text,
                                'section_title': section_title,
                                'section_id': section_id,
                                'start_char': section_start,
                                'end_char': section_start + len(chunk_text)
                            })
                            chunk_id += 1
                            temp_chunk = [sentence]
                            temp_words = sent_words
                        else:
                            temp_chunk.append(sentence)
                            temp_words += sent_words

                    if temp_chunk:
                        chunk_text = ' '.join(temp_chunk)
                        all_chunks.append({
                            'id': f'chunk_{chunk_id}',
                            'text': chunk_text,
                            'section_title': section_title,
                            'section_id': section_id,
                            'start_char': section_start,
                            'end_char': section_start + len(chunk_text)
                        })
                        chunk_id += 1

            # If adding this block would exceed max, flush current chunk
            elif current_word_count + block_words > max_words and current_chunk_blocks:
                chunk_text = '\n\n'.join(current_chunk_blocks)
                all_chunks.append({
                    'id': f'chunk_{chunk_id}',
                    'text': chunk_text,
                    'section_title': section_title,
                    'section_id': section_id,
                    'start_char': section_start,
                    'end_char': section_start + len(chunk_text)
                })
                chunk_id += 1
                current_chunk_blocks = [block_text]
                current_word_count = block_words

            # Add block to current chunk
            else:
                current_chunk_blocks.append(block_text)
                current_word_count += block_words

        # Flush remaining blocks for this section
        if current_chunk_blocks:
            chunk_text = '\n\n'.join(current_chunk_blocks)
            all_chunks.append({
                'id': f'chunk_{chunk_id}',
                'text': chunk_text,
                'section_title': section_title,
                'section_id': section_id,
                'start_char': section_start,
                'end_char': section_start + len(chunk_text)
            })
            chunk_id += 1

    return all_chunks


def main():
    """Main smart chunking workflow."""
    pdf_path = "data/Nvidia Report.pdf"

    if not os.path.exists(pdf_path):
        print(f"Error: {pdf_path} not found")
        return

    print("=" * 60)
    print("SMART CHUNKING WORKFLOW")
    print("=" * 60)
    print()

    # Load PDF
    print("Step 1: Loading PDF...")
    full_text, pages = load_pdf(pdf_path)
    print(f"Loaded {len(pages)} pages with {len(full_text)} characters\n")

    # Extract sections
    print("Step 2: Extracting sections...")
    sections = extract_sections(full_text)
    print(f"\nFound sections:")
    for section in sections[:10]:  # Show first 10
        print(f"  - ITEM {section['section_id']}: {section['title'][:60]}...")
    if len(sections) > 10:
        print(f"  ... and {len(sections) - 10} more")
    print()

    # Create smart chunks
    print("Step 3: Creating smart chunks...")
    chunks = chunk_text_smart(sections, max_tokens=600)
    print(f"\nCreated {len(chunks)} smart chunks")
    print(f"Average words per chunk: {sum(len(c['text'].split()) for c in chunks) // len(chunks)}")
    print()

    # Show statistics by section
    print("Chunks per section:")
    section_counts = {}
    for chunk in chunks:
        section_id = chunk['section_id']
        section_counts[section_id] = section_counts.get(section_id, 0) + 1

    for section_id, count in sorted(section_counts.items())[:10]:
        print(f"  ITEM {section_id}: {count} chunks")
    print()

    # Show sample chunks
    print("Sample chunks:")
    for i in [0, len(chunks)//2, -1]:
        chunk = chunks[i]
        preview = chunk['text'][:200].replace('\n', ' ')
        print(f"\n  [{chunk['id']}] {chunk['section_title']}")
        print(f"  {preview}...")

    print("\n" + "=" * 60)
    print("SMART CHUNKING COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
