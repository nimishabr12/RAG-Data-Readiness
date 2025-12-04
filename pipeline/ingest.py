"""
Document ingestion module for RAG pipeline.
Handles loading and preprocessing of PDF documents.
"""

import os
from pathlib import Path
from pypdf import PdfReader


def load_pdf(file_path: str) -> tuple[str, list[str]]:
    """
    Load and extract text from a PDF file page by page.

    Args:
        file_path: Path to the PDF file

    Returns:
        A tuple containing:
        - full_text: Single concatenated string of all pages
        - pages: List of individual page texts
    """
    reader = PdfReader(file_path)
    pages = []

    for page in reader.pages:
        text = page.extract_text()
        pages.append(text)

    full_text = "\n".join(pages)

    return full_text, pages


def main():
    """Main ingestion workflow."""
    pdf_path = "data/Nvidia Report.pdf"

    if not os.path.exists(pdf_path):
        print(f"Error: {pdf_path} not found")
        return

    print(f"Loading PDF: {pdf_path}")
    full_text, pages = load_pdf(pdf_path)

    print(f"\nDocument loaded successfully!")
    print(f"Total pages: {len(pages)}")
    print(f"Total characters: {len(full_text)}")

    # Preview first 500 characters (replace problematic chars for console display)
    preview = full_text[:500].encode('ascii', errors='ignore').decode('ascii')
    print(f"\nFirst 500 characters (ASCII preview):\n{preview}")


if __name__ == "__main__":
    main()
