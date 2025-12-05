"""
Evaluation script for RAG system.
Runs both baseline and improved modes on evaluation questions and records metrics.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import csv
import time
from datetime import datetime
from dotenv import load_dotenv

from rag.query import answer_question


def load_questions(filepath: str = "eval/questions.json") -> list[dict]:
    """
    Load evaluation questions from JSON file.

    Args:
        filepath: Path to questions JSON file

    Returns:
        List of question dicts
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Handle both array format and object with "questions" key
    if isinstance(data, list):
        questions = data
    elif isinstance(data, dict) and 'questions' in data:
        questions = data['questions']
    else:
        raise ValueError("Invalid questions.json format")

    return questions


def run_evaluation_query(question_id: int, question: str, mode: str, top_k: int = 5) -> dict:
    """
    Run a single evaluation query and collect metrics.

    Args:
        question_id: ID of the question
        question: Question text
        mode: 'baseline' or 'improved'
        top_k: Number of chunks to retrieve

    Returns:
        Dict with evaluation metrics
    """
    print(f"  Running {mode} mode...")

    # Time the query
    start_time = time.time()

    try:
        answer, used_mode, chunk_ids = answer_question(question, mode=mode, top_k=top_k)
        elapsed_time = time.time() - start_time

        # Calculate total context characters
        # We need to get the actual chunk texts to calculate context size
        import chromadb
        collection_name = "nvidia_improved" if mode == "improved" else "nvidia_report"
        client = chromadb.PersistentClient(path="./chroma_db")
        collection = client.get_collection(name=collection_name)

        chunk_data = collection.get(ids=chunk_ids, include=['documents'])
        chunk_texts = chunk_data['documents']

        total_context_chars = sum(len(text) for text in chunk_texts)

        # Prepare result
        result = {
            'question_id': question_id,
            'mode': used_mode,
            'question': question,
            'answer_text': answer,
            'chunk_ids': ','.join(chunk_ids),
            'num_chunks': len(chunk_ids),
            'total_context_chars': total_context_chars,
            'elapsed_time_sec': round(elapsed_time, 2),
            'timestamp': datetime.now().isoformat(),
            'status': 'success'
        }

        print(f"    ✓ Completed in {elapsed_time:.2f}s, {total_context_chars:,} chars context")

        return result

    except Exception as e:
        elapsed_time = time.time() - start_time

        result = {
            'question_id': question_id,
            'mode': mode,
            'question': question,
            'answer_text': f"ERROR: {str(e)}",
            'chunk_ids': '',
            'num_chunks': 0,
            'total_context_chars': 0,
            'elapsed_time_sec': round(elapsed_time, 2),
            'timestamp': datetime.now().isoformat(),
            'status': 'error'
        }

        print(f"    ✗ Error: {str(e)}")

        return result


def run_full_evaluation(questions_file: str = "eval/questions.json",
                        output_file: str = "eval/results.csv",
                        top_k: int = 5) -> None:
    """
    Run full evaluation on all questions with both modes.

    Args:
        questions_file: Path to questions JSON
        output_file: Path to output CSV file
        top_k: Number of chunks to retrieve per query
    """
    # Load environment variables
    load_dotenv()
    cohere_api_key = os.getenv("COHERE_API_KEY")

    if not cohere_api_key:
        print("Error: COHERE_API_KEY not found in .env file")
        return

    # Load questions
    print("=" * 70)
    print("RAG EVALUATION - BASELINE VS IMPROVED")
    print("=" * 70)
    print()

    questions = load_questions(questions_file)
    print(f"Loaded {len(questions)} evaluation questions")
    print(f"Output file: {output_file}")
    print(f"Top-k chunks: {top_k}")
    print()

    # CSV header
    fieldnames = [
        'question_id',
        'mode',
        'question',
        'answer_text',
        'chunk_ids',
        'num_chunks',
        'total_context_chars',
        'elapsed_time_sec',
        'timestamp',
        'status'
    ]

    # Check if output file exists
    file_exists = os.path.exists(output_file)

    # Open CSV file
    with open(output_file, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write header if file is new
        if not file_exists:
            writer.writeheader()
            print(f"Created new results file: {output_file}")
        else:
            print(f"Appending to existing results file: {output_file}")

        print()
        print("=" * 70)

        # Process each question
        for i, q in enumerate(questions, 1):
            question_id = q.get('id', i)
            question_text = q['question']

            print(f"\n[{i}/{len(questions)}] Question {question_id}")
            print(f"Q: {question_text[:100]}...")

            # Run baseline mode
            baseline_result = run_evaluation_query(question_id, question_text, 'baseline', top_k)
            writer.writerow(baseline_result)
            csvfile.flush()

            # Run improved mode
            improved_result = run_evaluation_query(question_id, question_text, 'improved', top_k)
            writer.writerow(improved_result)
            csvfile.flush()

    print()
    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {output_file}")
    print(f"Total queries run: {len(questions) * 2} ({len(questions)} questions × 2 modes)")


def print_summary(results_file: str = "eval/results.csv") -> None:
    """
    Print summary statistics from evaluation results.

    Args:
        results_file: Path to results CSV file
    """
    if not os.path.exists(results_file):
        print(f"Results file not found: {results_file}")
        return

    with open(results_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print("No results found in file")
        return

    # Separate by mode
    baseline_rows = [r for r in rows if r['mode'] == 'baseline']
    improved_rows = [r for r in rows if r['mode'] == 'improved']

    print()
    print("=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print()

    for mode, mode_rows in [('BASELINE', baseline_rows), ('IMPROVED', improved_rows)]:
        if not mode_rows:
            continue

        print(f"--- {mode} MODE ---")
        print(f"Total queries: {len(mode_rows)}")

        # Success rate
        successes = [r for r in mode_rows if r['status'] == 'success']
        success_rate = len(successes) / len(mode_rows) * 100 if mode_rows else 0
        print(f"Success rate: {success_rate:.1f}%")

        if successes:
            # Average metrics
            avg_time = sum(float(r['elapsed_time_sec']) for r in successes) / len(successes)
            avg_context = sum(int(r['total_context_chars']) for r in successes) / len(successes)
            avg_chunks = sum(int(r['num_chunks']) for r in successes) / len(successes)

            print(f"Average elapsed time: {avg_time:.2f}s")
            print(f"Average context size: {avg_context:,.0f} chars")
            print(f"Average chunks used: {avg_chunks:.1f}")

        print()

    print("=" * 70)


def main():
    """Main evaluation workflow."""
    import argparse

    parser = argparse.ArgumentParser(description='Run RAG evaluation')
    parser.add_argument('--questions', default='eval/questions.json', help='Path to questions JSON file')
    parser.add_argument('--output', default='eval/results.csv', help='Path to output CSV file')
    parser.add_argument('--top-k', type=int, default=5, help='Number of chunks to retrieve')
    parser.add_argument('--summary', action='store_true', help='Print summary of existing results')

    args = parser.parse_args()

    if args.summary:
        print_summary(args.output)
    else:
        run_full_evaluation(args.questions, args.output, args.top_k)
        print()
        print_summary(args.output)


if __name__ == "__main__":
    main()
