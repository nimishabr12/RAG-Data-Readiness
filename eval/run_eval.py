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
import cohere


def evaluate_correctness(question: str, answer: str, cohere_client) -> float:
    """
    Evaluate answer correctness using LLM-as-a-judge.

    Args:
        question: The question asked
        answer: The generated answer
        cohere_client: Cohere client instance

    Returns:
        Score: 1.0 (correct), 0.5 (partially correct), 0.0 (incorrect)
    """
    prompt = f"""You are evaluating the correctness of an answer to a question.

Question: {question}

Answer: {answer}

Rate the correctness of the answer on this scale:
- 1.0: The answer is accurate and directly addresses the question
- 0.5: The answer is partially correct but incomplete or somewhat inaccurate
- 0.0: The answer is incorrect or does not address the question

Respond with ONLY a single number: 1.0, 0.5, or 0.0"""

    try:
        response = cohere_client.chat(
            model="command-r-plus-08-2024",
            message=prompt,
            temperature=0.0
        )

        score_text = response.text.strip()
        # Extract number from response
        if '1.0' in score_text or '1 ' in score_text or score_text == '1':
            return 1.0
        elif '0.5' in score_text:
            return 0.5
        else:
            return 0.0

    except Exception as e:
        print(f"    Warning: Correctness evaluation failed: {e}")
        return -1.0  # Indicates evaluation failure


def evaluate_faithfulness(question: str, answer: str, context: str, cohere_client) -> float:
    """
    Evaluate answer faithfulness to retrieved context using LLM-as-a-judge.

    Args:
        question: The question asked
        answer: The generated answer
        context: The retrieved context/chunks
        cohere_client: Cohere client instance

    Returns:
        Score: 1.0 (faithful) or 0.0 (not faithful)
    """
    # Truncate context if too long
    max_context_chars = 10000
    if len(context) > max_context_chars:
        context = context[:max_context_chars] + "...[truncated]"

    prompt = f"""You are evaluating whether an answer is faithful to (supported by) the provided context.

Question: {question}

Context:
{context}

Answer: {answer}

Is the answer faithful to the context? That is, are all claims in the answer supported by information in the context?

Rate faithfulness:
- 1.0: The answer is fully supported by the context (faithful)
- 0.0: The answer contains information not in the context or contradicts it (not faithful)

Respond with ONLY a single number: 1.0 or 0.0"""

    try:
        response = cohere_client.chat(
            model="command-r-plus-08-2024",
            message=prompt,
            temperature=0.0
        )

        score_text = response.text.strip()
        # Extract number from response
        if '1.0' in score_text or '1 ' in score_text or score_text == '1':
            return 1.0
        else:
            return 0.0

    except Exception as e:
        print(f"    Warning: Faithfulness evaluation failed: {e}")
        return -1.0  # Indicates evaluation failure


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


def run_evaluation_query(question_id: int, question: str, mode: str, top_k: int = 5,
                          evaluate_quality: bool = False, cohere_client=None) -> dict:
    """
    Run a single evaluation query and collect metrics.

    Args:
        question_id: ID of the question
        question: Question text
        mode: 'baseline' or 'improved'
        top_k: Number of chunks to retrieve
        evaluate_quality: Whether to evaluate correctness and faithfulness
        cohere_client: Cohere client for quality evaluation

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
        context_combined = "\n\n".join(chunk_texts)

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
            'status': 'success',
            'correctness': None,
            'faithfulness': None
        }

        # Evaluate quality if requested
        if evaluate_quality and cohere_client:
            print(f"    Evaluating quality...")
            correctness = evaluate_correctness(question, answer, cohere_client)
            faithfulness = evaluate_faithfulness(question, answer, context_combined, cohere_client)

            result['correctness'] = correctness if correctness >= 0 else None
            result['faithfulness'] = faithfulness if faithfulness >= 0 else None

            print(f"    ✓ Completed in {elapsed_time:.2f}s | Context: {total_context_chars:,} chars | Correctness: {correctness} | Faithfulness: {faithfulness}")
        else:
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
                        top_k: int = 5,
                        evaluate_quality: bool = False,
                        max_questions: int = None) -> None:
    """
    Run full evaluation on all questions with both modes.

    Args:
        questions_file: Path to questions JSON
        output_file: Path to output CSV file
        top_k: Number of chunks to retrieve per query
        evaluate_quality: Whether to evaluate correctness and faithfulness
        max_questions: Maximum number of questions to evaluate (None = all)
    """
    # Load environment variables
    load_dotenv()
    cohere_api_key = os.getenv("COHERE_API_KEY")

    if not cohere_api_key:
        print("Error: COHERE_API_KEY not found in .env file")
        return

    # Initialize Cohere client if quality evaluation enabled
    co = None
    if evaluate_quality:
        co = cohere.Client(cohere_api_key)

    # Load questions
    print("=" * 70)
    print("RAG EVALUATION - BASELINE VS IMPROVED")
    print("=" * 70)
    print()

    questions = load_questions(questions_file)

    # Limit number of questions if specified
    if max_questions is not None:
        questions = questions[:max_questions]

    print(f"Loaded {len(questions)} evaluation questions")
    print(f"Output file: {output_file}")
    print(f"Top-k chunks: {top_k}")
    print(f"Quality evaluation: {'Enabled' if evaluate_quality else 'Disabled'}")
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
        'correctness',
        'faithfulness',
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
            baseline_result = run_evaluation_query(
                question_id, question_text, 'baseline', top_k,
                evaluate_quality=evaluate_quality, cohere_client=co
            )
            writer.writerow(baseline_result)
            csvfile.flush()

            # Run improved mode
            improved_result = run_evaluation_query(
                question_id, question_text, 'improved', top_k,
                evaluate_quality=evaluate_quality, cohere_client=co
            )
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

            # Quality metrics if available
            correctness_scores = [float(r['correctness']) for r in successes if r.get('correctness') and r['correctness'] != '']
            faithfulness_scores = [float(r['faithfulness']) for r in successes if r.get('faithfulness') and r['faithfulness'] != '']

            if correctness_scores:
                avg_correctness = sum(correctness_scores) / len(correctness_scores)
                print(f"Average correctness: {avg_correctness:.2f} (n={len(correctness_scores)})")

            if faithfulness_scores:
                avg_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores)
                print(f"Average faithfulness: {avg_faithfulness:.2f} (n={len(faithfulness_scores)})")

        print()

    print("=" * 70)


def main():
    """Main evaluation workflow."""
    import argparse

    parser = argparse.ArgumentParser(description='Run RAG evaluation')
    parser.add_argument('--questions', default='eval/questions.json', help='Path to questions JSON file')
    parser.add_argument('--output', default='eval/results.csv', help='Path to output CSV file')
    parser.add_argument('--top-k', type=int, default=5, help='Number of chunks to retrieve')
    parser.add_argument('--evaluate-quality', action='store_true', help='Evaluate correctness and faithfulness using LLM-as-a-judge')
    parser.add_argument('--max-questions', type=int, default=None, help='Maximum number of questions to evaluate (default: all)')
    parser.add_argument('--summary', action='store_true', help='Print summary of existing results')

    args = parser.parse_args()

    if args.summary:
        print_summary(args.output)
    else:
        run_full_evaluation(
            args.questions,
            args.output,
            args.top_k,
            evaluate_quality=args.evaluate_quality,
            max_questions=args.max_questions
        )
        print()
        print_summary(args.output)


if __name__ == "__main__":
    main()
