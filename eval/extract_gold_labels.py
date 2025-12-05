"""
Helper script to extract quality scores from evaluation results.
Populates gold_labels.csv from results.csv
"""

import csv
import os


def extract_gold_labels(results_file: str = "eval/results.csv",
                        output_file: str = "eval/gold_labels.csv") -> None:
    """
    Extract correctness and faithfulness scores from results.csv into gold_labels.csv.

    Args:
        results_file: Path to evaluation results CSV
        output_file: Path to output gold labels CSV
    """
    if not os.path.exists(results_file):
        print(f"Error: Results file not found: {results_file}")
        print("Please run evaluation with --evaluate-quality first:")
        print("  python eval/run_eval.py --evaluate-quality --max-questions 10")
        return

    # Read results
    with open(results_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print("No results found in file")
        return

    # Check if quality scores are available
    has_scores = any(r.get('correctness') and r.get('correctness') != '' for r in rows)

    if not has_scores:
        print("Warning: No quality scores found in results file.")
        print("Run evaluation with --evaluate-quality to generate scores:")
        print("  python eval/run_eval.py --evaluate-quality --max-questions 10")
        print()
        print("Creating template gold_labels.csv for manual entry...")

    # Extract gold labels
    gold_labels = []
    for row in rows:
        question_id = row['question_id']
        mode = row['mode']
        correct = row.get('correctness', '') if row.get('correctness') else ''
        faithful = row.get('faithfulness', '') if row.get('faithfulness') else ''

        gold_labels.append({
            'question_id': question_id,
            'mode': mode,
            'correct': correct,
            'faithful': faithful
        })

    # Write to gold_labels.csv
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['question_id', 'mode', 'correct', 'faithful']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(gold_labels)

    print(f"✓ Gold labels extracted to: {output_file}")
    print(f"Total entries: {len(gold_labels)}")

    # Print summary
    filled_count = sum(1 for g in gold_labels if g['correct'] != '' and g['faithful'] != '')
    print(f"Entries with scores: {filled_count}/{len(gold_labels)}")

    if filled_count > 0:
        print()
        print("Summary by mode:")
        for mode in ['baseline', 'improved']:
            mode_labels = [g for g in gold_labels if g['mode'] == mode and g['correct'] != '']
            if mode_labels:
                avg_correct = sum(float(g['correct']) for g in mode_labels) / len(mode_labels)
                avg_faithful = sum(float(g['faithful']) for g in mode_labels) / len(mode_labels)
                print(f"  {mode.upper()}: avg_correct={avg_correct:.2f}, avg_faithful={avg_faithful:.2f} (n={len(mode_labels)})")


def print_table(gold_labels_file: str = "eval/gold_labels.csv") -> None:
    """
    Print gold labels in a formatted table.

    Args:
        gold_labels_file: Path to gold labels CSV
    """
    if not os.path.exists(gold_labels_file):
        print(f"Error: Gold labels file not found: {gold_labels_file}")
        return

    with open(gold_labels_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print("No data in gold labels file")
        return

    # Print table
    print()
    print("=" * 60)
    print("GOLD LABELS - QUALITY SCORES")
    print("=" * 60)
    print()
    print(f"{'QID':<5} {'Mode':<10} {'Correct':<10} {'Faithful':<10}")
    print("-" * 60)

    for row in rows:
        qid = row['question_id']
        mode = row['mode']
        correct = row['correct'] if row['correct'] else 'N/A'
        faithful = row['faithful'] if row['faithful'] else 'N/A'

        print(f"{qid:<5} {mode:<10} {correct:<10} {faithful:<10}")

    print()
    print("=" * 60)


def main():
    """Main workflow."""
    import argparse

    parser = argparse.ArgumentParser(description='Extract gold labels from evaluation results')
    parser.add_argument('--results', default='eval/results.csv', help='Path to results CSV file')
    parser.add_argument('--output', default='eval/gold_labels.csv', help='Path to output gold labels CSV')
    parser.add_argument('--print', action='store_true', dest='print_table', help='Print gold labels as table')

    args = parser.parse_args()

    if args.print_table:
        print_table(args.output)
    else:
        extract_gold_labels(args.results, args.output)


if __name__ == "__main__":
    main()
