"""
Summarize and analyze RAG evaluation results.
Joins results.csv with gold_labels.csv to compute accuracy metrics.
"""

import csv
import os
import pandas as pd
from typing import Dict, List


def load_results(results_file: str = "eval/results.csv") -> pd.DataFrame:
    """
    Load evaluation results from CSV.

    Args:
        results_file: Path to results CSV file

    Returns:
        DataFrame with results
    """
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Results file not found: {results_file}")

    df = pd.read_csv(results_file)
    return df


def load_gold_labels(gold_labels_file: str = "eval/gold_labels.csv") -> pd.DataFrame:
    """
    Load gold standard labels from CSV.

    Args:
        gold_labels_file: Path to gold labels CSV file

    Returns:
        DataFrame with gold labels
    """
    if not os.path.exists(gold_labels_file):
        raise FileNotFoundError(f"Gold labels file not found: {gold_labels_file}")

    df = pd.read_csv(gold_labels_file)

    # Convert empty strings to NaN for proper handling
    df['correct'] = pd.to_numeric(df['correct'], errors='coerce')
    df['faithful'] = pd.to_numeric(df['faithful'], errors='coerce')

    return df


def join_results_with_labels(results_df: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join results with gold labels on question_id and mode.

    Args:
        results_df: Results DataFrame
        labels_df: Gold labels DataFrame

    Returns:
        Joined DataFrame
    """
    # Ensure question_id is same type in both
    results_df['question_id'] = results_df['question_id'].astype(str)
    labels_df['question_id'] = labels_df['question_id'].astype(str)

    # Join on question_id and mode
    merged = results_df.merge(
        labels_df,
        on=['question_id', 'mode'],
        how='left',
        suffixes=('_llm', '_gold')
    )

    return merged


def compute_accuracy_metrics(merged_df: pd.DataFrame) -> Dict:
    """
    Compute accuracy and other quality metrics.

    Args:
        merged_df: Merged DataFrame with results and gold labels

    Returns:
        Dictionary of metrics
    """
    # Filter to only rows with gold labels
    labeled_df = merged_df[merged_df['correct_gold'].notna()].copy()

    if len(labeled_df) == 0:
        return {
            'total_evaluated': 0,
            'message': 'No gold labels found. Please populate gold_labels.csv'
        }

    metrics = {
        'total_evaluated': len(labeled_df),
        'by_mode': {}
    }

    # Compute metrics by mode
    for mode in ['baseline', 'improved']:
        mode_df = labeled_df[labeled_df['mode'] == mode]

        if len(mode_df) == 0:
            continue

        # Correctness metrics
        correct_scores = mode_df['correct_gold'].dropna()
        fully_correct = (correct_scores == 1.0).sum()
        partially_correct = (correct_scores == 0.5).sum()
        incorrect = (correct_scores == 0.0).sum()

        # Faithfulness metrics
        faithful_scores = mode_df['faithful_gold'].dropna()
        faithful = (faithful_scores == 1.0).sum()
        unfaithful = (faithful_scores == 0.0).sum()

        # Average scores
        avg_correct = correct_scores.mean() if len(correct_scores) > 0 else 0
        avg_faithful = faithful_scores.mean() if len(faithful_scores) > 0 else 0

        # LLM-as-judge metrics (if available)
        llm_correct = mode_df['correctness_llm'].dropna()
        llm_faithful = mode_df['faithfulness_llm'].dropna()

        avg_llm_correct = llm_correct.mean() if len(llm_correct) > 0 else None
        avg_llm_faithful = llm_faithful.mean() if len(llm_faithful) > 0 else None

        # Context metrics
        avg_context_chars = mode_df['total_context_chars'].mean()
        avg_elapsed_time = mode_df['elapsed_time_sec'].mean()

        metrics['by_mode'][mode] = {
            'count': len(mode_df),
            'correctness': {
                'fully_correct': fully_correct,
                'partially_correct': partially_correct,
                'incorrect': incorrect,
                'avg_score': avg_correct,
                'accuracy': fully_correct / len(correct_scores) if len(correct_scores) > 0 else 0
            },
            'faithfulness': {
                'faithful': faithful,
                'unfaithful': unfaithful,
                'avg_score': avg_faithful,
                'faithfulness_rate': faithful / len(faithful_scores) if len(faithful_scores) > 0 else 0
            },
            'llm_judge': {
                'avg_correct': avg_llm_correct,
                'avg_faithful': avg_llm_faithful
            },
            'efficiency': {
                'avg_context_chars': avg_context_chars,
                'avg_elapsed_time': avg_elapsed_time
            }
        }

    # Compute agreement between LLM-as-judge and gold labels (if both available)
    if 'correctness_llm' in merged_df.columns:
        agreement_df = labeled_df[
            labeled_df['correctness_llm'].notna() &
            labeled_df['correct_gold'].notna()
        ]

        if len(agreement_df) > 0:
            # Calculate correlation
            corr_correct = agreement_df['correctness_llm'].corr(agreement_df['correct_gold'])
            corr_faithful = agreement_df['faithfulness_llm'].corr(agreement_df['faithful_gold'])

            metrics['llm_judge_agreement'] = {
                'correctness_correlation': corr_correct,
                'faithfulness_correlation': corr_faithful,
                'sample_size': len(agreement_df)
            }

    return metrics


def print_summary(metrics: Dict) -> None:
    """
    Print formatted summary of metrics.

    Args:
        metrics: Dictionary of computed metrics
    """
    print()
    print("=" * 80)
    print("EVALUATION SUMMARY - RESULTS WITH GOLD LABELS")
    print("=" * 80)
    print()

    if 'message' in metrics:
        print(metrics['message'])
        return

    print(f"Total Evaluated: {metrics['total_evaluated']} queries")
    print()

    # Print metrics by mode
    for mode in ['baseline', 'improved']:
        if mode not in metrics['by_mode']:
            continue

        m = metrics['by_mode'][mode]

        print("-" * 80)
        print(f"{mode.upper()} MODE")
        print("-" * 80)
        print(f"Total queries: {m['count']}")
        print()

        # Correctness
        c = m['correctness']
        print("Correctness (Gold Labels):")
        print(f"  Fully correct:      {c['fully_correct']:3d} ({c['fully_correct']/m['count']*100:5.1f}%)")
        print(f"  Partially correct:  {c['partially_correct']:3d} ({c['partially_correct']/m['count']*100:5.1f}%)")
        print(f"  Incorrect:          {c['incorrect']:3d} ({c['incorrect']/m['count']*100:5.1f}%)")
        print(f"  Average score:      {c['avg_score']:.3f}")
        print(f"  Accuracy (1.0):     {c['accuracy']*100:.1f}%")
        print()

        # Faithfulness
        f = m['faithfulness']
        print("Faithfulness (Gold Labels):")
        print(f"  Faithful:           {f['faithful']:3d} ({f['faithful']/m['count']*100:5.1f}%)")
        print(f"  Unfaithful:         {f['unfaithful']:3d} ({f['unfaithful']/m['count']*100:5.1f}%)")
        print(f"  Average score:      {f['avg_score']:.3f}")
        print(f"  Faithfulness rate:  {f['faithfulness_rate']*100:.1f}%")
        print()

        # LLM-as-judge (if available)
        if m['llm_judge']['avg_correct'] is not None:
            print("LLM-as-Judge Scores:")
            print(f"  Avg correctness:    {m['llm_judge']['avg_correct']:.3f}")
            print(f"  Avg faithfulness:   {m['llm_judge']['avg_faithful']:.3f}")
            print()

        # Efficiency
        e = m['efficiency']
        print("Efficiency:")
        print(f"  Avg context size:   {e['avg_context_chars']:,.0f} chars")
        print(f"  Avg elapsed time:   {e['avg_elapsed_time']:.2f}s")
        print()

    # LLM-as-judge agreement
    if 'llm_judge_agreement' in metrics:
        a = metrics['llm_judge_agreement']
        print("-" * 80)
        print("LLM-AS-JUDGE AGREEMENT WITH GOLD LABELS")
        print("-" * 80)
        print(f"Sample size: {a['sample_size']} queries")
        print(f"Correctness correlation: {a['correctness_correlation']:.3f}")
        print(f"Faithfulness correlation: {a['faithfulness_correlation']:.3f}")
        print()

    print("=" * 80)


def create_comparison_table(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a comparison table showing baseline vs improved performance.

    Args:
        merged_df: Merged DataFrame

    Returns:
        Comparison DataFrame
    """
    # Filter to labeled data
    labeled = merged_df[merged_df['correct_gold'].notna()].copy()

    if len(labeled) == 0:
        return pd.DataFrame()

    # Group by question_id
    comparison_rows = []

    for qid in labeled['question_id'].unique():
        qid_data = labeled[labeled['question_id'] == qid]

        baseline = qid_data[qid_data['mode'] == 'baseline']
        improved = qid_data[qid_data['mode'] == 'improved']

        if len(baseline) == 0 or len(improved) == 0:
            continue

        baseline_row = baseline.iloc[0]
        improved_row = improved.iloc[0]

        comparison_rows.append({
            'question_id': qid,
            'question': baseline_row['question'][:60] + '...' if len(baseline_row['question']) > 60 else baseline_row['question'],
            'baseline_correct': baseline_row['correct_gold'],
            'improved_correct': improved_row['correct_gold'],
            'baseline_faithful': baseline_row['faithful_gold'],
            'improved_faithful': improved_row['faithful_gold'],
            'improvement': 'Yes' if improved_row['correct_gold'] > baseline_row['correct_gold'] else 'No'
        })

    return pd.DataFrame(comparison_rows)


def main():
    """Main analysis workflow."""
    import argparse

    parser = argparse.ArgumentParser(description='Summarize RAG evaluation results with gold labels')
    parser.add_argument('--results', default='eval/results.csv', help='Path to results CSV file')
    parser.add_argument('--gold-labels', default='eval/gold_labels.csv', help='Path to gold labels CSV file')
    parser.add_argument('--comparison', action='store_true', help='Show baseline vs improved comparison table')

    args = parser.parse_args()

    try:
        # Load data
        print("Loading data...")
        results_df = load_results(args.results)
        labels_df = load_gold_labels(args.gold_labels)

        print(f"Loaded {len(results_df)} results")
        print(f"Loaded {len(labels_df)} gold labels")

        # Join
        merged_df = join_results_with_labels(results_df, labels_df)

        # Compute metrics
        metrics = compute_accuracy_metrics(merged_df)

        # Print summary
        print_summary(metrics)

        # Show comparison table if requested
        if args.comparison:
            comparison_df = create_comparison_table(merged_df)

            if len(comparison_df) > 0:
                print()
                print("=" * 80)
                print("BASELINE vs IMPROVED COMPARISON")
                print("=" * 80)
                print()
                print(comparison_df.to_string(index=False))
                print()

                # Calculate improvement stats
                improved_count = (comparison_df['improvement'] == 'Yes').sum()
                print(f"Improved mode performed better on {improved_count}/{len(comparison_df)} questions ({improved_count/len(comparison_df)*100:.1f}%)")
                print()

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print()
        print("Make sure you have:")
        print("  1. Run evaluation: python eval/run_eval.py --evaluate-quality --max-questions 10")
        print("  2. Extract gold labels: python eval/extract_gold_labels.py")
        print("  3. Or manually populate eval/gold_labels.csv")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
