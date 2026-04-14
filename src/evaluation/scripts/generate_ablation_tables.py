#!/usr/bin/env python3
"""
Generate ablation experiment results tables.

Usage:
    python src/evaluation/scripts/generate_ablation_tables.py
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
import pandas as pd

def load_all_results(base_dir: Path = Path("ablation_experiments")) -> List[Dict[str, Any]]:
    """Load all results.json files from ablation experiments."""
    results = []
    for results_file in base_dir.rglob("results.json"):
        try:
            with open(results_file, 'r') as f:
                data = json.load(f)
                data['_source_file'] = str(results_file)
                results.append(data)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load {results_file}: {e}")
    return results

def safe_mean(values: List[Optional[float]]) -> Optional[float]:
    """Compute mean, ignoring None values."""
    valid_values = [v for v in values if v is not None]
    if not valid_values:
        return None
    return sum(valid_values) / len(valid_values)

def aggregate_by_split_and_condition(results: List[Dict[str, Any]], 
                                     model_filter: str = None,
                                     condition_filter: str = None,
                                     exclude_conditions: List[str] = None) -> pd.DataFrame:
    """Aggregate results by split and condition, computing means if multiple runs exist."""
    grouped = defaultdict(list)
    exclude_conditions = exclude_conditions or []
    
    for result in results:
        # Apply filters
        if model_filter and result.get('model') != model_filter:
            continue
        if condition_filter and result.get('condition') != condition_filter:
            continue
        if result.get('condition') in exclude_conditions:
            continue
            
        split = result.get('split', 'unknown')
        condition = result.get('condition', 'unknown')
        model = result.get('model', 'unknown')
        key = (split, condition, model)
        grouped[key].append(result)
    
    rows = []
    for (split, condition, model), runs in grouped.items():
        # Filter out runs with missing critical metrics
        valid_runs = [
            r for r in runs 
            if r.get('baseline_recall_100') is not None 
            and r.get('final_recall_100') is not None
            and r.get('baseline_ndcg_10') is not None
            and r.get('final_ndcg_10') is not None
        ]
        
        if not valid_runs:
            print(f"Warning: Skipping {model}/{split}/{condition} - no valid runs with complete metrics")
            continue
        
        n_runs = len(valid_runs)
        
        row = {
            'model': model,
            'condition': condition,
            'split': split,
            'n_runs': n_runs,
            'baseline_recall_100': safe_mean([r.get('baseline_recall_100') for r in valid_runs]),
            'final_recall_100': safe_mean([r.get('final_recall_100') for r in valid_runs]),
            'baseline_ndcg_10': safe_mean([r.get('baseline_ndcg_10') for r in valid_runs]),
            'final_ndcg_10': safe_mean([r.get('final_ndcg_10') for r in valid_runs]),
        }
        
        # Compute deltas
        row['delta_recall_100'] = row['final_recall_100'] - row['baseline_recall_100']
        row['delta_ndcg_10'] = row['final_ndcg_10'] - row['baseline_ndcg_10']
        
        # Optional: add latency stats
        latency_times = [r['latency']['total_wall_time_seconds'] for r in valid_runs if 'latency' in r and r['latency']]
        latency_tokens = [r['latency']['total_tokens'] for r in valid_runs if 'latency' in r and r['latency']]
        
        if latency_times:
            row['avg_wall_time'] = sum(latency_times) / len(latency_times)
        if latency_tokens:
            row['avg_tokens'] = sum(latency_tokens) / len(latency_tokens)
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df

def format_delta(value: float, format_type: str = 'markdown') -> str:
    """Format delta values with appropriate signs and colors."""
    if format_type == 'markdown':
        if value > 0.001:
            return f"**+{value:.3f}**"
        elif value < -0.001:
            return f"**{value:.3f}**"
        else:
            return f"{value:.3f}"
    elif format_type == 'latex':
        if value > 0.001:
            return f"\\textcolor{{ForestGreen}}{{\\textbf{{+{value:.3f}}}}}"
        elif value < -0.001:
            return f"\\textcolor{{red}}{{\\textbf{{{value:.3f}}}}}"
        else:
            return f"{value:.3f}"
    elif format_type == 'html':
        if value > 0.001:
            color = "green"
            sign = "+"
        elif value < -0.001:
            color = "red"
            sign = ""
        else:
            color = "black"
            sign = "" if value < 0 else "+"
        return f'<span style="color: {color}; font-weight: bold;">{sign}{value:.3f}</span>'
    else:
        return f"{value:+.3f}"

def generate_markdown_table(df: pd.DataFrame, title: str = "Results") -> str:
    """Generate a Markdown table."""
    # Sort by split name
    df = df.sort_values('split')
    
    table = f"## {title}\n\n"
    table += "| Split | Baseline R@100 | Final R@100 | Δ Recall | Baseline nDCG@10 | Final nDCG@10 | Δ nDCG@10 |\n"
    table += "|-------|----------------|-------------|----------|------------------|---------------|------------|\n"
    
    for _, row in df.iterrows():
        table += f"| {row['split']} "
        table += f"| {row['baseline_recall_100']:.3f} "
        table += f"| {row['final_recall_100']:.3f} "
        table += f"| {format_delta(row['delta_recall_100'], 'markdown')} "
        table += f"| {row['baseline_ndcg_10']:.3f} "
        table += f"| {row['final_ndcg_10']:.3f} "
        table += f"| {format_delta(row['delta_ndcg_10'], 'markdown')} |\n"
    
    table += "\n"
    return table

def generate_latex_table(df: pd.DataFrame, title: str = "Results", label: str = "tab:results") -> str:
    """Generate a LaTeX table."""
    # Sort by split name
    df = df.sort_values('split')
    
    table = "\\begin{table}[ht]\n"
    table += "\\centering\n"
    table += "\\begin{tabular}{|l|c|c|c|c|c|c|}\n"
    table += "\\hline\n"
    table += "\\textbf{Split} & \\textbf{Baseline} & \\textbf{Final} & \\textbf{$\\Delta$ Recall} & "
    table += "\\textbf{Baseline} & \\textbf{Final} & \\textbf{$\\Delta$ nDCG@10} \\\\\n"
    table += " & \\textbf{R@100} & \\textbf{R@100} &  & \\textbf{nDCG@10} & \\textbf{nDCG@10} &  \\\\\n"
    table += "\\hline\n"
    
    for _, row in df.iterrows():
        split_name = row['split'].replace('_', '\\_')
        table += f"{split_name} "
        table += f"& {row['baseline_recall_100']:.3f} "
        table += f"& {row['final_recall_100']:.3f} "
        table += f"& {format_delta(row['delta_recall_100'], 'latex')} "
        table += f"& {row['baseline_ndcg_10']:.3f} "
        table += f"& {row['final_ndcg_10']:.3f} "
        table += f"& {format_delta(row['delta_ndcg_10'], 'latex')} \\\\\n"
    
    table += "\\hline\n"
    table += "\\end{tabular}\n"
    table += f"\\caption{{{title}}}\n"
    table += f"\\label{{{label}}}\n"
    table += "\\end{table}\n\n"
    
    return table

def generate_html_table(df: pd.DataFrame, title: str = "Results") -> str:
    """Generate an HTML table with colored deltas."""
    # Sort by split name
    df = df.sort_values('split')
    
    table = f"<h2>{title}</h2>\n"
    table += '<table border="1" style="border-collapse: collapse; margin: 20px 0;">\n'
    table += '  <thead>\n'
    table += '    <tr style="background-color: #f2f2f2;">\n'
    table += '      <th style="padding: 8px;">Split</th>\n'
    table += '      <th style="padding: 8px;">Baseline R@100</th>\n'
    table += '      <th style="padding: 8px;">Final R@100</th>\n'
    table += '      <th style="padding: 8px;">Δ Recall</th>\n'
    table += '      <th style="padding: 8px;">Baseline nDCG@10</th>\n'
    table += '      <th style="padding: 8px;">Final nDCG@10</th>\n'
    table += '      <th style="padding: 8px;">Δ nDCG@10</th>\n'
    table += '    </tr>\n'
    table += '  </thead>\n'
    table += '  <tbody>\n'
    
    for _, row in df.iterrows():
        table += '    <tr>\n'
        table += f'      <td style="padding: 8px;"><strong>{row["split"]}</strong></td>\n'
        table += f'      <td style="padding: 8px;">{row["baseline_recall_100"]:.3f}</td>\n'
        table += f'      <td style="padding: 8px;">{row["final_recall_100"]:.3f}</td>\n'
        table += f'      <td style="padding: 8px;">{format_delta(row["delta_recall_100"], "html")}</td>\n'
        table += f'      <td style="padding: 8px;">{row["baseline_ndcg_10"]:.3f}</td>\n'
        table += f'      <td style="padding: 8px;">{row["final_ndcg_10"]:.3f}</td>\n'
        table += f'      <td style="padding: 8px;">{format_delta(row["delta_ndcg_10"], "html")}</td>\n'
        table += '    </tr>\n'
    
    table += '  </tbody>\n'
    table += '</table>\n\n'
    
    return table

def save_csv(df: pd.DataFrame, output_path: Path):
    """Save results as CSV."""
    df.to_csv(output_path, index=False)
    print(f"  ✓ Saved CSV: {output_path}")

def sanitize_filename(name: str) -> str:
    """Convert model/condition names to valid filenames."""
    return name.replace('/', '_').replace(' ', '_')

def main():
    parser = argparse.ArgumentParser(description="Generate ablation experiment tables")
    parser.add_argument('--format', type=str, default='all', 
                       choices=['markdown', 'latex', 'html', 'csv', 'all'],
                       help='Output format (default: all)')
    parser.add_argument('--output-dir', type=Path, default=Path('results_tables'),
                       help='Output directory for generated tables')
    parser.add_argument('--exclude-conditions', type=str, nargs='+', 
                       default=['one_shot'],
                       help='Conditions to exclude (default: one_shot)')
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(exist_ok=True)
    
    # Load all results
    print("Loading results from ablation_experiments/...")
    results = load_all_results()
    print(f"✓ Loaded {len(results)} results files")
    
    if not results:
        print("Error: No results found!")
        return
    
    # Get unique models and conditions
    all_models = sorted(set(r.get('model', 'unknown') for r in results))
    all_conditions = sorted(set(r.get('condition', 'unknown') for r in results 
                                if r.get('condition') not in args.exclude_conditions))
    
    print(f"\nExcluding conditions: {args.exclude_conditions}")
    print(f"\nProcessing:")
    print(f"  Models: {all_models}")
    print(f"  Conditions: {all_conditions}")
    
    # Generate tables for each (model, condition) combination
    for model in all_models:
        print(f"\n{'='*70}")
        print(f"Model: {model}")
        print(f"{'='*70}")
        
        for condition in all_conditions:
            # Aggregate data for this model + condition
            df = aggregate_by_split_and_condition(
                results, 
                model_filter=model, 
                condition_filter=condition,
                exclude_conditions=args.exclude_conditions
            )
            
            if df.empty:
                print(f"  ⊘ No data for {condition}")
                continue
            
            # Build title and filenames
            title = f"{model} - {condition}"
            filename_base = f"{sanitize_filename(model)}_{condition}"
            label = f"tab:{sanitize_filename(model)}_{condition}"
            
            print(f"\n  ✓ {condition}: {len(df)} split(s)")
            
            # Generate tables in requested formats
            if args.format in ['markdown', 'all']:
                md_table = generate_markdown_table(df, title)
                output_file = args.output_dir / f"{filename_base}.md"
                with open(output_file, 'w') as f:
                    f.write(md_table)
                print(f"    → {output_file}")
            
            if args.format in ['latex', 'all']:
                latex_table = generate_latex_table(df, title, label)
                output_file = args.output_dir / f"{filename_base}.tex"
                with open(output_file, 'w') as f:
                    f.write(latex_table)
                print(f"    → {output_file}")
            
            if args.format in ['html', 'all']:
                html_table = generate_html_table(df, title)
                output_file = args.output_dir / f"{filename_base}.html"
                with open(output_file, 'w') as f:
                    f.write(html_table)
                print(f"    → {output_file}")
            
            if args.format in ['csv', 'all']:
                output_file = args.output_dir / f"{filename_base}.csv"
                save_csv(df, output_file)
                print(f"    → {output_file}")
    
    # Also create a combined LaTeX file with all tables
    if args.format in ['latex', 'all']:
        print(f"\n{'='*70}")
        print("Creating combined LaTeX file...")
        combined_file = args.output_dir / "all_results_combined.tex"
        with open(combined_file, 'w') as f:
            f.write("% Required LaTeX packages:\n")
            f.write("% \\usepackage{xcolor}\n")
            f.write("% \\definecolor{ForestGreen}{RGB}{34,139,34}\n\n")
            
            for model in all_models:
                for condition in all_conditions:
                    df = aggregate_by_split_and_condition(
                        results, 
                        model_filter=model, 
                        condition_filter=condition,
                        exclude_conditions=args.exclude_conditions
                    )
                    if not df.empty:
                        title = f"{model} - {condition}"
                        label = f"tab:{sanitize_filename(model)}_{condition}"
                        table = generate_latex_table(df, title, label)
                        # Remove the package comments from individual tables
                        table = table.split('\n\n', 1)[-1]
                        f.write(table + "\n")
        
        print(f"✓ Combined file: {combined_file}")
    
    print(f"\n{'='*70}")
    print(f"✅ Done! All tables saved to: {args.output_dir}/")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()