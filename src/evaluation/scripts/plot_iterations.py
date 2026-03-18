"""
Plot iteration results for LLM agents across experiments.

Usage:
    # Plot all results (organized by split)
    python -m src.evaluation.scripts.plot_iterations
    
    # Filter by agent
    python -m src.evaluation.scripts.plot_iterations --agent lite_llm_agent
    
    # Filter by split
    python -m src.evaluation.scripts.plot_iterations --split code_retrieval
    
    # Specific agent + split
    python -m src.evaluation.scripts.plot_iterations --agent lite_llm_agent --split code_retrieval
    
    # Use separate subplots instead of overlay
    python -m src.evaluation.scripts.plot_iterations --separate-plots
    
    # Custom output directory
    python -m src.evaluation.scripts.plot_iterations --output-dir my_plots
"""

import json
import re
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Optional
import argparse

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np


@dataclass
class IterationResult:
    agent: str
    split: str
    iteration: int
    n_docs: int
    top_k: int
    timestamp: str
    recall_at_10: Optional[float]
    recall_at_100: Optional[float]
    ndcg_at_10: Optional[float]
    n_chunks: Optional[int]
    chunks_per_doc: Optional[float]
    filepath: Path


def parse_filename(filename: str) -> Optional[Dict[str, any]]:
    """
    Parse filename like: lite_llm_agent_232444docs_100_iter0.json
    Returns: {'agent': 'lite_llm_agent', 'n_docs': 232444, 'top_k': 100, 'iteration': 0}
    """
    pattern = r'^(.+?)_(\d+)docs_(\d+)_iter(\d+)\.json$'
    match = re.match(pattern, filename)
    if match:
        return {
            'agent': match.group(1),
            'n_docs': int(match.group(2)),
            'top_k': int(match.group(3)),
            'iteration': int(match.group(4))
        }
    return None


def load_results(results_dir: Path, agent_filter: Optional[str] = None, 
                 split_filter: Optional[str] = None) -> List[IterationResult]:
    """Load all iteration results from JSON files."""
    results = []
    
    if not results_dir.exists():
        print(f"⚠️  Results directory not found: {results_dir}")
        return results
    
    for json_file in results_dir.rglob("*.json"):
        parsed = parse_filename(json_file.name)
        if not parsed:
            continue
            
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Apply filters
            if agent_filter and data.get('agent') != agent_filter:
                continue
            if split_filter and data.get('split') != split_filter:
                continue
            
            metrics = data.get('metrics', {})
            config = data.get('config', {})
            
            result = IterationResult(
                agent=data.get('agent'),
                split=data.get('split'),
                iteration=data.get('iteration', parsed['iteration']),
                n_docs=config.get('n_docs', parsed['n_docs']),
                top_k=config.get('top_k', parsed['top_k']),
                timestamp=data.get('timestamp'),
                recall_at_10=metrics.get('recall_at_10'),
                recall_at_100=metrics.get('recall_at_100'),
                ndcg_at_10=metrics.get('ndcg_at_10'),
                n_chunks=config.get('n_chunks'),
                chunks_per_doc=config.get('chunks_per_doc'),
                filepath=json_file
            )
            results.append(result)
            
        except (json.JSONDecodeError, KeyError) as e:
            print(f"⚠️  Skipping {json_file.name}: {e}")
            continue
    
    return results


def group_by_experiment(results: List[IterationResult]) -> Dict[str, List[IterationResult]]:
    """Group results by (agent, split, n_docs, top_k)."""
    groups = defaultdict(list)
    for r in results:
        key = (r.agent, r.split, r.n_docs, r.top_k)
        groups[key].append(r)
    
    # Sort each group by iteration
    for key in groups:
        groups[key].sort(key=lambda x: x.iteration)
    
    return dict(groups)


def calculate_improvements(results: List[IterationResult], metric: str) -> Dict[str, any]:
    """Calculate best-so-far trajectory and improvement stats."""
    values = [getattr(r, metric) for r in results if getattr(r, metric) is not None]
    if not values:
        return None
    
    # For recall/ndcg, higher is better
    best_so_far = []
    current_best = -np.inf
    accepted_indices = []
    
    for i, val in enumerate(values):
        if val > current_best:
            current_best = val
            accepted_indices.append(i)
        best_so_far.append(current_best)
    
    initial = values[0]
    final = values[-1]
    best = max(values)
    improvement = ((final - initial) / initial * 100) if initial != 0 else 0
    
    return {
        'values': values,
        'best_so_far': best_so_far,
        'accepted_indices': accepted_indices,
        'initial': initial,
        'final': final,
        'best': best,
        'improvement_pct': improvement,
        'n_experiments': len(values)
    }


def plot_overlaid_metrics(ax, results: List[IterationResult], metrics_to_plot: List[str]):
    """Plot all metrics overlaid on a single axis."""
    
    metric_configs = {
        'recall_at_10': {'label': 'Recall@10', 'color': '#00ff00', 'marker': 'o'},
        'recall_at_100': {'label': 'Recall@100', 'color': '#00ccff', 'marker': 's'},
        'ndcg_at_10': {'label': 'nDCG@10', 'color': '#ff6600', 'marker': '^'}
    }
    
    has_data = False
    all_values = []
    best_overall = {'metric': None, 'value': -np.inf, 'iter': None}
    
    for metric in metrics_to_plot:
        stats = calculate_improvements(results, metric)
        if not stats:
            continue
        
        has_data = True
        config = metric_configs.get(metric, {'label': metric, 'color': 'white', 'marker': 'o'})
        iterations = list(range(len(stats['values'])))
        all_values.extend(stats['values'])
        
        # Track best overall across all metrics
        if stats['best'] > best_overall['value']:
            best_overall = {
                'metric': config['label'],
                'value': stats['best'],
                'iter': stats['values'].index(stats['best'])
            }
        
        # Plot all experiments as semi-transparent dots
        ax.scatter(iterations, stats['values'], 
                  c=config['color'], alpha=0.3, s=50, 
                  marker=config['marker'], zorder=2)
        
        # Plot best-so-far line
        ax.plot(iterations, stats['best_so_far'], 
               color=config['color'], linewidth=2.5, 
               label=f"{config['label']} (best: {stats['best']:.4f})", 
               zorder=3)
        
        # Highlight accepted/improved experiments
        accepted_iters = stats['accepted_indices']
        accepted_vals = [stats['values'][i] for i in accepted_iters]
        ax.scatter(accepted_iters, accepted_vals, 
                  c=config['color'], edgecolors='white',
                  s=120, zorder=4, linewidth=1.5, marker=config['marker'],
                  alpha=0.9)
    
    if not has_data:
        ax.text(0.5, 0.5, 'No metrics data', ha='center', va='center', 
                transform=ax.transAxes, fontsize=12, color='gray')
        return
    
    # Mark overall best with star
    if best_overall['iter'] is not None:
        ax.scatter([best_overall['iter']], [best_overall['value']], 
                  marker='*', s=500, c='gold', edgecolors='white', 
                  linewidth=2, zorder=5, label=f"Best: {best_overall['metric']}")
    
    # Add stats text
    n_iterations = len(results)
    stats_text = f"Iterations: {n_iterations}\n"
    
    for metric in metrics_to_plot:
        stats = calculate_improvements(results, metric)
        if stats:
            config = metric_configs.get(metric, {'label': metric})
            stats_text += f"{config['label']}: {stats['improvement_pct']:+.1f}%\n"
    
    ax.text(0.02, 0.98, stats_text.strip(), transform=ax.transAxes, 
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.8),
            color='white', family='monospace')
    
    # Styling
    ax.set_xlabel('Iteration', fontsize=11, fontweight='bold')
    ax.set_ylabel('Metric Value', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_facecolor('#1a1a1a')
    
    # Force integer x-axis
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_xlim(-0.5, len(results) - 0.5)
    
    # Legend
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9, 
             bbox_to_anchor=(0.98, 0.98))
    
    # Set y-axis limits with padding
    if all_values:
        y_min, y_max = min(all_values), max(all_values)
        y_range = y_max - y_min
        if y_range > 0:
            ax.set_ylim(max(0, y_min - 0.05 * y_range), 
                       min(1, y_max + 0.05 * y_range))
        else:
            ax.set_ylim(0, 1)


def plot_metric_trajectory(ax, results: List[IterationResult], metric: str, 
                          metric_label: str, higher_is_better: bool = True):
    """Plot single metric trajectory in autonomous ML research style."""
    stats = calculate_improvements(results, metric)
    if not stats:
        ax.text(0.5, 0.5, f'No {metric} data', ha='center', va='center', 
                transform=ax.transAxes, fontsize=12, color='gray')
        return
    
    iterations = list(range(len(stats['values'])))
    
    # Plot all experiments as gray dots
    ax.scatter(iterations, stats['values'], c='gray', alpha=0.5, s=60, 
               label='Experiments', zorder=2)
    
    # Plot best-so-far line
    ax.plot(iterations, stats['best_so_far'], 'b-', linewidth=2.5, 
            label='Best so far', zorder=3)
    
    # Highlight accepted/improved experiments
    accepted_iters = stats['accepted_indices']
    accepted_vals = [stats['values'][i] for i in accepted_iters]
    ax.scatter(accepted_iters, accepted_vals, c='lime', edgecolors='darkgreen',
               s=100, label='Accepted', zorder=4, linewidth=1.5)
    
    # Mark final best with star
    best_iter = stats['values'].index(stats['best'])
    ax.scatter([best_iter], [stats['best']], marker='*', s=400, 
               c='white', edgecolors='black', linewidth=2, zorder=5)
    
    # Add stats text
    stats_text = (
        f"Experiment: {len(iterations)} iterations\n"
        f"Best {metric_label}: {stats['best']:.4f}\n"
        f"Improvement: {stats['improvement_pct']:+.1f}%"
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7),
            color='white', family='monospace')
    
    # Styling
    ax.set_xlabel('Iteration', fontsize=11, fontweight='bold')
    ax.set_ylabel(metric_label, fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_facecolor('#1a1a1a')
    
    # Force integer x-axis
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_xlim(-0.5, len(stats['values']) - 0.5)
    
    # Legend
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    
    # Set y-axis limits with some padding
    y_min, y_max = min(stats['values']), max(stats['values'])
    y_range = y_max - y_min
    if y_range > 0:
        ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)


def create_experiment_plot(results: List[IterationResult], exp_key: tuple, 
                           metrics_to_plot: List[str], output_path: Optional[Path] = None,
                           separate_plots: bool = False):
    """Create plot for a single experiment - overlaid or separate subplots."""
    agent, split, n_docs, top_k = exp_key
    
    # Filter metrics that have data
    available_metrics = []
    metric_labels = {
        'recall_at_10': f'Recall@10',
        'recall_at_100': f'Recall@100',
        'ndcg_at_10': 'nDCG@10'
    }
    
    for metric in metrics_to_plot:
        if any(getattr(r, metric) is not None for r in results):
            available_metrics.append(metric)
    
    if not available_metrics:
        print(f"⚠️  No metrics data for {agent}/{split}")
        return
    
    # Create figure with dark background
    if separate_plots:
        n_metrics = len(available_metrics)
        fig = plt.figure(figsize=(14, 4 * n_metrics))
        fig.patch.set_facecolor('#0d0d0d')
        
        # Title
        title = f"{agent.replace('_', ' ').title()}: {split.replace('_', ' ').title()}"
        fig.suptitle(title, fontsize=16, fontweight='bold', color='white', y=0.98)
        
        # Create subplots
        gs = GridSpec(n_metrics, 1, figure=fig, hspace=0.3)
        
        for i, metric in enumerate(available_metrics):
            ax = fig.add_subplot(gs[i])
            ax.set_facecolor('#1a1a1a')
            
            # Set spine colors
            for spine in ax.spines.values():
                spine.set_edgecolor('#333333')
            
            # Set tick colors
            ax.tick_params(colors='white', which='both')
            
            plot_metric_trajectory(ax, results, metric, metric_labels[metric])
        
        footer_y = 0.01
    else:
        # Overlaid plot
        fig, ax = plt.subplots(figsize=(14, 8))
        fig.patch.set_facecolor('#0d0d0d')
        ax.set_facecolor('#1a1a1a')
        
        # Title
        title = f"{agent.replace('_', ' ').title()}: {split.replace('_', ' ').title()}"
        fig.suptitle(title, fontsize=16, fontweight='bold', color='white', y=0.96)
        
        # Set spine colors
        for spine in ax.spines.values():
            spine.set_edgecolor('#333333')
        
        # Set tick colors
        ax.tick_params(colors='white', which='both')
        
        plot_overlaid_metrics(ax, results, available_metrics)
        
        footer_y = 0.02
    
    # Add footer with experiment details
    footer_text = f"Documents: {n_docs:,} | Top-k: {top_k} | Iterations: {len(results)}"
    fig.text(0.5, footer_y, footer_text, ha='center', fontsize=10, 
             color='gray', family='monospace')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, facecolor='#0d0d0d', 
                    bbox_inches='tight', pad_inches=0.2)
        print(f"✅ Saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot iteration results for LLM preprocessing agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--agent', type=str, help='Filter by agent name')
    parser.add_argument('--split', type=str, help='Filter by split name')
    parser.add_argument('--results-dir', type=Path, default=Path('results'),
                       help='Directory containing result JSON files')
    parser.add_argument('--output', type=Path, help='Output file path (if not set, shows interactive plot)')
    parser.add_argument('--metrics', nargs='+', 
                       default=['recall_at_10', 'recall_at_100', 'ndcg_at_10'],
                       help='Metrics to plot')
    parser.add_argument('--output-dir', type=Path, default=Path('results_graphs'),
                       help='Base directory for saving plots (organized by split)')
    parser.add_argument('--separate-plots', action='store_true',
                       help='Use separate subplots for each metric instead of overlaying')
    
    args = parser.parse_args()
    
    # Load results
    print(f"📊 Loading results from {args.results_dir}...")
    results = load_results(args.results_dir, args.agent, args.split)
    
    if not results:
        print("❌ No results found matching filters")
        return
    
    print(f"✅ Loaded {len(results)} result files")
    
    # Group by experiment
    experiments = group_by_experiment(results)
    print(f"📈 Found {len(experiments)} unique experiment(s):")
    for (agent, split, n_docs, top_k), exp_results in experiments.items():
        print(f"  • {agent}/{split} ({n_docs} docs, top-{top_k}): {len(exp_results)} iterations")
    
    # Create plots
    if len(experiments) == 1 and args.output:
        # Single experiment with explicit output path
        exp_key, exp_results = list(experiments.items())[0]
        create_experiment_plot(exp_results, exp_key, args.metrics, args.output, args.separate_plots)
    elif len(experiments) == 1 and not args.output:
        # Single experiment, show interactively
        exp_key, exp_results = list(experiments.items())[0]
        create_experiment_plot(exp_results, exp_key, args.metrics, None, args.separate_plots)
    else:
        # Multiple experiments - organize by split
        print(f"\n📁 Saving plots to {args.output_dir}/")
        for exp_key, exp_results in experiments.items():
            agent, split, n_docs, top_k = exp_key
            
            # Organize: results_graphs/{split}/{agent}_{n_docs}docs_top{top_k}.png
            split_dir = args.output_dir / split
            filename = f"{agent}_{n_docs}docs_top{top_k}.png"
            output_path = split_dir / filename
            
            create_experiment_plot(exp_results, exp_key, args.metrics, output_path, args.separate_plots)
        
        print(f"\n✅ Generated {len(experiments)} plot(s)")


if __name__ == '__main__':
    main()