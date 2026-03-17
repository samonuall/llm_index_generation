"""
plot_iterations.py – Visualize agent improvement over iterations

Creates plots showing metric evolution across iterations for agents.

CLI usage:
    # Plot iterations for a specific agent
    python -m src.evaluation.scripts.plot_iterations --agent lite_llm_agent --split code_retrieval
    
    # Plot all agents on a split (comparison)
    python -m src.evaluation.scripts.plot_iterations --split code_retrieval --compare-all
    
    # Plot specific metrics only
    python -m src.evaluation.scripts.plot_iterations --agent lite_llm_agent --split code_retrieval --metrics ndcg recall
    
    # Custom output directory
    python -m src.evaluation.scripts.plot_iterations --agent lite_llm_agent --split code_retrieval --output plots/

Plots saved to: results/<split>/plots/<agent>_<metric>.png
"""

from __future__ import annotations

import sys
import pathlib
import json
import argparse
from typing import List, Dict, Optional

try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("⚠️  matplotlib not installed. Install with: pip install matplotlib")

_PROJECT_ROOT = pathlib.Path(__file__).parents[3]
RESULTS_DIR = _PROJECT_ROOT / "results"


def load_iteration_history(split: str, agent: str) -> Optional[Dict]:
    """Load iteration history for an agent on a split."""
    history_file = RESULTS_DIR / split / f"{agent}_iterations.json"
    
    if not history_file.exists():
        return None
    
    try:
        with history_file.open('r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {history_file}: {e}")
        return None


def find_all_agents_with_iterations(split: str) -> List[str]:
    """Find all agents that have iteration histories for this split."""
    split_dir = RESULTS_DIR / split
    if not split_dir.exists():
        return []
    
    agents = []
    for f in split_dir.glob("*_iterations.json"):
        agent_name = f.name.replace("_iterations.json", "")
        agents.append(agent_name)
    
    return sorted(agents)

def plot_single_agent(
    history: Dict,
    split: str,
    agent: str,
    metrics_to_plot: List[str],
    output_dir: pathlib.Path
) -> List[pathlib.Path]:
    """
    Plot metrics over iterations for a single agent.
    Creates combined view with absolute values and deltas.
    """
    if not PLOTTING_AVAILABLE:
        print("Cannot plot: matplotlib not installed")
        return []
    
    iterations = history.get('iterations', [])
    if not iterations:
        print(f"No iterations found for {agent}")
        return []
    
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_plots = []
    
    # Extract data
    iter_nums = [it.get('iteration', i+1) for i, it in enumerate(iterations)]
    
    # Define metrics
    # Define ONLY performance metrics (remove chunks_per_doc)
    metric_configs = {
        'recall_at_10': {
            'label': 'Recall@10',
            'color': '#2E86AB',
            'key': 'recall_at_10',
            'source': 'metrics'
        },
        'recall_at_100': {
            'label': 'Recall@100',
            'color': '#A23B72',
            'key': 'recall_at_100',
            'source': 'metrics'
        },
        'ndcg_at_10': {
            'label': 'nDCG@10',
            'color': '#F18F01',
            'key': 'ndcg_at_10',
            'source': 'metrics'
        }
    }
    
    # Filter to requested metrics
    if metrics_to_plot and 'all' not in metrics_to_plot:
        metric_configs = {k: v for k, v in metric_configs.items() if k in metrics_to_plot}
    
    # =========================================================================
    # COMBINED PLOT: Absolute values + Delta bars
    # =========================================================================
    
    fig = plt.figure(figsize=(18, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[2, 1], wspace=0.3)
    
    # Left subplot: All metrics over iterations (normalized to 0-100 scale)
    ax1 = fig.add_subplot(gs[0])
    
    # Store original values and deltas
    all_values = {}
    deltas = {}
    
    for metric_name, config in metric_configs.items():
        # Extract values
        if config['source'] == 'config':
            values = [it['config'].get(config['key'], 0) for it in iterations]
        else:
            values = [it['metrics'].get(config['key'], 0) for it in iterations]
        
        all_values[metric_name] = values
        
        # Calculate delta
        if len(values) > 1:
            deltas[metric_name] = values[-1] - values[0]
        else:
            deltas[metric_name] = 0
        
        # Normalize to percentage of initial value for better visualization
        if values[0] != 0:
            normalized = [(v / values[0]) * 100 for v in values]
        else:
            normalized = [v * 100 for v in values]
        
        # Plot
        ax1.plot(iter_nums, normalized,
                marker='o',
                linewidth=2.5,
                markersize=8,
                color=config['color'],
                label=config['label'],
                alpha=0.9)
    
    ax1.set_xlabel('Iteration', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Normalized Value (% of Initial)', fontsize=13, fontweight='bold')
    ax1.set_title(f'Metric Evolution Over Iterations\n{agent} on {split}',
                  fontsize=15, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xticks(iter_nums)
    ax1.legend(loc='best', fontsize=11, framealpha=0.95)
    ax1.axhline(y=100, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    
    # Right subplot: Delta (improvement) as horizontal bar chart
    ax2 = fig.add_subplot(gs[1])
    
    metric_labels = [config['label'] for config in metric_configs.values()]
    delta_values = list(deltas.values())
    colors_list = [config['color'] for config in metric_configs.values()]
    
    bars = ax2.barh(metric_labels, delta_values, color=colors_list, alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for i, (bar, delta) in enumerate(zip(bars, delta_values)):
        width = bar.get_width()
        label_x = width + (0.002 if width > 0 else -0.002)
        ha = 'left' if width > 0 else 'right'
        
        # Show absolute delta and percentage change
        original = all_values[list(metric_configs.keys())[i]][0]
        if original != 0:
            pct_change = (delta / original) * 100
            label_text = f'{delta:+.4f}\n({pct_change:+.1f}%)'
        else:
            label_text = f'{delta:+.4f}'
        
        ax2.text(label_x, bar.get_y() + bar.get_height()/2,
                label_text,
                ha=ha, va='center',
                fontsize=10,
                fontweight='bold')
    
    ax2.axvline(x=0, color='black', linewidth=1.5, linestyle='-')
    ax2.set_xlabel('Δ (Change from Iter 1 to Final)', fontsize=13, fontweight='bold')
    ax2.set_title('Total Improvement', fontsize=15, fontweight='bold', pad=20)
    ax2.grid(True, axis='x', alpha=0.3, linestyle='--')
    
    # Color the background based on positive/negative
    xlim = ax2.get_xlim()
    ax2.axvspan(0, xlim[1], alpha=0.1, color='green')
    ax2.axvspan(xlim[0], 0, alpha=0.1, color='red')
    
    plt.tight_layout()
    
    # Save combined plot
    combined_path = output_dir / f"{agent}_combined.png"
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    plt.close()
    saved_plots.append(combined_path)
    print(f"✓ Saved combined plot: {combined_path}")
    
    # =========================================================================
    # ABSOLUTE VALUES PLOT (all on one graph - single axis)
    # =========================================================================

    fig, ax = plt.subplots(figsize=(14, 8))

    for metric_name, config in metric_configs.items():
        values = all_values[metric_name]
        ax.plot(iter_nums, values,
            marker='o',
            linewidth=2.5,
            markersize=8,
            color=config['color'],
            label=config['label'],
            alpha=0.9)
        
        # Add value labels
        for x, y in zip(iter_nums, values):
            ax.annotate(f'{y:.4f}',
                        xy=(x, y),
                        xytext=(0, 8),
                        textcoords='offset points',
                        ha='center',
                        fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.3',
                                facecolor='white',
                                edgecolor=config['color'],
                                alpha=0.7))

    ax.set_xlabel('Iteration', fontsize=13, fontweight='bold')
    ax.set_ylabel('Metric Value', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xticks(iter_nums)
    ax.legend(loc='best', fontsize=12, framealpha=0.95)

    plt.title(f'Retrieval Metrics Over Iterations\n{agent} on {split}',
            fontsize=15, fontweight='bold', pad=20)
    plt.tight_layout()

    # Save
    absolute_path = output_dir / f"{agent}_absolute.png"
    plt.savefig(absolute_path, dpi=300, bbox_inches='tight')
    plt.close()
    saved_plots.append(absolute_path)
    print(f"✓ Saved absolute values plot: {absolute_path}")
    
    # =========================================================================
    # DELTA ONLY PLOT (bar chart with all metrics)
    # =========================================================================
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metric_labels = [config['label'] for config in metric_configs.values()]
    delta_values = list(deltas.values())
    colors_list = [config['color'] for config in metric_configs.values()]
    
    bars = ax.bar(range(len(metric_labels)), delta_values, 
                  color=colors_list, alpha=0.8, edgecolor='black', linewidth=2)
    
    # Add value labels on bars
    for i, (bar, delta) in enumerate(zip(bars, delta_values)):
        height = bar.get_height()
        label_y = height + (0.002 if height > 0 else -0.002)
        va = 'bottom' if height > 0 else 'top'
        
        # Show absolute delta and percentage change
        original = all_values[list(metric_configs.keys())[i]][0]
        if original != 0:
            pct_change = (delta / original) * 100
            label_text = f'{delta:+.4f}\n({pct_change:+.1f}%)'
        else:
            label_text = f'{delta:+.4f}'
        
        ax.text(bar.get_x() + bar.get_width()/2, label_y,
               label_text,
               ha='center', va=va,
               fontsize=11,
               fontweight='bold')
    
    ax.axhline(y=0, color='black', linewidth=2, linestyle='-')
    ax.set_xticks(range(len(metric_labels)))
    ax.set_xticklabels(metric_labels, fontsize=12, fontweight='bold')
    ax.set_ylabel('Δ (Improvement)', fontsize=13, fontweight='bold')
    ax.set_title(f'Total Improvement Across All Metrics\n{agent} on {split}',
                fontsize=15, fontweight='bold', pad=20)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    
    # Color background
    ylim = ax.get_ylim()
    ax.axhspan(0, ylim[1], alpha=0.1, color='green')
    ax.axhspan(ylim[0], 0, alpha=0.1, color='red')
    
    plt.tight_layout()
    
    # Save
    delta_path = output_dir / f"{agent}_delta.png"
    plt.savefig(delta_path, dpi=300, bbox_inches='tight')
    plt.close()
    saved_plots.append(delta_path)
    print(f"✓ Saved delta plot: {delta_path}")
    
    return saved_plots


def plot_comparison(
    histories: Dict[str, Dict],
    split: str,
    metric: str,
    output_dir: pathlib.Path
) -> pathlib.Path:
    """
    Plot comparison of multiple agents on the same metric.
    
    Args:
        histories: Dict of {agent_name: history_dict}
        split: Split name
        metric: Metric to compare (e.g., 'ndcg_at_10')
        output_dir: Where to save plot
    """
    if not PLOTTING_AVAILABLE:
        return None
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D', '#C73E1D', '#6A4C93']
    
    for idx, (agent, history) in enumerate(histories.items()):
        iterations = history.get('iterations', [])
        if not iterations:
            continue
        
        iter_nums = [it.get('iteration', i+1) for i, it in enumerate(iterations)]
        
        # Extract values
        if metric == 'chunks_per_doc':
            values = [it['config'].get(metric, 0) for it in iterations]
        else:
            values = [it['metrics'].get(metric, 0) for it in iterations]
        
        color = colors[idx % len(colors)]
        ax.plot(iter_nums, values,
                marker='o',
                linewidth=2.5,
                markersize=8,
                color=color,
                label=agent,
                alpha=0.8)
    
    ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
    ax.set_title(f"{metric.replace('_', ' ').title()} Comparison\n{split}", 
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    
    filename = f"comparison_{metric}.png"
    save_path = output_dir / filename
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved comparison: {save_path}")
    return save_path


def main():
    parser = argparse.ArgumentParser(description="Plot agent iteration metrics")
    parser.add_argument("--agent", type=str, help="Agent name")
    parser.add_argument("--split", type=str, required=True, help="Split name")
    parser.add_argument("--compare-all", action="store_true", 
                       help="Compare all agents on this split")
    parser.add_argument("--metrics", nargs='+', 
                       default=['all'],
                       choices=['all', 'recall_at_10', 'recall_at_100', 'ndcg_at_10', 'chunks_per_doc'],
                       help="Which metrics to plot")
    parser.add_argument("--output", type=str, 
                       help="Custom output directory (default: results/<split>/plots/)")
    
    args = parser.parse_args()
    
    if not PLOTTING_AVAILABLE:
        print("ERROR: matplotlib is required. Install with: pip install matplotlib")
        sys.exit(1)
    
    # Determine output directory
    if args.output:
        output_dir = pathlib.Path(args.output)
    else:
        output_dir = RESULTS_DIR / args.split / "plots"
    
    if args.compare_all:
        # Compare all agents
        agents = find_all_agents_with_iterations(args.split)
        if not agents:
            print(f"No agents with iteration histories found for split: {args.split}")
            sys.exit(1)
        
        print(f"Found {len(agents)} agents with iterations: {', '.join(agents)}")
        
        histories = {}
        for agent in agents:
            history = load_iteration_history(args.split, agent)
            if history:
                histories[agent] = history
        
        if not histories:
            print("Could not load any iteration histories")
            sys.exit(1)
        
        # Plot comparison for each metric
        metrics = ['ndcg_at_10', 'recall_at_10', 'recall_at_100', 'chunks_per_doc']
        if args.metrics and 'all' not in args.metrics:
            metrics = args.metrics
        
        for metric in metrics:
            plot_comparison(histories, args.split, metric, output_dir)
    
    elif args.agent:
        # Single agent plots
        history = load_iteration_history(args.split, args.agent)
        if not history:
            print(f"No iteration history found for {args.agent} on {args.split}")
            print(f"Expected file: {RESULTS_DIR / args.split / f'{args.agent}_iterations.json'}")
            sys.exit(1)
        
        plot_single_agent(history, args.split, args.agent, args.metrics, output_dir)
    
    else:
        print("ERROR: Must specify --agent or --compare-all")
        sys.exit(1)
    
    print(f"\n✓ All plots saved to: {output_dir}")


if __name__ == "__main__":
    main()