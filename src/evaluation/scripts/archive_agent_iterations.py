import os
import json
import shutil
from pathlib import Path
from datetime import datetime

def archive_agent_iterations(
    logs_dir: str,
    model: str,
    split: str,
    loops: int,
    max_distractors: int,
    condition: str,
    final_recall_100: float,
    baseline_recall_100: float,
    final_ndcg_10: float = None,
    final_recall_10: float = None,
):
    """
    Archive iteration logs into organized folder structure:
    results/{model}/{split}/loops{N}_dist{M}_{timestamp}/
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Build structured path
    run_name = f"loops{loops}_dist{max_distractors}_{timestamp}"
    archive_root = Path("results") / model / split / run_name
    archive_root.mkdir(parents=True, exist_ok=True)
    
    # Copy logs
    logs_archive = archive_root / "logs"
    if os.path.exists(logs_dir):
        shutil.copytree(logs_dir, logs_archive, dirs_exist_ok=True)
        print(f"[archive] Logs copied to {logs_archive}")
    
    # Save config metadata
    config_data = {
        "model": model,
        "split": split,
        "loops": loops,
        "max_distractors": max_distractors,
        "condition": condition,
        "timestamp": timestamp,
        "n_datapoints": max_distractors + count_relevant_docs(split),  # add helper if needed
    }
    with open(archive_root / "config.json", "w") as f:
        json.dump(config_data, f, indent=2)
    
    # Save final results
    results_data = {
        **config_data,
        "baseline_recall_100": baseline_recall_100,
        "final_recall_100": final_recall_100,
        "improvement": final_recall_100 - baseline_recall_100,
        "final_ndcg_10": final_ndcg_10,
        "final_recall_10": final_recall_10,
    }
    with open(archive_root / "final_results.json", "w") as f:
        json.dump(results_data, f, indent=2)
    
    # Generate per-iteration summary CSV
    generate_iteration_summary_csv(logs_archive, archive_root / "summary.csv")
    
    print(f"[archive] Archived to {archive_root}")
    return archive_root

def generate_iteration_summary_csv(logs_dir: Path, output_csv: Path):
    """Generate a CSV table of per-iteration results"""
    import glob
    import pandas as pd
    
    rows = []
    hyp_files = sorted(glob.glob(str(logs_dir / "iteration_*_hypotheses.json")))
    
    for hf in hyp_files:
        iteration = int(Path(hf).stem.split("_")[1])
        with open(hf) as f:
            hyps = json.load(f)
        
        # Baseline (from first hypothesis's baseline)
        if hyps:
            baseline = hyps[0]["test_results"]["baseline_recall_100"]
        else:
            baseline = None
        
        # Best/adopted hypothesis
        adopted = [h for h in hyps if h.get("adopted")]
        if adopted:
            best = adopted[0]
            best_id = best["id"]
            best_r100 = best["test_results"]["hypothesis_recall_100"]
            delta = best["test_results"]["delta_recall_100"]
        else:
            best_id = "none"
            best_r100 = baseline
            delta = 0.0
        
        rows.append({
            "iteration": iteration,
            "baseline_recall_100": baseline,
            "adopted_hypothesis": best_id,
            "adopted_recall_100": best_r100,
            "delta_recall_100": delta,
            "n_hypotheses_tested": len(hyps),
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"[archive] Summary CSV: {output_csv}")