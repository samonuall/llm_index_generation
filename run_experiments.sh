#!/usr/bin/env bash
# run_experiments.sh — Run all ablation conditions from a clean baseline each time.
# Usage: bash run_experiments.sh [SPLIT]
#   SPLIT: CRUMB split name (default: tip_of_the_tongue_5000docs)
# Results are written to results/{condition}_{timestamp}.json

set -euo pipefail

SPLIT="${1:-tip_of_the_tongue_5000docs}"

BASELINE_PREPROCESS="src/agents/baseline/preprocess.py"
AGENT_PREPROCESS="src/agents/analysis_code_agent/preprocess.py"
ONE_SHOT_PREPROCESS="src/agents/analysis_code_agent/preprocess.py"

reset_preprocess() {
    echo ""
    echo ">>> Resetting preprocess.py to baseline..."
    cp "$BASELINE_PREPROCESS" "$AGENT_PREPROCESS"
}

run_experiment() {
    local label="$1"
    shift
    echo ""
    echo "=============================================="
    echo "  RUNNING: $label"
    echo "=============================================="
    reset_preprocess
    uv run python main.py "$@"
    echo ">>> Done: $label"

    # Generate plots for this experiment (ok if it fails)
    LATEST_DIR=$(ls -td ablation_experiments/*_${label}_* 2>/dev/null | head -1)
    if [ -n "$LATEST_DIR" ]; then
        echo ">>> Generating plots for $LATEST_DIR ..."
        uv run python src/agents/analysis_code_agent/plot_experiment.py --experiment-dir "$LATEST_DIR" || true
    fi
}

# 1. One-shot (single LLM call, no loop)
run_experiment "one_shot" \
    --agent one_shot --split "$SPLIT"

# 2. Agent — no history, no contrastive
run_experiment "agent" \
    --agent analysis_code_agent --loops 3 --condition agent --split "$SPLIT"

# 3. Agent + History
run_experiment "agent_history" \
    --agent analysis_code_agent --loops 3 --condition agent_history --split "$SPLIT"

# 4. Agent + Contrastive (no history)
run_experiment "agent_contrastive_no_history" \
    --agent analysis_code_agent --loops 3 --condition agent_contrastive_no_history --split "$SPLIT"

# 5. Agent + History + Contrastive (3 loops)
run_experiment "agent_contrastive" \
    --agent analysis_code_agent --loops 3 --condition agent_contrastive --split "$SPLIT"

# 6. Agent + History + Contrastive (7 loops)
run_experiment "agent_contrastive_7loops" \
    --agent analysis_code_agent --loops 7 --condition agent_contrastive --split "$SPLIT"

echo ""
echo "=============================================="
echo "  ALL EXPERIMENTS COMPLETE"
echo "  Results in: results/"
echo "=============================================="
ls -lt results/*.json | head -10
