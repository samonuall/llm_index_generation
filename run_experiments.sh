#!/usr/bin/env bash
# run_experiments.sh — Run all ablation conditions from a clean baseline each time.
#
# Usage:
#   bash run_experiments.sh                                          # default (GPT-4o via UMass proxy)
#   bash run_experiments.sh --model gemini/gemini-2.5-pro           # Gemini 2.5 Pro (native API, needs GEMINI_API_KEY)
#   bash run_experiments.sh --model openai/gpt4o --api-base https://thekeymaker.umass.edu/
#
# Results are written to results/{condition}_{timestamp}.json

set -euo pipefail

SPLIT="${1:-tip_of_the_tongue_5000docs}"

BASELINE_PREPROCESS="src/agents/baseline/preprocess.py"
AGENT_PREPROCESS="src/agents/analysis_code_agent/preprocess.py"

# Parse optional --model and --api-base flags
MODEL_ARGS=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            MODEL_ARGS="$MODEL_ARGS --model $2"
            shift 2
            ;;
        --api-base)
            MODEL_ARGS="$MODEL_ARGS --api-base $2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

cleanup_port() {
    local port=8765
    local pid
    pid=$(lsof -i :$port -t 2>/dev/null || true)
    if [ -n "$pid" ]; then
        echo ">>> Killing leftover process on port $port (pid $pid)..."
        kill -9 "$pid" 2>/dev/null || true
        sleep 1
    fi
}

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
    cleanup_port
    reset_preprocess
    uv run python main.py "$@" $MODEL_ARGS
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
ls -lt results/**/*.json results/*.json 2>/dev/null | head -15
