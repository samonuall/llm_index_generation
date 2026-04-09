#!/usr/bin/env bash
# run_experiments.sh — Run agent+history experiments across CRUMB subsets.
#
# Usage:
#   bash run_experiments.sh                                                        # tip_of_the_tongue, default model
#   bash run_experiments.sh --split paper_retrieval                                # different subset
#   bash run_experiments.sh --split clinical_trial --model gemini/gemini-2.5-pro  # different subset + model
#   bash run_experiments.sh --model openai/gpt4o --api-base https://thekeymaker.umass.edu/
#
# Available --split values (must download data first):
#   tip_of_the_tongue            uv run python src/evaluation/scripts/get_data.py --split tip_of_the_tongue
#   paper_retrieval              uv run python src/evaluation/scripts/get_data.py --split paper_retrieval
#   clinical_trial               uv run python src/evaluation/scripts/get_data.py --split clinical_trial
#   legal_qa                     uv run python src/evaluation/scripts/get_data.py --split legal_qa
#   code_retrieval               uv run python src/evaluation/scripts/get_data.py --split code_retrieval
#   set_operation_entity_retrieval  uv run python src/evaluation/scripts/get_data.py --split set_operation_entity_retrieval
#
# Results are written to results/{model}/{condition}_{timestamp}.json

set -euo pipefail

BASELINE_PREPROCESS="src/agents/baseline/preprocess.py"
AGENT_PREPROCESS="src/agents/analysis_code_agent/preprocess.py"

# Parse flags
SPLIT="tip_of_the_tongue"
MODEL_ARGS=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --split)
            SPLIT="$2"
            shift 2
            ;;
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

# Check data is downloaded
if [ ! -f "data/${SPLIT}/documents.jsonl" ] || [ ! -f "data/${SPLIT}/queries.jsonl" ]; then
    echo "ERROR: data/${SPLIT}/ not found."
    echo "Download it first with:"
    echo "  uv run python src/evaluation/scripts/get_data.py --split ${SPLIT}"
    exit 1
fi

echo ""
echo "=============================================="
echo "  CRUMB split : ${SPLIT}"
echo "  $(wc -l < data/${SPLIT}/queries.jsonl) queries, $(wc -l < data/${SPLIT}/documents.jsonl | tr -d ' ') docs cached"
echo "=============================================="

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
    echo "  RUNNING: $label  [split=${SPLIT}]"
    echo "=============================================="
    cleanup_port
    reset_preprocess
    uv run python main.py --split "$SPLIT" "$@" $MODEL_ARGS
    echo ">>> Done: $label"

    # Generate plots for this experiment (ok if it fails)
    LATEST_DIR=$(ls -td ablation_experiments/*_${label}_* 2>/dev/null | head -1)
    if [ -n "$LATEST_DIR" ]; then
        echo ">>> Generating plots for $LATEST_DIR ..."
        uv run python src/agents/analysis_code_agent/plot_experiment.py --experiment-dir "$LATEST_DIR" || true
    fi
}

# Agent + History (5 loops)
run_experiment "agent_history" \
    --agent analysis_code_agent --loops 5 --condition agent_history

# --- Other conditions (commented out) ---
# # One-shot baseline
# run_experiment "one_shot" \
#     --agent one_shot
#
# # Agent — no history, no contrastive
# run_experiment "agent" \
#     --agent analysis_code_agent --loops 3 --condition agent
#
# # Agent + Contrastive (no history)
# run_experiment "agent_contrastive_no_history" \
#     --agent analysis_code_agent --loops 3 --condition agent_contrastive_no_history
#
# # Agent + History + Contrastive (3 loops)
# run_experiment "agent_contrastive" \
#     --agent analysis_code_agent --loops 3 --condition agent_contrastive
#
# # Agent + History + Contrastive (7 loops)
# run_experiment "agent_contrastive_7loops" \
#     --agent analysis_code_agent --loops 7 --condition agent_contrastive

echo ""
echo "=============================================="
echo "  ALL EXPERIMENTS COMPLETE"
echo "  Split   : ${SPLIT}"
echo "  Results in: results/"
echo "=============================================="
ls -lt results/**/*.json results/*.json 2>/dev/null | head -15
