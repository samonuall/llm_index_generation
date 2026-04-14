#!/usr/bin/env bash
set -euo pipefail

show_help() {
    cat <<EOF
Usage: $0 [--split NAME] [--max-distractors N] [--model MODEL] [--api-base URL]

--split            Split name (default: tip_of_the_tongue)
--max-distractors  Max non-relevant docs to sample (default: 9000)
--model            Model identifier (e.g., openai/gpt4o)
--api-base         API base URL
EOF
}

# defaults
SPLIT="tip_of_the_tongue"
MAX_DISTRACTORS="9000"
MODEL_ARGS_ARRAY=()

# parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --split)
            SPLIT="$2"; shift 2
            ;;
        --max-distractors)
            MAX_DISTRACTORS="$2"; shift 2
            ;;
        --model)
            MODEL_ARGS_ARRAY+=("--model" "$2"); shift 2
            ;;
        --api-base)
            MODEL_ARGS_ARRAY+=("--api-base" "$2"); shift 2
            ;;
        --help|-h)
            show_help; exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            show_help
            exit 1
            ;;
    esac
done

BASELINE_PREPROCESS="src/agents/baseline/preprocess.py"
AGENT_PREPROCESS="src/agents/analysis_code_agent/preprocess.py"

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
    local label="$1"; shift
    local extra_args=("$@")
    echo ""
    echo "=============================================="
    echo "  RUNNING: $label"
    echo "=============================================="
    cleanup_port
    reset_preprocess

    uv run python main.py "${extra_args[@]}" --split "$SPLIT" --max-distractors "$MAX_DISTRACTORS" "${MODEL_ARGS_ARRAY[@]}"

    echo ">>> Done: $label"

    LATEST_DIR=$(ls -td ablation_experiments/*_${label}_* 2>/dev/null | head -1)
    if [ -n "$LATEST_DIR" ]; then
        echo ">>> Generating plots for $LATEST_DIR ..."
        uv run python src/agents/analysis_code_agent/plot_experiment.py --experiment-dir "$LATEST_DIR" || true
    fi
}

# Experiments
run_experiment "one_shot" --agent one_shot
run_experiment "agent" --agent analysis_code_agent --loops 3 --condition agent
run_experiment "agent_history" --agent analysis_code_agent --loops 3 --condition agent_history
run_experiment "agent_contrastive_no_history" --agent analysis_code_agent --loops 3 --condition agent_contrastive_no_history
run_experiment "agent_contrastive" --agent analysis_code_agent --loops 3 --condition agent_contrastive
run_experiment "agent_contrastive_7loops" --agent analysis_code_agent --loops 7 --condition agent_contrastive

echo ""
echo "=============================================="
echo "  ALL EXPERIMENTS COMPLETE"
echo "=============================================="
ls -lt results/**/*.json results/*.json 2>/dev/null | head -15