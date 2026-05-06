#!/bin/bash
#SBATCH --job-name=throwaway_merge_test
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=150G
#SBATCH --cpus-per-task=12
#SBATCH --partition=cpu


set -euo pipefail

# Always run from the repo root (fixed path)
REPO_DIR="/work/pi_dagarwal_umass_edu/project_13/sam/llm_index_generation"
cd "$REPO_DIR"

# Preflight: ensure repo is writable (results/, ablation_experiments/, caches, etc.)
if ! ( : > .slurm_write_test 2>/dev/null ); then
    echo "ERROR: repo root is not writable on this node: $REPO_DIR" >&2
    echo "  Fix: copy the repo to a writable filesystem (e.g. $HOME or $SCRATCH) and sbatch from there." >&2
    exit 1
fi
rm -f .slurm_write_test

# Mirror all output to a stable logs/ folder (Slurm still writes slurm-*.out/err)
# Prefer writing inside the repo, but fall back if the repo is read-only on compute nodes.
_pick_log_dir() {
    local candidates=(
        "$REPO_DIR/run_splits_logs"
        "${SLURM_SUBMIT_DIR:-}/run_splits_logs"
        "${HOME:-}/run_splits_logs"
        "${TMPDIR:-}/run_splits_logs"
        "/tmp/run_splits_logs"
    )
    for d in "${candidates[@]}"; do
        [[ -z "$d" ]] && continue
        mkdir -p "$d" 2>/dev/null && { echo "$d"; return 0; }
    done
    return 1
}

LOG_DIR="$(_pick_log_dir)" || {
    echo "ERROR: could not create any run_splits_logs directory (repo may be read-only)." >&2
    exit 1
}

LOG_FILE="$LOG_DIR/run_all_splits_${SLURM_JOB_ID:-nojob}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging to: $LOG_FILE"

# ── Sanity checks ───────────────────────────────────────────────────────────
if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: 'uv' not found in PATH on this node."
    echo "  Fix: load your modules / activate env so 'uv' is available, then re-submit."
    exit 127
fi

if [ ! -f ".env" ]; then
    echo "WARNING: .env not found in repo root ($REPO_DIR/.env)."
    echo "  Agents typically load it via python-dotenv; alternatively export keys in the job environment."
fi

# ── Splits to run ────────────────────────────────────────────────────────────
SPLITS=(
    tip_of_the_tongue
    paper_retrieval
    clinical_trial
    legal_qa
    code_retrieval
    set_operation_entity_retrieval
    theorem_retrieval
    stack_exchange
)

# ── Parse flags ───────────────────────────────────────────────────────────────
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

# ── Run ───────────────────────────────────────────────────────────────────────
FAILED=()

for SPLIT in "${SPLITS[@]}"; do
    echo ""
    echo "=============================================="
    echo "  SPLIT: ${SPLIT}"
    echo "=============================================="

    # Check data is downloaded
    if [ ! -f "data/${SPLIT}/documents.jsonl" ] || [ ! -f "data/${SPLIT}/validation_queries.jsonl" ] || [ ! -f "data/${SPLIT}/evaluation_queries.jsonl" ]; then
        echo "  SKIP — data not found. Download with:"
        echo "    uv run python src/evaluation/scripts/get_data.py --split ${SPLIT}"
        FAILED+=("${SPLIT} (no data)")
        continue
    fi

    bash run_experiments.sh --split "${SPLIT}" $MODEL_ARGS || {
        echo "  ERROR: ${SPLIT} failed"
        FAILED+=("${SPLIT} (error)")
    }
done

echo ""
echo "=============================================="
echo "  ALL SPLITS COMPLETE"
echo "  Results in: results/"
echo "=============================================="
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "  Skipped/failed: ${FAILED[*]}"
fi
ls -lt results/**/*.json results/*.json 2>/dev/null | head -20 || true
