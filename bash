#!/bin/bash
# download_all_splits.sh

splits=(
    "tip_of_the_tongue"
    "paper_retrieval"
    "stack_exchange"
    "clinical_trial"
    "legal_qa"
    "theorem_retrieval"
    "code_retrieval"
    "set_operation_entity_retrieval"
)

for split in "${splits[@]}"; do
    echo "Downloading $split..."
    uv run python -m src.evaluation.scripts.get_data_extended_cache --split "$split"
done

echo "All splits downloaded!"