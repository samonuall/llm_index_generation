#!/bin/bash
# Evaluation Pipeline for 5k Documents

# Define all splits
SPLITS=(
    "paper_retrieval"
    "tip_of_the_tongue" 
    "clinical_trial"
    "code_retrieval"
    "legal_qa"
    "set_operation_entity_retrieval"
    "theorem_retrieval"
    "stack_exchange"
)

# Step 1: Download 5k docs for each split (if not already done)
echo "========================================="
echo "STEP 1: Downloading Data"
echo "========================================="
for split in "${SPLITS[@]}"; do
  if [ -d "data/${split}_5000docs" ] && [ -f "data/${split}_5000docs/documents.jsonl" ]; then
    echo "✓ Skipping $split - data already exists"
  else
    echo "📥 Downloading 5k docs from $split..."
    uv run python -m src.evaluation.scripts.get_data --split $split --limit 5000
  fi
done

# Step 2: Run baseline evaluation (for comparison)
echo ""
echo "========================================="
echo "STEP 2: Running Baseline"
echo "========================================="
for split in "${SPLITS[@]}"; do
  echo "🧪 Baseline: ${split}_5000docs..."
  uv run python -m src.evaluation.scripts.test_preprocessing_split \
    --agent baseline \
    --split ${split}_5000docs
done

# Step 3: Run lite_llm_agent with 3 iterations on each split
echo ""
echo "========================================="
echo "STEP 3: Running lite_llm_agent (3 iterations)"
echo "========================================="
for split in "${SPLITS[@]}"; do
  echo "🤖 Running lite_llm_agent on ${split}_5000docs..."
  
  for iter in 0 1 2; do
    echo "  📝 Iteration $iter..."
    uv run python -m src.evaluation.scripts.test_preprocessing_split \
      --agent lite_llm_agent \
      --split ${split}_5000docs \
      --iteration $iter \
      --track-iterations
  done
  echo "✓ Completed ${split}_5000docs"
  echo ""
done

# Step 4: Generate visualizations
echo "========================================="
echo "STEP 4: Generating Visualizations"
echo "========================================="
uv run python -m src.evaluation.scripts.plot_iterations

# Step 5: View aggregate results
echo ""
echo "========================================="
echo "STEP 5: Aggregate Results"
echo "========================================="
uv run python -m src.evaluation.scripts.aggregate_results

echo ""
echo "✅ All evaluations completed!"