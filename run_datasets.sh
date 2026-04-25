#!/bin/bash

# Dataset specifications: name|docs|queries|mem|time|cpus
DATASET_SPECS=(
    "theorem_retrieval|23,839|69|32G|02:00:00|4"
    "stack_exchange|40,956|107|32G|02:00:00|4"
    "code_retrieval|232,444|3,665|64G|06:00:00|8"
    "paper_retrieval|363,133|72|64G|06:00:00|8"
    "set_operation_entity_retrieval|651,704|423|96G|12:00:00|12"
    "clinical_trial|914,628|113|96G|12:00:00|12"
    "tip_of_the_tongue|1,083,337|135|96G|12:00:00|12"
    "legal_qa|1,182,626|6,753|96G|12:00:00|12"
)

echo "=========================================="
echo "   LLM Index Generation - Dataset Runner"
echo "=========================================="
echo ""
echo "Available Models:"
echo "[1] haiku"
echo "[2] gpt4o"
echo ""
read -p "Select model (1 or 2): " model_selection

case $model_selection in
    1)
        MODEL="haiku"
        ;;
    2)
        MODEL="gpt4o"
        ;;
    *)
        echo "Invalid model selection. Exiting."
        exit 1
        ;;
esac

echo ""
echo "Selected model: $MODEL"
echo ""
echo "Available Datasets:"
echo ""
printf "%-5s %-35s %-12s %-10s %-8s %-10s %-6s\n" "NUM" "DATASET" "DOCS" "QUERIES" "MEMORY" "TIME" "CPUS"
echo "---------------------------------------------------------------------------------------------------"

i=1
for spec in "${DATASET_SPECS[@]}"; do
    IFS='|' read -r name docs queries mem time cpus <<< "$spec"
    printf "%-5s %-35s %-12s %-10s %-8s %-10s %-6s\n" "[$i]" "$name" "$docs" "$queries" "$mem" "$time" "$cpus"
    ((i++))
done

echo ""
echo "Special options:"
echo "[9] Run ALL datasets"
echo "[0] Exit"
echo ""
read -p "Select dataset(s) to run (e.g., 1 3 5 or 9 for all): " selection

if [[ "$selection" == "0" ]]; then
    echo "Exiting."
    exit 0
fi

# Create ablation_slurm directory
mkdir -p ablation_slurm

# Function to submit a job
submit_job() {
    local spec=$1
    local model=$2
    IFS='|' read -r name docs queries mem time cpus <<< "$spec"
    
    echo ""
    echo "Submitting: $name (model: $model)"
    echo "  Documents: $docs | Queries: $queries"
    echo "  Resources: ${mem} RAM, ${cpus} CPUs, ${time}"
    
    cat > ablation_slurm/run_${name}_${model}.slurm << SLURM_EOF
#!/bin/bash
#SBATCH --job-name=llm_${name}_${model}
#SBATCH --output=ablation_slurm/${name}_${model}_%j.out
#SBATCH --error=ablation_slurm/${name}_${model}_%j.err
#SBATCH --time=${time}
#SBATCH --mem=${mem}
#SBATCH --cpus-per-task=${cpus}
#SBATCH --partition=cpu

# Go to the directory where the job was submitted from
cd \$SLURM_SUBMIT_DIR

source \$HOME/.local/bin/env

# Download data if not exists
if [ ! -d "data/${name}" ]; then
    echo "Downloading dataset: ${name}"
    uv run python -m src.evaluation.scripts.get_data --split ${name}
fi

# Run the agent with specified model
echo "Starting agent for ${name} with model ${model} at \$(date)"
uv run python main.py --agent analysis_code_agent --loops 3 --condition agent_history --split ${name} --model ${model}

echo "Job completed at \$(date)"
SLURM_EOF

    sbatch ablation_slurm/run_${name}_${model}.slurm
    echo "  ✓ Submitted as ablation_slurm/run_${name}_${model}.slurm"
}

# Process selection
if [[ "$selection" == "9" ]]; then
    echo ""
    echo "Submitting ALL datasets with model: $MODEL..."
    for spec in "${DATASET_SPECS[@]}"; do
        submit_job "$spec" "$MODEL"
        sleep 0.5
    done
else
    for num in $selection; do
        if [[ $num -ge 1 && $num -le ${#DATASET_SPECS[@]} ]]; then
            idx=$((num-1))
            submit_job "${DATASET_SPECS[$idx]}" "$MODEL"
            sleep 0.5
        else
            echo "Invalid selection: $num"
        fi
    done
fi

echo ""
echo "=========================================="
echo "All jobs submitted with model: $MODEL"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  watch -n 5 'squeue -u \$USER'"
echo ""
echo "Check logs:"
echo "  ls -lth ablation_slurm/"
echo "  tail -f ablation_slurm/<dataset>_${MODEL}_*.out"
echo "=========================================="