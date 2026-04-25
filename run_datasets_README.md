# Run the script
```
./run_datasets.sh
```

# Select llm model

Select llm model to run (e.g., 1, 2):

# Select dataset

Select dataset(s) to run (e.g., 1 3 5 or 9 for all):

# Monitor Jobs

```
# Check all running jobs
squeue -u $USER

# Watch in real-time
watch -n 5 'squeue -u $USER'

# Check specific dataset log
tail -f logs/theorem_retrieval_*.out
tail -f logs/tip_of_the_tongue_*.out

# Check all logs
ls -lth logs/
```

# Full workflow

```
cd /scratch/llm_index_generation

# 1. Run the interactive selector
./run_datasets.sh

# 2. When prompted, choose:
#    - Type "1" for smallest dataset (quick test)
#    - Type "1 2" for two small datasets
#    - Type "9" for all datasets

# 3. Monitor progress
squeue -u $USER

# 4. Check status anytime
./check_status.sh

# 5. View results when complete
uv run python -m src.evaluation.scripts.aggregate_results
```

# Results
Saved into /ablation_slurm