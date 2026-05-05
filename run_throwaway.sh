#!/bin/bash
#SBATCH --job-name=throwaway_merge_test
#SBATCH --output=run_splits_logs/throwaway_%j.out
#SBATCH --error=run_splits_logs/throwaway_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=150G
#SBATCH --cpus-per-task=12
#SBATCH --partition=cpu

cd /work/pi_dagarwal_umass_edu/project_13/sam/llm_index_generation
mkdir -p run_splits_logs

uv run python run_splits_throwaway.py
