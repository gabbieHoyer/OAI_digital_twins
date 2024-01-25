#!/bin/bash
#SBATCH --mem=32G
#SBATCH --partition=dgx
#SBATCH --gres=gpu:teslav100:16
#SBATCH --ntasks=96
#SBATCH --time=22:00:00        # Request more than the theoretical minimum to account for inefficiencies
#SBATCH --output=elastic_net_10K-%j.out

echo "Starting job on ${SLURM_NTASKS:-1} cores."

# Activate your Python environment if necessary
source /netopt/rhel7/versions/python/Anaconda3-5.2.0/bin/activate /data/VirtualAging/users/ghoyer/conda/envs/pytorch_env
cd /data/VirtualAging/users/ghoyer/OAI/digital_trial/mega_bootstrap

# Run your Python script
# python main_script_tkr.py
python statistical_modeling.py
