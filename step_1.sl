#!/bin/bash
#SBATCH --job-name=stability_all_parallel
#SBATCH --account=def-sblain
#SBATCH --mem=90000      # increase as needed
#SBATCH --time=0-02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mail-user=q2h3s6p4k0e9o7a5@biaptlab.slack.com # adjust this to match your email address
#SBATCH --mail-type=FAIL

module load python/3.7.9

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index scikit-learn
pip install --no-index pandas
pip install --no-index matplotlib
python -u 1_functional_connectivity.py $ANALYSIS_PARAM
