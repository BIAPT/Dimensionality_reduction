#!/bin/bash
#SBATCH --job-name=stability_all_parallel
#SBATCH --account=def-sblain
#SBATCH --mem=1000      # increase as needed
#SBATCH --time=0-02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mail-user=q2h3s6p4k0e9o7a5@biaptlab.slack.com # adjust this to match your email address
#SBATCH --mail-type=FAIL

module load python/3.9.6
module load scipy-stack
source dimred_env/bin/activate

python -u 1_functional_connectivity.py \
       -input_dir ~/DATA \
	   	 -output_dir ~/RESULTS \
		   -participants ~/DATA/data_2states.txt \
		   -frequencyband alpha \
		   -id WSAS02
