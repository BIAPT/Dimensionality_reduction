#!/bin/bash
#SBATCH --job-name=stability_all_parallel
#SBATCH --account=def-sblain
#SBATCH --mem=1000      # increase as needed
#SBATCH --time=0-00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mail-user=q2h3s6p4k0e9o7a5@biaptlab.slack.com # adjust this to match your email address
#SBATCH --mail-type=ALL
#SBATCH --array=WSAS02, WSAS09, WSAS20, WSAS27, WSAS28, WSAS10, WSAS11, WSAS12, WSAS13, WSAS17, WSAS18, WSAS22, WSAS25, WSAS29, WSAS05, WSAS19


module load python/3.9.6
module load scipy-stack
source dimred_env/bin/activate

python -u 1_functional_connectivity.py \
       -input_dir ~/Dim_Red/Dimensionality_reduction/DATA \
	   	 -output_dir ~/Dim_Red/Dimensionality_reduction/RESULTS \
		   -participants ~/Dim_Red/Dimensionality_reduction/DATA/data_2states.txt \
		   -frequencyband alpha \
		   -id ${SLURM_ARRAY_TASK_ID}
