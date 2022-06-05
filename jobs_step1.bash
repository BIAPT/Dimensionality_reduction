#!/bin/bash

# We assume running this from the script directory
P_IDS=("WSAS02" "WSAS09" "WSAS20" "WSAS27" "WSAS28" "WSAS10" "WSAS11" "WSAS12" "WSAS13" "WSAS17" "WSAS18" "WSAS22" "WSAS25" "WSAS29" "WSAS05" "WSAS19")
FREQUENCIES=("delta" "theta" "alpha" "beta" "gamma"  "fullband")

for pid in ${P_IDS[@]}; do
    for frequency in ${FREQUENCIES[@]}; do
	analysis_param="
		-input_dir ~/Documents/GitHub/Dimensionality_reduction/DATA 
		-output_dir ~/Documents/GitHub/Dimensionality_reduction/RESULTS 
		-participants ~/Documents/GitHub/Dimensionality_reduction/DATA/data_2states.txt
		-frequencyband ${frequency}
		-id ${pid}
		"
	echo "${analysis_param}"
	sbatch --export=ANALYSIS_PARAM=$analysis_param $1	
    done
done