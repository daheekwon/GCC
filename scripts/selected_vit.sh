#!/bin/bash

# Generate 50 random unique numbers between 0 and 49999
# shuf generates random permutation, -i specifies the range, -n selects the number of outputs

# selected_samples=(873 2938 29270 12307 38414 11185 32554 42415 1719 6715 2621 46842 4200 4132 44792 6800 4225 33289 4261 46902 46352 48475 4223 17567 39693 17567 33793 19371 47165 24323 39065 45288 46352 19460 6366 37626 45362 543 3245 11614 19405 30966 38825 47358 4239 )

selected_samples=(873 2938 29270 12307 38414 11185 4261 46902 46352 6366 37626 45362 543 3245 11614 19405 3096 6800 4225 33289 4261 46902 48475 19460)

# Iterate through the selected samples array
for i in "${selected_samples[@]}"
do
    export CUDA_VISIBLE_DEVICES=0
    echo "Running analysis for sample $i"
    python run_circuit_analysis.py --tgt_sample $i  --pot_threshold 85
done