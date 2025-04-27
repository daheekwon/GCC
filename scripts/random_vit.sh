#!/bin/bash

# Generate 50 random unique numbers between 0 and 49999
# shuf generates random permutation, -i specifies the range, -n selects the number of outputs
for i in $(shuf -i 0-49999 -n 50)
do
    export CUDA_VISIBLE_DEVICES=0
    echo "Running analysis for sample $i"
    python run_circuit_analysis.py --tgt_sample $i --pot_threshold 80
done