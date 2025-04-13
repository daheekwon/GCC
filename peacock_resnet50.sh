#!/bin/bash

# Loop from 4200 to 4249
for i in {4206..4249}
do
    echo "Running analysis for sample $i"
    python run_circuit_analysis.py --tgt_sample $i
done