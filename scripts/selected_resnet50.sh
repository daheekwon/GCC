selected_samples=(4223 7674 39065)


# Iterate through the selected samples array
for i in "${selected_samples[@]}"
do
    echo "Running analysis for sample $i"
    python run_circuit_analysis.py --tgt_sample $i --pot_threshold 90 --gpu 0
done