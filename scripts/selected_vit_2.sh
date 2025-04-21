# Process every 50th image from 0 to 49999
for ((i=4000; i<6000; i+=50))
do
    echo "Running analysis for sample $i"
    python run_circuit_analysis.py --tgt_sample $i --pot_threshold 90 --gpu 3 --model vit
done
