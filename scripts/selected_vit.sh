# !/bin/bash

# selected_samples=(873 2938 29270 12307 38414 11185 32554 42415 1719 6715 2621 46842 4200 4132 44792 6800 4225 33289 4261 46902 46352 48475 4223 17567 39693 17567 33793 19371 47165 24323 39065 45288 46352 19460 6366 37626 45362 543 3245 11614 19405 30966 38825 47358 4239)


# # Iterate through the selected samples array
# for i in "${selected_samples[@]}"
# do
#     echo "Running analysis for sample $i"
#     python run_circuit_analysis.py --tgt_sample $i --pot_threshold 99
# done


# Process every 50th image from 0 to 49999
# for ((i=4223; i<50000; i+=50))
# do
#     echo "Running analysis for sample $i"
#     python run_circuit_analysis.py --tgt_sample $i --pot_threshold 90 --gpu 0 --model vit
# done

DATASET_PATH="/project/data/external/ILSVRC/Data/CLS-LOC/val"
GPU_ID=3
export CUDA_VISIBLE_DEVICES=$GPU_ID
python -m get_highly_activated_samples \
    --output_dir "/project/PURE/results_circuit_v0227_POT_80/vit/imagenet" \
    --model_name vit \
    --dataset_name imagenet \
    --dataset_path $DATASET_PATH \
    --n_samples 500
