import os
import argparse
import pickle
import subprocess
import json
from typing import List, Tuple, Set
from tqdm.auto import tqdm

from model_configs import get_model_config
from find_active_channels import load_activation_samples, find_active_channels
from GCCs import CircuitAnalyzer


def parse_args():
    parser = argparse.ArgumentParser(description='Run Circuit Analysis Pipeline')
    
    # Required arguments
    parser.add_argument('--gpu', type=str, default='6',
                       help='GPU device number')
    parser.add_argument('--dataset', type=str, default='imagenet',
                       help='Dataset name')
    parser.add_argument('--model', type=str, default='resnet50',
                       help='Model name')
    parser.add_argument('--tgt_sample', type=int, default=4202,
                       help='Target sample index to analyze')
    parser.add_argument('--pot_threshold', type=float, default=80,
                       help='Percentile threshold for Peak Over Threshold method')
    parser.add_argument('--save_plots', type=bool, default=True,
                       help='Save plots')
    parser.add_argument('--save_dir', type=str, default='/GCC/results',
                       help='Save directory')
    parser.add_argument('--dataset_path', type=str, default='/data/ImageNet1k/val',
                       help='Dataset path')
    parser.add_argument('--corrupted_data', type=str, default='/GCC/corrupted_imagenet/corrupted_val_dataset.pt',
                       help='Corrupted data path')
    parser.add_argument('--patching_type', type=str, default='zero',
                       help='Patching type')
    
    return parser.parse_args()



def filter_valid_channels(active_channels: List[Tuple[str, int]], model_type: str) -> List[Tuple[str, int]]:
    """Filter channels based on model architecture."""
    if model_type == 'resnet':
        return [(layer, channel) for layer, channel in active_channels 
                if layer != 'layer4_block2']
                
    elif model_type == 'vit' or model_type == 'swin_t' or model_type == 'clip_vit':
        return [(layer, channel) for layer, channel in active_channels 
                if not layer.endswith('_11')]
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def main():
    args = parse_args()
    model_config = get_model_config(args.model)
    
    # Run pipeline with model-specific handling, run_find_active_channels
    save_dir = os.path.join(args.save_dir, args.model, args.dataset)
    
    highly_activated_samples = load_activation_samples(save_dir)
    active_channels = find_active_channels(highly_activated_samples, args.tgt_sample)

    def layer_sort_key(x):
        # Extract the layer number from the name (e.g., 'encoder_layer_2' -> 2)
        if args.model == 'vit':
            layer_num = int(x[0].split('_')[2])
        elif args.model == 'swin_t':
            layer_num = int(x[0].split('_')[3])
        elif args.model == 'clip_vit':
            layer_num = int(x[0].split('_')[2])
        return layer_num

    active_channels.sort(key=lambda x: x[0])

    if args.model == 'resnet50':
        active_channels.sort(key=lambda x: x[0])
    elif args.model == 'vit':
        active_channels.sort(key=layer_sort_key)
    elif args.model == 'swin_t':
        active_channels.sort(key=layer_sort_key)
    elif args.model == 'clip_vit':
        active_channels.sort(key=layer_sort_key)

    write_path = os.path.join(save_dir, f'pot_{int(args.pot_threshold)}', f'{args.tgt_sample}')
    save_path = os.path.join(write_path, f'active_channels_sample_{args.tgt_sample}.pkl')
    os.makedirs(write_path, exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(active_channels, f)
    
    # Filter channels based on model architecture
    valid_channels = filter_valid_channels(active_channels, model_config['model_type'])


    print(f"\nFound {len(active_channels)} active channels")
    print(f"After filtering the last layer: {len(valid_channels)} channels remain")
    print("================================================")
    print(f"Writing results to {write_path}")
    print("================================================")
    print()

    # Search channels that have not been searched yet
    def run_main_analysis(current_src_layer, current_src_channel, analyzer, layer_keys, visited, computed_results):
        cache_key = (current_src_layer, current_src_channel)

        # Skip re-searching already explored nodes
        if cache_key in visited:
            print(f"Skipping {current_src_layer} channel {current_src_channel} - already searched")
            return computed_results.get(cache_key, [])

        visited.add(cache_key)

        # Error handling
        if current_src_layer not in layer_keys:
            raise ValueError(f"Layer {current_src_layer} is not found in layer_keys")

        layer_idx = layer_keys.index(current_src_layer)

        # Base case: terminate when reaching the last layer
        if layer_idx + 1 >= len(layer_keys):
            computed_results[cache_key] = []
            print(f"Reached last layer {current_src_layer} channel {current_src_channel}")
            return []

        # Setup for analysis: set target layer and channel for analysis
        next_layer = layer_keys[layer_idx + 1]
        analyzer.src_layer_block = current_src_layer
        analyzer.src_channel = current_src_channel
        analyzer.tgt_layer_block = next_layer

        # Execute score-based filtering
        filtered_channels, filtered_scores, filtered_info_scores  = analyzer.analyze_channel_impacts()

        # Terminate when no connected channels found
        if len(filtered_channels) == 0:
            # print(f"No channels found for {current_src_layer} channel {current_src_channel}")
            computed_results[cache_key] = []
            return []

        # Save current node results
        results = [
            (current_src_layer, current_src_channel, next_layer, filtered_channels, filtered_scores, filtered_info_scores)
        ]

        # If not at the stopping layer, recursively explore child channels
        if next_layer not in {'layer4_block2', 'layers_11', 'block_11'}:
            for ch in filtered_channels:
                child_results = run_main_analysis(next_layer, ch, analyzer, layer_keys, visited, computed_results)
                results.extend(child_results)

        computed_results[cache_key] = results
        return results

    def save_circuit(circuit, root_layer_name, root_channel_idx, target_sample, save_dir):

        save_dir = os.path.join(save_dir, f'pot_{int(args.pot_threshold)}', f'{args.tgt_sample}')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(
            save_dir,
            f'connection_matrices_from_{root_layer_name}_ch{root_channel_idx}.pkl'
        )

        save_data = {
            'circuit': circuit,
            'src_layer_block': root_layer_name,
            'src_channel': root_channel_idx,
            'tgt_sample': target_sample
        }

        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)
        print(f"Saved {len(circuit)} circuits.")# {save_path}")
        
    # Add dummy argument for CircuitAnalyzer
    args.src_layer_block = None
    args.src_channel = None
    analyzer = CircuitAnalyzer(args)

    if args.model == 'resnet50':
        layer_keys = sorted(analyzer.avg_activated_samples.keys())
    elif args.model == 'vit':
        layer_keys = sorted(analyzer.avg_activated_samples.keys(), key=lambda x: int(x.split('_')[2]))
    elif args.model == 'swin_t':
        layer_keys = sorted(analyzer.avg_activated_samples.keys(), key=lambda x: int(x.split('_')[3]))
    elif args.model == 'clip_vit':
        layer_keys = sorted(analyzer.avg_activated_samples.keys(), key=lambda x: int(x.split('_')[2]))
    
    visited = set()
    computed_results = {}

    # Start exploration from root node
    for root_layer_name, root_channel_idx in tqdm(valid_channels, desc="Analyzing channels", total=len(valid_channels)):
        root_key = (root_layer_name, root_channel_idx)
        if root_key in visited:
            print(f"Root node {root_layer_name} channel {root_channel_idx} already searched — skipping.")
            continue
        else:
            print(f"Searching from {root_layer_name} ({root_channel_idx} Channel)")

        circuit = run_main_analysis(
            root_layer_name, root_channel_idx,
            analyzer, layer_keys,
            visited, computed_results
        )

        # Save only when exploration results exist
        if len(circuit) > 0:
            save_circuit(circuit, root_layer_name, root_channel_idx, args.tgt_sample, save_dir)
               # Save visited set as JSON
            visited_list = [[layer, int(channel)] for layer, channel in visited]  # Convert int64 to native int
            visited_save_path = os.path.join(save_dir, f'pot_{int(args.pot_threshold)}', f'{args.tgt_sample}', 'visited_channels.json')
            with open(visited_save_path, 'w') as f:
                json.dump(visited_list, f)


    # Final count of skipped channels
    print(f"\nCircuit analysis pipeline completed!")

if __name__ == '__main__':
    main() 