import os
import argparse
import pickle
import subprocess
import json
from typing import List, Tuple, Set

def parse_args():
    parser = argparse.ArgumentParser(description='Run Circuit Analysis Pipeline')
    
    # Required arguments
    parser.add_argument('--gpu', type=str, default='2',
                       help='GPU device number')
    parser.add_argument('--dataset', type=str, default='imagenet',
                       help='Dataset name')
    parser.add_argument('--model', type=str, default='vit',
                       help='Model name')
    parser.add_argument('--tgt_sample', type=int, required=True,
                       help='Target sample index to analyze')
    parser.add_argument('--pot_threshold', type=float, default=95,
                       help='Percentile threshold for Peak Over Threshold method')
    parser.add_argument('--save_plots', type=bool, default=True,
                       help='Save plots of filtered channels')
    return parser.parse_args()

def run_find_active_channels(args) -> str:
    """Run find_active_channels.py and return path to output pickle file."""
    cmd = [
        'python', 'find_active_channels.py',
        '--gpu', args.gpu,
        '--dataset', args.dataset,
        '--model', args.model,
        '--tgt_sample', str(args.tgt_sample)
    ]
    
    print("Finding active channels...")
    subprocess.run(cmd, check=True)
    
    # Construct path to output pickle file
    samples_dir = f"/data3/dahee/Concepts/results/{args.model}/{args.dataset}"
    save_dir = os.path.join(samples_dir, f"{args.tgt_sample}")
    pickle_path = os.path.join(save_dir, f'active_channels_sample_{args.tgt_sample}.pkl')
    
    return pickle_path

def get_metadata_path(args) -> str:
    """Get path to the metadata file."""
    samples_dir = f"/data3/dahee/Concepts/results/{args.model}/{args.dataset}"
    save_dir = os.path.join(samples_dir, f"{args.tgt_sample}")
    os.makedirs(save_dir, exist_ok=True)
    return os.path.join(save_dir, 'metadata.json')

def load_searched_channels(metadata_path: str) -> Set[Tuple[str, int]]:
    """Load set of previously searched channels."""
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            # Convert list of lists from JSON back to set of tuples
            channel_list = json.load(f)
            return channel_list
    return {}

def run_main_analysis(args, layer_name: str, channel_idx: int):
    """Run main.py with specified layer and channel."""
    cmd = [
        'python', 'main.py',
        '--gpu', args.gpu,
        '--dataset', args.dataset,
        '--model', args.model,
        '--src_layer', layer_name,
        '--src_channel', str(channel_idx),
        '--tgt_sample', str(args.tgt_sample),
        '--pot_threshold', str(args.pot_threshold),
        '--save_plots', str(args.save_plots)
    ]
    
    print(f"\nAnalyzing layer {layer_name}, channel {channel_idx}...")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running analysis for layer {layer_name}, channel {channel_idx}")
        print(f"Exit code: {e.returncode}")
        print(f"Error output:\n{e.stderr}")
        print("\nSkipping to next channel...")

def filter_valid_channels(active_channels: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
    """Filter out last encoder layer."""
    return [(layer, channel) for layer, channel in active_channels 
            if not layer.endswith('_11')]  # ViT-B/16 has 12 layers (0-11)

def main():
    args = parse_args()
    
    # First, run find_active_channels.py
    try:
        pickle_path = run_find_active_channels(args)
    except subprocess.CalledProcessError as e:
        print("Error running find_active_channels.py")
        print(f"Exit code: {e.returncode}")
        print(f"Error output:\n{e.stderr}")
        return
    
    # Load the results
    with open(pickle_path, 'rb') as f:
        active_channels = pickle.load(f)
    
    # Filter out layer4_block2 channels
    valid_channels = filter_valid_channels(active_channels)
    
    metadata_path = get_metadata_path(args)
    print(f"\nFound {len(active_channels)} active channels")
    print(f"After filtering last two layers: {len(valid_channels)} channels remain")

    # Run main.py for each valid channel that hasn't been searched
    skipped = 0
    for layer_name, channel_idx in valid_channels:
        # Load searched channels once at the start
        searched_channels = load_searched_channels(metadata_path)

        if len(searched_channels) > 0 and channel_idx in searched_channels[layer_name]['searched_channels']:
            print(f"Skipping {layer_name} channel {channel_idx} - already searched")
            skipped += 1
            continue
            
        success = run_main_analysis(args, layer_name, channel_idx)
    
    # Final count of skipped and failed channels
    print(f"\nCircuit analysis pipeline completed!")
    print(f"Skipped {skipped} among {len(valid_channels)} channels.")

if __name__ == '__main__':
    main() 