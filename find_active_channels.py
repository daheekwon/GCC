import os
import argparse
import pickle
from typing import Dict, List, Tuple

def parse_args():
    parser = argparse.ArgumentParser(description='Find Active Channels for Target Sample')
    
    # Required arguments
    parser.add_argument('--gpu', type=str, default='2',
                       help='GPU device number')
    
    # Analysis parameters
    parser.add_argument('--dataset', type=str, default='imagenet',
                        help='Dataset name')
    parser.add_argument('--model', type=str, default='resnet50',
                       help='Model name')
    parser.add_argument('--tgt_sample', type=int, required=True,
                       help='Target sample index to analyze')
    parser.add_argument('--pot_threshold', type=float, default=95,
                    help='Percentile threshold for Peak Over Threshold method (default: 80)')
    return parser.parse_args()

def load_activation_samples(samples_dir: str) -> Dict:
    """Load pre-computed highly activated samples."""
    high_file = os.path.join(samples_dir, 'highly_activated_samples_top500.pkl')
    
    with open(high_file, 'rb') as f:
        highly_activated_samples = pickle.load(f)
        
    return highly_activated_samples

def find_active_channels(highly_activated_samples: Dict, target_sample: int) -> List[Tuple[str, int]]:
    """
    Find all (layer, channel) pairs where target_sample appears in highly activated samples.
    
    Args:
        highly_activated_samples: Dictionary of activation samples
        target_sample: Sample index to search for
        
    Returns:
        List of (layer_name, channel_idx) tuples where target sample appears
    """
    active_channels = []
    
    for layer_name, layer_data in highly_activated_samples.items():
        for channel_idx, samples in layer_data.items():
            if target_sample in samples:
                active_channels.append((layer_name, channel_idx))
    
    return active_channels

def main():
    args = parse_args()
    
    # Setup paths
    samples_dir = f"/data8/dahee/circuit/results/{args.model}/{args.dataset}"
    save_dir = os.path.join(samples_dir, f"pot_{int(args.pot_threshold)}/{args.tgt_sample}")
    os.makedirs(save_dir, exist_ok=True)
    
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    # Load activation samples
    highly_activated_samples = load_activation_samples(samples_dir)
    
    # Find active channels
    active_channels = find_active_channels(highly_activated_samples, args.tgt_sample)
    
    # Sort by layer name with numeric order
    def layer_sort_key(x):
        # Extract the layer number from the name (e.g., 'encoder_layer_2' -> 2)
        layer_num = int(x[0].split('_')[-1])
        return layer_num
    
    # Sort using the custom key function
    active_channels.sort(key=layer_sort_key)
    
    # Save results
    save_path = os.path.join(save_dir, f'active_channels_sample_{args.tgt_sample}.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(active_channels, f)

if __name__ == '__main__':
    main() 