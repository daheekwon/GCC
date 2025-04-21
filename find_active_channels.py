import os
import argparse
import pickle
from typing import Dict, List, Tuple


def load_activation_samples(samples_dir: str) -> Dict:
    """Load pre-computed highly activated samples."""
    high_file = os.path.join(samples_dir, 'highly_activated_samples_top500_mlp.pkl')
    
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