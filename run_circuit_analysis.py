import os
import argparse
import pickle
import subprocess
import json
from typing import List, Tuple, Set
from tqdm.auto import tqdm

from model_configs import get_model_config
from find_active_channels import load_activation_samples, find_active_channels
from main import CircuitAnalyzer

def parse_args():
    parser = argparse.ArgumentParser(description='Run Circuit Analysis Pipeline')
    
    # Required arguments
    parser.add_argument('--gpu', type=str, default='0',
                       help='GPU device number')
    parser.add_argument('--dataset', type=str, default='imagenet',
                       help='Dataset name')
    parser.add_argument('--model', type=str, default='resnet50',
                       help='Model name')
    parser.add_argument('--tgt_sample', type=int, required=True,
                       help='Target sample index to analyze')
    parser.add_argument('--pot_threshold', type=float, default=95,
                       help='Percentile threshold for Peak Over Threshold method')
    parser.add_argument('--save_plots', type=bool, default=True,
                       help='Save plots')
    parser.add_argument('--dataset_path', type=str, default='/data/ImageNet1k/val',
                       help='Dataset path')
    
    return parser.parse_args()

def parse_args():
    parser = argparse.ArgumentParser(description='Run Circuit Analysis Pipeline')
    
    # Required arguments
    parser.add_argument('--gpu', type=str, default='1',
                       help='GPU device number')
    parser.add_argument('--dataset', type=str, default='imagenet',
                       help='Dataset name')
    parser.add_argument('--model', type=str, default='resnet50',
                       help='Model name')
    parser.add_argument('--tgt_sample', type=int, default=4202,
                       help='Target sample index to analyze')
    parser.add_argument('--pot_threshold', type=float, default=95,
                       help='Percentile threshold for Peak Over Threshold method')
    parser.add_argument('--save_plots', type=bool, default=True,
                       help='Save plots')
    parser.add_argument('--save_dir', type=str, default='/project/PURE/results_circuit_v0227_POT_80',
                       help='Save directory')
    parser.add_argument('--dataset_path', type=str, default='/project/data/external/ILSVRC/Data/CLS-LOC/val',
                       help='Dataset path')
    
    return parser.parse_args()

def run_find_active_channels(args) -> str:
    """Run find_active_channels.py and return path to output pickle file."""
    cmd = [
        'python', 'find_active_channels.py',
        '--gpu', args.gpu,
        '--dataset', args.dataset,
        '--model', args.model,
        '--tgt_sample', str(args.tgt_sample),
        '--pot_threshold', str(args.pot_threshold)
    ]
    
    print("Finding active channels...")
    subprocess.run(cmd, check=True)
    
    # Construct path to output pickle file
    samples_dir = f"/data8/dahee/circuit/results/{args.model}/{args.dataset}"
    save_dir = os.path.join(samples_dir, f"pot_{int(args.pot_threshold)}/{args.tgt_sample}")
    pickle_path = os.path.join(save_dir, f'active_channels_sample_{args.tgt_sample}.pkl')
    
    return pickle_path

def get_metadata_path(args) -> str:
    """Get path to the metadata file."""
    samples_dir = f"/data8/dahee/circuit/results/{args.model}/{args.dataset}/pot_{int(args.pot_threshold)}"
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
        '--src_layer_block', layer_name,
        '--src_channel', str(channel_idx),
        '--tgt_sample', str(args.tgt_sample),
        '--pot_threshold', str(args.pot_threshold),
        '--save_plots', str(args.save_plots),
        '--save_dir', args.save_dir,
        '--dataset_path', args.dataset_path,
    ]
    
    print(f"\nAnalyzing layer {layer_name}, channel {channel_idx}...")
    subprocess.run(cmd, check=True)

def filter_valid_channels(active_channels: List[Tuple[str, int]], model_type: str) -> List[Tuple[str, int]]:
    """Filter channels based on model architecture."""
    if model_type == 'resnet':
        return [(layer, channel) for layer, channel in active_channels 
                if layer != 'layer4_block2']
    elif model_type == 'vit':

        return [(layer, channel) for layer, channel in active_channels 
                if not layer.endswith('_11')]
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def main():
    args = parse_args()
    model_config = get_model_config(args.model)
    
    # Run pipeline with model-specific handling, run_find_active_channels
    samples_dir = os.path.join(args.save_dir, args.model, args.dataset)
    save_dir = os.path.join(samples_dir, f"{args.tgt_sample}")
    os.makedirs(save_dir, exist_ok=True)
    
    highly_activated_samples = load_activation_samples(samples_dir)
    active_channels = find_active_channels(highly_activated_samples, args.tgt_sample)
    active_channels.sort(key=lambda x: x[0])
    
    save_path = os.path.join(save_dir, f'active_channels_sample_{args.tgt_sample}.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(active_channels, f)
    
    # Filter channels based on model architecture
    valid_channels = filter_valid_channels(active_channels, model_config['model_type'])
    
    metadata_path = get_metadata_path(args)
    print(f"\nFound {len(active_channels)} active channels")
    print(f"After filtering the last layer: {len(valid_channels)} channels remain")

    # Search channels that have not been searched yet
    def run_main_analysis(current_src_layer, current_src_channel, analyzer, layer_keys, visited, computed_results):
        cache_key = (current_src_layer, current_src_channel)

        # 이미 탐색한 노드는 재탐색 생략
        if cache_key in visited:
            print(f"Skipping {current_src_layer} channel {current_src_channel} - already searched")
            return computed_results.get(cache_key, [])

        visited.add(cache_key)

        # Error handling
        if current_src_layer not in layer_keys:
            raise ValueError(f"Layer {current_src_layer} not found in layer_keys")

        layer_idx = layer_keys.index(current_src_layer)

        # Base case: 마지막 레이어 도달 시 종료
        if layer_idx + 1 >= len(layer_keys):
            computed_results[cache_key] = []
            print(f"Reached last layer {current_src_layer} channel {current_src_channel}")
            return []

        # Setup for analysis: 분석 대상 레이어 및 채널 설정
        next_layer = layer_keys[layer_idx + 1]
        analyzer.src_layer_block = current_src_layer
        analyzer.src_channel = current_src_channel
        analyzer.tgt_layer_block = next_layer

        # 스코어 기반 필터링 실행
        filtered_channels, filtered_scores, filtered_info_scores, return_level, shape, scale = analyzer.analyze_channel_impacts()

        # 연결된 채널이 없는 경우 종료
        if len(filtered_channels) == 0:
            print(f"No channels found for {current_src_layer} channel {current_src_channel}")
            computed_results[cache_key] = []
            return []

        # 현재 노드의 결과 저장
        results = [
            (current_src_layer, current_src_channel, next_layer, filtered_channels, filtered_scores, filtered_info_scores, return_level, shape, scale)
        ]

        # If not at the stopping layer, recursively explore child channels
        if next_layer not in {'layer4_block2', 'layers_11'}:
            for ch in filtered_channels:
                child_results = run_main_analysis(next_layer, ch, analyzer, layer_keys, visited, computed_results)
                results.extend(child_results)

        computed_results[cache_key] = results
        return results


    def save_circuit(circuit, root_layer_name, root_channel_idx, target_sample, save_dir):
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
        print(f"Saved connection matrices to {save_path}")
        
    # Add dummy argument for CircuitAnalyzer
    args.src_layer_block = None
    args.src_channel = None
    analyzer = CircuitAnalyzer(args)
    layer_keys = sorted(analyzer.avg_activated_samples.keys())
    
    visited = set()
    computed_results = {}

    # 루트 노드 기준으로 탐색 시작
    for root_layer_name, root_channel_idx in tqdm(valid_channels, desc="Analyzing channels", total=len(valid_channels)):
        root_key = (root_layer_name, root_channel_idx)

        if root_key in visited:
            print(f"Root node {root_layer_name} channel {root_channel_idx} already searched — skipping.")
            continue

        circuit = run_main_analysis(
            root_layer_name, root_channel_idx,
            analyzer, layer_keys,
            visited, computed_results
        )

        # 탐색 결과 있을 때만 저장
        if len(circuit) > 0:
            # save_circuit(circuit, root_layer_name, root_channel_idx, args.tgt_sample, save_dir)
            save_circuit(circuit, root_layer_name, root_channel_idx, args.tgt_sample, '/project/PURE/results_circuit_v0227_POT_80/resnet50/imagenet/pot_95_v2')
    
    # Final count of skipped channels
    print(f"\nCircuit analysis pipeline completed!")

if __name__ == '__main__':
    main() 