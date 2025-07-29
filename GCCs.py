import os
import sys
import argparse
import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle
from activation_analysis import *
import numpy as np
import torchvision.models as models
from torchvision.models import resnet50, ResNet50_Weights, ViT_B_16_Weights, ViT_L_16_Weights, ViT_H_14_Weights
from scipy import stats
from scipy import sparse  # Add this import at the top with other imports
import json  # Add this import at the top
from model_configs import get_model_config, create_model

def parse_args():
    parser = argparse.ArgumentParser(description='Neural Network Channel Analysis')
    
    # Required arguments
    parser.add_argument('--gpu', type=str, default='2',
                       help='GPU device number')
    
    # Analysis parameters
    parser.add_argument('--dataset', type=str, default='imagenet',
                        help='Dataset name')
    parser.add_argument('--model', type=str, default='resnet50',
                       help='Model name')
    parser.add_argument('--src_layer_block', type=str, default='layer1_block0',
                       help='Start layer block to analyze (e.g., layer1_block0)')
    parser.add_argument('--src_channel', type=int, default=0,
                       help='Start Channel number to analyze')
    parser.add_argument('--num_images', type=int, default=10,
                       help='Number of images to display')
    parser.add_argument('--tgt_sample', type=int, required=True,
                       help='Specific sample index to analyze')
    parser.add_argument('--dataset_path', type=str, default='/data/ImageNet1k/val',
                       help='Dataset path')
    
    # Output options
    parser.add_argument('--save_plots', type=bool, default=True,
                       help='Save plots instead of displaying')
    parser.add_argument('--save_dir', type=str, default='/GCC/results/',
                       help='Save directory')
    
    # Add POT threshold parameter
    parser.add_argument('--pot_threshold', type=float, default=95,
                       help='Percentile threshold for Peak Over Threshold method (default: 80)')
    
    return parser.parse_args()

class CircuitAnalyzer:
    def __init__(self, args):
        # Store individual args instead of the whole args object
        self.model_name = args.model
        self.dataset = args.dataset
        self.src_layer_block = args.src_layer_block
        self.src_channel = args.src_channel
        self.tgt_sample = args.tgt_sample
        self.save_plots = args.save_plots
        self.pot_threshold = args.pot_threshold
        
        # Get model configuration
        self.model_config = get_model_config(self.model_name)
        self.model_type = self.model_config['model_type']
        
        # Setup paths
        self.samples_dir = os.path.join(args.save_dir, self.model_name, self.dataset)
        if self.dataset == "imagenet":
            self.val_dir = args.dataset_path
        self.save_dir = os.path.join(self.samples_dir, f"pot_{int(self.pot_threshold)}/{self.tgt_sample}")
        
        # Setup environment
        setup_environment(args, self.save_dir)
        
        # Load data and model
        self.val_dataset, self.val_loader = load_data(self.val_dir)
        self.model = create_model(self.model_name).cuda()
        self.model.eval()
        
        self.patching_type = args.patching_type
        if self.patching_type == 'corrupted':
            self.corrupted_dataset = torch.load(args.corrupted_data)
            print(f"Corrupted dataset loaded from {args.corrupted_data}: {self.corrupted_dataset.shape}")
        else:
            self.corrupted_dataset = None

        # Load activation samples
        self.avg_activated_samples, self.highly_activated_samples = load_activation_samples(self.samples_dir)
        
    def initialize_connection_matrices(self):
        """Initialize sparse matrices for connections between consecutive layers."""
        connection_matrices = []
        layer_keys = list(self.avg_activated_samples.keys())
        
        # Get dimensions directly from activations
        def get_layer_dim(layer_name):
            if self.model_type == 'resnet':
                # Parse layer name (e.g., 'layer1_block0')
                layer_num = int(layer_name[5])  # gets '1' from 'layer1'
                block_num = int(layer_name.split('block')[1])
                
                # Get the correct layer and block
                layer = self.model._modules[f'layer{layer_num}']
                block = layer[block_num]
                
                # Get the actual output dimension from the block
                return block.conv3.out_channels if hasattr(block, 'conv3') else block.conv2.out_channels
            elif self.model_type == 'vit':  # ViT
                # Parse layer name (e.g., 'encoder_layer_0')
                layer_num = int(layer_name.split('_')[-1])
                
                # Get the correct encoder layer
                layer = self.model.encoder.layers[layer_num]
                
                # Return hidden dimension (768 for ViT-B/16)
                return layer.mlp[0].out_features
        
        # Find start index based on src_layer_block
        start_idx = layer_keys.index(self.src_layer_block)
        
        # Only create matrices from the start layer onwards
        for i in range(start_idx, len(layer_keys) - 1):
            src_layer = layer_keys[i]
            tgt_layer = layer_keys[i + 1]
            
            # Get dimensions from model structure
            src_channels = get_layer_dim(src_layer)
            tgt_channels = get_layer_dim(tgt_layer)
            
            # Create sparse matrix for connections between current and next layer
            connection_matrix = sparse.csr_matrix((src_channels, tgt_channels))
            connection_matrices.append(connection_matrix)
        
        return connection_matrices

    def filter_channels(self, scores):
        """Filter channels based on scores and activation overlap ratios using efficient POT method."""
        # Normalize scores and convert to numpy once
        normalized_scores = scores.cpu().numpy()
        if sum(normalized_scores) == 0:
            print(f"Warning: No meaningful scores for {self.tgt_layer_block}")
            import pdb; pdb.set_trace()
            
            return np.array([]), np.array([]), np.array([])
        normalized_scores /= normalized_scores.sum()
        
        # Create linked channels mask instead of list
        linked_mask = np.zeros(len(self.avg_activated_samples[self.tgt_layer_block]), dtype=bool)
        for i in range(len(linked_mask)):
            if self.tgt_sample in self.avg_activated_samples[self.tgt_layer_block][i]:
                linked_mask[i] = True
        
        # Filter non-zero linked channels more efficiently
        valid_mask = (~np.isnan(normalized_scores)) & (normalized_scores != 0) & linked_mask
        scores_ls = normalized_scores[valid_mask]
        next_channel_pool = np.where(valid_mask)[0]
        
        if len(scores_ls) == 0:
            return np.array([]), np.array([]), np.array([])

        # POT method
        threshold = np.percentile(scores_ls, self.pot_threshold)
        exceedances_mask = scores_ls > threshold
        exceedances = scores_ls[exceedances_mask] - threshold
        if len(exceedances) > 0:
            # Fit GPD only to exceedances
            shape, _, scale = stats.genpareto.fit(exceedances,floc=0)
            
            shape = np.max([shape, 10e-10])
            scale = np.min([scale, np.std(exceedances)])

            # Calculate return level
            p = 0.95
            N = len(scores_ls)
            Nx = len(exceedances)
            return_level = threshold + scale/shape * ((N/Nx * (1-p))**(-shape) - 1)

            if return_level < 0:
                return (np.array([]), np.array([]), np.array([]))
            else:
                # Apply return level threshold
                outlier_mask = scores_ls > return_level
        else:
            return (np.array([]), np.array([]), np.array([]))
        
        filtered_channels = next_channel_pool[outlier_mask]
        filtered_scores = scores_ls[outlier_mask]
        
        if len(filtered_channels) == 0:
            return (np.array([]), np.array([]), np.array([]))
        
        # Compute activation overlaps more efficiently
        src_activations = set(self.highly_activated_samples[self.src_layer_block][self.src_channel])
        
        # Vectorize ratio calculations
        ratios = []
        for tgt_ch in filtered_channels:

            tgt_activations = set(self.highly_activated_samples[self.tgt_layer_block][tgt_ch])
            ratio = len(tgt_activations & src_activations) / len(tgt_activations)
            ratios.append(ratio)
        
        # Calculate denominator ratios only once for all channels
        denom_ratios = []
        for tgt_ch in next_channel_pool:
            tgt_activations = set(self.highly_activated_samples[self.tgt_layer_block][tgt_ch])
            ratio = len(tgt_activations & src_activations) / len(tgt_activations)
            denom_ratios.append(ratio)
        
        # print(f"denom_ratios: {np.mean(denom_ratios), np.median(denom_ratios)}")
        ratio_threshold = max(np.mean(denom_ratios),0.1)  # Take maximum of mean and 0.1
        ratio_mask = np.array(ratios) > ratio_threshold
        return (filtered_channels[ratio_mask], 
                filtered_scores[ratio_mask],
                np.array(ratios)[ratio_mask])

    def analyze_channel_impacts(self):
        """Analyze channel impacts using scoring mechanism."""
        if self.model_type == 'resnet':
            return self._analyze_resnet_channel_impacts()
        elif self.model_type == 'vit':
            return self._analyze_vit_channel_impacts()
        elif self.model_type == 'swin_t':
            return self._analyze_swin_t_channel_impacts()
        elif self.model_type == 'clip_vit':
            return self._analyze_clip_vit_channel_impacts()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def _analyze_resnet_channel_impacts(self):
        # Get channel indices for the target sample
        channel_indices = get_channel_indices(
            self.highly_activated_samples,
            self.src_layer_block,
            self.tgt_sample 
        )
        
        if len(channel_indices) == 0:
            print(f"Warning: No channels found for target sample {self.tgt_sample} in {self.src_layer_block}")
            return [], [], [], [], [], []
        
        # Calculate impact scores
        scores = analyze_channel_score(
            model=self.model,
            src_layer_block=self.src_layer_block,
            tgt_layer_block=self.tgt_layer_block,
            val_dataset=self.val_dataset,
            channel_idx=channel_indices,
            src_channel=self.src_channel,
            model_type=self.model_type,
            patching_type=self.patching_type,
            corrupted_dataset=self.corrupted_dataset
        )
        
        # Filter channels and get significant ones
        filtered_channels, filtered_scores, filtered_info_scores = self.filter_channels(scores)
        
        return filtered_channels, filtered_scores, filtered_info_scores

    def _analyze_vit_channel_impacts(self):
        """Analyze channel impacts for ViT models."""
        # Get channel indices for the target sample
        channel_indices = get_channel_indices(
            self.highly_activated_samples,
            self.src_layer_block,
            self.tgt_sample 
        )
        
        if len(channel_indices) == 0:
            print(f"Warning: No channels found for target sample {self.tgt_sample} in {self.src_layer_block}")
            return [], [], [], [], [], []
        
        # Calculate impact scores
        scores = analyze_channel_score(
            model=self.model,
            src_layer_block=self.src_layer_block,
            tgt_layer_block=self.tgt_layer_block,
            val_dataset=self.val_dataset,
            channel_idx=channel_indices,
            src_channel=self.src_channel,
            model_type=self.model_type,
            patching_type=self.patching_type,
            corrupted_dataset=self.corrupted_dataset
        )
        
        # Filter channels and get significant ones
        filtered_channels, filtered_scores, filtered_info_scores = self.filter_channels(scores)
        
        return filtered_channels, filtered_scores, filtered_info_scores

    def _analyze_clip_vit_channel_impacts(self):
        """Analyze channel impacts for ViT models."""
        # Get channel indices for the target sample
        channel_indices = get_channel_indices(
            self.highly_activated_samples,
            self.src_layer_block,
            self.tgt_sample 
        )
        
        if len(channel_indices) == 0:
            print(f"Warning: No channels found for target sample {self.tgt_sample} in {self.src_layer_block}")
            return [], [], [], [], [], []
        
        # Calculate impact scores
        scores = analyze_channel_score(
            model=self.model,
            src_layer_block=self.src_layer_block,
            tgt_layer_block=self.tgt_layer_block,
            val_dataset=self.val_dataset,
            channel_idx=channel_indices,
            src_channel=self.src_channel,
            model_type=self.model_type,
            patching_type=self.patching_type,
            corrupted_dataset=self.corrupted_dataset
        )
        
        # Filter channels and get significant ones
        filtered_channels, filtered_scores, filtered_info_scores = self.filter_channels(scores)
        
        return filtered_channels, filtered_scores, filtered_info_scores

    def _analyze_swin_t_channel_impacts(self):
        """Analyze channel impacts for Swin-T models."""
        # Get channel indices for the target sample
        channel_indices = get_channel_indices(
            self.highly_activated_samples,
            self.src_layer_block,
            self.tgt_sample
        )

        if len(channel_indices) == 0:
            print(f"Warning: No channels found for target sample {self.tgt_sample} in {self.src_layer_block}")
            return [], [], [], [], [], []
        # Calculate impact scores
        scores = analyze_channel_score(
            model=self.model,
            src_layer_block=self.src_layer_block,
            tgt_layer_block=self.tgt_layer_block,
            val_dataset=self.val_dataset,
            channel_idx=channel_indices,
            src_channel=self.src_channel,
            model_type=self.model_type,
            patching_type=self.patching_type,
            corrupted_dataset=self.corrupted_dataset
        )
        
        # Filter channels and get significant ones
        filtered_channels, filtered_scores, filtered_info_scores = self.filter_channels(scores)
        
        return filtered_channels, filtered_scores, filtered_info_scores
        


def setup_environment(args, save_dir):
    """Setup GPU and output directory."""
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.save_plots:
        os.makedirs(save_dir, exist_ok=True)

def load_data(val_dir):
    """Load ImageNet validation dataset."""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_dataset = datasets.ImageFolder(val_dir, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False,
                           num_workers=8, pin_memory=True)
    
    return val_dataset, val_loader

def load_activation_samples(samples_dir):
    """Load pre-computed activation samples."""
    avg_file = os.path.join(samples_dir, 'highly_activated_samples_top5000.pkl')
    high_file = os.path.join(samples_dir, 'highly_activated_samples_top500.pkl')
    
    with open(avg_file, 'rb') as f:
        avg_activated_samples = pickle.load(f)
    with open(high_file, 'rb') as f:
        highly_activated_samples = pickle.load(f)
        
    return avg_activated_samples, highly_activated_samples