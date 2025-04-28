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
    parser.add_argument('--save_dir', type=str, default='/data8/dahee/circuit/results/',
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
        
        # Load activation samples
        self.avg_activated_samples, self.highly_activated_samples = load_activation_samples(self.samples_dir)
        
        # Initialize matrices for layer connections
        # self.connection_matrices = self.initialize_connection_matrices()
        
        # Load or create metadata with model
        self.metadata = load_or_create_metadata(self.save_dir, self.model, self.model_type)
        
        # Load or create metadata with model
        self.metadata = load_or_create_metadata(self.save_dir, self.model)

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
                return layer.mlp[3].out_features

        
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
            print('!!!!!!!!!!!!!!!!!!')
            print(f"Warning: No meaningful scores for {self.tgt_layer_block}")
            return np.array([]), np.array([]), np.array([]), np.array([])
        normalized_scores /= normalized_scores.sum()
        
        # Create linked channels mask instead of list
        linked_mask = np.zeros(len(self.avg_activated_samples[self.tgt_layer_block]), dtype=bool)
        for i in range(len(linked_mask)):
            if self.tgt_sample in self.avg_activated_samples[self.tgt_layer_block][i]:
                linked_mask[i] = True
        
        # Filter non-zero linked channels more efficiently
        # valid_mask = (normalized_scores != 0) & linked_mask
        valid_mask = (~np.isnan(normalized_scores)) & (normalized_scores != 0) & linked_mask
        scores_ls = normalized_scores[valid_mask]
        next_channel_pool = np.where(valid_mask)[0]
        
        if len(scores_ls) == 0:
            return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

        # POT method
        threshold = np.percentile(scores_ls, self.pot_threshold)
        exceedances_mask = scores_ls > threshold
        exceedances = scores_ls[exceedances_mask] - threshold
        if len(exceedances) > 0:
            # Fit GPD only to exceedances
            shape, _, scale = stats.genpareto.fit(exceedances,floc=0)
            
            shape = np.max([shape, 0.00000000000000001])
            scale = np.min([scale, np.std(exceedances)])

            shape, loc, scale = stats.genpareto.fit(exceedances)
            # Calculate return level
            p = 0.95
            N = len(scores_ls)
            Nx = len(exceedances)
            return_level = threshold + scale/shape * ((N/Nx * (1-p))**(-shape) - 1)

            if return_level < 0:
                return (np.array([]), np.array([]), np.array([]), return_level, shape, scale)
            else:
                # Apply return level threshold
                outlier_mask = scores_ls > return_level
        else:
            return (np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]))
        
        filtered_channels = next_channel_pool[outlier_mask]
        filtered_scores = scores_ls[outlier_mask]
        
        if len(filtered_channels) == 0:
            return (np.array([]), np.array([]), np.array([]), return_level, shape, scale)
        
        # Compute activation overlaps more efficiently
        # src_activations = set(self.avg_activated_samples[self.src_layer_block][self.src_channel])
        src_activations = set(self.highly_activated_samples[self.src_layer_block][self.src_channel])
        
        # Vectorize ratio calculations
        ratios = []
        for tgt_ch in filtered_channels:
            # tgt_activations = set(self.avg_activated_samples[self.tgt_layer_block][tgt_ch])
            tgt_activations = set(self.highly_activated_samples[self.tgt_layer_block][tgt_ch])
            ratio = len(tgt_activations & src_activations) / len(tgt_activations)
            ratios.append(ratio)
        
        # Calculate denominator ratios only once for all channels
        denom_ratios = []
        for tgt_ch in next_channel_pool:
            # tgt_activations = set(self.avg_activated_samples[self.tgt_layer_block][tgt_ch])
            tgt_activations = set(self.highly_activated_samples[self.tgt_layer_block][tgt_ch])
            ratio = len(tgt_activations & src_activations) / len(tgt_activations)
            denom_ratios.append(ratio)
        
        # print(f"denom_ratios: {np.mean(denom_ratios), np.median(denom_ratios)}")
        ratio_threshold = max(np.mean(denom_ratios),0.05)  # Take maximum of mean and 0.1
        ratio_mask = np.array(ratios) > ratio_threshold
        return (filtered_channels[ratio_mask], 
                filtered_scores[ratio_mask],
                np.array(ratios)[ratio_mask],
                return_level,
                shape,
                scale)

    def analyze_channel_impacts(self):
        """Analyze channel impacts using scoring mechanism."""
        if self.model_type == 'resnet':
            return self._analyze_resnet_channel_impacts()
        elif self.model_type == 'vit':
            return self._analyze_vit_channel_impacts()
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
            model_type=self.model_type
        )
        
        # Filter channels and get significant ones
        filtered_channels, filtered_scores, filtered_info_scores, return_level, shape, scale = self.filter_channels(scores)
        
        return filtered_channels, filtered_scores, filtered_info_scores, return_level, shape, scale

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
            model_type=self.model_type
        )
        
        # Filter channels and get significant ones
        filtered_channels, filtered_scores, filtered_info_scores, return_level, shape, scale = self.filter_channels(scores)
        
        return filtered_channels, filtered_scores, filtered_info_scores, return_level, shape, scale

    def visualize_activations(self):
        """Visualize channel activations and their feature maps."""
        # Create args-like object with required attributes
        args_dict = {
            'src_layer_block': self.src_layer_block,
            'src_channel': self.src_channel,
            'num_images': 10,
            'save_plots': self.save_plots
        }
        args = type('Args', (), args_dict)
        
        visualize_channel_activations(
            self.model, 
            args, 
            self.val_dataset, 
            self.avg_activated_samples,
            self.save_dir
        )

    def visualize_filtered_channels(self, filtered_channels):
        """Visualize activation maps and highly activated samples for filtered channels."""
        for ch_idx in filtered_channels:
            # Check if file already exists - simplified filename
            save_filename = os.path.join(
                self.save_dir,
                f'{self.tgt_layer_block}_ch_{ch_idx}_act.png'
            )
            
            if os.path.exists(save_filename):
                # print(f"Skipping visualization - file already exists: {self.tgt_layer_block}_ch_{ch_idx}_act")
                continue
                
            # Create figure for this channel
            fig = plt.figure(figsize=(20, 4.2))
            gs = gridspec.GridSpec(2, 10)
            gs.update(wspace=0, hspace=0)
            
            if self.save_plots:
            # Get highly activated samples for this channel
                imgs = save_vis_images(
                    self.highly_activated_samples[self.tgt_layer_block][ch_idx],
                    self.val_dataset,
                    self.tgt_sample,
                    10
                )
            
            # Plot original images
            for i, (idx, img) in enumerate(imgs):
                ax = plt.subplot(gs[i])
                ax.imshow(img)
                if i == 0:
                    ax.set_ylabel(f"Channel {ch_idx}", fontsize=15)
                ax.set_xticks([])
                ax.set_yticks([])
            
            # Plot activation maps
            for i, (idx, _) in enumerate(imgs):
                ax = plt.subplot(gs[i + 10])
                activation_map = get_activation(
                    self.model,
                    self.tgt_layer_block,
                    self.val_dataset,
                    idx,
                    ch_idx,
                    model_type=self.model_type
                )

                if self.model_type == 'resnet':
                    activation_map = activation_map.squeeze().cpu().numpy()
                else:
                    activation_map = activation_map.squeeze()[1:].reshape(14, 14).cpu().numpy()
                ax.imshow(activation_map)
                if i == 0:
                    ax.set_ylabel(f"Channel {ch_idx}", fontsize=15)
                ax.set_xticks([])
                ax.set_yticks([])
            
            plt.tight_layout()
            
            # Save the figure with simplified filename
            if self.save_plots:
                plt.savefig(save_filename)
                plt.close()
            else:
                plt.show()

    def analyze_channel_connections_iteratively(self):
        """Iteratively analyze connections starting from src_channel through all subsequent layers."""
        layer_keys = list(self.avg_activated_samples.keys())
        
        current_src_layer_idx = layer_keys.index(self.src_layer_block)
        current_src_channels = [self.src_channel]
        
        # Add initial source channel to metadata if not exists  ####여기가 좀 다르긴 함. 
        if self.src_layer_block not in self.metadata:
            self.metadata[self.src_layer_block] = {
                'searched_channels': [],
                # 'return_levels': {},
                # 'scale': {},
                # 'shape': {}
            }
        
        if self.src_channel not in self.metadata[self.src_layer_block]['searched_channels']:
            self.metadata[self.src_layer_block]['searched_channels'].append(self.src_channel)
        
        # Iterate through subsequent layers
        for layer_idx in range(current_src_layer_idx, len(layer_keys) - 1):
            current_src_layer = layer_keys[layer_idx]
            next_layer = layer_keys[layer_idx + 1]
            matrix_idx = layer_idx - current_src_layer_idx
            next_src_channels = []
            
            print(f"Analyzing connections from {current_src_layer} to {next_layer}")
            
            # For each current source channel
            for src_ch in current_src_channels:
                self.src_layer_block = current_src_layer
                self.src_channel = src_ch
                self.tgt_layer_block = next_layer

                filtered_channels, filtered_scores, filtered_info_scores, return_level, shape, scale = self.analyze_channel_impacts()
                
                # Store return level in metadata
                # if next_layer not in self.metadata[current_src_layer]['return_levels']:
                #     self.metadata[current_src_layer]['return_levels'] = {}
                # print(return_level)
                # self.metadata[current_src_layer]['return_levels'][str(src_ch)] = float(return_level)

                # if next_layer not in self.metadata[current_src_layer]['scale']:
                #     self.metadata[current_src_layer]['scale'] = {}
                # self.metadata[current_src_layer]['scale'][str(src_ch)] = float(scale)

                # if next_layer not in self.metadata[current_src_layer]['shape']:
                #     self.metadata[current_src_layer]['shape'] = {}
                # self.metadata[current_src_layer]['shape'][str(src_ch)] = float(shape)
                
                if len(filtered_channels) > 0:
                    # Update connection matrix
                    rows = np.full(len(filtered_channels), src_ch)
                    cols = filtered_channels
                    data = filtered_scores
                    new_entries = sparse.csr_matrix(
                        (data, (rows, cols)),
                        shape=self.connection_matrices[matrix_idx].shape
                    )
                    self.connection_matrices[matrix_idx] = self.connection_matrices[matrix_idx] + new_entries
                    
                    # Update metadata with filtered channels
                    for ch in filtered_channels:
                        if ch not in self.metadata[next_layer]['searched_channels']:
                            self.metadata[next_layer]['searched_channels'].append(int(ch))
                    
                    self.visualize_filtered_channels(filtered_channels)
                    next_src_channels.extend(filtered_channels)
            
            # Save metadata after each layer analysis
            self._save_metadata()
            
            current_src_channels = list(set(next_src_channels))
            
            if not current_src_channels:
                print(f"No more connections found after layer {current_src_layer}")
                break
        
        # Reset to original source layer and channel
        self.src_layer_block = layer_keys[current_src_layer_idx]
        self.src_channel = self.src_channel
        
        return self.connection_matrices

    def _save_metadata(self):
        """Save metadata to file."""
        metadata_path = os.path.join(self.save_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=4)

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

def visualize_channel_activations(model, args, val_dataset, avg_activated_samples, save_dir):
    """Visualize channel activations and their feature maps."""
    fig = plt.figure(figsize=(20, 4.2))
    gs = gridspec.GridSpec(2, args.num_images)
    gs.update(wspace=0, hspace=0)

    imgs = save_vis_images(
        avg_activated_samples[args.src_layer_block][args.src_channel],
        val_dataset,
        args.tgt_sample,
        args.num_images
    )

    # Plot original images
    for i, (idx, img) in enumerate(imgs):
        ax = plt.subplot(gs[i])
        ax.imshow(img)
        if i == 0:
            ax.set_ylabel(f"Channel {args.src_channel}", fontsize=15)
        ax.set_xticks([])
        ax.set_yticks([])

    # Plot activation maps
    for i, (idx, _) in enumerate(imgs):
        ax = plt.subplot(gs[i + args.num_images])
        activation_map = get_activation(
            model, 
            args.src_layer_block,
            val_dataset,
            idx,
            args.src_channel
        )
        ax.imshow(activation_map.squeeze().cpu().numpy())
        if i == 0:
            ax.set_ylabel(f"Channel {args.src_channel}", fontsize=15)
        ax.set_xticks([])
        ax.set_yticks([])

    if args.save_plots:
        plt.savefig(os.path.join(save_dir, f'{args.src_layer_block}_channel_{args.src_channel}_to_{args.tgt_layer_block}_activations.png'))
        plt.close()
    else:
        plt.show()

def load_or_create_metadata(save_dir, model=None, model_type='resnet'):
    """Load existing metadata or create default if not exists."""
    metadata_path = os.path.join(save_dir, 'metadata.json')
    
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            return json.load(f)
    
    # Create default metadata with layer structure automatically
    default_metadata = {}
    
    if model_type == 'resnet':
        # ResNet layer structure
        for layer_idx in range(1, 5):  # layer1 to layer4
            layer = getattr(model, f'layer{layer_idx}')
            for block_idx in range(len(layer)):
                key = f'layer{layer_idx}_block{block_idx}'
                default_metadata[key] = {
                    'searched_channels': [],
                }
    elif model_type == 'vit':  # ViT
        # ViT layer structure - ensure numerical ordering
        num_layers = len(model.encoder.layers)
        # Create sorted layer keys
        layer_keys = [f'encoder_layer_{i}' for i in range(num_layers)]
        # Sort numerically based on the layer number
        layer_keys.sort(key=lambda x: int(x.split('_')[-1]))
        
        for key in layer_keys:
            default_metadata[key] = {
                'searched_channels': [],
            }
    
    return default_metadata

def main():
    args = parse_args()
    analyzer = CircuitAnalyzer(args)
    
    # Perform iterative analysis
    if args.tgt_sample is not None:
        connection_matrices = analyzer.analyze_channel_connections_iteratively()
        
        # Save connection matrices
        save_path = os.path.join(
            analyzer.save_dir,
            f'connection_matrices_from_{args.src_layer_block}_ch{args.src_channel}.pkl'
        )
        
        save_data = {
            'matrices': connection_matrices,
            'src_layer_block': args.src_layer_block,
            'src_channel': args.src_channel,
            'tgt_sample': args.tgt_sample
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)
        print(f"Saved connection matrices to {save_path}")

if __name__ == '__main__':
    main() 