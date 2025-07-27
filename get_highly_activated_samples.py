import argparse
import os
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import pickle
import heapq
from model_configs import get_model_config, create_model

class IntermediateLayerGetter(nn.Module):
    def __init__(self, model, model_type: str):
        super().__init__()
        self.model = model
        self.model_type = model_type
        self.activations = {}
        self.hooks = []
        self.block_idx = 0
        self.hidden_dims = []
        
        if model_type == 'resnet':
            self._register_resnet_hooks()
        elif model_type == 'vit':
            self._register_vit_hooks()
        elif model_type == 'swin_t':
            self._register_swin_t_hooks()
        elif model_type == 'clip_vit':
            self._register_clip_vit_hooks()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def _register_resnet_hooks(self):
        """Register hooks for ResNet architecture."""
        for layer_idx, layer in enumerate([self.model.layer1, self.model.layer2, 
                                         self.model.layer3, self.model.layer4]):
            for block_idx, block in enumerate(layer):
                name = f'layer{layer_idx + 1}_block{block_idx}'
                hook = block.register_forward_hook(
                    lambda m, inp, out, name=name: self.activations.update({name: out})
                )
                self.hooks.append(hook)
    
    def _register_vit_hooks(self):
        """Register hooks for ViT architecture."""
        
        for layer_idx, layer in enumerate(self.model.encoder.layers):
            name = f'encoder_layer_{layer_idx}'
            # Hook for the output of each transformer block
            hook = layer.mlp[1].register_forward_hook(
                lambda m, inp, out, name=name: self.activations.update({name: out})
            )
            self.hooks.append(hook)
    
    def _register_clip_vit_hooks(self):
        """Register hooks for ViT architecture."""
        
        for layer_idx, layer in enumerate(self.model.visual.transformer.resblocks):
            name = f'clip-vit_block_{layer_idx}'
            # Hook for the output of each transformer block
            hook = layer.mlp[1].register_forward_hook(
                lambda m, inp, out, name=name: self.activations.update({name: out})
            )
            self.hooks.append(hook)

    def _register_swin_t_hooks(self):

        def hook_fn(name, module, input, output):
            self.activations[name] = output
        
        """Register hooks for Swin-T architecture."""
        for stage in self.model.features:
            if isinstance(stage, nn.Sequential):
                for block in stage:
                    if block.__class__.__name__ == 'SwinTransformerBlock':
                        block_name = f'swin_t_block_{self.block_idx}'

                        try:
                            # Get hidden dim from MLP
                            hidden_dim = block.mlp[0].out_features
                            self.hidden_dims.append(hidden_dim)

                            def make_hook(curr_name):
                                return lambda m, i, o: hook_fn(curr_name, m, i, o)

                            hook = block.mlp[1].register_forward_hook(make_hook(block_name))
                            self.hooks.append(hook)
                            self.block_idx += 1
                        except AttributeError as e:
                            print(f"Block {block_name} missing expected MLP structure: {e}")

    def forward(self, x):
        """Forward pass through the model."""
        with torch.no_grad():
            if self.model_type == 'clip_vit':
                _ = self.model.encode_image(x)
            else:
                _ = self.model(x)
            if self.model_type == 'vit' or self.model_type == 'clip_vit':
                # Process ViT activations to match expected format
                processed_activations = {}
                # Sort keys numerically based on layer number
                sorted_keys = sorted(self.activations.keys(), 
                                   key=lambda x: int(x.split('_')[2]))
                
                for name in sorted_keys:
                    activation = self.activations[name]
                    # Reshape from [B, N, H] to [B, H, N] to match channel-first format
                    processed_activations[name] = activation.transpose(1, 2)
                return processed_activations
            return self.activations
    
    def __del__(self):
        for hook in self.hooks:
            hook.remove()

def get_top_k_activations(feature_map, k=0.1, model_type='resnet'):
    """Calculate mean of top k% activations based on model type."""
    if model_type == 'resnet':
        B, C, H, W = feature_map.shape
        k_pixels = int(H * W * k)
        feature_map_flat = feature_map.reshape(B, C, -1).permute(0, 2, 1)
    elif model_type == 'vit':  # ViT
        B, N, C = feature_map.shape  # N is number of tokens
        k_pixels = int(C * k)
        feature_map_flat = feature_map.permute(0, 2, 1)  # [B, C, N] C = 197
    elif model_type == 'clip_vit':
        C, N, B = feature_map.shape  # N is number of tokens
        k_pixels = int(C * k)
        feature_map_flat = feature_map.permute(2, 0, 1)  # [B, C, N] C = 197
    elif model_type == 'swin_t':
        B, H, W, C = feature_map.shape  # N is number of tokens
        k_pixels = int(H * W * k)
        feature_map_flat = feature_map.permute(0, 3, 1, 2).reshape(B, C, -1)  # [B, C, H, W] 
        feature_map_flat = feature_map_flat.permute(0, 2, 1)

    # Use torch.kthvalue for memory efficiency
    top_k_vals = torch.sort(feature_map_flat, dim=1, descending=True)[0]
    top_k_vals = top_k_vals[:, :k_pixels, :]
    return top_k_vals.mean(dim=1)

def save_checkpoint(results, args):
    """Save intermediate results to a checkpoint file."""
    output_dir = os.path.join(args.output_dir, args.model_name, args.dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    
    checkpoint_file = os.path.join(
        output_dir, 
        f'highly_activated_samples_{"cls" + str(args.class_idx) + "_" if args.class_idx is not None else ""}top{args.n_samples}.pkl'
    )
    print(f"Saving checkpoint to {checkpoint_file}")
    
    # Convert current heaps to sorted lists
    checkpoint_results = {}
    for block_name, channels in results.items():
        checkpoint_results[block_name] = {}
        for channel_idx, heap in enumerate(channels):
            sorted_data = sorted(heap, reverse=True)
            checkpoint_results[block_name][channel_idx] = [idx for _, idx in sorted_data]
    
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(checkpoint_results, f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='/data8/dahee/circuit/results',
                      help='Base directory for saving results')
    parser.add_argument('--model_name', type=str, default='resnet50',
                      help='Model name (default: resnet50)')
    parser.add_argument('--dataset_name', type=str, default='imagenet',
                      help='Dataset name (default: imagenet)')
    parser.add_argument('--class_idx', type=int, default=None,
                      help='Class index')
    parser.add_argument('--n_samples', type=int, default=500,
                      help='Number of samples to consider')
    args = parser.parse_args()
    
    # Get model configuration and create model
    model_config = get_model_config(args.model_name)
    model = create_model(args.model_name)
    model.eval()
    model = model.cuda()
    
    # Setup data loading
    transform = transforms.Compose([
        transforms.Resize(model_config.get('image_size', 224)),
        transforms.CenterCrop(model_config.get('image_size', 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_dir = '/data/ImageNet1k/val'  # Update with your path
    val_dataset = datasets.ImageFolder(val_dir, transform=transform)
    if args.class_idx is not None:
        subset_indices = (torch.tensor(val_dataset.targets) == args.class_idx).nonzero().squeeze().tolist()
        val_dataset = torch.utils.data.Subset(val_dataset, subset_indices)
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    
    # Create intermediate layer getter
    intermediate_getter = IntermediateLayerGetter(model, model_config['model_type'])
    intermediate_getter = intermediate_getter.cuda()
    
    # Do one forward pass to get layer names and shapes
    with torch.no_grad():
        dummy_input = torch.zeros(1, 3, 
                                model_config.get('image_size', 224), 
                                model_config.get('image_size', 224)).cuda()
        activations = intermediate_getter(dummy_input)
    
    # Initialize results dictionary
    results = {}
    for name, activation in activations.items():
        if model_config['model_type'] == 'resnet':
            channel_size = activation.shape[1]  # B, C, H, W
        elif model_config['model_type'] == 'vit' or model_config['model_type'] == 'clip_vit':
            channel_size = activation.shape[1]  # B, H, N (after transpose)
        elif model_config['model_type'] == 'swin_t':
            channel_size = activation.shape[3]  # B, H, W, C
        # print(f"Layer {name} has {channel_size} channels")
        results[name] = [[] for _ in range(channel_size)]
    
    # Process batches
    total_samples = len(val_dataset)
    processed_samples = 0
    last_checkpoint = 0
    
    for batch_idx, (images, _) in enumerate(val_loader):
        images = images.to('cuda', non_blocking=True)
        batch_size = images.size(0)
        
        with torch.no_grad(), torch.cuda.amp.autocast():
            activations = intermediate_getter(images)
            for block_name, feature_maps in activations.items():
                top_activations = get_top_k_activations(
                    feature_maps, 
                    k=0.1, 
                    model_type=model_config['model_type']
                )
                # Process each channel
                for channel_idx in range(top_activations.shape[1]):
                    channel_acts = top_activations[:, channel_idx]
                    current_heap = results[block_name][channel_idx]
                    
                    batch_acts = list(zip(
                        channel_acts.cpu().tolist(),
                        range(processed_samples, processed_samples + batch_size)
                    ))
                    
                    for act, idx in batch_acts:
                        if len(current_heap) < args.n_samples:
                            heapq.heappush(current_heap, (act, idx))
                        elif act > current_heap[0][0]:
                            heapq.heapreplace(current_heap, (act, idx))
        
        processed_samples += batch_size
        progress = (processed_samples / total_samples) * 100
        
        if batch_idx % 20 == 0:
            print(f'Processed {processed_samples}/{total_samples} images ({progress:.1f}%)')
        
        # Save checkpoint every 10% progress
        current_checkpoint = int(progress // 10) * 10
        if current_checkpoint > last_checkpoint:
            save_checkpoint(results, args)
            last_checkpoint = current_checkpoint
    
    # Save final results
    print("Processing final results...")
    save_checkpoint(results, args)

if __name__ == '__main__':
    main()
