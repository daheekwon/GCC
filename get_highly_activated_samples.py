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

MODEL_CONFIGS = {
    'resnet18': {'channels': [64, 128, 256, 512]},
    'resnet34': {'channels': [64, 128, 256, 512]},
    'resnet50': {'channels': [256, 512, 1024, 2048]},
    'resnet101': {'channels': [256, 512, 1024, 2048]},
}

class IntermediateLayerGetter(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.activations = {}
        
        # Register hooks for each block within each layer
        self.hooks = []
        for layer_idx, layer in enumerate([model.layer1, model.layer2, model.layer3, model.layer4]):
            for block_idx, block in enumerate(layer):
                name = f'layer{layer_idx + 1}_block{block_idx}'
                hook = block.register_forward_hook(
                    lambda m, inp, out, name=name: self.activations.update({name: out})
                )
                self.hooks.append(hook)
                
    def forward(self, x):
        _ = self.model(x)
        return self.activations
    
    def __del__(self):
        for hook in self.hooks:
            hook.remove()

def get_top_k_activations(feature_map, k=0.1):
    """Calculate mean of top k% pixels for each channel more efficiently."""
    B, C, H, W = feature_map.shape
    k_pixels = int(H * W * k)
    
    # More memory efficient reshape
    feature_map_flat = feature_map.permute(0, 1, 2, 3).reshape(B, C, -1)
    
    # Use torch.kthvalue instead of topk for memory efficiency
    top_k_vals = torch.sort(feature_map_flat, dim=2, descending=True)[0]
    top_k_vals = top_k_vals[:, :, :k_pixels]
    
    return top_k_vals.mean(dim=2)

def save_checkpoint(results, args):
    """Save intermediate results to a checkpoint file."""
    output_dir = os.path.join(args.output_dir, args.model_name, args.dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    
    if args.class_idx is not None:
        checkpoint_file = os.path.join(output_dir, f'highly_activated_samples_cls{args.class_idx}_top{args.n_samples}.pkl')
    else:
        checkpoint_file = os.path.join(output_dir, f'highly_activated_samples_top{args.n_samples}.pkl')
    print(f"Saving checkpoint to {checkpoint_file}")
    
    # Convert current heaps to sorted lists
    checkpoint_results = {}
    for block_name, channels in results.items():
        checkpoint_results[block_name] = {}
        for channel_idx, heap in enumerate(channels):
            sorted_data = sorted(heap, reverse=True)
            checkpoint_results[block_name][channel_idx] = [idx for _, idx in sorted_data]
    
    try:
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint_results, f)
    except Exception as e:
        print(f"Error saving checkpoint: {e}")

def main():
    # Add arguments
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
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load ImageNet validation set
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if args.dataset_name == 'imagenet':
        val_dir = '/data/ImageNet1k/val'  # Replace with your ImageNet validation path
        val_dataset = datasets.ImageFolder(val_dir, transform=transform)
        if args.class_idx is not None:
            subset_indices = (torch.tensor(val_dataset.targets) == args.class_idx).nonzero().squeeze().tolist()
            val_dataset = torch.utils.data.Subset(val_dataset, subset_indices)

        val_loader = DataLoader(
            val_dataset, 
            batch_size=128, 
            shuffle=False,
            num_workers=8, 
            pin_memory=True
        )
    
    # Load pretrained ResNet
    if args.model_name == 'resnet50':
        model = torchvision.models.resnet50(pretrained=True)
    elif args.model_name == 'resnet18':
        model = torchvision.models.resnet18(pretrained=True)
    elif args.model_name == 'resnet34':
        model = torchvision.models.resnet34(pretrained=True)
    elif args.model_name == 'resnet101':
        model = torchvision.models.resnet101(pretrained=True)
    model.eval()
    model = model.to(device)
    
    # Create intermediate layer getter
    intermediate_getter = IntermediateLayerGetter(model)
    intermediate_getter = intermediate_getter.to(device)
    
    # Validate model name
    if args.model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported model: {args.model_name}")
    
    # Use model specific channel sizes
    channels = MODEL_CONFIGS[args.model_name]['channels']
    
    # Initialize results with the new naming structure
    results = {}
    for layer_idx in range(4):
        for block_idx in range(len(model.layer1 if layer_idx == 0 else 
                              model.layer2 if layer_idx == 1 else 
                              model.layer3 if layer_idx == 2 else 
                              model.layer4)):
            block_name = f'layer{layer_idx + 1}_block{block_idx}'
            channel_size = channels[layer_idx]
            results[block_name] = [[] for _ in range(channel_size)]
    
    # Optimize data loading
    val_loader = DataLoader(
        val_dataset,
        batch_size=256,  # Increased batch size
        shuffle=False,
        num_workers=16,  # Increased workers
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    
    # Process batches more efficiently
    total_samples = len(val_dataset)
    processed_samples = 0
    last_checkpoint = 0
    
    for batch_idx, (images, _) in enumerate(val_loader):
        images = images.to(device, non_blocking=True)
        batch_size = images.size(0)
        
        with torch.no_grad(), torch.cuda.amp.autocast():
            activations = intermediate_getter(images)
            
            for block_name, feature_maps in activations.items():
                top_activations = get_top_k_activations(feature_maps, k=0.1)
                
                # Process each channel
                for channel_idx in range(top_activations.shape[1]):
                    channel_acts = top_activations[:, channel_idx]
                    
                    # Use the new block naming convention
                    current_heap = results[block_name][channel_idx]
                    
                    # Convert to list of (activation, index) pairs
                    batch_acts = list(zip(
                        channel_acts.cpu().tolist(),
                        torch.arange(processed_samples, processed_samples + batch_size, device=device).cpu().tolist()
                    ))
                    
                    # Maintain only top args.n_samples activations using a min-heap
                    for act, idx in batch_acts:
                        if len(current_heap) < args.n_samples:
                            heapq.heappush(current_heap, (act, idx))
                        elif act > current_heap[0][0]:  # If larger than smallest in heap
                            heapq.heapreplace(current_heap, (act, idx))
        
        processed_samples += batch_size
        progress = (processed_samples / total_samples) * 100
        
        if batch_idx % 20 == 0:
            print(f'Processed {processed_samples}/{total_samples} images '
                  f'({progress:.1f}%)')
        
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
