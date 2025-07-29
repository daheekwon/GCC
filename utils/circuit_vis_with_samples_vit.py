import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle
import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import sys
from tqdm import tqdm
sys.path.append('/GCC/utils')
from plot_crop_mask import crop_mask

# Load the highly activated samples data
with open('/GCC/results/vit/imagenet/highly_activated_samples_top500.pkl', 'rb') as f:
    highly_activated_samples = pickle.load(f)

# Set up image loading and transformation
val_dir = '/data/ImageNet1k/val'
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

def is_vit_model(model):
    """Check if the model is a ViT model."""
    return (hasattr(model, 'encoder') and 
            hasattr(model.encoder, 'layers') and 
            hasattr(model.encoder.layers[0], 'mlp'))

def get_mlp_activations_batch(model, images, layer_idx):
    """
    Get MLP activations for a batch of images.
    
    Args:
        model: ViT model
        images: Batch of images [B, 3, 224, 224]
        layer_idx: Index of the transformer layer
    
    Returns:
        Batch of activation maps [B, hidden_dim, 224, 224]
    """
    model.eval()
    activations = None
    
    def hook_fn(module, input, output):
        nonlocal activations
        activations = output
    
    hook = model.encoder.layers[layer_idx].mlp[1].register_forward_hook(hook_fn)
    
    with torch.no_grad():
        _ = model(images)
    
    hook.remove()
    
    # Process all channels at once
    batch_size = activations.size(0)
    hidden_dim = activations.size(2)
    
    # Reshape to [B, hidden_dim, 14, 14] (excluding CLS token)
    grid_activations = activations[:, 1:, :].reshape(batch_size, 14, 14, hidden_dim)
    grid_activations = grid_activations.permute(0, 3, 1, 2)
    
    # Upsample to [B, hidden_dim, 224, 224]
    upsampled = torch.nn.functional.interpolate(
        grid_activations,
        size=(224, 224),
        mode='bilinear',
        align_corners=False
    )
    
    return upsampled

def load_and_preprocess_image(img_path):
    img = Image.open(img_path).convert('RGB')
    return transform(img)

def get_sample_images_and_masks(model, node_name, channel_idx, num_samples=4):
    if node_name not in highly_activated_samples:
        return None, None
    
    if not is_vit_model(model):
        print("Only ViT models are currently supported")
        return None, None
    
    sample_indices = highly_activated_samples[node_name][channel_idx][:num_samples]
    images = []
    image_paths = []
    
    # Get layer index from node name
    layer_idx = get_layer_number(node_name)
    
    # Get the class folders
    class_folders = sorted(os.listdir(val_dir))
    
    # Collect all image paths first
    for idx in sample_indices:
        class_idx = idx // 50
        img_idx = idx % 50
        
        class_folder = class_folders[class_idx]
        class_path = os.path.join(val_dir, class_folder)
        img_files = sorted(os.listdir(class_path))
        img_path = os.path.join(class_path, img_files[img_idx])
        image_paths.append(img_path)
    
    # Load all images
    for img_path in image_paths:
        img = load_and_preprocess_image(img_path)
        images.append(img)
    
    # Stack images into a batch
    images_batch = torch.stack(images)
    
    # Get activations for all images at once
    if images_batch.device != next(model.parameters()).device:
        images_batch = images_batch.to(next(model.parameters()).device)
    
    activations_batch = get_mlp_activations_batch(model, images_batch, layer_idx)
    
    # Extract masks for the specific channel
    masks = activations_batch[:, channel_idx].cpu()
    
    # Normalize masks
    masks = (masks - masks.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]) / \
           (masks.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0] - 
            masks.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0] + 1e-8)
    
    return images, masks

def get_layer_number_vit(node_name):
    if 'encoder_layer_' in node_name:
        return int(node_name.split('_')[2])
    return -1

def get_layer_number_resnet(node_name):
    if 'layer' in node_name:
        return int(list(node_name.split('_')[0])[-1]), int(list(node_name.split('_')[1])[-1])
    return -1

def visualize_layer_samples(model, circuit, layers_to_show, num_samples=4, crop_th=0.8, mask_th=0.4, alpha=0.5):
    """
    Visualize cropped sample images for each channel in selected layers.
    """
    # Group nodes by layer
    layer_groups = {}
    for node in circuit.nodes():
        if model == 'vit':
            layer_num = get_layer_number_vit(node[0])
        else:
            layer_num, block_num = get_layer_number_resnet(node[0])
        if layer_num in layers_to_show:
            if layer_num not in layer_groups:
                layer_groups[layer_num] = []
            layer_groups[layer_num].append(node)
    
    # Sort layers
    sorted_layers = sorted(layer_groups.keys())
    
    for layer_num in tqdm(sorted_layers, desc="Processing layers"):
        nodes = sorted(layer_groups[layer_num], key=lambda x: x[1])
        n_channels = len(nodes)
        
        # Create figure with minimal size
        fig = plt.figure(figsize=(5, n_channels * 0.8))
        
        # Create layout without any margins
        gs = gridspec.GridSpec(n_channels, num_samples + 1, figure=fig)
        gs.update(wspace=0, hspace=0)
        plt.subplots_adjust(top=0.95, bottom=0.05, left=0.2, right=0.99, hspace=0, wspace=0)
        
        plt.figtext(0.5, 0.98, f'Layer {layer_num} - Sample Images by Channel', 
                   fontsize=12, ha='center')
        
        for idx, node in enumerate(nodes):
            ax_label = fig.add_subplot(gs[idx, 0])
            ax_label.text(1.0, 0.5, f'Channel {node[1]}', 
                        ha='right', va='center', fontsize=10)
            ax_label.axis('off')
            
            images, masks = get_sample_images_and_masks(model, node[0], node[1], num_samples)
            if images is None or masks is None:
                continue
            
            for i, (img, mask) in enumerate(zip(images, masks)):
                ax = fig.add_subplot(gs[idx, i+1])
                cropped = crop_mask(img, mask, 
                                  crop_th=crop_th,
                                  mask_th=mask_th,
                                  alpha=alpha)
                cropped_np = cropped.permute(1, 2, 0).numpy()
                ax.imshow(cropped_np)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_frame_on(False)
        
        plt.show()