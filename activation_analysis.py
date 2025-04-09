import os
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from matplotlib import gridspec
import pickle
from pytorch_msssim import ssim
import cv2 
import matplotlib.gridspec as gridspec
from typing import Dict, List, Tuple, Union

class LayerActivationHelper:
    """Helper class to manage layer and block access in ResNet models."""
    
    @staticmethod
    def get_target_block(model: nn.Module, layer_name: str, block_idx: int) -> nn.Module:
        """Get the target block from specified layer.
        
        Args:
            model: ResNet model
            layer_name: Name of layer (e.g., 'layer1', 'layer2')
            block_idx: Index of block within layer
            
        Returns:
            Target block module
        
        Raises:
            ValueError: If layer_name is invalid
        """
        layer_mapping = {
            'layer1': model.layer1,
            'layer2': model.layer2, 
            'layer3': model.layer3,
            'layer4': model.layer4
        }

        
        if layer_name not in layer_mapping:
            raise ValueError(f"Invalid layer name: {layer_name}")
            
        return layer_mapping[layer_name][block_idx]

def parse_layer_block(layer_block_str: str) -> Tuple[str, int]:
    """Parse layer_block string into layer name and block index.
    
    Args:
        layer_block_str: String in format 'layerX_blockY'
        
    Returns:
        Tuple of (layer_name, block_idx)
        
    Example:
        >>> parse_layer_block('layer3_block2')
        ('layer3', 2)
    """
    parts = layer_block_str.split('_')
    if len(parts) != 2 or not parts[1].startswith('block'):
        raise ValueError(f"Invalid layer_block format: {layer_block_str}")
    
    layer_name = parts[0]
    block_idx = int(parts[1][5:])  # Remove 'block' prefix
    return layer_name, block_idx

def get_channel_indices(highly_activated_samples: Dict, 
                       layer_block: str,
                       target_sample: int) -> List[Tuple[int, List[int]]]:
    """Get channel indices and their corresponding samples up to target sample.
    
    Args:
        highly_activated_samples: Dict of activation samples
        layer_block: Layer/block identifier (e.g., 'layer3_block2')
        target_sample: Target sample index
        
    Returns:
        List of tuples (channel_index, sample_indices)
    """
    channel_idx = []
    for channel, samples in highly_activated_samples[layer_block].items():
        for i, sample in enumerate(samples):
            if sample == target_sample:
                channel_idx.append((channel, samples[:i+1]))
                break
                
    return channel_idx

def get_activation(model: nn.Module,
                  layer_block: str, 
                  val_dataset: Dataset,
                  image_idx: int,
                  channel_idx: int) -> torch.Tensor:
    """Get activation feature map for specific channel.
    
    Args:
        model: The model
        layer_block: Layer/block identifier (e.g., 'layer3_block2')
        val_dataset: Validation dataset
        image_idx: Index of image
        channel_idx: Channel to extract
        
    Returns:
        Channel activation tensor
    """
    activation = []
    layer_name, block_idx = parse_layer_block(layer_block)
    
    def hook_fn(module, input, output):
        channel_activation = output[:, channel_idx, :, :]
        activation.append(channel_activation.detach().cpu())

    img, _ = val_dataset[image_idx]
    img = img.unsqueeze(0).cuda()
    
    target_block = LayerActivationHelper.get_target_block(model, layer_name, block_idx)
    
    with torch.no_grad():
        handle = target_block.register_forward_hook(hook_fn)
        _ = model(img)
        handle.remove()
        
    return activation[0]

def save_vis_images(indices, val_dataset,tgt_sample, num_images=5):
    imgs = []
    for i, idx in enumerate(indices[:num_images-1]):
        img, label = val_dataset[idx]        

        # Convert tensor to image for display
        img = img.permute(1, 2, 0).numpy()
        # Denormalize the image
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        imgs.append((idx,img))

    # Add the last image
    img, label = val_dataset[tgt_sample]

    img = img.permute(1, 2, 0).numpy()
    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img = np.clip(img, 0, 1)
    imgs.append((tgt_sample, img))

    return imgs


def get_activation_subset(model: nn.Module,
                        layer_block: str,
                        val_dataset: Dataset,
                        indices: List[int],
                        channel_idx: int,
                        batch_size: int = 32) -> torch.Tensor:
    """Get full feature maps for specified channels without taking mean.
    
    Args:
        model: The neural network model
        layer_block: Layer/block identifier (e.g., 'layer3_block2')
        val_dataset: Validation dataset
        indices: List of sample indices
        channel_idx: Channel to extract
        batch_size: Batch size for processing
        
    Returns:
        torch.Tensor: Activation maps for the specified channel
    """
    activations = []
    layer_name, block_idx = parse_layer_block(layer_block)
    
    def hook_fn(module, input, output):
        channel_activation = output[:, channel_idx, :, :]
        activations.append(channel_activation.detach().cpu())
    
    subset = Subset(val_dataset, indices)
    subset_loader = DataLoader(subset, batch_size=batch_size, shuffle=False, 
                             num_workers=4, pin_memory=True)
    
    target_block = LayerActivationHelper.get_target_block(model, layer_name, block_idx)
    
    with torch.no_grad(), target_block.register_forward_hook(hook_fn):
        for batch in subset_loader:
            images = batch[0].cuda()
            _ = model(images)
            
    return torch.cat(activations, dim=0)

def get_output_with_modified_activation(model: nn.Module,
                                      src_layer_block: str,
                                      tgt_layer_block: str,
                                      image: torch.Tensor,
                                      original_act: torch.Tensor,
                                      channel_idx: int) -> torch.Tensor:
    """Get output by modifying a specific block's output for a specific channel.
    
    Args:
        model: The neural network model
        source_layer_block: Source layer/block identifier (e.g., 'layer3_block2')
        target_layer_block: Target layer/block identifier (e.g., 'layer4_block1')
        image: Input image tensor
        original_act: Original activation to modify
        channel_idx: Channel to modify
        
    Returns:
        torch.Tensor: Modified output activation
    """
    outputs = []
    src_layer_name, src_block_idx = parse_layer_block(src_layer_block)
    tgt_layer_name, tgt_block_idx = parse_layer_block(tgt_layer_block)
    
    def hook_fn_source_block(module, input, output):
        output_clone = output.clone()
        output_clone[:, channel_idx, :, :] = original_act
        return output_clone
    
    def hook_fn_target_block(module, input, output):
        outputs.append(output.detach().cpu())
    
    src_block = LayerActivationHelper.get_target_block(model, src_layer_name, src_block_idx)
    tgt_block = LayerActivationHelper.get_target_block(model, tgt_layer_name, tgt_block_idx)
    
    handle1 = src_block.register_forward_hook(hook_fn_source_block)
    handle2 = tgt_block.register_forward_hook(hook_fn_target_block)
    
    with torch.no_grad():
        _ = model(image.unsqueeze(0).cuda())
    
    handle1.remove()
    handle2.remove()

    return outputs[0]

def analyze_channel_score(model: nn.Module,
                        src_layer_block: str,
                        tgt_layer_block: str,
                        val_dataset: Dataset,
                        channel_idx: List[Tuple[int, List[int]]],
                        src_channel: int) -> torch.Tensor:
    """Analyzes the impact of a channel by comparing original, masked, and amplified activations.
    
    Args:
        model: The neural network model
        source_layer_block: Source layer/block identifier (e.g., 'layer3_block2')
        target_layer_block: Target layer/block identifier (e.g., 'layer4_block1')
        val_dataset: Validation dataset
        channel_idx: List of channel indices and their samples
        src_channel: Index of the channel to analyze
    
    Returns:
        torch.Tensor: Impact scores for the channel
    """
    # Find the target channel in channel_idx
    for i, (c, idx) in enumerate(channel_idx):
        if c == src_channel:
            break
            
    # Get the original image
    img, _ = val_dataset[channel_idx[i][1][-1]]

    # Get original activation
    input_A = get_activation_subset(
        model=model,
        layer_block=src_layer_block,
        val_dataset=val_dataset,
        indices=[channel_idx[i][1][-1]],
        channel_idx=channel_idx[i][0]
    ).squeeze()

    # Create masked and amplified versions
    masked_A = torch.zeros_like(input_A)
    amplified_A = input_A * 2

    # Get outputs for different activation versions
    output_org_A = get_output_with_modified_activation(
        model=model,
        src_layer_block=src_layer_block,
        tgt_layer_block=tgt_layer_block,
        image=img,
        original_act=input_A,
        channel_idx=channel_idx[i][0]
    )
    
    output_masked_A = get_output_with_modified_activation(
        model=model,
        src_layer_block=src_layer_block,
        tgt_layer_block=tgt_layer_block,
        image=img,
        original_act=masked_A,
        channel_idx=channel_idx[i][0]
    )
    
    output_amplified_A = get_output_with_modified_activation(
        model=model,
        src_layer_block=src_layer_block,
        tgt_layer_block=tgt_layer_block,
        image=img,
        original_act=amplified_A,
        channel_idx=channel_idx[i][0]
    )

    # Calculate differences
    diff_masked = (output_org_A[0] - output_masked_A[0]).mean(dim=(1,2))
    diff_amplified = (output_amplified_A[0] - output_org_A[0]).mean(dim=(1,2))

    # Filter negative values
    diff_masked = torch.where(diff_masked < 0, torch.zeros_like(diff_masked), diff_masked)
    diff_amplified = torch.where(diff_amplified < 0, torch.zeros_like(diff_amplified), diff_amplified)

    # Calculate final scores
    scores = diff_masked
    
    return scores.clone().detach()