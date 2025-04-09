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
import cv2 
import matplotlib.gridspec as gridspec
from typing import Dict, List, Tuple, Union

class LayerActivationHelper:
    """Helper class to manage layer access in ViT models."""
    
    @staticmethod
    def get_target_layer(model: nn.Module, layer_name: str) -> nn.Module:
        """Get the target encoder layer.
        
        Args:
            model: ViT model
            layer_name: Name of layer (e.g., 'encoder_layer_0')
            
        Returns:
            Target encoder layer module
        """
        layer_num = int(layer_name.split('_')[-1])
        return model.encoder.layers[layer_num]

def parse_layer_block(layer_block_str: str) -> Tuple[str, int]:
    """Parse layer string into layer number.
    
    Args:
        layer_block_str: String in format 'encoder_layer_X'
        
    Returns:
        Tuple of (layer_name, layer_num)
        
    Example:
        >>> parse_layer_block('encoder_layer_3')
        ('encoder_layer', 3)
    """
    if not layer_block_str.startswith('encoder_layer_'):
        raise ValueError(f"Invalid layer format: {layer_block_str}")
    
    layer_num = int(layer_block_str.split('_')[-1])
    return 'encoder_layer', layer_num

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
                  layer_name: str, 
                  val_dataset: Dataset,
                  image_idx: int,
                  feature_idx: int) -> torch.Tensor:
    """Get activation feature map for specific feature in ViT.
    
    Args:
        model: The model
        layer_name: Layer identifier (e.g., 'encoder_layer_3')
        val_dataset: Validation dataset
        image_idx: Index of image
        feature_idx: Feature to extract
        
    Returns:
        Feature activation tensor
    """
    activation = []
    _, layer_num = parse_layer_block(layer_name)
    
    def hook_fn(module, input, output):
        # For ViT, output shape is (batch_size, sequence_length, hidden_dim)
        # Extract specific feature across all patch tokens
        feature_activation = output[0, :, feature_idx]
        activation.append(feature_activation.detach().cpu())
    
    img, _ = val_dataset[image_idx]
    img = img.unsqueeze(0).cuda()
    
    target_layer = LayerActivationHelper.get_target_layer(model, layer_name)
    
    with torch.no_grad():
        handle = target_layer.register_forward_hook(hook_fn)
        _ = model(img)
        handle.remove()
        
    # Reshape activation to 2D for visualization
    patch_size = int(np.sqrt(activation[0].shape[0] - 1))  # -1 for CLS token
    act_map = activation[0][1:].reshape(patch_size, patch_size)
    return act_map

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
    
    img, label = val_dataset[tgt_sample]
    img = img.permute(1, 2, 0).numpy()
    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img = np.clip(img, 0, 1)
    imgs.append((tgt_sample, img))
    
    return imgs

def get_activation_subset(model: nn.Module,
                        layer_name: str,
                        val_dataset: Dataset,
                        indices: List[int],
                        feature_idx: int,
                        batch_size: int = 32) -> torch.Tensor:
    """Get full feature maps for specified features without taking mean.
    
    Args:
        model: The neural network model
        layer_name: Layer identifier (e.g., 'encoder_layer_3')
        val_dataset: Validation dataset
        indices: List of sample indices
        feature_idx: Feature to extract
        batch_size: Batch size for processing
        
    Returns:
        torch.Tensor: Activation maps for the specified feature
    """
    activations = []
    _, layer_num = parse_layer_block(layer_name)
    
    def hook_fn(module, input, output):
        # For ViT, output shape is (batch_size, sequence_length, hidden_dim)
        # Extract specific feature across all patch tokens
        feature_activation = output[:, :, feature_idx]
        activations.append(feature_activation.detach().cpu())
    
    subset = Subset(val_dataset, indices)
    subset_loader = DataLoader(subset, batch_size=batch_size, shuffle=False, 
                             num_workers=4, pin_memory=True)
    
    target_layer = LayerActivationHelper.get_target_layer(model, layer_name)
    
    with torch.no_grad(), target_layer.register_forward_hook(hook_fn):
        for batch in subset_loader:
            images = batch[0].cuda()
            _ = model(images)
            
    return torch.cat(activations, dim=0)

def get_output_with_modified_activation(model: nn.Module,
                                      src_layer_name: str,
                                      tgt_layer_name: str,
                                      image: torch.Tensor,
                                      original_act: torch.Tensor,
                                      feature_idx: int) -> torch.Tensor:
    """Get output by modifying a specific layer's output for a specific feature.
    
    Args:
        model: The neural network model
        source_layer_name: Source layer identifier (e.g., 'encoder_layer_3')
        target_layer_name: Target layer identifier (e.g., 'encoder_layer_4')
        image: Input image tensor
        original_act: Original activation to modify
        feature_idx: Feature to modify
        
    Returns:
        torch.Tensor: Modified output activation
    """
    outputs = []
    _, src_layer_num = parse_layer_block(src_layer_name)
    _, tgt_layer_num = parse_layer_block(tgt_layer_name)
    
    def hook_fn_source_layer(module, input, output):
        output_clone = output.clone()
        output_clone[:, :, feature_idx] = original_act
        return output_clone
    
    def hook_fn_target_layer(module, input, output):
        outputs.append(output.detach().cpu())
    
    src_layer = LayerActivationHelper.get_target_layer(model, src_layer_name)
    tgt_layer = LayerActivationHelper.get_target_layer(model, tgt_layer_name)
    
    handle1 = src_layer.register_forward_hook(hook_fn_source_layer)
    handle2 = tgt_layer.register_forward_hook(hook_fn_target_layer)
    
    with torch.no_grad():
        _ = model(image.unsqueeze(0).cuda())
    
    handle1.remove()
    handle2.remove()

    return outputs[0]

def analyze_channel_score(model: nn.Module,
                        src_layer_name: str,
                        tgt_layer_name: str,
                        val_dataset: Dataset,
                        feature_idx: int,
                        src_feature: int) -> torch.Tensor:
    """Analyzes the impact of a feature by comparing original, masked, and amplified activations.
    
    Args:
        model: The neural network model
        source_layer_name: Source layer identifier (e.g., 'encoder_layer_3')
        target_layer_name: Target layer identifier (e.g., 'encoder_layer_4')
        val_dataset: Validation dataset
        feature_idx: Index of the feature to analyze
        src_feature: Index of the feature to analyze
    
    Returns:
        torch.Tensor: Impact scores for the feature
    """
    # Get the original image
    img, _ = val_dataset[feature_idx]

    # Get original activation
    input_A = get_activation_subset(
        model=model,
        layer_name=src_layer_name,
        val_dataset=val_dataset,
        indices=[feature_idx],
        feature_idx=src_feature
    ).squeeze()

    # Create masked and amplified versions
    masked_A = torch.zeros_like(input_A)
    amplified_A = input_A * 2

    # Get outputs for different activation versions
    output_org_A = get_output_with_modified_activation(
        model=model,
        src_layer_name=src_layer_name,
        tgt_layer_name=tgt_layer_name,
        image=img,
        original_act=input_A,
        feature_idx=src_feature
    )
    
    output_masked_A = get_output_with_modified_activation(
        model=model,
        src_layer_name=src_layer_name,
        tgt_layer_name=tgt_layer_name,
        image=img,
        original_act=masked_A,
        feature_idx=src_feature
    )
    
    output_amplified_A = get_output_with_modified_activation(
        model=model,
        src_layer_name=src_layer_name,
        tgt_layer_name=tgt_layer_name,
        image=img,
        original_act=amplified_A,
        feature_idx=src_feature
    )

    # Calculate differences
    diff_masked = (output_org_A[0] - output_masked_A[0]).mean(dim=0)
    diff_amplified = (output_amplified_A[0] - output_org_A[0]).mean(dim=0)

    # Filter negative values
    diff_masked = torch.where(diff_masked < 0, torch.zeros_like(diff_masked), diff_masked)
    diff_amplified = torch.where(diff_amplified < 0, torch.zeros_like(diff_amplified), diff_amplified)

    # Calculate final scores
    scores = diff_masked
    
    return scores.clone().detach()