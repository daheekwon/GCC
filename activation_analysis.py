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
    """Helper class to manage layer access in different model architectures."""
    
    def __init__(self, model, model_type: str):
        self.model = model
        self.model_type = model_type
    
    def get_target_layer(self, layer_name: str) -> nn.Module:
        """Get target layer based on model type."""
        if self.model_type == 'resnet':
            return self._get_resnet_layer(layer_name)
        elif self.model_type == 'vit':
            return self._get_vit_layer(layer_name)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def _get_resnet_layer(self, layer_name: str) -> nn.Module:
        """Get ResNet layer."""
        if isinstance(layer_name, tuple):
            layer_name, block_idx = layer_name
        else:
            layer_name, block_idx = parse_layer_block(layer_name)
            
        layer_mapping = {
            'layer1': self.model.layer1,
            'layer2': self.model.layer2,
            'layer3': self.model.layer3,
            'layer4': self.model.layer4
        }
        
        if layer_name not in layer_mapping:
            raise ValueError(f"Invalid ResNet layer name: {layer_name}")
        
        return layer_mapping[layer_name][block_idx]
    
    def _get_vit_layer(self, layer_name: str) -> nn.Module:
        """Get ViT layer."""
        if isinstance(layer_name, tuple):
            _, layer_idx = layer_name
        else:
            _, layer_idx = parse_layer_block(layer_name)
            
        try:
            # Access the transformer encoder layers
            return self.model.encoder.layers[layer_idx]
        except (IndexError, AttributeError):
            raise ValueError(f"Invalid ViT layer index: {layer_idx}")

def parse_layer_block(layer_name: str) -> Tuple[str, int]:
    """Parse layer name into components based on model architecture."""
    if layer_name.startswith('layer'):  # ResNet format
        parts = layer_name.split('_')
        if len(parts) != 2 or not parts[1].startswith('block'):
            raise ValueError(f"Invalid ResNet layer format: {layer_name}")
        return parts[0], int(parts[1][5:])
    elif layer_name.startswith('encoder_layer'):  # ViT format
        try:
            layer_num = int(layer_name.split('_')[-1])
            return 'encoder_layer', layer_num
        except (IndexError, ValueError):
            raise ValueError(f"Invalid ViT layer format: {layer_name}")
    else:  # Unknown format
        raise ValueError(f"Unrecognized layer format: {layer_name}")

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
                  channel_idx: int,
                  model_type: str = 'resnet') -> torch.Tensor:
    """Get activation feature map for specific channel."""
    activation = []
    
    def hook_fn(module, input, output):
        if isinstance(output, tuple):  # Handle ViT output format
            output = output[0]  # Get the main output tensor
        channel_activation = output[:, channel_idx, :, :] if model_type == 'resnet' else output[:, :, channel_idx]
        activation.append(channel_activation.detach().cpu())

    img, _ = val_dataset[image_idx]
    img = img.unsqueeze(0).cuda()
    
    helper = LayerActivationHelper(model, model_type)
    target_block = helper.get_target_layer(layer_block)
    
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
                        model_type: str = 'resnet',
                        batch_size: int = 32) -> torch.Tensor:
    """Get full feature maps for specified channels without taking mean."""
    activations = []
    
    def hook_fn(module, input, output):
        if isinstance(output, tuple):  # Handle ViT output format
            output = output[0]  # Get the main output tensor
        channel_activation = output[:, channel_idx, :, :] if model_type == 'resnet' else output[:, :, channel_idx]
        activations.append(channel_activation.detach().cpu())
    
    subset = Subset(val_dataset, indices)
    subset_loader = DataLoader(subset, batch_size=batch_size, shuffle=False, 
                             num_workers=4, pin_memory=True)
    
    helper = LayerActivationHelper(model, model_type)
    target_block = helper.get_target_layer(layer_block)
    
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
                                      channel_idx: int,
                                      model_type: str = 'resnet') -> torch.Tensor:
    """Get output by modifying a specific block's output for a specific channel."""
    outputs = []
    src_layer_name = parse_layer_block(src_layer_block)
    tgt_layer_name = parse_layer_block(tgt_layer_block)
    
    def hook_fn_source_block(module, input, output):
        output_clone = output.clone()
        if model_type == 'resnet':
            output_clone[:, channel_idx, :, :] = original_act
        elif model_type == 'vit':  # ViT
            output_clone[:, :, channel_idx] = original_act
        return output_clone
    
    def hook_fn_target_block(module, input, output):
            outputs.append(output.detach().cpu())
    
    helper = LayerActivationHelper(model, model_type)
    src_block = helper.get_target_layer(src_layer_name)
    tgt_block = helper.get_target_layer(tgt_layer_name)

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
                        src_channel: int,
                        model_type: str = 'resnet') -> torch.Tensor:
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
        model_type=model_type,
        layer_block=src_layer_block,
        val_dataset=val_dataset,
        indices=[channel_idx[i][1][-1]],
        channel_idx=src_channel
    ).squeeze()

    
    # Create masked and amplified versions
    masked_A = torch.zeros_like(input_A)
    # amplified_A = input_A * 2

    # Get outputs for different activation versions
    output_org_A = get_output_with_modified_activation(
        model=model,
        model_type=model_type,
        src_layer_block=src_layer_block,
        tgt_layer_block=tgt_layer_block,
        image=img,
        original_act=input_A,
        channel_idx=src_channel
    )
    
    output_masked_A = get_output_with_modified_activation(
        model=model,
        model_type=model_type,
        src_layer_block=src_layer_block,
        tgt_layer_block=tgt_layer_block,
        image=img,
        original_act=masked_A,
        channel_idx=src_channel
    )
    
    # output_amplified_A = get_output_with_modified_activation(
    #     model=model,
    #     model_type=model_type,
    #     src_layer_block=src_layer_block,
    #     tgt_layer_block=tgt_layer_block,
    #     image=img,
    #     original_act=amplified_A,
    #     channel_idx=src_channel
    # )

    # Calculate differences
    
    if model_type == 'resnet':
        # For ResNet: average over spatial dimensions (height, width)
        diff_masked = (output_org_A[0] - output_masked_A[0]).mean(dim=(1,2))
        # diff_amplified = (output_amplified_A[0] - output_org_A[0]).mean(dim=(1,2))
    else:
        # For ViT: average over sequence length dimension
        diff_masked = (output_org_A[0] - output_masked_A[0]).mean(dim=0)
        # diff_amplified = (output_amplified_A[0] - output_org_A[0]).mean(dim=0)

    # Filter negative values
    diff_masked = torch.where(diff_masked < 0, torch.zeros_like(diff_masked), diff_masked)
    # diff_amplified = torch.where(diff_amplified < 0, torch.zeros_like(diff_amplified), diff_amplified)

    # Calculate final scores
    scores = diff_masked
    
    return scores.clone().detach()