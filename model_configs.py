from typing import Dict, Any
import torchvision.models as torch_models
from torchvision.models import (
    ViT_B_16_Weights,
    ViT_L_16_Weights,
    ViT_H_14_Weights
)

MODEL_CONFIGS = {
    # ResNet Configurations
    'resnet18': {
        'channels': [64, 128, 256, 512],
        'model_type': 'resnet',
        'model_fn': lambda: torch_models.resnet18(weights=torch_models.ResNet18_Weights.IMAGENET1K_V1)
    },
    'resnet34': {
        'channels': [64, 128, 256, 512],
        'model_type': 'resnet',
        'model_fn': lambda: torch_models.resnet34(weights=torch_models.ResNet34_Weights.IMAGENET1K_V1)
    },
    'resnet50': {
        'channels': [256, 512, 1024, 2048],
        'model_type': 'resnet',
        'model_fn': lambda: torch_models.resnet50(weights=torch_models.ResNet50_Weights.IMAGENET1K_V1)
    },
    'resnet101': {
        'channels': [256, 512, 1024, 2048],
        'model_type': 'resnet',
        'model_fn': lambda: torch_models.resnet101(weights=torch_models.ResNet101_Weights.IMAGENET1K_V1)
    },
    
    # ViT Configurations
    'vit': {
        'hidden_size': 768,
        'num_attention_heads': 12,
        'num_hidden_layers': 12,
        'patch_size': 16,
        'image_size': 224,
        'model_type': 'vit',
        'model_fn': lambda: torch_models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    },
    'vit_large': {
        'hidden_size': 1024,
        'num_attention_heads': 16,
        'num_hidden_layers': 24,
        'patch_size': 16,
        'image_size': 224,
        'model_type': 'vit',
        'model_fn': lambda: torch_models.vit_l_16(weights=ViT_L_16_Weights.IMAGENET1K_V1)
    },
    'vit_huge': {
        'hidden_size': 1280,
        'num_attention_heads': 16,
        'num_hidden_layers': 32,
        'patch_size': 14,
        'image_size': 224,
        'model_type': 'vit',
        'model_fn': lambda: torch_models.vit_h_14(weights=ViT_H_14_Weights.IMAGENET1K_V1)
    }
}

def get_model_config(model_name: str) -> Dict[str, Any]:
    """Get model configuration by name."""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported model: {model_name}")
    return MODEL_CONFIGS[model_name]

def create_model(model_name: str):
    """Create model instance based on model name."""
    config = get_model_config(model_name)
    return config['model_fn']()