import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
import pickle
import os
import argparse
from tqdm import tqdm

class FeatureExtractor:
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.activations = {layer: {} for layer in target_layers}
        
        # Register hooks
        for name, module in model.named_modules():
            if name in target_layers:
                module.register_forward_hook(self.get_activation(name))
    
    def get_activation(self, name):
        def hook(module, input, output):
            # For each sample in the batch
            for idx, activation in enumerate(output):
                # Convert to numpy and store mean activation across spatial dimensions
                act = activation.detach().cpu().numpy().mean(axis=(1, 2))
                self.activations[name][self.current_indices[idx]] = act
        return hook
    
    def __call__(self, x, indices):
        self.current_indices = indices
        with torch.no_grad():
            self.model(x)
        return self.activations

def get_resnet_block_names():
    return {
        'block_0': 'layer1',
        'block_1': 'layer2',
        'block_2': 'layer3',
        'block_3': 'layer4'
    }

def main():
    parser = argparse.ArgumentParser(description='Generate activation values for all channels')
    parser.add_argument('--model_name', type=str, default='resnet50')
    parser.add_argument('--dataset_name', type=str, default='imagenet')
    parser.add_argument('--output_dir', type=str, default='/data8/dahee/circuit/results')
    parser.add_argument('--batch_size', type=int, default=32)
    
    args = parser.parse_args()
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load pre-trained model
    model = models.resnet50(pretrained=True)
    model = model.to(device)
    model.eval()
    
    # Get layer names
    block_names = get_resnet_block_names()
    
    # Initialize feature extractor
    feature_extractor = FeatureExtractor(model, list(block_names.values()))
    
    # Data loading and preprocessing
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_dir = '/data/ImageNet1k/val/'
    val_dataset = datasets.ImageFolder(val_dir, transform=transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Create output directory
    base_path = f'{args.output_dir}/{args.model_name}/{args.dataset_name}'
    os.makedirs(base_path, exist_ok=True)
    output_path = f'{base_path}/activations.pkl'
    
    print("Generating activations...")
    activations = {block: {} for block in block_names.keys()}
    
    # Process all samples
    for batch_idx, (images, _) in enumerate(tqdm(val_loader)):
        images = images.to(device)
        indices = list(range(batch_idx * args.batch_size,
                           min((batch_idx + 1) * args.batch_size, len(val_dataset))))
        
        # Get activations
        batch_activations = feature_extractor(images, indices)
        
        # Store activations
        for block_name, layer_name in block_names.items():
            activations[block_name].update(batch_activations[layer_name])
    
    # Save activations
    print(f"Saving activations to {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(activations, f)
    
    print("Done!")

if __name__ == '__main__':
    main() 