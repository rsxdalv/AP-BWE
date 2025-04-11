"""Utility functions for model format conversion and loading."""

import os
import torch
from safetensors.torch import save_file, load_file

def convert_zip_to_ckpt(zip_path):
    """Convert a .zip model file to .ckpt format."""
    if not os.path.exists(zip_path):
        print(f"Source file {zip_path} does not exist")
        return None
    
    # Create output path by replacing .zip with .ckpt
    ckpt_path = zip_path.replace('.zip', '.ckpt')
    
    # Skip if ckpt already exists
    if os.path.exists(ckpt_path):
        print(f"CKPT file already exists at {ckpt_path}")
        return ckpt_path
    
    print(f"Converting {zip_path} to CKPT format...")
    try:
        # Load the model from zip
        state_dict = torch.load(zip_path, map_location='cpu')
        
        # Save as ckpt (essentially just renaming, but ensuring it loads correctly)
        torch.save(state_dict, ckpt_path)
        print(f"Successfully converted to {ckpt_path}")
        return ckpt_path
    except Exception as e:
        print(f"Error converting to CKPT: {e}")
        return None

def convert_ckpt_to_safetensors(ckpt_path):
    """Convert a .ckpt model file to .safetensors format."""
    if not os.path.exists(ckpt_path):
        print(f"Source file {ckpt_path} does not exist")
        return None
    
    # Create output path by replacing .ckpt with .safetensors
    safetensors_path = ckpt_path.replace('.ckpt', '.safetensors')
    
    # Skip if safetensors already exists
    if os.path.exists(safetensors_path):
        print(f"SafeTensors file already exists at {safetensors_path}")
        return safetensors_path
    
    print(f"Converting {ckpt_path} to SafeTensors format...")
    try:
        # Load the model from ckpt
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        
        # Extract the generator state dict
        if 'generator' in checkpoint:
            state_dict = checkpoint['generator']
        else:
            state_dict = checkpoint  # Assume it's already a state dict
        
        # Save as safetensors
        save_file(state_dict, safetensors_path)
        print(f"Successfully converted to {safetensors_path}")
        return safetensors_path
    except Exception as e:
        print(f"Error converting to SafeTensors: {e}")
        return None

def convert_model_to_all_formats(model_path):
    """Convert a model file to all formats (zip → ckpt → safetensors)."""
    if model_path.endswith('.zip'):
        # Convert zip to ckpt
        ckpt_path = convert_zip_to_ckpt(model_path)
        if ckpt_path:
            # Convert ckpt to safetensors
            safetensors_path = convert_ckpt_to_safetensors(ckpt_path)
            return model_path, ckpt_path, safetensors_path
    elif model_path.endswith('.ckpt'):
        # Convert ckpt to safetensors
        safetensors_path = convert_ckpt_to_safetensors(model_path)
        return None, model_path, safetensors_path
    
    return model_path, None, None
