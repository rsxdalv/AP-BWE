"""Gradio interface for audio bandwidth extension using AP-BWE models."""

import os
import json
import torch
import torchaudio
import torchaudio.functional as aF
import gradio as gr
from huggingface_hub import hf_hub_download
from ap_bwe.env import AttrDict
from ap_bwe.datasets.dataset import amp_pha_stft, amp_pha_istft
from ap_bwe.models.model import APNet_BWE_Model
from safetensors.torch import load_file


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_DIR = "./data/models/ap_bwe"
HUGGINGFACE_REPO = "rsxdalv/AP-BWE"

def ensure_model_downloaded(model_name):
    """Check if model exists locally, download from HuggingFace if not."""
    # Extract directory and base name
    model_dir = os.path.dirname(model_name)
    base_name = os.path.basename(model_name)
    
    # Update paths to include weights in the directory structure
    full_model_dir = os.path.join(MODEL_DIR, "weights", model_dir)
    safetensors_path = os.path.join(MODEL_DIR, "weights", model_dir, base_name)
    config_path = os.path.join(full_model_dir, "config.json")
    
    # Create directory if it doesn't exist
    os.makedirs(full_model_dir, exist_ok=True)
    
    # Download files if they don't exist locally
    files_to_download = [
        (f"weights/{model_dir}/{base_name}", safetensors_path),
        (f"weights/{model_dir}/config.json", config_path)
    ]
    
    for remote_path, local_path in files_to_download:
        if not os.path.isfile(local_path):
            try:
                hf_hub_download(
                    repo_id=HUGGINGFACE_REPO,
                    filename=remote_path,
                    local_dir=MODEL_DIR,
                    local_dir_use_symlinks=False
                )
                print(f"Downloaded {remote_path} successfully.")
            except Exception as e:
                print(f"Error downloading {remote_path}: {e}")
                return False
    
    return True

def load_config_and_model(checkpoint_path):
    """Load configuration and model from checkpoint path."""
    assert os.path.isfile(checkpoint_path)
    
    # Load configuration from the checkpoint directory
    config_file = os.path.join(os.path.split(checkpoint_path)[0], 'config.json')
    with open(config_file) as f:
        config = AttrDict(json.loads(f.read()))
    
    # Load checkpoint based on its format
    if checkpoint_path.endswith('.safetensors'):
        # Load safetensors format
        device_str = 'cpu' if device.type == 'cpu' else 'cuda'
        state_dict = load_file(checkpoint_path, device=device_str)
    else:
        # Load traditional PyTorch format (.zip or .ckpt)
        checkpoint_dict = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint_dict['generator']
    
    # Initialize model and load weights
    model = APNet_BWE_Model(config).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    
    return config, model

def process_audio(audio_path, checkpoint_path):
    """Process audio using the specified model checkpoint."""
    # Load config and model
    config, model = load_config_and_model(checkpoint_path)
    
    # Load and prepare audio
    audio, orig_sampling_rate = torchaudio.load(audio_path)
    audio = audio.to(device)
    
    # Resample to target rates
    audio_hr = aF.resample(audio, orig_freq=orig_sampling_rate, new_freq=config.hr_sampling_rate)
    audio_lr = aF.resample(audio, orig_freq=orig_sampling_rate, new_freq=config.lr_sampling_rate)
    audio_lr = aF.resample(audio_lr, orig_freq=config.lr_sampling_rate, new_freq=config.hr_sampling_rate)
    audio_lr = audio_lr[:, :audio_hr.size(1)]
    
    # Process with model
    with torch.no_grad():
        amp_nb, pha_nb, _ = amp_pha_stft(audio_lr, config.n_fft, config.hop_size, config.win_size)
        amp_wb_g, pha_wb_g, _ = model(amp_nb, pha_nb)
        audio_hr_g = amp_pha_istft(amp_wb_g, pha_wb_g, config.n_fft, config.hop_size, config.win_size)
    
    # Return audio in Gradio format
    return (config.hr_sampling_rate, audio_hr_g.squeeze().cpu().numpy())

def get_default_checkpoint(model_type):
    """Get the default checkpoint path based on model type and ensure it's downloaded."""
    # Map model types to their respective safetensors paths
    model_paths = {
        "16kHz (2kHz input)": "2kto16k/g_2kto16k.safetensors",
        "16kHz (4kHz input)": "4kto16k/g_4kto16k.safetensors",
        "16kHz (8kHz input)": "8kto16k/g_8kto16k.safetensors",
        "48kHz (8kHz input)": "8kto48k/g_8kto48k.safetensors",
        "48kHz (12kHz input)": "12kto48k/g_12kto48k.safetensors",
        "48kHz (16kHz input)": "16kto48k/g_16kto48k.safetensors", 
        "48kHz (24kHz input)": "24kto48k/g_24kto48k.safetensors"
    }
    
    # Get model path or return empty string if not found
    model_path = model_paths.get(model_type, "")
    if not model_path:
        return ""
    
    # Ensure model is downloaded
    ensure_model_downloaded(model_path)
    
    # Return full path to the safetensors model
    return os.path.join(MODEL_DIR, "weights", model_path)

def create_interface():
    """Create the Gradio interface."""
    model_choices = [
        "16kHz (2kHz input)",
        "16kHz (4kHz input)",
        "16kHz (8kHz input)",
        "48kHz (8kHz input)",
        "48kHz (12kHz input)",
        "48kHz (16kHz input)",
        "48kHz (24kHz input)"
    ]
    
    with gr.Blocks(title="AP-BWE Audio Bandwidth Extension") as demo:
        gr.Markdown("# AP-BWE: Audio Bandwidth Extension")
        gr.Markdown("Upload audio files and enhance their bandwidth using AP-BWE models.")
        
        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(label="Input Audio", type="filepath")
                model_type = gr.Dropdown(label="Model Type", choices=model_choices, value=model_choices[0])
                process_btn = gr.Button("Process")
            
            with gr.Column():
                audio_output = gr.Audio(label="Enhanced Audio")
        
        def process_with_model_type(audio_path, model_type):
            if not audio_path:
                return None
                
            checkpoint_path = get_default_checkpoint(model_type)
            if not os.path.exists(checkpoint_path):
                return None
            
            return process_audio(audio_path, checkpoint_path)
            
        process_btn.click(
            fn=process_with_model_type,
            inputs=[audio_input, model_type],
            outputs=audio_output
        )
        
        gr.Markdown("### About")
        gr.Markdown("""
        This interface uses AP-BWE (Amplitude-Phase Bandwidth Extension) to enhance audio quality.
        
        - 16kHz BWE: Supports upsampling from 2kHz, 4kHz, or 8kHz to 16kHz
        - 48kHz BWE: Supports upsampling from 8kHz, 12kHz, 16kHz, or 24kHz to 48kHz
        
        Select the appropriate model based on your source audio and desired output.
        """)
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch()
