"""Gradio interface for audio bandwidth extension using AP-BWE models."""

from __future__ import absolute_import, division, print_function, unicode_literals
import os
import json
import torch
import time
import tempfile
import torchaudio
import torchaudio.functional as aF
import gradio as gr
from ap_bwe.env import AttrDict
from ap_bwe.datasets.dataset import amp_pha_stft, amp_pha_istft
from ap_bwe.models.model import APNet_BWE_Model
import soundfile as sf

# Global variables
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_checkpoint(filepath, device):
    """Load model checkpoint from file."""
    assert os.path.isfile(filepath)
    print(f"Loading '{filepath}'")
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict

def process_audio(audio_path, checkpoint_path):
    """Process audio using the specified model checkpoint."""
    # Load configuration from the checkpoint directory
    config_file = os.path.join(os.path.split(checkpoint_path)[0], 'config.json')
    with open(config_file) as f:
        json_config = json.loads(f.read())
    h = AttrDict(json_config)
    
    # Initialize model
    model = APNet_BWE_Model(h).to(device)
    
    # Load checkpoint
    state_dict = load_checkpoint(checkpoint_path, device)
    model.load_state_dict(state_dict['generator'])
    
    # Load audio
    audio, orig_sampling_rate = torchaudio.load(audio_path)
    audio = audio.to(device)
    
    # Resample to high resolution and low resolution
    audio_hr = aF.resample(audio, orig_freq=orig_sampling_rate, new_freq=h.hr_sampling_rate)
    audio_lr = aF.resample(audio, orig_freq=orig_sampling_rate, new_freq=h.lr_sampling_rate)
    audio_lr = aF.resample(audio_lr, orig_freq=h.lr_sampling_rate, new_freq=h.hr_sampling_rate)
    audio_lr = audio_lr[:, : audio_hr.size(1)]
    
    # Process with model
    model.eval()
    with torch.no_grad():
        pred_start = time.time()
        amp_nb, pha_nb, com_nb = amp_pha_stft(audio_lr, h.n_fft, h.hop_size, h.win_size)
        amp_wb_g, pha_wb_g, com_wb_g = model(amp_nb, pha_nb)
        audio_hr_g = amp_pha_istft(amp_wb_g, pha_wb_g, h.n_fft, h.hop_size, h.win_size)
        processing_time = time.time() - pred_start
    
    # Save result to temporary file
    output_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    output_path = output_file.name
    sf.write(output_path, audio_hr_g.squeeze().cpu().numpy(), h.hr_sampling_rate, 'PCM_16')
    
    return output_path, f"Processing time: {processing_time:.4f} seconds"

def process_audio_16k(audio_path, checkpoint_path):
    """Process audio using 16k model."""
    return process_audio(audio_path, checkpoint_path)

def process_audio_48k(audio_path, checkpoint_path):
    """Process audio using 48k model.""" 
    return process_audio(audio_path, checkpoint_path)

def create_interface():
    """Create and launch the Gradio interface."""
    with gr.Blocks(title="AP-BWE Audio Bandwidth Extension") as demo:
        gr.Markdown("# AP-BWE: Audio Bandwidth Extension")
        gr.Markdown("Upload audio files and enhance their bandwidth using AP-BWE models.")
        
        with gr.Tab("16kHz Bandwidth Extension"):
            gr.Markdown("## 16kHz Bandwidth Extension")
            gr.Markdown("Extend bandwidth of lower resolution audio to 16kHz")
            
            with gr.Row():
                with gr.Column():
                    audio_input_16k = gr.Audio(label="Input Audio", type="filepath")
                    checkpoint_input_16k = gr.Textbox(
                        label="Checkpoint Path", 
                        placeholder="Path to checkpoint file (e.g., checkpoints/2kto16k/g_2kto16k.zip)",
                        value="checkpoints/2kto16k/g_2kto16k.zip"
                    )
                    process_btn_16k = gr.Button("Process")
                
                with gr.Column():
                    audio_output_16k = gr.Audio(label="Enhanced Audio")
                    info_output_16k = gr.Textbox(label="Processing Info")
            
            process_btn_16k.click(
                fn=process_audio_16k,
                inputs=[audio_input_16k, checkpoint_input_16k],
                outputs=[audio_output_16k, info_output_16k]
            )
        
        with gr.Tab("48kHz Bandwidth Extension"):
            gr.Markdown("## 48kHz Bandwidth Extension")
            gr.Markdown("Extend bandwidth of lower resolution audio to 48kHz")
            
            with gr.Row():
                with gr.Column():
                    audio_input_48k = gr.Audio(label="Input Audio", type="filepath")
                    checkpoint_input_48k = gr.Textbox(
                        label="Checkpoint Path", 
                        placeholder="Path to checkpoint file (e.g., checkpoints/16kto48k/g_16kto48k.zip)",
                        value="checkpoints/16kto48k/g_16kto48k.zip"
                    )
                    process_btn_48k = gr.Button("Process")
                
                with gr.Column():
                    audio_output_48k = gr.Audio(label="Enhanced Audio")
                    info_output_48k = gr.Textbox(label="Processing Info")
            
            process_btn_48k.click(
                fn=process_audio_48k,
                inputs=[audio_input_48k, checkpoint_input_48k],
                outputs=[audio_output_48k, info_output_48k]
            )
        
        gr.Markdown("### About")
        gr.Markdown("""
        This interface uses AP-BWE (Amplitude-Phase Bandwidth Extension) to enhance audio quality.
        
        - 16kHz BWE: Supports upsampling from 2kHz, 4kHz, or 8kHz to 16kHz
        - 48kHz BWE: Supports upsampling from 8kHz, 12kHz, 16kHz, or 24kHz to 48kHz
        
        Use the appropriate tab and model checkpoint based on your source audio and desired output.
        """)
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=True)
