"""API functions for audio bandwidth extension using AP-BWE models."""

import os
import json
import torch
import torchaudio
import torchaudio.functional as aF
from ap_bwe.env import AttrDict
from ap_bwe.datasets.dataset import amp_pha_stft, amp_pha_istft
from ap_bwe.models.model import APNet_BWE_Model
from safetensors.torch import load_file

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_config_and_model(checkpoint_path):
    """Load configuration and model from checkpoint path."""
    assert os.path.isfile(checkpoint_path)
    
    config_file = os.path.join(os.path.split(checkpoint_path)[0], 'config.json')
    with open(config_file) as f:
        config = AttrDict(json.loads(f.read()))
    
    if checkpoint_path.endswith('.safetensors'):
        device_str = 'cpu' if device.type == 'cpu' else 'cuda'
        state_dict = load_file(checkpoint_path, device=device_str)
    else:
        checkpoint_dict = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint_dict['generator']
    
    model = APNet_BWE_Model(config).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    
    return config, model

def process_audio(audio_path, checkpoint_path):
    """Process audio using the specified model checkpoint.
    
    Args:
        audio_path: Path to the audio file to process
        checkpoint_path: Path to the model checkpoint
    """
    config, model = load_config_and_model(checkpoint_path)
    
    audio, orig_sampling_rate = torchaudio.load(audio_path)
    audio = audio.to(device)
    
    audio_hr = aF.resample(audio, orig_freq=orig_sampling_rate, new_freq=config.hr_sampling_rate)
    audio_lr = aF.resample(audio, orig_freq=orig_sampling_rate, new_freq=config.lr_sampling_rate)
    audio_lr = aF.resample(audio_lr, orig_freq=config.lr_sampling_rate, new_freq=config.hr_sampling_rate)
    audio_lr = audio_lr[:, :audio_hr.size(1)]
    
    with torch.no_grad():
        amp_nb, pha_nb, _ = amp_pha_stft(audio_lr, config.n_fft, config.hop_size, config.win_size)
        amp_wb_g, pha_wb_g, _ = model(amp_nb, pha_nb)
        audio_hr_g = amp_pha_istft(amp_wb_g, pha_wb_g, config.n_fft, config.hop_size, config.win_size)
    
    return config.hr_sampling_rate, audio_hr_g.squeeze().cpu().numpy()


