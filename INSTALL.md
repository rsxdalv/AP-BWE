# AP-BWE Installation Guide

AP-BWE (Amplitude and Phase Bandwidth Extension) can be installed as a Python package, allowing you to use it in your own projects or from the command line.

## Installation

You can install the package directly from the repository:

```bash
# Clone the repository
git clone https://github.com/yxlu-0102/AP-BWE.git
cd AP-BWE

# Install the package
pip install -e .
```

This will install the package in development mode, which means changes to the code will be immediately reflected without needing to reinstall.

## Requirements

AP-BWE requires Python 3.9 or later. The main dependencies are:

- PyTorch >= 1.9.0
- torchaudio
- numpy
- matplotlib
- librosa
- soundfile
- rich
- joblib

## Command-line Usage

After installation, you can use AP-BWE from the command line in two ways:

### Using the unified command:

```bash
# For 16kHz bandwidth extension
ap-bwe train-16k --config /path/to/config_2kto16k.json --checkpoint_path /path/to/checkpoints/AP-BWE_2kto16k
ap-bwe inference-16k --checkpoint_file /path/to/checkpoint/g_2kto16k --input_wavs_dir /path/to/input/wavs --output_dir /path/to/output

# For 48kHz bandwidth extension
ap-bwe train-48k --config /path/to/config_16kto48k.json --checkpoint_path /path/to/checkpoints/AP-BWE_16kto48k
ap-bwe inference-48k --checkpoint_file /path/to/checkpoint/g_16kto48k --input_wavs_dir /path/to/input/wavs --output_dir /path/to/output
```

### Using direct commands:

```bash
# For 16kHz bandwidth extension
ap-bwe-train-16k --config /path/to/config_2kto16k.json --checkpoint_path /path/to/checkpoints/AP-BWE_2kto16k
ap-bwe-inference-16k --checkpoint_file /path/to/checkpoint/g_2kto16k --input_wavs_dir /path/to/input/wavs --output_dir /path/to/output

# For 48kHz bandwidth extension
ap-bwe-train-48k --config /path/to/config_16kto48k.json --checkpoint_path /path/to/checkpoints/AP-BWE_16kto48k
ap-bwe-inference-48k --checkpoint_file /path/to/checkpoint/g_16kto48k --input_wavs_dir /path/to/input/wavs --output_dir /path/to/output
```

## Python API Usage

You can also use AP-BWE as a Python package in your own code:

```python
import torch
import torchaudio
from ap_bwe.models.model import APNet_BWE_Model
from ap_bwe.datasets.dataset import amp_pha_stft, amp_pha_istft
from ap_bwe.env import AttrDict
import json

# Load configuration
with open('configs/config_2kto16k.json') as f:
    config = json.load(f)
h = AttrDict(config)

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = APNet_BWE_Model(h).to(device)

# Load checkpoint
checkpoint = torch.load('checkpoints/g_2kto16k', map_location=device)
model.load_state_dict(checkpoint['generator'])
model.eval()

# Process audio
audio, sr = torchaudio.load('input.wav')
audio = audio.to(device)

# Resample to low resolution
audio_lr = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=h.lr_sampling_rate)
audio_lr = torchaudio.functional.resample(audio_lr, orig_freq=h.lr_sampling_rate, new_freq=h.hr_sampling_rate)

# Process through the model
with torch.no_grad():
    mag_lr, pha_lr, com_lr = amp_pha_stft(audio_lr, h.n_fft, h.hop_size, h.win_size)
    mag_hr, pha_hr, com_hr = model(mag_lr, pha_lr)
    audio_hr = amp_pha_istft(mag_hr, pha_hr, h.n_fft, h.hop_size, h.win_size)

# Save output
torchaudio.save('output.wav', audio_hr.cpu(), h.hr_sampling_rate)
```
