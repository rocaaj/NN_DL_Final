"""
precompute_mfccs.py

Author: Anthony Roca
---------------------

This script preprocesses audio files from the UrbanSound8K dataset and extracts
Mel-frequency cepstral coefficients (MFCCs) for each clip. The resulting MFCC tensors
are normalized, padded/truncated to a fixed length, and cached as .pt files for fast loading
during training.

Key Features:
-------------
- Input: UrbanSound8K WAV files located in fold subdirectories (e.g., fold1, fold2, ...)
- Output: Normalized MFCC tensors saved to ../data/mfcc_cache/
- MFCC Parameters:
    * n_mfcc: 40
    * FFT size: 2048
    * Hop length: 512
    * n_mels: 128
- Normalization: Per-feature standardization
- Padding: Zero-padded or truncated to a fixed MAX_LEN of 200 frames
- Error Handling: Skips and logs any failed audio files

Usage:
------
$ python precompute_mfcc.py

Output:
-------
- A tensor file for each audio clip saved as {fold}_{filename}.pt
- Printed status message on completion

Acknowledgment:
---------------
This implementation benefited from iterative debugging and generalization guidance provided by OpenAI's ChatGPT.
"""


import os
import torch
import torchaudio
import pandas as pd
import numpy as np
from tqdm import tqdm

# -----------------------------
# Config
# -----------------------------
DATA_DIR = '../data/'
METADATA_CSV = '../data/UrbanSound8K.csv'
MFCC_CACHE_DIR = '../data/mfcc_cache/'
N_MFCC = 40
MAX_LEN = 200

os.makedirs(MFCC_CACHE_DIR, exist_ok=True)

# -----------------------------
# Load Metadata
# -----------------------------
df = pd.read_csv(METADATA_CSV)
df = df.rename(columns={'slice_file_name': 'filename', 'class': 'label'})

# -----------------------------
# Process and Save MFCCs
# -----------------------------
for idx, row in tqdm(df.iterrows(), total=len(df)):
    fold = row['fold']
    filename = row['filename']
    file_path = os.path.join(DATA_DIR, f"fold{fold}", filename)

    try:
        waveform, sr = torchaudio.load(file_path)
        waveform = waveform.mean(dim=0, keepdim=True)  # convert to mono
        mfcc = torchaudio.transforms.MFCC(
            sample_rate=sr,
            n_mfcc=N_MFCC,
            melkwargs={"n_fft": 2048, "hop_length": 512, "n_mels": 128}
        )(waveform)

        mfcc = mfcc.squeeze(0).T  # shape: (time, n_mfcc)
        mfcc = (mfcc - mfcc.mean(0)) / (mfcc.std(0) + 1e-6)  # normalize

        if mfcc.shape[0] < MAX_LEN:
            pad = torch.zeros((MAX_LEN - mfcc.shape[0], N_MFCC))
            mfcc = torch.cat([mfcc, pad], dim=0)
        else:
            mfcc = mfcc[:MAX_LEN, :]

        save_path = os.path.join(MFCC_CACHE_DIR, f"{fold}_{filename}.pt")
        torch.save(mfcc, save_path)
    except Exception as e:
        print(f"Failed to process {file_path}: {e}")

print("✅ MFCC precomputation complete. Tensors saved to:", MFCC_CACHE_DIR)
