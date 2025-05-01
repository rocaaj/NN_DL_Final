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
# Configuration
# -----------------------------
DATA_DIR = '../data/'                        # Directory containing UrbanSound8K fold subfolders
METADATA_CSV = '../data/UrbanSound8K.csv'    # Metadata with labels and filenames
MFCC_CACHE_DIR = '../data/mfcc_cache/'       # Directory to store precomputed MFCC tensors
N_MFCC = 40                                  # Number of MFCC coefficients
MAX_LEN = 200                                # Max length in time steps for MFCC feature matrices

# Ensure cache directory exists
os.makedirs(MFCC_CACHE_DIR, exist_ok=True)

# -----------------------------
# Load and clean metadata
# -----------------------------
df = pd.read_csv(METADATA_CSV)
df = df.rename(columns={'slice_file_name': 'filename', 'class': 'label'})  # unify naming conventions

# -----------------------------
# Process and save MFCC features
# -----------------------------
for idx, row in tqdm(df.iterrows(), total=len(df)):
    fold = row['fold']
    filename = row['filename']
    file_path = os.path.join(DATA_DIR, f"fold{fold}", filename)

    try:
        # Load waveform and convert stereo to mono if needed
        waveform, sr = torchaudio.load(file_path)
        waveform = waveform.mean(dim=0, keepdim=True)  # (1, num_samples)

        # Compute MFCCs
        mfcc = torchaudio.transforms.MFCC(
            sample_rate=sr,
            n_mfcc=N_MFCC,
            melkwargs={
                "n_fft": 2048,
                "hop_length": 512,
                "n_mels": 128
            }
        )(waveform)  # shape: (1, n_mfcc, time)

        mfcc = mfcc.squeeze(0).T  # reshape to (time, n_mfcc)

        # Normalize each coefficient across time (z-score normalization)
        mfcc = (mfcc - mfcc.mean(0)) / (mfcc.std(0) + 1e-6)

        # Pad or truncate to MAX_LEN
        if mfcc.shape[0] < MAX_LEN:
            pad = torch.zeros((MAX_LEN - mfcc.shape[0], N_MFCC))
            mfcc = torch.cat([mfcc, pad], dim=0)
        else:
            mfcc = mfcc[:MAX_LEN, :]

        # Save tensor to disk using a consistent filename format
        save_path = os.path.join(MFCC_CACHE_DIR, f"{fold}_{filename}.pt")
        torch.save(mfcc, save_path)

    except Exception as e:
        # Gracefully handle files that fail to load or process
        print(f"Failed to process {file_path}: {e}")

print("MFCC precomputation complete. Tensors saved to:", MFCC_CACHE_DIR)
