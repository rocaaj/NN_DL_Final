import os
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# 1) Provide paths manually
AUDIO_DIR = "_________________/Audio/dataset/audio"   # relative path to audio
META_CSV = "_________________/Audio/dataset/UrbanSound8K.csv"  # relative path to csv

# 2) Read metadata
df = pd.read_csv(META_CSV)
print(f"Total examples: {len(df)}")
print(df.head())

# 3) Build full file paths and check existence
def make_path(row):
    return os.path.join(AUDIO_DIR, f"fold{row.fold}", row.slice_file_name)

df["file_path"] = df.apply(make_path, axis=1)
missing = df[~df.file_path.map(os.path.exists)]
if not missing.empty:
    print("Missing files:", missing.slice_file_name.unique())
else:
    print("All files found.")

