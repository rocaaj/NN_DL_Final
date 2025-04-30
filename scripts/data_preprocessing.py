def preprocess_audio(file_path, sr_target=22050, duration=4, n_mfcc=40, mode="mfcc"):
    try:
        # Load audio
        y, sr = librosa.load(file_path, sr=None, mono=False)

        # Convert mono to stereo if needed
        if y.ndim == 1:
            y = np.stack([y, y], axis=0)

        # Resample (if original sr ≠ target sr)
        if sr != sr_target:
            y = librosa.resample(y, orig_sr=sr, target_sr=sr_target, axis=1)
            sr = sr_target

        # Resize (pad or crop to fixed duration)
        target_len = sr * duration
        if y.shape[1] < target_len:
            pad_width = target_len - y.shape[1]
            y = np.pad(y, ((0, 0), (0, pad_width)), mode='constant')
        else:
            y = y[:, :target_len]

        # Convert to mono by averaging stereo
        y_mono = np.mean(y, axis=0)

        # Feature extraction
        if mode == "mfcc":
            mfcc = librosa.feature.mfcc(y=y_mono, sr=sr, n_mfcc=n_mfcc)
            mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)
            return mfcc  # shape: [n_mfcc, time_steps]

        elif mode == "mel":
            mel = librosa.feature.melspectrogram(y=y_mono, sr=sr, n_mels=128)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            mel_db = (mel_db - np.mean(mel_db)) / np.std(mel_db)
            return mel_db  # shape: [128, time_steps]

        else:
            raise ValueError(f"Invalid mode: {mode}. Use 'mfcc' or 'mel'.")

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return np.zeros((n_mfcc if mode == "mfcc" else 128, sr_target * duration // 512))  # fallback

# Example Usage

# For MFCC-based LSTM
df["mfcc"] = df["file_path"].apply(lambda p: preprocess_audio(p, mode="mfcc"))

# For MelSpec-based CNN-LSTM
df["mel"] = df["file_path"].apply(lambda p: preprocess_audio(p, mode="mel"))
