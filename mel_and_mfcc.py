import matplotlib.pyplot as plt
import librosa.display

# Utility functions to plot
def plot_mel(mel, sr=22050, title="Mel Spectrogram"):
    plt.figure(figsize=(8, 3))
    librosa.display.specshow(mel, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_mfcc(mfcc, sr=22050, title="MFCC"):
    plt.figure(figsize=(8, 3))
    librosa.display.specshow(mfcc, sr=sr, x_axis='time')
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.show()
  
# Loop over each unique class
for cls in sorted(df["class"].unique()):
    sample = df[df["class"] == cls].iloc[0]
    
    print(f"\n Class: {cls}")
    
    plot_mel(sample["mel"], title=f"Mel Spectrogram - {cls}")
    plot_mfcc(sample["mfcc"], title=f"MFCC - {cls}")
