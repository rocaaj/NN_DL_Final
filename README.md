# CSC 375 Final Project
## Established paper: [Link to Paper](https://doi.org/10.1016/j.autcon.2024.105485)

---
### Baseline model

```bash


```



---
### CNN+Transfomer Hybrid model

```bash


```


---
### Improved CNN+LSTM model (not adopted because of time constraints)
This script performs 10-fold cross-validation on the UrbanSound8K dataset, saves the best model for each fold, and generates plots for metrics such as loss, accuracy, macro F1-score, precision, and recall.

#### Dependencies
- Python 3.8+
- **audioread 3.0.1**
- **librosa 0.11.0**
- **matplotlib 3.10.1**
- **numpy 2.2.5**
- **pandas 2.2.3**
- **scikit-learn 1.6.1**
- **scipy 1.15.2**
- **soundfile 0.13.1**
- **torch 2.7.0+cu118**
- **torchaudio 2.7.0+cu118**
- **torchvision 0.22.0+cu118**


### How to run
```bash
python train_improved_concat_model.py
```