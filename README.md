# Parkinson's Disease EEG Diagnostics — Khoshnevis et al. Implementation

> **An academic reproduction of a high-precision Parkinson's Disease diagnostic
> pipeline built on multimodal, nonlinear feature fusion over resting-state EEG.**

---

## 📄 About

This repository contains a Google Colab notebook that implements the
methodology described in *Khoshnevis et al.* for detecting Parkinson's Disease
(PD) from resting-state EEG recordings.

The pipeline extracts three blocks of interpretable features from EEG alpha and
beta sub-bands, fuses them into a master feature tensor, and feeds them into a
compact 1-D convolutional neural network.

| Metric | Score |
|---|---|
| Validation Accuracy | **83.22 %** |
| ROC-AUC | **~0.90** |
| Healthy Control precision | 0.94 |
| PD recall | 0.90 |

---

## 📦 Dataset

The notebook uses the publicly available OpenNeuro dataset **ds004584**
(resting-state EEG, 149 subjects — 50 PD, 99 healthy controls).

> **Download:** <https://openneuro.org/datasets/ds004584>

After downloading, upload the dataset folder to your Google Drive.  
Then open the notebook and fill in the paths in **Cell 0 (Configuration)**.

---

## 🚀 Quick Start

1. **Open** `Collab_notebook.ipynb` in [Google Colab](https://colab.research.google.com/).
2. **Mount** your Google Drive when prompted.
3. **Edit Cell 0** — the only cell you need to change:

```python
DATASET_URL   = "https://openneuro.org/datasets/ds004584"   # optional reference
DRIVE_DATA_DIR = "/content/drive/MyDrive/Dataset/ds004584-download"  # raw .set files
EXPORT_DIR     = "/content/drive/MyDrive/Dataset/processed_features" # per-subject .npy
BASE_DIR       = "/content/drive/MyDrive/Dataset"                    # final arrays
```

4. **Run all cells** in order (`Runtime → Run all`).

> The cohort processing cell (~Cell 12) has fault-tolerant **resume support** —
> if the runtime disconnects, re-run from that cell and already-processed
> subjects are automatically skipped.

---

## 🔬 Pipeline Overview

```
Raw EEG (.set)
    │
    ├─ [Cell 1]  Environment setup  (mne, nolds, antropy, torch)
    ├─ [Cell 2]  Dataset path verification
    │
    ├─ [Cell 3]  Pre-processing
    │               • 60 Hz notch filter
    │               • Alpha (8–12 Hz) & Beta (13–30 Hz) bandpass
    │
    ├─ [Cell 4]  ICA artifact rejection  (eye-blink removal via Fp1)
    │
    ├─ [Cell 5]  Epoch segmentation  (20 × 2-second windows)
    │
    ├─ [Cells 6–9]  Feature extraction per epoch × channel
    │               Block 1 – Nonlinear / chaotic (4 features)
    │                   • Lyapunov volatility proxy
    │                   • Hurst Exponent (HE)
    │                   • Approximate Entropy (ApEn)
    │                   • Higuchi Fractal Dimension (HFD)
    │               Block 2 – Higher-order moments (6 features, 3rd–8th)
    │               Block 3 – Frequency-domain PSD (3 features)
    │                   • Peak frequency
    │                   • Mean spectral centroid
    │                   • Total band power
    │               → Master tensor: (20 epochs × 63 channels × 26 features)
    │
    ├─ [Cell 10]  Cohort batch processing  (all 149 subjects, resumable)
    ├─ [Cell 11]  Alternative high-speed batch pass
    ├─ [Cell 12]  Aggregation → X_final.npy, y_final.npy
    │               Labels: PD (sub-001 – sub-050 = 1), HC (sub-051 – sub-149 = 0)
    │
    ├─ [Cell 13]  Stratified train/test split + RobustScaler
    │
    ├─ [Cell 14]  KhoshnevisNet training  (30 epochs, AdamW, BCEWithLogitsLoss)
    │
    └─ [Cell 15]  Evaluation  (confusion matrix, ROC-AUC, classification report)
```

---

## 🧠 Model Architecture — KhoshnevisNet

```
Input (Batch, 26 features, 63 channels)
    │
    ├─ LayerNorm (spatial normalisation)
    ├─ Conv1D(26→64, k=3) + BN + ReLU + Dropout2D(0.2)
    ├─ Conv1D(64→128, k=3) + BN + ReLU
    ├─ MaxPool1D(k=2)  →  128 × 31
    ├─ Dropout(0.4)
    ├─ Flatten  →  3968
    ├─ Linear(3968→64) + BN + ReLU + Dropout(0.5)
    └─ Linear(64→1)  →  sigmoid  →  PD probability
```

Training details:
- Optimiser: `AdamW` (lr = 1e-4, weight_decay = 1e-2)
- Loss: `BCEWithLogitsLoss` with class-imbalance weighting (pos_weight ≈ 1.98)
- Gradient clipping: `max_norm = 1.0`
- GPU: T4 (Google Colab)

---

## 📁 Repository Structure

```
Khoshnevis_et_al_Implementation/
├── Collab_notebook.ipynb   # Full pipeline notebook
├── requirements.txt        # Python dependencies
├── .gitignore
└── README.md
```

> **Note:** Dataset files (`.set`, `.fdt`) and processed arrays (`.npy`) are
> excluded from version control via `.gitignore`.

---

## 🛠 Dependencies

See [`requirements.txt`](requirements.txt) for pinned versions, or install directly:

```bash
pip install mne nolds antropy torch numpy scipy scikit-learn matplotlib seaborn
```

---

## 📝 Citation

If you use this implementation, please cite the original paper:

```
Khoshnevis et al. — Parkinson's Disease detection from resting-state EEG
using multimodal nonlinear feature fusion.
(Full citation to be updated once publication details are available.)
```

---

## ⚖️ License

This project is released under the [MIT License](LICENSE).
