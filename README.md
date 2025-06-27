# DL\_BrainAge\_CoPath

A compact PyTorch pipeline that predicts **brain age** from 3-D T1-weighted MRI. It is further used to quantify how combined Alzheimer’s-disease (AD) and Lewy-body (LB) pathology accelerates structural ageing.

---

## 🚀 What’s in this repo?

| File                                   | Role                                                                                                                                      |
| -------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `densenet3d.py`                        | 3-D DenseNet-121 (growth-32, compression 0.5) exactly as described in the manuscript.                                                     |
| `main.py`                              | Command-line script for **training** (`train` mode) or **inference** (`predict` mode) with early-stopping, LR decay and MAE/R² reporting. |
| `requirements.txt` / `environment.yml` | Pinned package versions (Python 3.11, CUDA 12.1, PyTorch 2.1.1, etc.).                                                                    |
| `demo/`                                | Minimal example data & weights (add later).                                                                                     |

> Raw MRI and biomarker data are **not redistributed**. Users must obtain them from ADNI, NACC, AIBL, CamCAN and HCP-A under their respective licences.

---

## 🔧 Installation

```bash
conda env create -f environment.yml      # creates "brainage" env ▸ ~10 min
conda activate brainage
```

or, without Conda:

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

A CUDA 12.1-capable GPU (≥ 12 GB VRAM) is recommended for training but **not required** for inference.

---

## 🏃 Quick start

### 1 · Training from scratch

```bash
python main.py train \
    --x_train data/x_train.npy --y_train data/age_train.npy \
    --x_val   data/x_val.npy   --y_val   data/age_val.npy   \
    --outdir  checkpoints
```

`main.py` will iterate 5×15 epochs with LR decay 0.7 and early-stop once validation MAE plateaus, saving `checkpoints/best_model.pt`.

### 2 · Inference

```bash
python main.py predict \
    --x_test data/x_test.npy \
    --weights checkpoints/best_model.pt \
    --y_test data/age_test.npy   # optional → prints MAE & R²
```

Predictions are saved to `predictions.npy`.

---

## 📈 Package versions

Core stack (see `requirements.txt` for full list & exact pins):

* Python 3.11
* PyTorch 2.1.1 + CUDA 12.1
* NumPy 1.26 · SciPy 1.11 · scikit-learn 1.5 

---

## 🔍 Interpretability

The convolutional class-activation map (CAM) head in `densenet3d.py` feeds saliency analysis identical to Fig. 3 in the manuscript. An example notebook will be added under `notebooks/` once the manuscript is accepted.

---

## 📄 License

BSD-3-Clause. See `LICENSE`.

---

## 📬 Contact

Questions or issues → open a GitHub issue or email **Babak Ahmadi** [babak.ahmadi@ufl.edu](mailto:babak.ahmadi@ufl.edu).
