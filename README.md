# DL_BrainAge_CoPath

A deep‑learning pipeline for predicting brain age from structural MRI and dissecting the impact of Alzheimer’s disease (AD), Lewy‑body (LB) pathology, and their co‑occurrence on accelerated brain ageing.

---

## 🚀 Project Overview

This repository accompanies the manuscript **“Lewy‑body co‑pathology exacerbates deep‑learning brain‑age and accelerates neurodegeneration”** (In Review at Nature Communications, 2025). We:

1. Train a **3‑D DenseNet** on >4k cognitively‑unimpaired (CU) T1‑w MRIs pooled from **ADNI, NACC, AIBL, CamCAN & HCP‑A** to model normative ageing.
2. Compare brain age gap (BAG) across biomarker‑defined AD, LB, and AD+LB subgroups.
3. Localize cortical & subcortical drivers of elevated BAG with **saliency maps**.
4. Track longitudinal BAG & regional atrophy and relate them to cognitive decline.

> **Note** All raw MRI and biomarker data remain subject to the original cohort licences. We provide links & DOI’s but do **not** redistribute the datasets.

---

## 📂 Repository Structure

```text
BACPDL/
├── src/                       # Core Python code
│   ├── densenet3d.py          # 3‑D DenseNet architecture
│   ├── train_brainage.py      # Training loop & cross‑validation
│   ├── predict_brainage.py    # Inference on a pre‑processed scan
│   ├── saliency.py            # Saliency‑map generation utilities
│   └── utils/                 # I/O, metrics, visualization helpers
├── demo/
│   ├── sample_T1.nii.gz       # 1× anonymised CU scan (299 kB)
│   └── run_demo.py            # End‑to‑end demo (load ➜ predict ➜ plot)
├── notebooks/                 # Interactive Jupyter notebooks
│   └── brain_age_workflow.ipynb
├── requirements.txt           # Lightweight pip spec (see below)
├── environment.yml            # Reproducible Conda env (GPU, CUDA 12.1)
├── LICENSE                    # BSD‑3‑Clause
└── README.md                  # You are here 📑
```

---

## 🔧 Installation

### 1. Clone repo & create Conda env

```bash
conda env create -f environment.yml  # ≈10 min on a laptop
conda activate bacpdl
```

### 2. Install extra system packages (Ubuntu 20.04+)

```bash
sudo apt-get install -y ants
```

> **Tip** A Dockerfile is provided in `extras/` if you prefer containers.

---

## 🏃 Quick‑start Demo  (≤ 60 s on CPU)

```bash
python demo/run_demo.py --img demo/sample_T1.nii.gz \
                        --weights checkpoints/densenet3d_cu.pth
```

Output ⇢ JSON with predicted age, BAG, and a sagittal saliency overlay PNG.

---

## 🧠 Training From Scratch

```bash
python src/train_brainage.py \
       --train_csv data/cu_train.csv \
       --val_csv   data/cu_val.csv   \
       --epochs 75 --batch 8 --gpu 0
```

*Five‑fold* site‑stratified cross‑validation is enabled by default.

---

## 📊 Reproducing Manuscript Figures

1. Follow *Training From Scratch* or download released weights.
2. Run `predict_brainage.py` on each pathology CSV.
3. Execute `analysis/plot_paper_figures.py` to regenerate Figs 2–5.

Full provenance scripts are in `analysis/` and are invoked by `make all`.

---

## 📈 Requirements (software)

Core: `python==3.11`, `torch==2.1`, `nibabel==5.1`, `scikit‑learn==1.5`, `scipy==1.11`, `statsmodels==0.14`, `nilearn==0.10`.

See `requirements.txt` / `environment.yml` for pinned versions.

Hardware:

* GPU with ≥12 GB VRAM for training (tested on NVIDIA A100).

---

## 🗄️ Data Access & Pre‑processing

| Cohort | URL / DOI                                                                | Licence            |
| ------ | ------------------------------------------------------------------------ | ------------------ |
| ADNI   | [https://adni.loni.usc.edu](https://adni.loni.usc.edu)                   | Data‑use agreement |
| NACC   | [https://naccdata.org](https://naccdata.org)                             | Data‑use agreement |
| AIBL   | [https://aibl.csiro.au](https://aibl.csiro.au)                           | Data‑use agreement |
| CamCAN | [https://doi.org/10.17863/CAM.16671](https://doi.org/10.17863/CAM.16671) | CC BY‑NC‑SA 4.0    |
| HCP‑A  | [https://doi.org/10.15154/1520707](https://doi.org/10.15154/1520707)     | Custom DUA         |

Pre‑processing is fully automated via `preprocessing/run_freesurfer_pipe.sh` plus affine MNI registration with ANTs.

---

## 🔍 Interpretability

`saliency.py` implements gradient saliency & Gaussian smoothing to map voxel‑level contributions.

---

## 📑 Citation

Please cite both the **paper** and this **software**:

```bibtex
@article{babakian2025copath,
  title   = {Lewy-body co-pathology exacerbates deep-learning brain-age and accelerates neurodegeneration},
  author  = {Ahmadi, B. et al.},
  journal = {Nature Communications},
  year    = {2025}
}
```

---

## 📄 License

BSD‑3‑Clause. See `LICENSE` for details.

---

## 📬 Contact

For questions, please open an issue or email **Babak Ahmadi** [babak.ahmadi@ufl.edu](mailto:babak.ahmadi@ufl.edu).

