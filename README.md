# NTIRE 2026 Night-Time Dehazing Challenge Inference

This repository provides the inference code and environment files used to reproduce our challenge submission results.

## Environment

Python version used: **Python 3.8.20**

We provide two environment files for reproducibility:

- `environment.yml`
- `requirements.txt`

### Option 1: Conda

```bash
conda env create -f environment.yml
conda activate dehazedct
```

### Option 2: Pip

```bash
pip install -r requirements.txt
```

## Dataset Setup

Place the 5 test images in:

```bash
data/challenge_test/hazy/
```

Example structure:

```bash
.
├── data/
│   └── challenge_test/
│       └── hazy/
│           ├── 31_NTHazy.png
│           ├── 32_NTHazy.png
│           ├── 33_NTHazy.png
│           ├── 34_NTHazy.png
│           └── 35_NTHazy.png
├── DCNv4_op/
├── predict_stage2_ensemble.py
├── environment.yml
├── requirements.txt
├── final.pth
└── flash_intern_image_l_22kto1k_384.pth
```

## Checkpoints

Download the following files from the Google Drive folder below and place them in the repository root directory:

- `final.pth` — best challenge checkpoint
- `flash_intern_image_l_22kto1k_384.pth` — pretrained FlashInternImage backbone

Google Drive folder:

```text
https://drive.google.com/drive/folders/1uP5ZUUcnkXYPO_PgCiayVXFcZ27ftzIi?usp=sharing
```

## DCNv4 Setup

To install the DCNv4 operator used in the model, run:

```bash
cd DCNv4_op
bash make.sh
cd ..
```

If needed, this step may require minor adjustment depending on the local CUDA and PyTorch build environment.

## Inference

Run the following command from the repository root:

```bash
python predict_stage2_ensemble.py \
    --data_root ./data/challenge_test \
    --ckpt ./final.pth \
    --out_dir ./out \
    --device cuda:0 \
    --mode fullpad \
    --mod 32 \
    --pad_mode reflect
```

## Output

The dehazed results will be saved to:

```bash
./out/
```

## Notes

- This README is intended for inference-time reproduction of the released challenge results.
- Please use the provided environment files, released checkpoints, and the same inference settings.
- Minor numerical differences can occur across different hardware or software environments, but the reproduced results should remain at the same performance level.
