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

## Reproducibility of Paper Results

To reproduce the paper results, including the generation of dehazed images on the test data and the quantitative metrics reported in the paper, download the **`artifacts`** folder from the Google Drive link below and place the `dataset` and `checkpoints` folders in a local directory with the following structure.

Google Drive (artifacts):
https://drive.google.com/drive/folders/14UvIxnu40E0EYuOfGfAjX_IbFzWZwBCm?usp=sharing

Expected directory structure:

    artifacts/
    ├── dataset/
    │   ├── Dense-Haze/
    │   │   ├── train/
    │   │   └── test/
    │   ├── NH-Haze/
    │   │   ├── train/
    │   │   └── test/
    │   └── NH-Haze2/
    │       ├── train/
    │       └── test/
    └── checkpoints/
        └── HistoFusionNet/
            ├── Checkpoints_DenseHaze/
            │   └── best_psnr.pkl
            ├── Checkpoints_NH_Haze/
            │   └── best_psnr.pkl
            └── Checkpoints_NH_Haze2/
                └── best_psnr.pkl

> **Note:** The `train/` and `test/` subdirectory structure is the same for all three datasets.

### Evaluation on NH-Haze

Run:

    python predict.py \
      --data_root ./artifacts/dataset/NH-Haze/test/ \
      --ckpt ./artifacts/checkpoints/HistoFusionNet/Checkpoints_NH_Haze/best_psnr.pkl \
      --out_dir results_NH_Haze \
      --device cuda:0 \
      --mode fullpad \
      --mod 32 \
      --pad_mode reflect \
      --num_workers 0 \
      --sanity

### Evaluation on NH-Haze2

Run:

    python predict.py \
      --data_root ./artifacts/dataset/NH-Haze2/test/ \
      --ckpt ./artifacts/checkpoints/HistoFusionNet/Checkpoints_NH_Haze2/best_psnr.pkl \
      --out_dir results_NH_Haze2 \
      --device cuda:0 \
      --mode fullpad \
      --mod 32 \
      --pad_mode reflect \
      --num_workers 0 \
      --sanity

### Evaluation on Dense-Haze

Run:

    python predict_dense.py \
      --data_root ./artifacts/dataset/Dense-Haze/test/ \
      --ckpt ./artifacts/checkpoints/HistoFusionNet/Checkpoints_DenseHaze/best_psnr.pkl \
      --out_dir results_DenseHaze \
      --device cuda:0 \
      --mode fullpad \
      --mod 32 \
      --pad_mode reflect \
      --num_workers 0 \
      --sanity

### Output

The generated dehazed images will be saved in the corresponding output directories:

- `results_NH_Haze/`
- `results_NH_Haze2/`
- `results_DenseHaze/`

These commands are intended to reproduce the generation of dehazed images on the test sets and the quantitative results reported in the paper.

## Training Commands

The following commands correspond to training on the different datasets before applying frequency-adaptive refinement fine-tuning.

### Training on NH-Haze

Run:

```bash
python train.py \
  --dataset_root ./artifacts/dataset/NH-Haze \
  --crop_size 384 \
  -train_batch_size 2 \
  -train_epoch 5000 \
  --model_save_dir ./artifacts/checkpoints/HistoFusionNet/Checkpoints_NH_Haze \
  --denet_save_dir ./artifacts/checkpoints/HistoFusionNet/Checkpoints_NH_Haze \
  --log_txt train_loss_NH_Haze.txt \
  --excel_txt test_metrics_NH_Haze.xlsx
```

### Training on NH-Haze2

Run:

```bash
python train.py \
  --dataset_root ./artifacts/dataset/NH-Haze2 \
  --crop_size 384 \
  -train_batch_size 2 \
  -train_epoch 5000 \
  --model_save_dir ./artifacts/checkpoints/HistoFusionNet/Checkpoints_NH_Haze2 \
  --denet_save_dir ./artifacts/checkpoints/HistoFusionNet/Checkpoints_NH_Haze2 \
  --log_txt train_loss_NH_Haze2.txt \
  --excel_txt test_metrics_NH_Haze2.xlsx
```

### Training on Dense-Haze

Run:

```bash
python train.py \
  --dataset_root ./artifacts/dataset/Dense-Haze \
  --crop_size 384 \
  -train_batch_size 2 \
  -train_epoch 5000 \
  --model_save_dir ./artifacts/checkpoints/HistoFusionNet/Checkpoints_DenseHaze \
  --denet_save_dir ./artifacts/checkpoints/HistoFusionNet/Checkpoints_DenseHaze \
  --log_txt train_loss_dense_haze.txt \
  --excel_txt test_metrics_dense_haze.xlsx
```

## Contact

For future contact, please email:

**mohammadheydari.eduu@gmail.com**


## Notes

- This README is intended for inference-time reproduction of the released challenge results.
- Please use the provided environment files, released checkpoints, and the same inference settings.
- Minor numerical differences can occur across different hardware or software environments, but the reproduced results should remain at the same performance level.



