# BraTS 2021 Missing-Modality Robust Segmentation

This project implements a robust 3D segmentation pipeline for brain tumors using the BraTS 2021 dataset. It specifically targets the challenge of missing MRI modalities during inference.

This repository currently implements a **3D U-Net baseline** trained with random modality dropout to serve as a robust baseline against which more advanced missing-modality architectures (e.g., Transformers) can be compared.

## End Goal & Roadmap

This project is part of a broader 21-day learning implementation plan. The ultimate goals of this project include:
1. **Missing-Modality-Robust Segmentation Pipeline**: Comparing the baseline 3D U-Net against a from-scratch masked-fusion transformer.
2. **Custom Fused Attention Kernel**: Implementing a custom fused Triton kernel for masked multi-head self-attention and benchmarking its speed/memory against PyTorch.
3. **Interactive GUI**: Building a Streamlit/Gradio frontend to demo the pipeline—allowing users to select cases, toggle modalities on/off, and see live predictions alongside kernel speed comparisons.
## Features

- **Robust 3D U-Net Baseline:** A 3D U-Net architecture adapted for multi-modal brain tumor segmentation.
- **Modality Dropout:** A training strategy that randomly zeroes out a subset of the 4 MRI modalities (T1, T1ce, T2, FLAIR) to simulate missing data and force the model to learn robust representations.
- **Efficient Patch-based Training:** Extracts 96³ 3D patches centered around tumor regions to handle the large 3D MRI volumes efficiently within hardware constraints.
- **Sliding Window Evaluation:** Performs full-volume evaluation using a sliding-window approach with overlapping patches.
- **Mixed Precision Training:** Utilizes PyTorch AMP (Automatic Mixed Precision) for memory efficiency and faster training.

## Requirements

The core dependencies for this project are:
- `torch` (with CUDA support recommended)
- `numpy`
- `nibabel`

Install them via pip:
```bash
pip install torch numpy nibabel
```

## Dataset Setup

1. Download the **BraTS 2021** dataset.
2. Extract the data.
3. Update the `ROOT_DIR` and `SPLIT_DIR` paths in `train.py` and `eval.py` to point to your local dataset directory. The default paths are set to `D:\MMD\IPA Assignment\BraTS2021\BraTS2021_Training_Data` and `D:\MMD\IPA Assignment\BraTS2021\splits` respectively.

## Usage

### Training

To train the 3D U-Net baseline from scratch, run:
```bash
python train.py
```
This script handles the train/validation split automatically and saves the model checkpoints to the `checkpoints/` directory.

You can enable a quick test mode by setting `QUICK_TEST = True` in `train.py` to verify the pipeline end-to-end without running a full training session.

### Evaluation

To evaluate the model and compute Dice scores across different modality combinations, run:
```bash
python eval.py
```
The script will evaluate the `best.pt` model checkpoint using sliding-window inference and report the Dice scores for Whole Tumor (WT), Tumor Core (TC), and Enhancing Tumor (ET) for representative modality combinations (e.g., all 4, T1ce+FLAIR, FLAIR only, etc.).

## Project Structure

- `models/unet_baseline.py`: PyTorch implementation of the 3D U-Net architecture.
- `dataset.py`: PyTorch Dataset class, handles patch sampling and random modality dropout.
- `preprocess.py`: Contains data normalization (z-score on brain mask) and patch extraction logic.
- `losses.py`: Implementation of the combined Dice and BCE loss function.
- `train.py`: Training loop with validation and early stopping.
- `eval.py`: Sliding-window evaluation across different missing-modality scenarios.
- `implementation-plan.md` / `specs.md`: Project notes, hardware specs, and implementation plan.
