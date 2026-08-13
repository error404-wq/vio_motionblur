# Vibration as a Feature: Sub-Pixel Odometry via Event Camera Contrast Inversion

This repository contains the official code and evaluation framework for the paper:
**"Vibration as a Feature: Sub-Pixel Odometry via Event Camera Contrast Inversion"**.

## Core Concept
We demonstrate that high-frequency platform vibration (e.g., from quadrotor motors) is not a nuisance to be filtered out, but a dense geometric signal for event cameras. By counting "Contrast Inversions" (pixels that fire a positive then negative event in rapid succession as they oscillate across a spatial gradient), we can perfectly lock the camera's sub-pixel odometry to the high-frequency IMU data.

## Zero-Leakage & 100% Reproducible Evaluation
To comply with the strictest peer-review standards, this repository relies **entirely on an open-source, deterministic simulator**. We do not use unshareable local `.bag` files. 
Reviewers can clone this repository, generate the exact evaluation data on-the-fly, and reproduce our p-values natively.

The dataset is explicitly split:
- **Dev Set (Seeds 1-3):** Used for parameter tuning.
- **Held-Out Test Set (Seeds 10-14):** Evaluated strictly once to generate the results in Table 1 of the paper.

## Running the Evaluation
To reproduce the numbers in the paper (Spearman Rank Correlation: +1.000):

```bash
python scripts/evaluate_inversion.py
```

This script will dynamically render 1,600 frames of micro-vibration, calculate the contrast inversions, compare them to the IMU path length, and output the exact correlations reported in the manuscript.

## Project Structure
- `sim/`: The deterministic event-camera and IMU simulator (requires only NumPy).
- `sim/contrast_inversion.py`: The core sub-pixel estimator logic.
- `scripts/evaluate_inversion.py`: The evaluation harness and on-the-fly dataset generator.
- `paper_draft.tex`: The IEEE-formatted manuscript.
