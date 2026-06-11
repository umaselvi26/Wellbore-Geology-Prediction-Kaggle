# ROGII - Wellbore Geology Prediction (Kaggle Competition)
An optimized Machine Learning pipeline using **LightGBM** to predict True Vertical Thickness (TVT) from complex subsurface well-log geological data. This approach successfully minimized the prediction error rate, driving the Kaggle Leaderboard RMSE down from a baseline of **152.687** to **112.397**.

## Project Overview
In wellbore geology, accurate estimation of formations is critical for drilling efficiency. This repository features a structured end-to-end pipeline designed to handle high-volume sequential data (over **4 Million rows**), implementing robust data preprocessing, specialized spatial feature engineering, and high-performance gradient boosting.

## Key Features & Methodology

### 1. Advanced Feature Engineering (The Game Changer)
Raw features like 'GR' (Gamma Ray) and coordinates ('X', 'Y', 'Z') were transformed to capture sequential and spatial geological contexts:
**Sequential Trend Modeling:** Extracted moving windows via rolling statistics ('GR_roll3', 'GR_roll10', 'GR_roll20') and deviation spreads ('GR_std10') to smooth continuous log noises.
**Temporal Lag Features:** Utilized directional data shifts ('GR_lag1', 'GR_lag3') to teach the model how prior formations affect the current layer thickness.
**3D Spatial Displacements:** Computed a dynamic Euclidean distance vector ('dist_3d') utilizing coordinate deltas:
  $$\text{dist\_3d} = \sqrt{\Delta X^2 + \Delta Y^2 + \Delta Z^2}$$
  This captures the physical spatial path of the horizontal wellbore trajectory.

### 2. Validation & Modeling Strategy
**Algorithm:** High-Performance **LightGBM Regressor**, configured to train safely and rapidly across heavy data splits using optimized thread management ('force_row_wise=True', 'n_jobs=-1').
**Hyperparameter Fine-Tuning:**
  * 'n_estimators': Increased trees up to '1000' to allow stable learning boundaries.
  * 'learning_rate': Dropped to '0.03' for fine-grained gradient updates.
  * 'max_depth' & 'num_leaves': Capped at '8' and '63' respectively to strictly prevent structural overfitting.
**Validation Realignment:** Evaluated alignment discrepancies using 'GroupKFold' vs standard 'KFold' to explicitly handle unseen out-of-sample well distributions.

## Results & Leaderboard Performance
**Initial Baseline RMSE:** '152.687'
**Final Optimized Score:** '112.397' (Substantial error drop!)
**Status:** Ranked safely within a highly competitive territory on the Public Leaderboard.

## Tech Stack & Dependencies
**Language:** Python 3.12
**Core Libraries:** 'pandas', 'numpy', 'scikit-learn', 'lightgbm', 'glob', 'os'
**Platform:** Kaggle Notebooks (Cloud Compute Engine)

## Repository Structure
* 'notebook.ipynb' - The complete execution script including feature extraction, training loops, and predictions block.
* 'README.md' - Documentation of the challenge approach.
