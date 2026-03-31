# 🏭 Quality2025 — Hybrid Anomaly Detection via Denoising AE + Robust Latent-Space SPC

> Quality Control Final Report | NTUST M11401xxx  
> Multivariate time-series anomaly detection combining Denoising Autoencoder, MCD-based Mahalanobis Distance, and Statistical Process Control charts.

---

## 📋 Table of Contents
- [Overview](#overview)
- [Method](#method)
- [Dataset](#dataset)
- [Results](#results)
- [Team](#team)

---

## Overview

Modern manufacturing and data-center systems generate high-frequency, high-dimensional sensor data where early anomaly detection is critical. This project proposes a **hybrid monitoring framework** that:

1. Trains a **Denoising Autoencoder (AE)** on normal windows to learn a compact latent representation.
2. Fits a **Minimum Covariance Determinant (MCD)** model in latent space for robust Mahalanobis Distance (MD) estimation.
3. Monitors three complementary statistics — **Shewhart MD**, **EWMA-smoothed MD**, and **Reconstruction Error (RE)** — via control charts with empirical quantile-based UCLs.
4. Fuses binary alarm flags using **OR / VOTE / AND** logic for flexible precision–recall trade-off.

The best configuration achieves **Point-Adjusted F1 = 0.988** on the held-out test set, up from 0.673 at baseline.

---

## Method

### Architecture

```
Input window (30 × 10 = 300-dim)
        │
  [Gaussian Noise, σ=0.03]   ← training only
        │
   Encoder: 300 → 64 → 32 → 6   (ReLU + Dropout)
        │
   Latent z ∈ ℝ⁶
        │
   Decoder: 6 → 32 → 64 → 300   (ReLU)
        │
   Reconstruction x̂
```

### Anomaly Statistics

| Statistic | Description |
|-----------|-------------|
| **MD (Shewhart)** | MCD-based Mahalanobis distance in latent space |
| **EWMA** | Exponentially weighted moving average of MD (`λ=0.1`) |
| **RE** | log(1 + MSE) reconstruction error |

### Decision Fusion

| Logic | Rule | Characteristic |
|-------|------|----------------|
| OR    | Any detector triggers | High recall |
| VOTE  | Majority triggers | Balanced |
| AND   | All detectors trigger | High precision |

### Control Limits

UCLs are set empirically from training data:

```
UCL_Shewhart = Q_q(MD_train)
UCL_EWMA     = Q_q(EWMA_train)
UCL_RE       = Q_q(log(1 + RE_train))
```

---

## Dataset

**Server Machine Dataset (SMD)** — public benchmark for anomaly detection in data-center servers.

- 28 machines × 38 sensors
- This project focuses on **machine 1-1**, using **10 selected sensors**
- Sliding window: `length=30`, `stride=1`
- Split: Training → Validation (30% of test) → Final Test (70% of test)
- Label strategy: stratified by anomaly segments to maintain anomaly ratio

---

## Results

### Validation Set — Baseline (q=0.997, λ=0.2, OR)

| Type | Accuracy | Precision | Recall | F1 | ROC-AUC |
|------|----------|-----------|--------|----|---------|
| Point-wise | 0.849 | 0.300 | 0.986 | 0.460 | 0.948 |
| Point-Adjusted | 0.850 | 0.303 | 1.000 | 0.465 | 0.920 |

### Fusion Logic Comparison (Validation, PA)

| Logic | PA Precision | PA Recall | PA F1 |
|-------|-------------|-----------|-------|
| OR    | 0.303 | 1.000 | 0.465 |
| VOTE  | 0.450 | 0.995 | 0.619 |
| AND   | 0.535 | 0.995 | 0.696 |

### Top Grid Search Results (Validation, PA)

| Rank | q | λ | Logic | PA Precision | PA Recall | PA F1 |
|------|---|---|-------|-------------|-----------|-------|
| 1 | 0.999 | 0.1 | AND | 0.982 | 0.994 | **0.988** |
| 2 | 0.999 | 0.3 | AND | 0.897 | 0.994 | 0.943 |
| 3 | 0.999 | 0.1 | VOTE | 0.788 | 0.994 | 0.879 |

### Final Test Set — Baseline vs. Best

| Setting | q | λ | Logic | PA Precision | PA Recall | **PA F1** |
|---------|---|---|-------|-------------|-----------|-----------|
| Baseline | 0.997 | 0.2 | OR | 0.507 | 0.999 | 0.673 |
| **Best** | **0.999** | **0.1** | **AND** | **0.978** | **0.998** | **0.988** |

---



---

## Getting Started

### Requirements

```bash
pip install -r requirements.txt
```

Main dependencies: `torch`, `scikit-learn`, `numpy`, `pandas`, `matplotlib`

**Environment:** Python 3.12.9, seed = 42




## Team

| Name | Student ID |
|------|-----------|
| 溫翎傑 | M11401204 |
| 陳喻宣 | M11401102 |
| 林侑萱 | M11401303 |
| 吳展佑 | M11401202 |

---

## References

- Zong et al. (2018). Deep Autoencoding Gaussian Mixture Model. *ICLR 2018*
- Rousseeuw (1984). Least Median of Squares Regression. *JASA*
- Sakurada & Yairi (2014). Anomaly Detection Using Autoencoders. *MLSDA*
- Kim et al. (2020). Robust Multivariate Process Monitoring Using Deep Autoencoder. *Journal of Process Control*
