# Quality2025 — Multivariate Anomaly Detection with a Denoising Autoencoder and Robust SPC

**品質管制期末專案（4 人團隊）｜NTUST 工業管理所**

把 10 個感測器的時間序列先用去噪自編碼器（Denoising Autoencoder）壓縮成 6 維，再用 MCD 穩健共變異數計算 Mahalanobis 距離，搭配 Shewhart、EWMA 管制圖與重建誤差三種警報，以 OR／VOTE／AND 規則融合。最終測試集的 Point-Adjusted F1 從基準的 **0.673 提升到 0.988**：誤報大幅減少（precision 0.507 → 0.978），漏報幾乎沒有增加（recall 0.999 → 0.998）。

| Final test set | Control-limit quantile q | EWMA λ | Fusion | PA Precision | PA Recall | **PA F1** |
|---|---:|---:|:---:|---:|---:|---:|
| Baseline | 0.997 | 0.2 | OR | 0.507 | 0.999 | 0.673 |
| **Best (grid search)** | **0.999** | **0.1** | **AND** | **0.978** | **0.998** | **0.988** |

---

## Why this project

Production lines and data-center servers are monitored by dozens of correlated sensors. Running a separate Shewhart chart on each variable ignores how the variables move together, and a single global threshold on a deep model's output is hard to explain to process engineers.

This project keeps the control-chart workflow that quality engineers already use, and puts it on top of a learned representation:

1. A **denoising autoencoder** learns what normal operation looks like and compresses each 30-step window into a 6-dimensional latent vector.
2. A **Minimum Covariance Determinant (MCD)** estimate in that latent space gives a robust Mahalanobis distance that is not dragged around by outliers in the training data.
3. Three statistics are charted with empirical control limits: **Shewhart MD**, **EWMA-smoothed MD**, and **reconstruction error**.
4. The three alarm flags are **fused** (OR / VOTE / AND), which gives a direct dial between catching everything and raising fewer false alarms.

## Method

### Autoencoder

```text
Input window (30 time steps × 10 sensors = 300 values)
        │
  Gaussian noise, σ = 0.03          (training only)
        │
  Encoder: 300 → 64 → 32 → 6        (ReLU + Dropout)
        │
  Latent z ∈ ℝ⁶
        │
  Decoder: 6 → 32 → 64 → 300        (ReLU)
        │
  Reconstruction x̂
```

### Monitoring statistics

| Statistic | Definition |
|---|---|
| Shewhart MD | MCD-based Mahalanobis distance of the latent vector |
| EWMA MD | Exponentially weighted moving average of MD (smoothing λ) |
| RE | log(1 + MSE) reconstruction error |

Control limits are empirical quantiles of each statistic on normal training windows:

```text
UCL_Shewhart = Q_q(MD_train)
UCL_EWMA     = Q_q(EWMA_train)
UCL_RE       = Q_q(log(1 + RE_train))
```

### Alarm fusion

| Logic | Rule | Effect |
|---|---|---|
| OR | Any statistic exceeds its UCL | Highest recall, most false alarms |
| VOTE | At least two of three exceed | Balanced |
| AND | All three exceed | Highest precision |

## Data

**Server Machine Dataset (SMD)**, a public benchmark of server telemetry with labeled anomaly segments ([OmniAnomaly repository](https://github.com/NetManAIOps/OmniAnomaly)).

- Machine **1-1**, **10 selected sensors** (the `_s10` files)
- Sliding windows of length 30, stride 1
- The labeled test period is split into a validation set (30%) for tuning and a final test set (70%). The split keeps whole anomaly segments together, so the final set contains anomalies the tuning never saw.

## Results

All tuning was done on the validation set; the final test set was used once at the end. The grid search covered 36 settings: q ∈ {0.99, 0.995, 0.997, 0.999} × λ ∈ {0.1, 0.2, 0.3} × {OR, VOTE, AND}, ranked by point-adjusted F1.

### Validation — baseline (q = 0.997, λ = 0.2, OR)

| Evaluation | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Point-wise | 0.849 | 0.300 | 0.986 | 0.460 | 0.948 |
| Point-adjusted | 0.850 | 0.303 | 1.000 | 0.465 | 0.920 |

The baseline catches every anomaly segment but raises too many false alarms.

### Validation — fusion rule (point-adjusted)

| Logic | Precision | Recall | F1 |
|---|---:|---:|---:|
| OR | 0.303 | 1.000 | 0.465 |
| VOTE | 0.450 | 0.995 | 0.619 |
| AND | 0.535 | 0.995 | 0.696 |

### Validation — top grid-search settings (point-adjusted)

| Rank | q | λ | Logic | Precision | Recall | F1 |
|---:|---:|---:|:---:|---:|---:|---:|
| 1 | 0.999 | 0.1 | AND | 0.982 | 0.994 | **0.988** |
| 2 | 0.999 | 0.3 | AND | 0.897 | 0.994 | 0.943 |
| 3 | 0.999 | 0.1 | VOTE | 0.788 | 0.994 | 0.879 |

A tighter control limit (q = 0.999), lighter EWMA smoothing (λ = 0.1) and requiring all three statistics to agree (AND) removed most false alarms while still flagging almost every anomaly segment. The same setting held up on the final test set (table at the top).

> **About point-adjusted metrics.** Point adjustment counts a whole anomaly segment as detected if any point inside it raises an alarm. It is the usual protocol for SMD and reflects how an operator responds to an incident, but it reads higher than point-wise scoring, so both are reported for the baseline.

## Repository structure

```text
.
├── scripts/
│   ├── run_sanity.py        # end-to-end check on the full test period
│   ├── run_baseline.py      # baseline settings (q=0.997, λ=0.2, OR)
│   ├── run_gridsearch.py    # grid search over q, λ and fusion logic
│   └── run_final.py         # best setting evaluated on the final test set
├── src/aemcd/
│   ├── models.py            # denoising autoencoder (PyTorch)
│   ├── detector.py          # AE + MCD detector, control limits, EWMA
│   ├── windowing.py         # sliding windows ↔ point labels
│   ├── data_io.py           # segment-preserving validation/final split
│   ├── scoring.py           # alarm fusion
│   ├── evaluation.py        # point-wise and point-adjusted metrics
│   ├── gridsearch.py
│   ├── viz.py               # control charts and score histograms
│   └── seed.py
├── Quality Control Final Report v1.pdf
└── requirements.txt
```

## How to run

Python 3.12, seed 42.

```bash
pip install -r requirements.txt
```

Put the three SMD files for machine 1-1 (10 selected sensors) in a `data/` folder. The folder is git-ignored, so the data is not included in this repository.

```text
data/
├── machine-1-1_train_s10.csv
├── machine-1-1_test_s10.csv
└── machine-1-1_test_label.csv
```

Run the scripts from inside `data/` with `src/` on the Python path:

```bash
cd data
PYTHONPATH=../src python ../scripts/run_baseline.py
PYTHONPATH=../src python ../scripts/run_gridsearch.py
PYTHONPATH=../src python ../scripts/run_final.py
```

On Windows PowerShell, set the path first with `$env:PYTHONPATH = "..\src"`. The `.vscode/launch.json` file has the same four configurations for VS Code.

## Team

| Name | Student ID |
|---|---|
| 溫翎傑 | M11401204 |
| 陳喻宣 | M11401102 |
| 林侑萱 | M11401303 |
| 吳展佑 | M11401202 |

## References

- Zong, B. et al. (2018). Deep Autoencoding Gaussian Mixture Model for Unsupervised Anomaly Detection. *ICLR*.
- Rousseeuw, P. J. (1984). Least Median of Squares Regression. *Journal of the American Statistical Association*.
- Sakurada, M. & Yairi, T. (2014). Anomaly Detection Using Autoencoders with Nonlinear Dimensionality Reduction. *MLSDA Workshop*.
- Kim et al. (2020). Robust Multivariate Process Monitoring Using Deep Autoencoder. *Journal of Process Control*.
