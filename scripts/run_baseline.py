#import necessary libraries py 3.12.9
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.covariance import MinCovDet
from sklearn.metrics import precision_recall_fscore_support , confusion_matrix ,classification_report
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import random
import os

from aemcd.seed import set_all_seeds
from aemcd.models import Autoencoder
from aemcd.detector import AE_MCD_Detector
from aemcd.windowing import make_windows, window_to_point
from aemcd.data_io import stratified_split_anomalies
from aemcd.scoring import fused_score
from aemcd.evaluation import point_adjust, evaluate_and_report
from aemcd.viz import plot_score_hist, plot_monitoring_charts

if __name__ == "__main__":
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    
    set_all_seeds(42)

    print("="*60)
    print("AE + MCD + Shewhart/EWMA expreimemt")
    print("="*60)

    # ---- 1. import data ----
    print("\n1. import data...")
    train  = pd.read_csv("machine-1-1_train_s10.csv", header=None).values
    test   = pd.read_csv("machine-1-1_test_s10.csv",  header=None).values
    labels = pd.read_csv("machine-1-1_test_label.csv", header=None).values

    # point-level label
    y = (labels.max(axis=1) > 0).astype(int)

    # standardize d data
    scaler = StandardScaler().fit(train)
    Xtrain = scaler.transform(train)
    Xtest  = scaler.transform(test)

    WIN = 30
    STRIDE_TR = 1
    STRIDE_TE = 1

    (X_val, y_val), (X_final, y_final) = stratified_split_anomalies(
        Xtest, y, val_ratio=0.3, random_state=42
    )
    
    
    print(f"\n split data :")
    print(f"  total samples : {len(y)},  all anomaly: {y.sum()} ({y.mean():.1%})")
    print(f"  VAL:   {len(y_val)} samples, {int(y_val.sum())} anomaly ({y_val.mean():.1%})")
    print(f"  FINAL: {len(y_final)} samples, {int(y_final.sum())} anomaly ({y_final.mean():.1%})")
    
    # check anomaly ratio difference
    ratio_diff = abs(y_val.mean() - y_final.mean())
    if ratio_diff < 0.05:
        print(f"    (difference  {ratio_diff:.1%})")
    else:
        print(f"   {ratio_diff:.1%})")
    # ==========================================================

    # windows
    Xtr_w,    _, _ = make_windows(Xtrain, y=None,  win=WIN, stride=STRIDE_TR)
    Xval_w,   _, _ = make_windows(X_val,  y=None,  win=WIN, stride=STRIDE_TE)
    Xfinal_w, _, _ = make_windows(X_final, y=None, win=WIN, stride=STRIDE_TE)

    # ---- 2. Establish a basic model ----
    print("\n2. Establish a basic model...")
    input_dim = Xtr_w.shape[1]
    latent_dim = 6
    ae = Autoencoder(input_dim=input_dim, latent_dim=latent_dim, noise_std=0.03)

    # ---- 3. fundemental experiment（q=0.997, λ=0.2, logic=OR）----
    set_all_seeds(42)
    print("\n" + "="*60)
    print("3. fundemental experiment（q=0.997, λ=0.2, logic=OR）")
    print("="*60)

    det_base = AE_MCD_Detector(
        ae, lr=1e-3, batch_size=128, epochs=60,
        q=0.997, ewma_lambda=0.2, use_re=True
    )
    det_base.fit(Xtr_w, val_split=0.1, patience=10)

    print("\n predict plot :")
    det_base.plot_training_history()

    # ---- Val predict ----
    pred_w, d_w, ew_w, re_w = det_base.predict(Xval_w, logic='or', return_details=True)

    pred_point = window_to_point(pred_w, n=len(y_val), win=WIN, stride=STRIDE_TE, mode="max")
    d_point    = window_to_point(d_w,    n=len(y_val), win=WIN, stride=STRIDE_TE, mode="max")
    ew_point   = window_to_point(ew_w,   n=len(y_val), win=WIN, stride=STRIDE_TE, mode="max")
    re_point   = window_to_point(re_w,   n=len(y_val), win=WIN, stride=STRIDE_TE, mode="max")

    pred_adj = point_adjust(y_val, pred_point)

    # baseline score ）
    score_base = fused_score(d_point, ew_point, re_point, det_base)
    evaluate_and_report(y_val, pred_point, pred_adj, title="Window-based 基礎實驗 (VAL)", y_score=score_base)

    plot_monitoring_charts(d_point, ew_point, re_point, y_val, det_base, title_suffix="(Val)")

    # ---- 4. adjust q value）----
    set_all_seeds(42)

    q_results = []
    for q_val in [0.99, 0.995, 0.997, 0.999]:
        print(f"\ntest q={q_val:.3f}...")
        ae_q = Autoencoder(input_dim=input_dim, latent_dim=latent_dim, noise_std=0.03)
        det_q = AE_MCD_Detector(
            ae_q, lr=1e-3, batch_size=128, epochs=60,
            q=q_val, ewma_lambda=0.2, use_re=True
        )
        det_q.fit(Xtr_w, val_split=0.1, patience=10, verbose=False)

        pred_q_w = det_q.predict(Xval_w, logic="or", return_details=False)
        pred_q = window_to_point(pred_q_w, n=len(y_val), win=WIN, stride=STRIDE_TE, mode="max")
        pred_q_adj = point_adjust(y_val, pred_q)

        p, r, f1, _ = precision_recall_fscore_support(y_val, pred_q_adj, average="binary", zero_division=0)
        q_results.append({"q": q_val, "P": p, "R": r, "F1": f1})
        print(f"  result (PA): P={p:.3f}, R={r:.3f}, F1={f1:.3f}")

    df_q = pd.DataFrame(q_results)
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(df_q))
    width = 0.25
    ax.bar(x - width, df_q["P"], width, label="Precision", alpha=0.85)
    ax.bar(x,          df_q["R"], width, label="Recall",    alpha=0.85)
    ax.bar(x + width,  df_q["F1"], width, label="F1",       alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{q:.3f}" for q in df_q["q"]])
    ax.set_xlabel("q value")
    ax.set_ylabel("Score (Point-Adjusted)")
    ax.set_title("Impact of q (Quantile for UCL) [VAL]")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.show()

    # ---- 5. different logics（on VAL）----
    print("\n" + "=" * 70)
    print(" 5. test on different logics(on VAL)")
    print("=" * 70)

    logic_results = []
    for logic in ["or", "vote", "and"]:
        pred_logic_w = det_base.predict(Xval_w, logic=logic, return_details=False)
        pred_logic = window_to_point(pred_logic_w, n=len(y_val), win=WIN, stride=STRIDE_TE, mode="max")
        pred_logic_adj = point_adjust(y_val, pred_logic)

        p, r, f1, _ = precision_recall_fscore_support(y_val, pred_logic_adj, average="binary", zero_division=0)
        logic_results.append({"logic": logic, "P": p, "R": r, "F1": f1})
        print(f"logic={logic:>4} -> (PA) P={p:.3f}, R={r:.3f}, F1={f1:.3f}")

    df_logic = pd.DataFrame(logic_results)
    print("\nlogic compare:")
    print(df_logic.to_string(index=False))

    # ---- 6. plotting （VAL windows）----
    print("\n VAL windows...")
    Xval_w_vis, yval_w, _ = make_windows(X_val, y=y_val, win=WIN, stride=STRIDE_TE)
    X_normal_vis = Xval_w_vis[yval_w == 0][:500]
    X_anomaly_vis = Xval_w_vis[yval_w == 1][:200]
    det_base.plot_latent_space(X_normal_vis, X_anomaly_vis)

    print("\n plotting re ...")
    det_base.plot_reconstruction(Xval_w_vis[:5], n_examples=5)

    print("\n analyse anomaly samples...")
    pred_val_w = det_base.predict(Xval_w_vis, logic="or", return_details=False)
    anom_win_idx = np.where((yval_w == 1) & (pred_val_w == 1))[0]
    if len(anom_win_idx) > 0:
        j = anom_win_idx[0]
        print(f"\n analyse anomaly window #{j}:")
        det_base.explain_anomaly(Xval_w_vis[j])
    else:
        print("VAL No")
