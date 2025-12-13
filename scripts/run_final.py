import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support

from aemcd.seed import set_all_seeds
from aemcd.models import Autoencoder
from aemcd.detector import AE_MCD_Detector
from aemcd.windowing import make_windows, window_to_point
from aemcd.data_io import stratified_split_anomalies
from aemcd.scoring import fused_score
from aemcd.evaluation import point_adjust, evaluate_and_report
from aemcd.viz import plot_score_hist, plot_monitoring_charts
from aemcd.gridsearch import grid_search_hyperparameters

if __name__ == "__main__":
    set_all_seeds(42)

    print("="*60)
    print("AE + MCD + Shewhart/EWMA expreimemt (FINAL)")
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

    # ---- 7. Grid search（用 VAL 的 point label y_val）----
    set_all_seeds(42)
    print("\n" + "=" * 70)
    print("🔎 7.  Grid search (VAL on PA-F1 rank)")
    print("=" * 70)

    best_det, best_params, df_grid = grid_search_hyperparameters(
        AE_MCD_Detector, input_dim, Xtr_w, Xval_w, y_val,
        win=WIN, stride=STRIDE_TE,
        latent_dim=latent_dim, noise_std=0.03,
        q_values=[0.99, 0.995, 0.997, 0.999],
        lambda_values=[0.1, 0.2, 0.3],
        logic_values=['or', 'vote',"and"],
        score_on="pa_f1",
        verbose=True
    )

    # ---- 8. Final （FINAL set）----
    # (A) Baseline on FINAL
    pred_base_w, d_base_w, ew_base_w, re_base_w = det_base.predict(
        Xfinal_w, logic="or", return_details=True
    )
    pred_base_point = window_to_point(pred_base_w, n=len(y_final), win=WIN, stride=STRIDE_TE, mode="max")
    d_base_point    = window_to_point(d_base_w,    n=len(y_final), win=WIN, stride=STRIDE_TE, mode="max")
    ew_base_point   = window_to_point(ew_base_w,   n=len(y_final), win=WIN, stride=STRIDE_TE, mode="max")
    re_base_point   = window_to_point(re_base_w,   n=len(y_final), win=WIN, stride=STRIDE_TE, mode="max")

    pred_base_adj = point_adjust(y_final, pred_base_point)

    #  baseline score 
    score_base_final = fused_score(d_base_point, ew_base_point, re_base_point, det_base)
    plot_score_hist(y_final, score_base_final, title="Baseline fused score (FINAL)")
    evaluate_and_report(
        y_final, pred_base_point, pred_base_adj,
        title="Baseline on FINAL (q=0.997, λ=0.2, logic=or)",
        y_score=score_base_final
    )

    # (B) Best params on FINAL
    pred_best_w, d_best_w, ew_best_w, re_best_w = best_det.predict(
        Xfinal_w, logic=best_params["logic"], return_details=True
    )
    pred_best_point = window_to_point(pred_best_w, n=len(y_final), win=WIN, stride=STRIDE_TE, mode="max")
    d_best_point    = window_to_point(d_best_w,    n=len(y_final), win=WIN, stride=STRIDE_TE, mode="max")
    ew_best_point   = window_to_point(ew_best_w,   n=len(y_final), win=WIN, stride=STRIDE_TE, mode="max")
    re_best_point   = window_to_point(re_best_w,   n=len(y_final), win=WIN, stride=STRIDE_TE, mode="max")

    pred_best_adj = point_adjust(y_final, pred_best_point)
    score_best_final = fused_score(d_best_point, ew_best_point, re_best_point, best_det)

    evaluate_and_report(
        y_final, pred_best_point, pred_best_adj,
        title=f"Best on FINAL (q={best_params['q']:.3f}, λ={best_params['lambda']:.2f}, logic={best_params['logic']})",
        y_score=score_best_final
    )
    plot_score_hist(y_final, score_best_final, title="Best fused score (FINAL)")
    plot_monitoring_charts(d_best_point, ew_best_point, re_best_point, y_final, best_det, title_suffix="(最佳參數, FINAL)")

    # ---- 9. Summary (Point-Adjusted) ----
    p_base, r_base, f1_base, _ = precision_recall_fscore_support(
        y_final, pred_base_adj, average="binary", zero_division=0
    )
    p_best, r_best, f1_best, _ = precision_recall_fscore_support(
        y_final, pred_best_adj, average="binary", zero_division=0
    )

    print("\n" + "=" * 70)
    print(" 9. summary (Point-Adjusted, FINAL)")
    print("=" * 70)
    print(f"base: P={p_base:.3f} R={r_base:.3f} F1={f1_base:.3f}")
    print(f"best: P={p_best:.3f} R={r_best:.3f} F1={f1_best:.3f}")
    if f1_base > 0:
        print(f"Degree of improvement: {(f1_best / f1_base - 1) * 100:+.1f}%")
