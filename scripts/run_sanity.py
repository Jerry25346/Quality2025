import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from aemcd.seed import set_all_seeds
from aemcd.models import Autoencoder
from aemcd.detector import AE_MCD_Detector
from aemcd.windowing import make_windows, window_to_point
from aemcd.scoring import fused_score
from aemcd.evaluation import point_adjust, evaluate_and_report
from aemcd.viz import plot_monitoring_charts

if __name__ == "__main__":
    set_all_seeds(42)

    print("="*60)
    print("Sanity Check - FULL TEST")
    print("="*60)

    train  = pd.read_csv("machine-1-1_train_s10.csv", header=None).values
    test   = pd.read_csv("machine-1-1_test_s10.csv",  header=None).values
    labels = pd.read_csv("machine-1-1_test_label.csv", header=None).values

    y = (labels.max(axis=1) > 0).astype(int)

    scaler = StandardScaler().fit(train)
    Xtrain = scaler.transform(train)
    Xtest  = scaler.transform(test)

    WIN = 30
    STRIDE_TR = 1
    STRIDE_TE = 1

    Xtr_w, _, _ = make_windows(Xtrain, y=None, win=WIN, stride=STRIDE_TR)

    input_dim = Xtr_w.shape[1]
    latent_dim = 6
    ae = Autoencoder(input_dim=input_dim, latent_dim=latent_dim, noise_std=0.03)

    det_base = AE_MCD_Detector(
        ae, lr=1e-3, batch_size=128, epochs=60,
        q=0.997, ewma_lambda=0.2, use_re=True
    )
    det_base.fit(Xtr_w, val_split=0.1, patience=10, verbose=False)

    #sanity check on full test set
    Xfull_w, _, _ = make_windows(Xtest, y=None, win=WIN, stride=STRIDE_TE)

    # baseline (or/vote/and )
    pred_full_w, d_full_w, ew_full_w, re_full_w = det_base.predict(Xfull_w, logic="and", return_details=True)

    pred_full_point = window_to_point(pred_full_w, n=len(y), win=WIN, stride=STRIDE_TE, mode="max")
    d_full_point    = window_to_point(d_full_w,    n=len(y), win=WIN, stride=STRIDE_TE, mode="max")
    ew_full_point   = window_to_point(ew_full_w,   n=len(y), win=WIN, stride=STRIDE_TE, mode="max")
    re_full_point   = window_to_point(re_full_w,   n=len(y), win=WIN, stride=STRIDE_TE, mode="max")

    pred_full_adj = point_adjust(y, pred_full_point)
    score_full = fused_score(d_full_point, ew_full_point, re_full_point, det_base)

    evaluate_and_report(y, pred_full_point, pred_full_adj, title="Sanity Check - det_base on FULL TEST", y_score=score_full)
    plot_monitoring_charts(d_full_point, ew_full_point, re_full_point, y, det_base, title_suffix="(FULL TEST)")
