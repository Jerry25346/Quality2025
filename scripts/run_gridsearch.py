import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from aemcd.seed import set_all_seeds
from aemcd.windowing import make_windows
from aemcd.data_io import stratified_split_anomalies
from aemcd.gridsearch import grid_search_hyperparameters
from aemcd.detector import AE_MCD_Detector

if __name__ == "__main__":
    set_all_seeds(42)

    print("="*60)
    print("Grid search only")
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

    (X_val, y_val), (X_final, y_final) = stratified_split_anomalies(
        Xtest, y, val_ratio=0.3, random_state=42
    )

    Xtr_w,  _, _ = make_windows(Xtrain, y=None, win=WIN, stride=STRIDE_TR)
    Xval_w, _, _ = make_windows(X_val,  y=None, win=WIN, stride=STRIDE_TE)

    input_dim = Xtr_w.shape[1]
    latent_dim = 6

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
