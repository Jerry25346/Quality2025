import numpy as np
from itertools import product
from sklearn.metrics import precision_recall_fscore_support

from aemcd.models import Autoencoder
from aemcd.detector import AE_MCD_Detector
from aemcd.windowing import window_to_point
from aemcd.evaluation import point_adjust

# Grid search
def grid_search_hyperparameters(detector_class, input_dim, Xtrain_w, Xval_w, y_val_point, win=30, stride=1,
                                latent_dim=12, noise_std=0.05,
                                q_values=[0.99, 0.995, 0.997],
                                lambda_values=[0.1, 0.2, 0.3],
                                logic_values=['or', 'vote', 'and'],
                                score_on="pa_f1",
                                verbose=True):
   
    print(f"\n{'='*60}")
    print(f"grid search start ...")
    print(f"{'='*60}\n")
    
    results = []
    best_score = -1.0
    best_params = None
    best_detector = None
    
    
    
    for q, lam, logic in product(q_values, lambda_values, logic_values):
        ae = Autoencoder(input_dim=input_dim, latent_dim=latent_dim, noise_std=noise_std)
        det = detector_class(ae, q=q, ewma_lambda=lam, use_re=True)
        
        # train
        det.fit(Xtrain_w, val_split=0.1, patience=10, verbose=False)
        
        # predict on val windows
        pred_w = det.predict(Xval_w, logic=logic, return_details=False)
        # map window preds -> point preds
        pred_point = window_to_point(pred_w, n=len(y_val_point), win=win, stride=stride, mode="max")
        pred_pa = point_adjust(y_val_point, pred_point)

        
        p_raw, r_raw, f1_raw, _ = precision_recall_fscore_support(
            y_val_point, pred_point, average="binary", zero_division=0
        )
        p_pa, r_pa, f1_pa, _ = precision_recall_fscore_support(
            y_val_point, pred_pa, average="binary", zero_division=0
        )

        # choose score
        score = f1_pa if score_on == "pa_f1" else f1_raw

        if verbose:
            print(f"[q={q:.3f}, λ={lam:.2f}, logic={logic:>4}] "
                  f"RAW(F1={f1_raw:.3f}, P={p_raw:.3f}, R={r_raw:.3f}) | "
                  f"PA(F1={f1_pa:.3f}, P={p_pa:.3f}, R={r_pa:.3f}) -> score={score:.3f}")

        results.append({
            "q": q,
            "lambda": lam,
            "logic": logic,
            "p_raw": p_raw, "r_raw": r_raw, "f1_raw": f1_raw,
            "p_pa": p_pa,   "r_pa": r_pa,   "f1_pa": f1_pa,
            "score": score
        })

        if score > best_score:
            best_score = score
            best_params = {"q": q, "lambda": lam, "logic": logic}
            best_detector = det

    import pandas as pd
    df_results = pd.DataFrame(results).sort_values("score", ascending=False)

    print(f"\n{'='*60}")
    print("grid search (Top 10 by score):")
    print(f"{'='*60}")
    print(df_results.head(10).to_string(index=False))

    print(f"\n{'='*60}")
    print("best parameters:")
    print(f"{'='*60}")
    print(f"  score_on = {score_on}")
    print(f"  q        = {best_params['q']:.3f}")
    print(f"  lambda   = {best_params['lambda']:.2f}")
    print(f"  logic    = {best_params['logic']}")
    print(f"  best_score = {best_score:.3f}")
    print(f"{'='*60}\n")

    return best_detector, best_params, df_results
