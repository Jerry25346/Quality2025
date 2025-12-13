import numpy as np

# Utility functions for windowing and evaluation
def make_windows(X, y=None, win=30, stride=1):
    
    X = np.asarray(X, dtype=np.float32)
    T, D = X.shape
    # length of windows striding from 0 to T-win
    idx = np.arange(0, T - win + 1, stride)
    Xw = np.stack([X[i:i+win].reshape(-1) for i in idx], axis=0) # shape: (num_windows, win*D) input for AE
    center_idx = idx + win // 2  #record center index of each window

    if y is None:
        return Xw, None, center_idx
    # windowed labels: if any point in window is anomaly, label the window as anomaly
    y = np.asarray(y).astype(int)
    yw = np.array([1 if y[i:i+win].max() > 0 else 0 for i in idx], dtype=int)
    return Xw, yw, center_idx

# Convert windowed scores back to pointwise scores
def window_to_point(scores_w, n, win=30, stride=1, mode="max"):
    scores_w = np.asarray(scores_w)
    out = np.full(n, -np.inf if mode == "max" else 0.0, dtype=float)
    cnt = np.zeros(n, dtype=int)

    starts = np.arange(0, n - win + 1, stride)
    for w_i, s in enumerate(starts):
        e = s + win
        if mode == "max":
            out[s:e] = np.maximum(out[s:e], scores_w[w_i])
        else:
            out[s:e] += scores_w[w_i]
            cnt[s:e] += 1

    if mode != "max":
        cnt[cnt == 0] = 1
        out = out / cnt
    else:
        out[np.isneginf(out)] = 0.0
    return out    