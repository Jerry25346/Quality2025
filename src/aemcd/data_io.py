import numpy as np

def split_test_for_val(Xtest, y, val_ratio=0.2):
    
    n = len(y)
    n_val = int(n * val_ratio)
    X_val, y_val = Xtest[:n_val], y[:n_val]
    X_te,  y_te  = Xtest[n_val:], y[n_val:]
    return (X_val, y_val), (X_te, y_te), n_val

#split test into val and final sets with stratified anomaly segments
def stratified_split_anomalies(X, y, val_ratio=0.3, random_state=42):
        
    np.random.seed(random_state)
    T = len(y)
    n_val = int(T * val_ratio)
        
    # detect anomaly segments
    anomaly_segments = []
    in_seg = False
    start = 0
        
    for i, label in enumerate(y):
        if label == 1 and not in_seg:
            in_seg = True
            start = i
        elif (label == 0 or i == T-1) and in_seg:
            end = i if label == 0 else i+1
            anomaly_segments.append((start, end))
            in_seg = False
        
    if not anomaly_segments:
        return (X[:n_val], y[:n_val]), (X[n_val:], y[n_val:])
        
    # randomly select segments for val set
    n_segments = len(anomaly_segments)
    n_val_segments = max(1, int(n_segments * val_ratio))
    val_segment_indices = np.random.choice(n_segments, n_val_segments, replace=False)
        
    # construct mask
    val_mask = np.zeros(T, dtype=bool)
    for idx in val_segment_indices:
        s, e = anomaly_segments[idx]
        val_mask[s:e] = True
        
    # ensure val set has enough normal points
    normal_indices = np.where((y == 0) & (~val_mask))[0]
    np.random.shuffle(normal_indices)
    current_val_size = val_mask.sum()
    need_more = n_val - current_val_size
        
    if need_more > 0:
        extra_normal = normal_indices[:need_more]
        val_mask[extra_normal] = True
        
    # split data
    val_indices = np.where(val_mask)[0]
    final_indices = np.where(~val_mask)[0]
        
    return (X[val_indices], y[val_indices]), (X[final_indices], y[final_indices])
