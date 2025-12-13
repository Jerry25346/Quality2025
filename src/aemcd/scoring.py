import numpy as np

# Fused score calculation
def fused_score(d, ew, re, det):
    
    s = d / (det.ucl_shewhart + 1e-12)
    e = ew / (det.ucl_ewma + 1e-12)
    if det.use_re and det.ucl_re is not None:
        r = re / (det.ucl_re + 1e-12)
        return np.maximum(np.maximum(s, e), r)
    return np.maximum(s, e)
