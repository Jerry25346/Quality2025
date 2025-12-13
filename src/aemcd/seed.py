import torch

# set seed 
def set_all_seeds(seed=42):
    import os, random
    import numpy as np


    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    #force deterministic algorithms
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
