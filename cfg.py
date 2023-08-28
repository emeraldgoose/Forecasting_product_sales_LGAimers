import os
import random
import numpy as np
import pandas as pd
import torch

class CFG:
    train_window_size = 90
    predict_size = 21
    epochs = 10
    learning_rate = 1e-3
    batch_size = 1024 * 3
    seed = 41

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True