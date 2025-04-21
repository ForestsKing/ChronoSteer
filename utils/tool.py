import os
import random

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def evaluate(pred, true):
    scaler = StandardScaler()
    true, pred = np.array(true), np.array(pred)
    true = scaler.fit_transform(true.reshape(-1, 1)).reshape(-1)
    pred = scaler.transform(pred.reshape(-1, 1)).reshape(-1)

    mse = np.mean(np.square(true - pred))
    mae = np.mean(np.abs(true - pred))

    return mse, mae
