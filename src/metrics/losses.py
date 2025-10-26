import numpy as np

def binary_ce_from_probs(p1, y):
    eps = 1e-8
    p1 = np.clip(p1, eps, 1 - eps)
    y = y.astype(float)
    return -np.mean(y * np.log(p1) + (1 - y) * np.log(1 - p1))
