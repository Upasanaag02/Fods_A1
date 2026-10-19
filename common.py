import numpy as np
import pandas as pd

SEED = 42

def load(path):
    df = pd.read_csv(path)
    return df["x"].to_numpy(), df["y"].to_numpy()

def vander(x, d):
    return np.vander(x, d + 1, increasing=True)

def rmse(y, yhat):
    return float(np.sqrt(np.mean((y - yhat) ** 2)))
