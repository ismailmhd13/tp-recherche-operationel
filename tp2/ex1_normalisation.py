
import numpy as np

np.set_printoptions(precision=4, suppress=True)





def my_min_max_scaler(X):
   
    X = np.asarray(X, dtype=float)
    x_min = X.min(axis=0, keepdims=True)          
    x_max = X.max(axis=0, keepdims=True)          
    etendue = x_max - x_min
    etendue[etendue == 0] = 1.0

    X_scaled = (X - x_min) / etendue
    return X_scaled, x_min, x_max







