

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from ex1_normalisation import my_min_max_scaler
np.set_printoptions(precision=4, suppress=True)


def minkowski(x, y, p=2):
    return np.sum(np.abs(x - y) ** p, axis=-1) ** (1.0 / p)





def apply_min_max(X, x_min, x_max):
    etendue = np.where((x_max - x_min) == 0, 1.0, x_max - x_min)
    return (np.asarray(X, dtype=float) - x_min) / etendue


class KNNClassifier:
    def __init__(self, k=5, p=2):
        self.k = k
        self.p = p

    def fit(self, X, y):
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y)
        self.classes_ = np.unique(self.y_train)
        return self

    def predict(self, X_test):
        X_test = np.atleast_2d(np.asarray(X_test, dtype=float))
        diff = np.abs(X_test[:, None, :] - self.X_train[None, :, :])
        D = np.sum(diff ** self.p, axis=2) ** (1.0 / self.p)
        voisins = np.argsort(D, axis=1, kind="stable")[:, : self.k]
        y_pred = []
        for idx in voisins:
            votes = np.array([np.sum(self.y_train[idx] == c) for c in self.classes_])
            y_pred.append(self.classes_[np.argmax(votes)])   
        return np.array(y_pred)


if __name__ == "__main__":
   
   

    
    bc = load_breast_cancer()

    X_tr, X_te, y_tr, y_te = train_test_split(
        bc.data, bc.target, test_size=0.25, random_state=42, stratify=bc.target)
    print(f"Train : {X_tr.shape}, Test : {X_te.shape}")

    X_tr_s, bc_min, bc_max = my_min_max_scaler(X_tr)
    X_te_s = apply_min_max(X_te, bc_min, bc_max)

    ks = list(range(1, 26, 2))
    acc = {1: [], 2: []}
    print(f"\n{'k':>3} | {'Acc L1 (Manhattan)':>19} | {'Acc L2 (Euclid.)':>17}")
    for k in ks:
        for p in (1, 2):
            y_pred = KNNClassifier(k=k, p=p).fit(X_tr_s, y_tr).predict(X_te_s)
            acc[p].append(np.mean(y_pred == y_te))
        print(f"{k:>3} | {acc[1][-1]:>19.4f} | {acc[2][-1]:>17.4f}")

    for p in (1, 2):
        b = int(np.argmax(acc[p]))
        print(f"Meilleur L{p} : k = {ks[b]}, accuracy = {acc[p][b]:.4f}")

   


