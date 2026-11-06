"""
TP n°2 — Exercice 2 (Partie I : K-Means), Étape 2 — section 2.3
Application sur base réelle : Wine Dataset (N = 178, d = 13)

Exécution : python ex2_partie2_kmeans_wine.py
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine

np.set_printoptions(precision=4, suppress=True)


# --- Normalisation (fonction de l'exercice 1) ---------------------------------
def my_min_max_scaler(X):
    """Retourne : X_scaled, x_min (1 x d), x_max (1 x d)"""
    X = np.asarray(X, dtype=float)
    x_min = X.min(axis=0, keepdims=True)
    x_max = X.max(axis=0, keepdims=True)
    etendue = x_max - x_min
    etendue[etendue == 0] = 1.0
    return (X - x_min) / etendue, x_min, x_max


# --- Étapes de K-Means (fonctions de la partie 2.2) ---------------------------
def assign_clusters(X, centroids):
    """Étape E : affectation au centroïde le plus proche en L2^2."""
    dist2 = np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    return np.argmin(dist2, axis=1)


def update_centroids(X, labels, K, old_centroids=None):
    """Étape M : barycentre de chaque cluster."""
    new = np.zeros((K, X.shape[1]))
    for j in range(K):
        membres = X[labels == j]
        if len(membres) > 0:
            new[j] = membres.mean(axis=0)
        elif old_centroids is not None:
            new[j] = old_centroids[j]
    return new


def inertie(X, labels, centroids):
    """Somme des distances L2^2 intra-cluster (ce que K-Means minimise)."""
    return np.sum((X - centroids[labels]) ** 2)


def kmeans(X, K, eps=1e-6, max_iter=100, seed=0):
    """K-Means complet (Lloyd) avec affichage de Δ(t) à chaque itération."""
    rng = np.random.default_rng(seed)
    mu = X[rng.choice(len(X), size=K, replace=False)].copy()   # 1. Initialisation
    deltas = []
    for t in range(max_iter):
        labels = assign_clusters(X, mu)                         # 2. Étape E
        mu_new = update_centroids(X, labels, K, mu)             # 3. Étape M
        delta = np.sum((mu_new - mu) ** 2)                      # Δ(t)
        deltas.append(delta)
        print(f"  t = {t:2d} | Δ(t) = {delta:.6e} | inertie = {inertie(X, labels, mu_new):.4f}")
        mu = mu_new
        if delta <= eps:                                        # 4. Arrêt
            print(f"  -> Convergence : Δ(t) ≤ ε = {eps} après {t + 1} itérations")
            break
    return mu, assign_clusters(X, mu), deltas


if __name__ == "__main__":
    wine = load_wine()
    X_wine, y_wine = wine.data, wine.target

    # 1. Normalisation (pas de test ici -> toute la base)
    X_wine_s, _, _ = my_min_max_scaler(X_wine)
    print("Après normalisation : min =", X_wine_s.min(), " max =", X_wine_s.max())

    # 2. K-Means complet avec K = 3
    print("\nK-Means sur Wine, K = 3")
    mu_w, lab_w, deltas_w = kmeans(X_wine_s, K=3, eps=1e-6, seed=42)

    # 3. Vérification de la décroissance de Δ(t)
    d = np.array(deltas_w)
    print("\nΔ(t) décroissant :", bool(np.all(np.diff(d) <= 1e-12)))
    print("Taille des clusters :", np.bincount(lab_w))

    # Bonus : comparaison avec les vraies classes (cluster -> classe majoritaire)
    correct = sum(np.bincount(y_wine[lab_w == j]).max() for j in range(3) if np.any(lab_w == j))
    print(f"Pureté des clusters vs vraies classes : {correct / len(y_wine):.2%}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(range(len(deltas_w)), np.maximum(deltas_w, 1e-16), "o-")
    ax.axhline(1e-6, color="r", ls="--", label="ε = 1e-6")
    ax.set_xlabel("Itération t"); ax.set_ylabel("Δ(t)  (échelle log)")
    ax.set_title("Wine : déplacement total des centroïdes")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("kmeans_wine_delta.png", dpi=120)
    plt.show()
