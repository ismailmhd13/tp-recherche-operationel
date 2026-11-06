import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)


def assign_clusters(X, centroids):
    dist2 = np.zeros((X.shape[0], centroids.shape[0]))
    for l in range(centroids.shape[0]):
        dist2[:, l] = np.sum((X - centroids[l]) ** 2, axis=1)
    return np.argmin(dist2, axis=1)   
   

def update_centroids(X, labels, K, old_centroids=None):
    new = np.zeros((K, X.shape[1]))
    for j in range(K):
        membres = X[labels == j]
        if len(membres) > 0:
            new[j] = membres.mean(axis=0)
        elif old_centroids is not None:          
            new[j] = old_centroids[j]
    return new


if __name__ == "__main__":
    X_toy = np.array([[1.0, 1.0], [1.5, 2.0], [2.0, 1.5],
                      [8.0, 8.0], [9.0, 8.5], [8.5, 9.0]])
    mu_init = np.array([[1.0, 2.0], [9.0, 9.0]])   
    K = 2

    mu = mu_init.copy()
    historique = [mu.copy()]
    for t in range(20):
        labels = assign_clusters(X_toy, mu)            
        mu_new = update_centroids(X_toy, labels, K, mu)  
        delta = np.sum((mu_new - mu) ** 2)
       
        mu = mu_new
        historique.append(mu.copy())
        if delta <= 1e-9:
            break

    couleurs = np.array(["tab:blue", "tab:orange"])
    n = len(historique)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.2))
    for s, ax in enumerate(np.atleast_1d(axes)):
        mu_s = historique[s]
        lab_s = assign_clusters(X_toy, mu_s)
        ax.scatter(X_toy[:, 0], X_toy[:, 1], c=couleurs[lab_s], s=70, edgecolor="k")
        for j in range(K):
            traj = np.array([h[j] for h in historique[: s + 1]])
            ax.plot(traj[:, 0], traj[:, 1], "--", color=couleurs[j], alpha=0.7)
        ax.scatter(mu_s[:, 0], mu_s[:, 1], c=couleurs, marker="X", s=250, edgecolor="k")
        ax.set_title("Initialisation" if s == 0 else f"Après itération {s}")
        ax.set_xlim(0, 10); ax.set_ylim(0, 10)
        ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.grid(alpha=0.3)
    fig.suptitle("K-Means (L2²) : déplacement des centroïdes (X = centroïde)")
    fig.tight_layout()
    fig.savefig("kmeans_toy.png", dpi=120)
    plt.show()
