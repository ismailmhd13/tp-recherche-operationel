import numpy as np
from scipy.optimize import linprog, minimize_scalar


# Fonction objectif
def f(x):
    return (x[0] - 2)**2 + (x[1] - 1)**2 + 3 * (x[2] - 3)**2


# Gradient de f
def df(x):
    return np.array([
        2 * (x[0] - 2),
        2 * (x[1] - 1),
        6 * (x[2] - 3)
    ])


def gradient(f, x):
    return df(x)


# Vérification de convexité
def convex(x, y, lam):
    lam = np.clip(lam, 0, 1)

    gauche = f(lam * x + (1 - lam) * y)
    droite = lam * f(x) + (1 - lam) * f(y)

    return gauche <= droite


# Sous-problème linéaire
def sous_probleme(grad):
    resultat = linprog(
        c=grad,
        A_ub=[[1, 1, 1]],
        b_ub=[4],
        bounds=[
            (0, None),
            (0, None),
            (0, None)
        ],
        method="highs"
    )

    return resultat.x


# Recherche du pas optimal
def pas_optimal(f, x, d):

    def phi(gamma):
        return f(x + gamma * d)

    resultat = minimize_scalar(
        phi,
        bounds=(0, 1),
        method="bounded"
    )

    return resultat.x




x = np.array([0.0, 0.0, 0.0])



for i in range(11):

    # Gradient au point courant
    grad = gradient(f, x)

    # Résolution du sous-problème
    s = sous_probleme(grad)
    print("s0=", s)

    # Direction
    d = s - x

    # Pas optimal
    gamma = pas_optimal(f, x, d)
    print('Pas optimal:', gamma)

    # Nouveau point
    x = x + gamma * d
    
    print(x)
    