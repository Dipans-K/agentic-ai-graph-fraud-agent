import numpy as np


def normalized_laplacian(adjacency):
    A = np.asarray(adjacency, float)
    degree = A.sum(axis=1)
    inv = np.diag(1 / np.sqrt(np.maximum(degree, 1e-12)))
    return np.eye(len(A)) - inv @ A @ inv


def spectral_analysis(adjacency, k=2):
    values, vectors = np.linalg.eigh(normalized_laplacian(adjacency))
    return values[:k], vectors[:, :k]


def random_walk(adjacency, seed, alpha=.85, steps=30):
    A = np.asarray(adjacency, float)
    P = A / (A.sum(axis=1, keepdims=True) + 1e-12)
    restart = np.zeros(len(A)); restart[seed] = 1
    score = restart.copy()
    for _ in range(steps):
        score = alpha * (score @ P) + (1-alpha) * restart
    return score


def agent(adjacency):
    values, _ = spectral_analysis(adjacency)
    walk = random_walk(adjacency, 0)
    return {"eigenvalues": values.tolist(), "random_walk_score": walk.tolist()}


if __name__ == "__main__":
    graph = [[0,1,1,0],[1,0,1,0],[1,1,0,1],[0,0,1,0]]
    print(agent(graph))
