import numpy as np
from src.main import normalized_laplacian, spectral_analysis, random_walk


def test_laplacian_symmetry():
    A = [[0,1,1],[1,0,1],[1,1,0]]
    L = normalized_laplacian(A)
    assert np.allclose(L, L.T)


def test_graph_scores():
    A = [[0,1,1],[1,0,1],[1,1,0]]
    values, vectors = spectral_analysis(A)
    assert len(values) == 2
    assert vectors.shape == (3,2)
    assert len(random_walk(A,0)) == 3
