import numpy as np
import scipy.sparse as sp
from scipy.sparse import linalg


def calculate_scaled_laplacian(adj: np.ndarray, lambda_max: int = 2, undirected: bool = True) -> np.matrix:
    """Rescale the normalized Laplacian eigenvalues to [-1, 1]."""
    if undirected:
        adj = np.maximum.reduce([adj, adj.T])
    laplacian_matrix = calculate_symmetric_normalized_laplacian(adj)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(laplacian_matrix, 1, which="LM")
        lambda_max = lambda_max[0]
    laplacian_matrix = sp.csr_matrix(laplacian_matrix)
    num_nodes, _ = laplacian_matrix.shape
    identity_matrix = sp.identity(num_nodes, format="csr", dtype=laplacian_matrix.dtype)
    return (2 / lambda_max * laplacian_matrix) - identity_matrix


def calculate_symmetric_message_passing_adj(adj: np.ndarray) -> np.matrix:
    """Calculate the renormalized message-passing adjacency D^{-1/2}(A+I)D^{-1/2}."""
    adj = adj + np.diag(np.ones(adj.shape[0], dtype=np.float32))
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    degree_inv_sqrt = np.power(row_sum, -0.5).flatten()
    degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0.0
    matrix_degree_inv_sqrt = sp.diags(degree_inv_sqrt)
    message_passing_adj = matrix_degree_inv_sqrt.dot(adj).transpose().dot(matrix_degree_inv_sqrt).astype(np.float32)
    return message_passing_adj


def calculate_transition_matrix(adj: np.ndarray) -> np.matrix:
    """Calculate the random-walk transition matrix P = D^{-1} A."""
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1)).flatten()
    degree_inv = np.power(row_sum, -1).flatten()
    degree_inv[np.isinf(degree_inv)] = 0.0
    degree_matrix = sp.diags(degree_inv)
    transition_matrix = degree_matrix.dot(adj).astype(np.float32).todense()
    return transition_matrix
