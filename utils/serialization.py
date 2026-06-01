import pickle

import numpy as np

from utils.graph_topology_normalization import (
    calculate_scaled_laplacian,
    calculate_symmetric_message_passing_adj,
    calculate_symmetric_normalized_laplacian,
    calculate_transition_matrix,
)


def load_pkl(pickle_file: str) -> object:
    """Load pickle data."""
    try:
        with open(pickle_file, "rb") as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError:
        with open(pickle_file, "rb") as f:
            pickle_data = pickle.load(f, encoding="latin1")
    except Exception as e:
        print("Unable to load data ", pickle_file, ":", e)
        raise
    return pickle_data


def load_npz(file_path: str) -> dict:
    """Load an npz file and return a dictionary of numpy arrays."""
    try:
        data = np.load(file_path, allow_pickle=True)
        return {key: data[key] for key in data.files}
    except Exception as e:
        print(f"Unable to load .npz file {file_path}: {e}")
        raise


def load_npy(file_path: str) -> np.ndarray:
    """Load an npy file."""
    try:
        return np.load(file_path, allow_pickle=True)
    except Exception as e:
        print(f"Unable to load .npy file {file_path}: {e}")
        raise


def dump_pkl(obj: object, file_path: str):
    """Dump an object to a pickle file."""
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)


def load_matrix(file_path: str):
    return load_pkl(file_path)


def load_adj(file_path: str, adj_type: str):
    """Load and preprocess the adjacency matrix for DHTS-Net."""
    try:
        _, _, adj_mx = load_pkl(file_path)
    except ValueError:
        adj_mx = load_pkl(file_path)

    if adj_type == "scalap":
        adj = [calculate_scaled_laplacian(adj_mx).astype(np.float32).todense()]
    elif adj_type == "normlap":
        adj = [calculate_symmetric_normalized_laplacian(adj_mx).astype(np.float32).todense()]
    elif adj_type == "symnadj":
        adj = [calculate_symmetric_message_passing_adj(adj_mx).astype(np.float32).todense()]
    elif adj_type == "transition":
        adj = [calculate_transition_matrix(adj_mx).T]
    elif adj_type == "doubletransition":
        adj = [calculate_transition_matrix(adj_mx).T, calculate_transition_matrix(adj_mx.T).T]
    elif adj_type == "identity":
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    elif adj_type == "original":
        adj = [adj_mx]
    else:
        raise ValueError(f"adj_type is not defined: {adj_type}")
    return adj, adj_mx
