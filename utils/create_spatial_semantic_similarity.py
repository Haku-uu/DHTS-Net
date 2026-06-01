import os
import pickle
import random

import networkx as nx
import numpy as np
from fastdtw import fastdtw
from gensim.models import Word2Vec
from joblib import Parallel, delayed
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


def vectorized_range(starts, stops):
    stops = np.asarray(stops)
    lengths = stops - starts
    assert lengths.min() == lengths.max(), "Lengths of each range should be equal."
    indices = np.repeat(stops - lengths.cumsum(), lengths) + np.arange(lengths.sum())
    return indices.reshape(-1, lengths[0])


def calculate_temporal_dtw_matrix_for_spatial_semantics(data_dir="./data/PEMS08/", radius=6, n_jobs=-1):
    """Calculate a DTW matrix from training data for spatial semantic similarity mining."""
    try:
        data = np.load(os.path.join(data_dir, "data.npz"))["data"].astype(np.float32)
        index = np.load(os.path.join(data_dir, "index.npz"))
        train_index = index["train"]
    except Exception as e:
        raise ValueError(f"Failed to load the data: {str(e)}")

    x_train_index = vectorized_range(train_index[:, 0], train_index[:, 1])
    train_data = data[x_train_index]
    train_data = train_data.transpose(0, 2, 1, 3)

    points_per_day = 24 * 12
    total_time_points = train_data.shape[0]
    num_days = total_time_points // points_per_day
    complete_days_data = train_data[: num_days * points_per_day]

    daily_data = complete_days_data.reshape(
        num_days,
        points_per_day,
        train_data.shape[1],
        train_data.shape[2],
        train_data.shape[3],
    )
    daily_average = np.mean(daily_data, axis=0)
    cache_path = os.path.join(data_dir, "dtw_matrix.pkl")

    if not os.path.exists(cache_path):
        num_nodes = daily_average.shape[1]
        dtw_matrix = np.zeros((num_nodes, num_nodes))

        def compute_dtw(i, j):
            if i <= j:
                distance, _ = fastdtw(daily_average[:, i, 0], daily_average[:, j, 0], radius=radius)
                return i, j, distance
            return i, j, 0

        pairs = [(i, j) for i in range(num_nodes) for j in range(num_nodes)]
        results = Parallel(n_jobs=n_jobs)(
            delayed(compute_dtw)(i, j) for i, j in tqdm(pairs, desc="Calculating DTW matrix")
        )

        for i, j, distance in results:
            dtw_matrix[i, j] = distance
        dtw_matrix = np.maximum(dtw_matrix, dtw_matrix.T)

        try:
            with open(cache_path, "wb") as f:
                pickle.dump(dtw_matrix, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            if os.path.exists(cache_path):
                os.remove(cache_path)
            raise IOError(f"Failed to save the DTW matrix: {str(e)}")

    try:
        with open(cache_path, "rb") as f:
            dtw_matrix = pickle.load(f)
    except Exception as e:
        raise ValueError(f"Failed to load cached DTW matrix: {str(e)}")

    return dtw_matrix


def build_spatial_semantic_graph_from_dtw(
    dtw_matrix,
    threshold_percentile=90,
    max_edges_per_node=None,
    use_inverse_dtw_as_weight=True,
):
    num_nodes = dtw_matrix.shape[0]

    if use_inverse_dtw_as_weight:
        semantic_similarity = 1 / (dtw_matrix + 1e-8)
    else:
        dtw_min = np.min(dtw_matrix)
        dtw_max = np.max(dtw_matrix)
        semantic_similarity = 1 - (dtw_matrix - dtw_min) / (dtw_max - dtw_min + 1e-8)

    np.fill_diagonal(semantic_similarity, 0)
    threshold = np.percentile(semantic_similarity[semantic_similarity > 0], threshold_percentile)

    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))

    for i in range(num_nodes):
        neighbors = [(j, semantic_similarity[i, j]) for j in range(num_nodes) if j != i]
        neighbors.sort(key=lambda x: x[1], reverse=True)
        if max_edges_per_node is not None:
            neighbors = neighbors[:max_edges_per_node]

        for j, similarity in neighbors:
            if similarity >= threshold:
                graph.add_edge(i, j, weight=similarity)

    return graph


def alias_sample(accept, alias):
    length = len(accept)
    idx = int(np.random.random() * length)
    rand = np.random.random()
    if rand < accept[idx]:
        return idx
    return alias[idx]


class SpatialSemanticNode2Vec:
    """Node2Vec encoder for spatial semantic similarity mining."""

    def __init__(self, graph, walk_length=80, num_walks=10, workers=4):
        self.graph = graph
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.workers = workers
        self.alias_nodes = {}
        self.preprocess_transition_probabilities()

    def preprocess_transition_probabilities(self):
        for node in self.graph.nodes():
            neighbors = list(self.graph.neighbors(node))
            if not neighbors:
                continue
            weights = [self.graph[node][neighbor].get("weight", 1.0) for neighbor in neighbors]
            weight_sum = sum(weights)
            probabilities = [weight / weight_sum for weight in weights]
            self.alias_nodes[node] = create_alias_table(probabilities)

    def unbiased_walk(self, start_node):
        walk = [start_node]
        while len(walk) < self.walk_length:
            current_node = walk[-1]
            neighbors = list(self.graph.neighbors(current_node))
            if len(neighbors) == 0:
                break
            next_node = neighbors[alias_sample(*self.alias_nodes[current_node])]
            walk.append(next_node)
        return walk

    def simulate_walks(self):
        walks = []
        nodes = list(self.graph.nodes())
        for _ in range(self.num_walks):
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.unbiased_walk(node))
        return walks

    def learn_embeddings(self, dimensions=128, window_size=10, min_count=1, sg=1):
        walks = self.simulate_walks()
        walks = [list(map(str, walk)) for walk in walks]
        model = Word2Vec(
            walks,
            vector_size=dimensions,
            window=window_size,
            min_count=min_count,
            sg=sg,
            workers=self.workers,
            epochs=10,
        )
        return model


def generate_spatial_semantic_similarity_matrix(
    dtw_matrix,
    threshold_percentile=85,
    max_edges_per_node=20,
    walk_length=80,
    num_walks=10,
    dimensions=64,
):
    graph = build_spatial_semantic_graph_from_dtw(
        dtw_matrix,
        threshold_percentile,
        max_edges_per_node,
        use_inverse_dtw_as_weight=True,
    )

    return graph, embedding_matrix, node2vec_model, spatial_semantic_similarity


if __name__ == "__main__":
    data_dir = "./data/PEMS07"
    dtw_matrix = calculate_temporal_dtw_matrix_for_spatial_semantics(data_dir=data_dir)
    graph, node_embeddings, node2vec_model, spatial_semantic_similarity = generate_spatial_semantic_similarity_matrix(
        dtw_matrix,
        threshold_percentile=85,
        max_edges_per_node=20,
        walk_length=80,
        num_walks=10,
        dimensions=64,
    )

    os.makedirs(data_dir, exist_ok=True)
    output_path = os.path.join(data_dir, "spatial_semantic_similarity_matrix.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(spatial_semantic_similarity, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Spatial semantic similarity matrix saved to {output_path}")
