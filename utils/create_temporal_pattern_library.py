import os
import pickle

import numpy as np
from tqdm import tqdm
from tslearn.clustering import KShape, TimeSeriesKMeans


class TemporalPatternLibraryBuilder:
    """Mine the temporal pattern library used by the Heterogeneous Temporal Path."""

    def __init__(self, config):
        self.config = config
        self.data_dir = config.get("data_dir", "./data")
        self.time_intervals = config.get("time_intervals", 300)
        self.candidate_key_days = config.get("candidate_key_days", config.get("cand_key_days", 14))
        self.pattern_length = config.get("pattern_length", config.get("s_attn_size", 5))
        self.num_patterns = config.get("num_patterns", config.get("n_cluster", 16))
        self.cluster_max_iter = config.get("cluster_max_iter", 5)
        self.cluster_method = config.get("cluster_method", "kshape")
        self.target_dim = config.get("target_dim", 0)

        self.points_per_hour = 3600 // self.time_intervals
        self.points_per_day = 24 * self.points_per_hour

        self.generate_and_save_temporal_pattern_library()

    def load_training_traffic_series(self):
        data = np.load(os.path.join(self.data_dir, "data.npz"))["data"]
        index = np.load(os.path.join(self.data_dir, "index.npz"))
        return data[index["train"], :, self.target_dim]

    def mine_temporal_pattern_library(self, traffic_series):
        candidate_key_steps = self.candidate_key_days * self.points_per_day
        traffic_series = traffic_series[: min(len(traffic_series), candidate_key_steps)]

        if self.cluster_method == "kshape":
            cluster_model = KShape(n_clusters=self.num_patterns, max_iter=self.cluster_max_iter, random_state=42)
        else:
            cluster_model = TimeSeriesKMeans(
                n_clusters=self.num_patterns,
                metric="softdtw",
                max_iter=self.cluster_max_iter,
                random_state=42,
            )

        cluster_model.fit(pattern_array)
        return cluster_model.cluster_centers_

    def generate_and_save_temporal_pattern_library(self):
        pkl_path = os.path.join(self.data_dir, "temporal_pattern_library.pkl")
        if os.path.exists(pkl_path):
            print(f"Temporal pattern library already exists at {pkl_path}")
            return

        print("Generating temporal pattern library...")
        traffic_series = self.load_training_traffic_series()
        temporal_pattern_library = self.mine_temporal_pattern_library(traffic_series)

        with open(pkl_path, "wb") as f:
            pickle.dump(temporal_pattern_library, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Temporal pattern library saved to {pkl_path}")


if __name__ == "__main__":
    config = {
        "data_dir": "./data/PEMS08",
        "time_intervals": 300,
        "candidate_key_days": 14,
        "pattern_length": 3,
        "num_patterns": 16,
        "cluster_max_iter": 5,
        "cluster_method": "kshape",
    }
    TemporalPatternLibraryBuilder(config)
