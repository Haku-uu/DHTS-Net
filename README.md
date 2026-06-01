# DHTS-Net: Dual-Heterogeneity Temporal-Spatial Network for Traffic Flow Prediction

This repository provides the PyTorch implementation of **DHTS-Net**, a Dual-Heterogeneity Temporal-Spatial Network for traffic flow prediction.

## Project Structure

```text
DHTS-Net/
├── DHTSNet.yaml
├── train.py
├── model/
│   ├── __init__.py
│   └── DHTSNet.py
├── lib/
│   ├── data_prepare.py
│   ├── dhtsnet_losses.py
│   ├── metrics.py
│   └── utils.py
├── utils/
│   ├── create_spatial_semantic_similarity.py
│   ├── create_temporal_pattern_library.py
│   ├── graph_topology_normalization.py
│   └── serialization.py
└── README.md
````

## Requirements

The implementation is based on Python and PyTorch. The main dependencies include:

```text
python >= 3.8
torch
numpy
pandas
scipy
scikit-learn
matplotlib
pyyaml
tqdm
einops
timm
torchinfo
fastdtw
joblib
networkx
gensim
tslearn
```

You can install the required packages manually or create a virtual environment before running the code.

Example:

```bash
pip install torch numpy pandas scipy scikit-learn matplotlib pyyaml tqdm einops timm torchinfo fastdtw joblib networkx gensim tslearn
```

## Data Preparation

The dataset directory is expected to follow the commonly used traffic forecasting format:

```text
data/
└── DATASET_NAME/
    ├── data.npz
    ├── index.npz
    └── adj_mx.pkl
```

where:

* `data.npz` stores traffic observations.
* `index.npz` stores train/validation/test split indices.
* `adj_mx.pkl` stores the road network adjacency matrix.

The input tensor follows the format:

```text
[B, T, N, C]
```

where:

* `B` is the batch size.
* `T` is the historical input length.
* `N` is the number of traffic sensors or road nodes.
* `C` is the number of input features.

## Preprocessing

DHTS-Net uses temporal pattern libraries and spatial semantic similarity information. Before training, you can generate the required auxiliary files using:

```bash
python utils/create_temporal_pattern_library.py
```

and

```bash
python utils/create_spatial_semantic_similarity.py
```

Please check and modify the dataset path in the corresponding scripts according to your local data directory.

## Training

The training configuration is provided in:

```text
DHTSNet.yaml
```

To train DHTS-Net, run:

```bash
python train.py
```

## Evaluation

During training, the model is evaluated using common traffic forecasting metrics:

* MAE
* RMSE
* MAPE

The testing results will be printed after training is completed.

## Citation

If you find this repository useful for your research, please consider citing our work:

```bibtex
@article{
  title={DHTS-Net: Dual-Heterogeneity Temporal-Spatial Network for Traffic Flow Prediction},
  author={ },
  journal={Under Review},
  year={2026}
}
```

## License

This project is released for research purposes only.
