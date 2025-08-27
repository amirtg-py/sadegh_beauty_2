import os
import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

DEFAULT_GEN_PATH = r"D:\\work_station\\radar_co\\radar\\final\\sadegh_beauty-codex-improve-data-generation-script-accuracy\\generated_data"
DEFAULT_REAL_PATH = r"D:\\work_station\\data"

@dataclass
class GraphWindow:
    features: torch.Tensor  # [N, 2] time and azimuth
    adjacency: torch.Tensor  # [N, N] binary adjacency matrix


def load_csv_files(directory: str) -> List[pd.DataFrame]:
    """Load all CSV files inside a directory."""
    csv_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.csv')]
    datasets = []
    for path in csv_files:
        try:
            df = pd.read_csv(path)
            if {'Time', 'Azimuth'}.issubset(df.columns):
                datasets.append(df[['Time', 'Azimuth']])
        except Exception as exc:
            print(f"Failed loading {path}: {exc}")
    return datasets


def create_windows(df: pd.DataFrame, window: float, step: float) -> List[pd.DataFrame]:
    """Split dataframe into sliding windows."""
    df = df.sort_values('Time').reset_index(drop=True)
    start, end = df['Time'].iloc[0], df['Time'].iloc[-1]
    windows = []
    t = start
    while t < end:
        mask = (df['Time'] >= t) & (df['Time'] < t + window)
        win = df[mask]
        if not win.empty:
            windows.append(win.copy())
        t += step
    return windows


def window_to_graph(win: pd.DataFrame, k: int = 5) -> GraphWindow:
    """Convert windowed dataframe to graph with k-NN adjacency."""
    coords = torch.tensor(win[['Time', 'Azimuth']].values, dtype=torch.float32)
    feats = (coords - coords.mean(dim=0)) / (coords.std(dim=0) + 1e-6)
    N = feats.size(0)
    # pairwise distances
    dist = torch.cdist(feats, feats, p=2)
    adj = torch.zeros(N, N)
    for i in range(N):
        # select k nearest neighbours excluding self
        nn_idx = torch.topk(dist[i], k + 1, largest=False).indices[1:]
        adj[i, nn_idx] = 1
    # make symmetric
    adj = torch.maximum(adj, adj.t())
    return GraphWindow(feats, adj)


class SimpleGAE(torch.nn.Module):
    """A very small graph autoencoder without external dependencies."""

    def __init__(self, in_dim: int, hidden: int = 32, latent: int = 16):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_dim, hidden)
        self.fc2 = torch.nn.Linear(hidden, latent)

    def encode(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.fc1(adj @ x))
        z = self.fc2(adj @ h)
        return z

    def recon_loss(self, z: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        logits = z @ z.t()
        return torch.nn.functional.binary_cross_entropy_with_logits(logits, adj)


def train_model(graphs: List[GraphWindow], epochs: int = 50, lr: float = 1e-2) -> SimpleGAE:
    model = SimpleGAE(in_dim=2)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        total = 0.0
        random.shuffle(graphs)
        for g in graphs:
            optimiser.zero_grad()
            z = model.encode(g.features, g.adjacency)
            loss = model.recon_loss(z, g.adjacency)
            loss.backward()
            optimiser.step()
            total += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: loss={total/len(graphs):.4f}")
    return model


def infer_clusters(model: SimpleGAE, g: GraphWindow) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        z = model.encode(g.features, g.adjacency).cpu().numpy()
    clustering = DBSCAN(eps=0.5, min_samples=3)
    labels = clustering.fit_predict(z)
    return labels


def plot_window(win: pd.DataFrame, labels: np.ndarray, path: str) -> None:
    plt.figure(figsize=(6, 4))
    plt.scatter(win['Time'], win['Azimuth'], c=labels, cmap='tab20', s=10)
    plt.xlabel('Time')
    plt.ylabel('Azimuth')
    plt.title('Tracklets by GNN/DBSCAN')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def process_dataset(data_dir: str, model: SimpleGAE | None, train: bool,
                    results_dir: str, window: float, step: float,
                    plots: int = 10) -> SimpleGAE:
    datasets = load_csv_files(data_dir)
    all_windows: List[pd.DataFrame] = []
    for df in datasets:
        all_windows.extend(create_windows(df, window, step))
    graph_windows = [window_to_graph(w) for w in all_windows if len(w) > 1]

    if train:
        model = train_model(graph_windows)

    os.makedirs(results_dir, exist_ok=True)
    summaries = []
    plot_indices = set(random.sample(range(len(graph_windows)), min(plots, len(graph_windows))))
    for idx, g in enumerate(graph_windows):
        labels = infer_clusters(model, g)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        summaries.append({'window_id': idx, 'num_points': g.features.size(0), 'clusters': n_clusters})
        if idx in plot_indices:
            win = all_windows[idx]
            plot_path = os.path.join(results_dir, f'window_{idx}.png')
            plot_window(win, labels, plot_path)
    pd.DataFrame(summaries).to_csv(os.path.join(results_dir, 'summary.csv'), index=False)
    return model


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Tracklet detection using simple GNN.')
    parser.add_argument('--generated_data', default=DEFAULT_GEN_PATH)
    parser.add_argument('--real_data', default=DEFAULT_REAL_PATH)
    parser.add_argument('--results', default='results')
    parser.add_argument('--window', type=float, default=1000.0)
    parser.add_argument('--step', type=float, default=500.0)
    parser.add_argument('--plots', type=int, default=10)
    args = parser.parse_args()

    gen_res = os.path.join(args.results, 'generated')
    real_res = os.path.join(args.results, 'real')

    model = process_dataset(args.generated_data, model=None, train=True,
                            results_dir=gen_res, window=args.window, step=args.step,
                            plots=args.plots)
    process_dataset(args.real_data, model=model, train=False,
                    results_dir=real_res, window=args.window, step=args.step,
                    plots=args.plots)


if __name__ == '__main__':
    main()
