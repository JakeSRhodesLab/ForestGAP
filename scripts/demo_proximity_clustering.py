"""Visualize forest-proximity clusters on a shared 2D RF-PHATE embedding.

This demo requires the optional PHATE dependencies: ``pip install '.[phate]'``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier

from forestgap import Proximity, embed, proximity_clustering


def _cluster_counts(labels: np.ndarray) -> dict[int, int]:
    values, counts = np.unique(labels, return_counts=True)
    return {int(value): int(count) for value, count in zip(values, counts)}


def _plot_clusters(figure, axis, coordinates, labels, method: str) -> None:
    unique_labels = np.unique(labels)
    color_indices = np.searchsorted(unique_labels, labels)
    colormap = plt.get_cmap("tab10", len(unique_labels))
    normalization = BoundaryNorm(
        np.arange(len(unique_labels) + 1) - 0.5,
        colormap.N,
    )
    points = axis.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        c=color_indices,
        cmap=colormap,
        norm=normalization,
        s=28,
    )
    colorbar = figure.colorbar(points, ax=axis, ticks=np.arange(len(unique_labels)))
    colorbar.set_ticklabels([str(label) for label in unique_labels])
    colorbar.set_label("Cluster label")
    axis.set_title(method.upper())
    axis.set_xlabel("RF-PHATE 1")
    axis.set_ylabel("RF-PHATE 2")

def main() -> None:

    X, y = make_blobs(n_samples=150, centers=3, cluster_std=0.65, random_state=0)
    proximity = Proximity(
        RandomForestClassifier(n_estimators=200, random_state=0), relation="original",
    ).fit(X, y)
    coordinates = embed(proximity, method="phate", random_state=0)

    configurations = {
        "spectral": {"n_clusters": 3, "random_state": 0},
        "hdbscan": {"min_cluster_size": 10},
        "kmeans": {"n_clusters": 3, "random_state": 0},
    }
    labels_by_method = {}
    for method, parameters in configurations.items():
        labels = proximity_clustering(proximity, method=method, **parameters)
        labels_by_method[method] = labels
        print(f"{method}: {_cluster_counts(labels)}")

    figure, axes = plt.subplots(
        1,
        len(labels_by_method),
        figsize=(14, 4),
        sharex=True,
        sharey=True,
    )
    for axis, (method, labels) in zip(axes, labels_by_method.items()):
        _plot_clusters(figure, axis, coordinates, labels, method)
    axes[0].set_ylabel("RF-PHATE 2")
    figure.suptitle("Random-forest proximity clustering")
    figure.tight_layout()
    output_dir = Path("results/figures/proximity_clustering_demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = output_dir / "rf_phate_clustering_comparison.png"
    figure.savefig(comparison_path, dpi=180, bbox_inches="tight")
    print(f"saved: {comparison_path}")

    for method, labels in labels_by_method.items():
        method_figure, method_axis = plt.subplots(figsize=(5, 4))
        _plot_clusters(method_figure, method_axis, coordinates, labels, method)
        method_figure.tight_layout()
        method_path = output_dir / f"rf_phate_{method}_clustering.png"
        method_figure.savefig(method_path, dpi=180, bbox_inches="tight")
        plt.close(method_figure)
        print(f"saved: {method_path}")
    plt.show()


if __name__ == "__main__":
    main()
