"""Unified adapters for supported additive boosted-tree estimators.

The optional backends are identified from estimator type metadata and are not
imported here.  This keeps importing :mod:`forestgap` independent of XGBoost,
LightGBM, and CatBoost while still giving the proximity implementation one
stable leaf/path/contribution contract.
"""

from __future__ import annotations

import json
from typing import Any, Protocol, runtime_checkable

import numpy as np
from scipy import sparse


BOOSTED_FAMILIES = {
    "gradient_boosting",
    "xgboost",
    "lightgbm",
    "catboost",
}


@runtime_checkable
class BoostedTreeAdapter(Protocol):
    """Runtime contract consumed by :class:`forestgap.Proximity`."""

    estimator: Any
    family: str

    @property
    def tree_count(self) -> int:
        """Return the number of scalar tree partitions exposed by the model."""

    @property
    def path_capable(self) -> bool:
        """Whether stable root-to-leaf paths and node depths are available."""

    def leaf_ids(self, X: Any) -> np.ndarray:
        """Return native leaf IDs with shape ``(n_samples, n_trees)``."""

    def decision_path(self, X: Any) -> tuple[sparse.csr_matrix, np.ndarray]:
        """Return a stacked path indicator and stable per-tree offsets."""

    def node_depths(self) -> list[np.ndarray]:
        """Return one node-depth array per tree in path-indicator order."""

    def contribution_weights(self, X: Any) -> np.ndarray:
        """Return normalized empirical fitted-contribution variances."""


def boosted_family(estimator: Any) -> str | None:
    """Identify a supported boosted estimator without importing its package."""

    name = type(estimator).__name__
    module = type(estimator).__module__.split(".", 1)[0]
    if module == "sklearn" and name in {
        "GradientBoostingClassifier",
        "GradientBoostingRegressor",
    }:
        return "gradient_boosting"
    if module == "xgboost" and name in {"XGBClassifier", "XGBRegressor"}:
        return "xgboost"
    if module == "lightgbm" and name in {"LGBMClassifier", "LGBMRegressor"}:
        return "lightgbm"
    if module == "catboost" and name in {"CatBoostClassifier", "CatBoostRegressor"}:
        return "catboost"
    return None


def make_boosted_adapter(estimator: Any) -> BoostedTreeAdapter:
    """Construct the adapter for a supported boosted estimator."""

    family = boosted_family(estimator)
    adapters = {
        "gradient_boosting": SklearnGradientBoostingAdapter,
        "xgboost": XGBoostAdapter,
        "lightgbm": LightGBMAdapter,
        "catboost": CatBoostAdapter,
    }
    try:
        return adapters[family](estimator)  # type: ignore[index, return-value]
    except KeyError as exc:
        raise TypeError(
            f"No boosted-tree adapter matches {type(estimator).__module__}."
            f"{type(estimator).__name__}."
        ) from exc


def _normalized_variance_weights(contributions: np.ndarray) -> np.ndarray:
    """Normalize per-tree contribution variance with a uniform zero fallback."""

    values = np.asarray(contributions, dtype=np.float64)
    if values.ndim not in (2, 3) or values.shape[1] == 0:
        raise RuntimeError("A fitted boosted model must expose at least one tree.")
    variances = np.var(values, axis=0)
    if variances.ndim == 2:
        # Multidimensional outputs use the covariance trace for one scalar
        # partition weight per CatBoost tree.
        variances = variances.sum(axis=1)
    variances = np.asarray(variances, dtype=np.float64)
    variances[~np.isfinite(variances)] = 0.0
    variances = np.maximum(variances, 0.0)
    total = float(variances.sum())
    if total <= np.finfo(np.float64).eps:
        variances.fill(1.0 / variances.size)
    else:
        variances /= total
    return variances


def _node_depths_from_sklearn_tree(tree: Any) -> np.ndarray:
    children_left = np.asarray(tree.children_left)
    children_right = np.asarray(tree.children_right)
    depths = np.zeros(int(tree.node_count), dtype=np.int32)
    stack = [0]
    while stack:
        node = stack.pop()
        for child in (children_left[node], children_right[node]):
            if child >= 0:
                depths[child] = depths[node] + 1
                stack.append(int(child))
    return depths


def _paths_to_indicator(
    leaves: np.ndarray,
    paths_by_tree: list[dict[int, np.ndarray]],
    node_counts: list[int],
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Build a stacked CSR path indicator from native terminal-leaf IDs."""

    leaves = np.asarray(leaves)
    if leaves.ndim != 2 or leaves.shape[1] != len(paths_by_tree):
        raise RuntimeError("Leaf and path inventories have different tree counts.")
    node_ptr = np.concatenate(([0], np.cumsum(node_counts))).astype(np.int64)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    for tree_idx, path_map in enumerate(paths_by_tree):
        offset = int(node_ptr[tree_idx])
        for row_idx, leaf in enumerate(leaves[:, tree_idx]):
            try:
                path = path_map[int(leaf)]
            except KeyError as exc:
                raise RuntimeError(
                    f"Tree {tree_idx} returned unknown native leaf ID {int(leaf)}."
                ) from exc
            rows.append(np.full(path.size, row_idx, dtype=np.int64))
            cols.append(offset + path)
    if rows:
        row_ids = np.concatenate(rows)
        col_ids = np.concatenate(cols)
    else:
        row_ids = np.zeros(0, dtype=np.int64)
        col_ids = np.zeros(0, dtype=np.int64)
    values = np.ones(row_ids.size, dtype=np.float32)
    indicator = sparse.csr_matrix(
        (values, (row_ids, col_ids)),
        shape=(leaves.shape[0], int(node_ptr[-1])),
    )
    return indicator, node_ptr


class SklearnGradientBoostingAdapter:
    """Adapter for sklearn GradientBoosting classifier/regressor models."""

    family = "gradient_boosting"

    def __init__(self, estimator: Any) -> None:
        self.estimator = estimator

    def _trees(self) -> list[Any]:
        inventory = np.asarray(self.estimator.estimators_, dtype=object)
        return list(inventory.ravel())

    @property
    def tree_count(self) -> int:
        return len(self._trees())

    @property
    def path_capable(self) -> bool:
        return self.tree_count > 0

    def leaf_ids(self, X: Any) -> np.ndarray:
        return np.column_stack([tree.apply(X) for tree in self._trees()]).astype(
            np.int64, copy=False,
        )

    def decision_path(self, X: Any) -> tuple[sparse.csr_matrix, np.ndarray]:
        trees = self._trees()
        blocks = [sparse.csr_matrix(tree.decision_path(X)) for tree in trees]
        counts = [int(tree.tree_.node_count) for tree in trees]
        node_ptr = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
        indicator = sparse.hstack(blocks, format="csr")
        return indicator, node_ptr

    def node_depths(self) -> list[np.ndarray]:
        return [_node_depths_from_sklearn_tree(tree.tree_) for tree in self._trees()]

    def contribution_weights(self, X: Any) -> np.ndarray:
        leaves = self.leaf_ids(X)
        learning_rate = float(self.estimator.learning_rate)
        contributions = np.empty(leaves.shape, dtype=np.float64)
        for tree_idx, tree in enumerate(self._trees()):
            leaf_values = np.asarray(tree.tree_.value).reshape(-1)
            contributions[:, tree_idx] = learning_rate * leaf_values[leaves[:, tree_idx]]
        return _normalized_variance_weights(contributions)


class XGBoostAdapter:
    """Adapter for XGBoost's sklearn classifier/regressor interface."""

    family = "xgboost"

    def __init__(self, estimator: Any) -> None:
        self.estimator = estimator

    def _booster(self) -> Any:
        return self.estimator.get_booster()

    def _tree_dumps(self) -> list[dict[str, Any]]:
        if not hasattr(self, "_tree_dumps_cache"):
            self._tree_dumps_cache = [
                json.loads(tree_json)
                for tree_json in self._booster().get_dump(dump_format="json")
            ]
        return self._tree_dumps_cache

    def _tree_geometry(
        self,
    ) -> tuple[list[dict[int, np.ndarray]], list[np.ndarray], list[dict[int, float]]]:
        if hasattr(self, "_tree_geometry_cache"):
            return self._tree_geometry_cache
        paths_by_tree: list[dict[int, np.ndarray]] = []
        depths_by_tree: list[np.ndarray] = []
        values_by_tree: list[dict[int, float]] = []
        for tree in self._tree_dumps():
            paths: dict[int, np.ndarray] = {}
            depths: list[int] = []
            values: dict[int, float] = {}

            def visit(node: dict[str, Any], parent_path: list[int], depth: int) -> None:
                position = len(depths)
                depths.append(depth)
                path = [*parent_path, position]
                if "leaf" in node:
                    leaf_id = int(node["nodeid"])
                    paths[leaf_id] = np.asarray(path, dtype=np.int64)
                    values[leaf_id] = float(node["leaf"])
                    return
                for child in node.get("children", ()):
                    visit(child, path, depth + 1)

            visit(tree, [], 0)
            paths_by_tree.append(paths)
            depths_by_tree.append(np.asarray(depths, dtype=np.int32))
            values_by_tree.append(values)
        self._tree_geometry_cache = (paths_by_tree, depths_by_tree, values_by_tree)
        return self._tree_geometry_cache

    @property
    def tree_count(self) -> int:
        return len(self._tree_dumps())

    @property
    def path_capable(self) -> bool:
        return self.tree_count > 0

    def leaf_ids(self, X: Any) -> np.ndarray:
        leaves = np.asarray(self.estimator.apply(X), dtype=np.int64)
        if leaves.ndim == 1:
            leaves = leaves.reshape(-1, 1)
        elif leaves.ndim > 2:
            leaves = leaves.reshape(leaves.shape[0], -1)
        return leaves

    def decision_path(self, X: Any) -> tuple[sparse.csr_matrix, np.ndarray]:
        paths, depths, _ = self._tree_geometry()
        return _paths_to_indicator(
            self.leaf_ids(X), paths, [int(depth.size) for depth in depths],
        )

    def node_depths(self) -> list[np.ndarray]:
        return self._tree_geometry()[1]

    def contribution_weights(self, X: Any) -> np.ndarray:
        leaves = self.leaf_ids(X)
        value_maps = self._tree_geometry()[2]
        contributions = np.empty(leaves.shape, dtype=np.float64)
        for tree_idx, values in enumerate(value_maps):
            contributions[:, tree_idx] = [
                values[int(leaf)] for leaf in leaves[:, tree_idx]
            ]
        return _normalized_variance_weights(contributions)


class LightGBMAdapter:
    """Adapter for LightGBM's sklearn classifier/regressor interface."""

    family = "lightgbm"

    def __init__(self, estimator: Any) -> None:
        self.estimator = estimator

    def _booster(self) -> Any:
        return self.estimator.booster_

    def _tree_info(self) -> list[dict[str, Any]]:
        if not hasattr(self, "_tree_info_cache"):
            self._tree_info_cache = self._booster().dump_model()["tree_info"]
        return self._tree_info_cache

    def _tree_geometry(
        self,
    ) -> tuple[list[dict[int, np.ndarray]], list[np.ndarray], list[dict[int, float]]]:
        if hasattr(self, "_tree_geometry_cache"):
            return self._tree_geometry_cache
        paths_by_tree: list[dict[int, np.ndarray]] = []
        depths_by_tree: list[np.ndarray] = []
        values_by_tree: list[dict[int, float]] = []
        for info in self._tree_info():
            paths: dict[int, np.ndarray] = {}
            depths: list[int] = []
            values: dict[int, float] = {}

            def visit(node: dict[str, Any], parent_path: list[int], depth: int) -> None:
                position = len(depths)
                depths.append(depth)
                path = [*parent_path, position]
                if "leaf_index" in node:
                    leaf_id = int(node["leaf_index"])
                    paths[leaf_id] = np.asarray(path, dtype=np.int64)
                    values[leaf_id] = float(node["leaf_value"])
                    return
                visit(node["left_child"], path, depth + 1)
                visit(node["right_child"], path, depth + 1)

            visit(info["tree_structure"], [], 0)
            paths_by_tree.append(paths)
            depths_by_tree.append(np.asarray(depths, dtype=np.int32))
            values_by_tree.append(values)
        self._tree_geometry_cache = (paths_by_tree, depths_by_tree, values_by_tree)
        return self._tree_geometry_cache

    @property
    def tree_count(self) -> int:
        return len(self._tree_info())

    @property
    def path_capable(self) -> bool:
        return self.tree_count > 0

    def leaf_ids(self, X: Any) -> np.ndarray:
        leaves = np.asarray(self._booster().predict(X, pred_leaf=True), dtype=np.int64)
        if leaves.ndim == 1:
            leaves = leaves.reshape(-1, 1)
        elif leaves.ndim > 2:
            leaves = leaves.reshape(leaves.shape[0], -1)
        return leaves

    def decision_path(self, X: Any) -> tuple[sparse.csr_matrix, np.ndarray]:
        paths, depths, _ = self._tree_geometry()
        return _paths_to_indicator(
            self.leaf_ids(X), paths, [int(depth.size) for depth in depths],
        )

    def node_depths(self) -> list[np.ndarray]:
        return self._tree_geometry()[1]

    def contribution_weights(self, X: Any) -> np.ndarray:
        leaves = self.leaf_ids(X)
        value_maps = self._tree_geometry()[2]
        contributions = np.empty(leaves.shape, dtype=np.float64)
        for tree_idx, values in enumerate(value_maps):
            contributions[:, tree_idx] = [
                values[int(leaf)] for leaf in leaves[:, tree_idx]
            ]
        return _normalized_variance_weights(contributions)


class CatBoostAdapter:
    """Adapter for CatBoost classifier/regressor models."""

    family = "catboost"

    def __init__(self, estimator: Any) -> None:
        self.estimator = estimator

    @property
    def grow_policy(self) -> str:
        params = self.estimator.get_all_params()
        return str(params.get("grow_policy", "SymmetricTree"))

    def _leaf_counts(self) -> np.ndarray:
        return np.asarray(self.estimator.get_tree_leaf_counts(), dtype=np.int64)

    @property
    def tree_count(self) -> int:
        return int(self._leaf_counts().size)

    @property
    def path_capable(self) -> bool:
        return self.tree_count > 0 and self.grow_policy == "SymmetricTree"

    def leaf_ids(self, X: Any) -> np.ndarray:
        # Do not coerce X: CatBoost needs DataFrame/Pool categorical metadata.
        leaves = np.asarray(self.estimator.calc_leaf_indexes(X), dtype=np.int64)
        if leaves.ndim == 1:
            leaves = leaves.reshape(-1, 1)
        return leaves

    def _symmetric_geometry(self) -> tuple[list[dict[int, np.ndarray]], list[np.ndarray]]:
        if not self.path_capable:
            raise ValueError(
                "CatBoost path-derived relations require grow_policy='SymmetricTree'; "
                f"the fitted model uses grow_policy={self.grow_policy!r}."
            )
        if hasattr(self, "_tree_geometry_cache"):
            return self._tree_geometry_cache
        paths_by_tree: list[dict[int, np.ndarray]] = []
        depths_by_tree: list[np.ndarray] = []
        for leaf_count in self._leaf_counts():
            depth = int(round(np.log2(int(leaf_count))))
            if 2**depth != int(leaf_count):
                raise RuntimeError(
                    "A SymmetricTree CatBoost model returned a non-power-of-two leaf count."
                )
            node_count = 2 ** (depth + 1) - 1
            node_depth = np.empty(node_count, dtype=np.int32)
            for level in range(depth + 1):
                start = 2**level - 1
                node_depth[start:start + 2**level] = level
            paths: dict[int, np.ndarray] = {}
            for leaf_id in range(int(leaf_count)):
                path = [0]
                prefix = 0
                for level in range(depth):
                    # CatBoost's native leaf code stores the root split in bit 0.
                    prefix = (prefix << 1) | ((leaf_id >> level) & 1)
                    path.append(2 ** (level + 1) - 1 + prefix)
                paths[leaf_id] = np.asarray(path, dtype=np.int64)
            paths_by_tree.append(paths)
            depths_by_tree.append(node_depth)
        self._tree_geometry_cache = (paths_by_tree, depths_by_tree)
        return self._tree_geometry_cache

    def decision_path(self, X: Any) -> tuple[sparse.csr_matrix, np.ndarray]:
        paths, depths = self._symmetric_geometry()
        return _paths_to_indicator(
            self.leaf_ids(X), paths, [int(depth.size) for depth in depths],
        )

    def node_depths(self) -> list[np.ndarray]:
        return self._symmetric_geometry()[1]

    def contribution_weights(self, X: Any) -> np.ndarray:
        leaves = self.leaf_ids(X)
        leaf_counts = self._leaf_counts()
        flat_values = np.asarray(self.estimator.get_leaf_values(), dtype=np.float64)
        total_leaves = int(leaf_counts.sum())
        if total_leaves == 0 or flat_values.size % total_leaves:
            raise RuntimeError("CatBoost leaf-value inventory is inconsistent with its trees.")
        output_dimension = flat_values.size // total_leaves
        leaf_values = flat_values.reshape(total_leaves, output_dimension)
        contributions = np.empty(
            (leaves.shape[0], leaves.shape[1], output_dimension), dtype=np.float64,
        )
        offset = 0
        for tree_idx, leaf_count in enumerate(leaf_counts):
            tree_values = leaf_values[offset:offset + int(leaf_count)]
            contributions[:, tree_idx, :] = tree_values[leaves[:, tree_idx]]
            offset += int(leaf_count)
        if output_dimension == 1:
            return _normalized_variance_weights(contributions[:, :, 0])
        return _normalized_variance_weights(contributions)


__all__ = [
    "BOOSTED_FAMILIES",
    "BoostedTreeAdapter",
    "CatBoostAdapter",
    "LightGBMAdapter",
    "SklearnGradientBoostingAdapter",
    "XGBoostAdapter",
    "boosted_family",
    "make_boosted_adapter",
]
