"""treeple-specific honesty and oblique-forest contracts."""

from __future__ import annotations

from typing import Any


def patch_treeple_regressor_tags() -> bool:
    """Patch treeple 0.10.3 regressor tags for supported sklearn versions."""

    try:
        from treeple._lib.sklearn.ensemble._forest import ForestRegressor
    except Exception:
        return False
    if getattr(ForestRegressor, "_forestgap_tags_patch", False):
        return True

    def _patched_sklearn_tags(self):
        tags = super(ForestRegressor, self).__sklearn_tags__()
        regressor_tags = getattr(tags, "regressor_tags", None)
        if regressor_tags is not None:
            try:
                regressor_tags.multi_label = True
            except AttributeError:
                pass
        return tags

    ForestRegressor.__sklearn_tags__ = _patched_sklearn_tags
    ForestRegressor._forestgap_tags_patch = True
    return True


def patch_treeple_sample_weight_signature() -> bool:
    """Adapt treeple's vendored positional dtype call to sklearn>=1.7."""

    try:
        from sklearn.utils.validation import _check_sample_weight as sklearn_check_sample_weight
        import treeple._lib.sklearn.tree._classes as treeple_tree_classes
    except Exception:
        return False
    if getattr(treeple_tree_classes, "_forestgap_sample_weight_patch", False):
        return True

    def _patched_check_sample_weight(
        sample_weight: Any,
        X: Any,
        dtype: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if args:
            if dtype is not None:
                raise TypeError("dtype provided both positionally and by keyword")
            dtype = args[0]
            if len(args) > 1:
                raise TypeError("too many positional arguments after X")
        return sklearn_check_sample_weight(sample_weight, X, dtype=dtype, **kwargs)

    treeple_tree_classes._check_sample_weight = _patched_check_sample_weight
    treeple_tree_classes._forestgap_sample_weight_patch = True
    return True


def honest_roles(estimator: Any) -> dict[str, Any]:
    """Return fitted HonestForest structure/estimation row roles."""

    required = ("honest_indices_", "structure_indices_")
    missing = [name for name in required if not hasattr(estimator, name)]
    if missing:
        raise TypeError(f"estimator is missing honest role state: {', '.join(missing)}")
    return {
        "honest_indices": estimator.honest_indices_,
        "structure_indices": estimator.structure_indices_,
    }
