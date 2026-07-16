from pathlib import Path

from scripts.validate_notebooks import EXPECTED, validate


ROOT = Path(__file__).resolve().parents[2]


def test_all_notebooks_have_fresh_composition_smoke_outputs():
    paths = sorted((ROOT / "notebooks").glob("*.ipynb"))
    assert {path.name for path in paths} == EXPECTED
    failures = {
        path.name: errors
        for path in paths
        if (errors := validate(path, execute=False))
    }
    assert failures == {}

def test_rotation_forest_tutorial_has_a_fresh_targeted_smoke_output():
    path = ROOT / "notebooks" / "rotation_forest_proximities.ipynb"
    assert validate(path, execute=False) == []

