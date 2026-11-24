"""The standalone example works with only the installed public package API."""
from pathlib import Path
import runpy


def test_multiobjective_catalog_example():
    namespace = runpy.run_path(str(Path(__file__).resolve().parents[3] / "examples" / "05_multiobjective_catalog.py"))
    result = namespace["run_example"]()
    assert len(result.history_df) == 10
    assert result.history_df.x.is_unique
    assert (result.history_df.x <= .8).all()
    assert result.best_y_physical is None
    assert result.best_x_physical is None
    assert result.pareto_rows
    assert len(result.hypervolume_history) == 3
    assert len(result.knowledge["specs"]) == 1
    spec, = result.knowledge["specs"]
    assert spec["pattern"] == "monotone" and spec["version"] == "2.0"
    assert spec["scope"]["objectives"] == ["gain"]
