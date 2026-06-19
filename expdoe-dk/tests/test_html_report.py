"""
v0.4 — HTML report tests.

Cover the rendered output against expected sections without depending on
JavaScript execution (the page must degrade gracefully when JS is off).
"""
from __future__ import annotations

import base64
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from expdoe_dk import (
    Campaign,
    Knowledge,
    Parameter,
    Space,
)
from expdoe_dk.bo.loop import Result


def _make_dummy_result() -> Result:
    """Build a Result without running BO so tests are fast and deterministic."""
    rng = np.random.default_rng(0)
    n = 8
    df = pd.DataFrame({
        "T": np.linspace(70, 110, n),
        "conc_A": np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float),
        "y": rng.uniform(50, 90, size=n),
        "kind": ["doe"] * 3 + ["bo"] * 5,
    })
    return Result(
        best_x_physical={"T": 95.0, "conc_A": 7.0},
        best_y_physical=92.5,
        history_df=df,
        n_doe=3,
        n_bo=5,
        objectives=["yield_pct"],
        maximize=[True],
        knowledge_summary=[
            {"kind": "arrhenius", "param": "T", "frozen": True,
             "activation_energy": 1.0, "amplitude_init": -1.0},
            {"kind": "random_augment", "n": 20},
        ],
        notes=["Example note rendered in the report footer."],
        param_units={"T": "°C", "conc_A": "mL"},
        param_kinds={"T": "continuous", "conc_A": "discrete"},
    )


# --------------------------------------------------------------------- #
# render_html — string-level checks
# --------------------------------------------------------------------- #
def test_report_contains_core_sections():
    result = _make_dummy_result()
    html = result.to_html_string(title="Reaction optimization")
    for needle in [
        "<!doctype html>",
        "<title>Reaction optimization</title>",
        '<canvas id="conv-chart"',
        "Best yield_pct",
        "Download CSV",
        "Knowledge applied",
        "Arrhenius prior on",  # quotes around T are HTML-escaped
        "Random augment (n=20)",
        "cdn.jsdelivr.net",
    ]:
        assert needle in html, f"missing piece: {needle!r}"


def test_report_renders_units_for_each_param():
    result = _make_dummy_result()
    html = result.to_html_string()
    # Best block shows both units.
    assert "°C" in html
    assert "mL" in html
    # Discrete param formatted without trailing decimal.
    assert ">7 mL<" in html


def test_result_to_dict_preserves_param_report_metadata():
    result = _make_dummy_result()

    payload = result.to_dict()

    assert payload["param_units"] == {"T": "°C", "conc_A": "mL"}
    assert payload["param_kinds"] == {"T": "continuous", "conc_A": "discrete"}


def test_report_includes_notes_when_present():
    result = _make_dummy_result()
    html = result.to_html_string()
    assert "Example note rendered in the report footer." in html


def test_report_csv_data_url_decodes_to_valid_history():
    result = _make_dummy_result()
    html = result.to_html_string()
    m = re.search(r'href="data:text/csv;base64,([A-Za-z0-9+/=]+)"', html)
    assert m is not None, "CSV download URL not found"
    csv = base64.b64decode(m.group(1)).decode("utf-8")
    lines = csv.strip().splitlines()
    # Header + 8 rows.
    assert lines[0] == "T,conc_A,y,kind"
    assert len(lines) == 1 + len(result.history_df)
    # First and last data rows match the DataFrame numerically.
    first = lines[1].split(",")
    assert float(first[0]) == pytest.approx(70.0)
    assert first[-1] == "doe"


def test_report_marks_doe_vs_bo_in_table():
    result = _make_dummy_result()
    html = result.to_html_string()
    assert 'class="kind-doe">doe<' in html
    assert 'class="kind-bo">bo<' in html


def test_report_handles_empty_knowledge_summary():
    result = _make_dummy_result()
    bare = Result(
        best_x_physical=result.best_x_physical,
        best_y_physical=result.best_y_physical,
        history_df=result.history_df,
        n_doe=result.n_doe,
        n_bo=result.n_bo,
        objectives=result.objectives,
        maximize=result.maximize,
        knowledge_summary=[],
        notes=[],
        param_units=result.param_units,
        param_kinds=result.param_kinds,
    )
    html = bare.to_html_string()
    assert "Knowledge applied" in html
    assert "Campaign ran with the baseline GP" in html


# --------------------------------------------------------------------- #
# to_html — file IO
# --------------------------------------------------------------------- #
def test_to_html_writes_file(tmp_path: Path):
    result = _make_dummy_result()
    out = result.to_html(tmp_path / "subdir" / "report.html")
    assert out.exists()
    assert out.stat().st_size > 0
    # Idempotent: another write succeeds.
    out2 = result.to_html(tmp_path / "subdir" / "report.html")
    assert out2 == out


def test_to_html_string_matches_file_contents(tmp_path: Path):
    result = _make_dummy_result()
    out = result.to_html(tmp_path / "r.html")
    assert out.read_text(encoding="utf-8") == result.to_html_string()


# --------------------------------------------------------------------- #
# Result populated by real Campaign.finalize() includes units/kinds
# --------------------------------------------------------------------- #
@pytest.mark.slow
def test_campaign_finalize_populates_units_for_report(tmp_path: Path):
    warnings.simplefilter("ignore")
    space = Space(
        params=[
            Parameter("T", bounds=(60.0, 120.0), unit="°C"),
            Parameter("conc", bounds=(1.0, 10.0), unit="mL",
                      kind="discrete", step=1.0),
        ],
        objectives="y",
        maximize=True,
    )

    def oracle(df):
        return -((df["T"] - 95) ** 2) * 0.02 - ((df["conc"] - 7) ** 2) * 1.5

    campaign = Campaign(space, Knowledge(), seed=0)
    result = campaign.run(oracle, n_doe=6, n_iter=5)
    assert result.param_units == {"T": "°C", "conc": "mL"}
    assert result.param_kinds == {"T": "continuous", "conc": "discrete"}

    out = result.to_html(tmp_path / "real.html",
                         title="real-campaign smoke")
    text = out.read_text(encoding="utf-8")
    assert "°C" in text
    assert "mL" in text
