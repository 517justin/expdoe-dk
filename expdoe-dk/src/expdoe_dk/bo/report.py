"""
HTML report generator for a Campaign Result.

Produces a single self-contained HTML file with the chemist's view of the
campaign:

  - Best conditions in physical units + natural-language summary
  - Convergence curve (best-so-far yield vs trial number)
  - Per-trial history table (DoE vs BO marked)
  - Knowledge spec used + auto-default notes
  - CSV download embedded as a data URL

Charts use Chart.js via CDN (cdn.jsdelivr.net is on the standard allowlist
for sandboxed clients). The page is functional without JavaScript — the
chart simply doesn't render. Tables, summary, and CSV all work statically.

Measured tables use physical units. Directional diagnostics explicitly distinguish
physical endpoints, model coordinates, and normalized likelihood noise.
"""
from __future__ import annotations

import base64
import csv
import html
import io
import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .loop import Result


# ---------------------------------------------------------------------- #
# Style — flat, minimal, A4-friendly. Dark-mode-aware via media query.
# ---------------------------------------------------------------------- #
_CSS = """
:root {
  --bg: #FFFFFF;
  --bg-soft: #F7F6F2;
  --text: #1B1B1B;
  --text-muted: #5F5E5A;
  --border: rgba(0,0,0,0.10);
  --accent: #185FA5;
  --good: #3B6D11;
  --bad: #A32D2D;
  --font: -apple-system, "Helvetica Neue", Arial, sans-serif;
  --mono: "SFMono-Regular", Menlo, Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #1B1B1B;
    --bg-soft: #2A2A2A;
    --text: #ECEAE5;
    --text-muted: #999;
    --border: rgba(255,255,255,0.10);
    --accent: #6FA8DC;
    --good: #B5D17F;
    --bad: #E08A8A;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: var(--font);
  color: var(--text);
  background: var(--bg);
  line-height: 1.55;
}
main { max-width: 920px; margin: 0 auto; padding: 32px 24px 64px; }
h1 { font-size: 26px; font-weight: 500; margin: 0 0 4px; }
h2 { font-size: 18px; font-weight: 500; margin: 28px 0 10px; padding-bottom: 4px; border-bottom: 1px solid var(--border); }
h3 { font-size: 14px; font-weight: 500; margin: 16px 0 6px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; }
.meta { color: var(--text-muted); font-size: 13px; margin-bottom: 24px; }
.best {
  background: var(--bg-soft);
  border-radius: 8px;
  padding: 16px 20px;
  margin: 12px 0 24px;
}
.best-val { font-size: 30px; font-weight: 500; color: var(--accent); }
.best-cap { font-size: 12px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th, td { padding: 6px 8px; text-align: right; border-bottom: 0.5px solid var(--border); font-family: var(--mono); }
th { font-family: var(--font); font-weight: 500; color: var(--text-muted); font-size: 11.5px; text-transform: uppercase; letter-spacing: 0.5px; }
th:first-child, td:first-child { text-align: left; }
.kind-doe { color: var(--text-muted); }
.kind-bo  { color: var(--accent); font-weight: 500; }
.kv { display: grid; grid-template-columns: max-content 1fr; gap: 4px 16px; font-size: 14px; }
.kv dt { color: var(--text-muted); }
.kv dd { margin: 0; font-family: var(--mono); }
ul.knowledge { list-style: none; padding: 0; margin: 0; }
ul.knowledge li { padding: 6px 0; border-bottom: 0.5px solid var(--border); font-family: var(--mono); font-size: 13px; }
ul.knowledge li:last-child { border-bottom: none; }
.note { background: #FFFDF0; color: #5A3D00; border-left: 3px solid #C9A227; padding: 10px 14px; margin: 12px 0; font-size: 13px; }
@media (prefers-color-scheme: dark) {
  .note { background: #3A3000; color: #F0D880; border-left-color: #C9A227; }
}
a.download {
  display: inline-block;
  margin-top: 12px;
  font-size: 12px;
  color: var(--accent);
  text-decoration: none;
  border: 0.5px solid var(--border);
  border-radius: 4px;
  padding: 4px 10px;
}
a.download:hover { border-color: var(--accent); }
.chart-wrap { position: relative; height: 280px; margin: 12px 0; }
footer { color: var(--text-muted); font-size: 11.5px; margin-top: 48px; text-align: center; }
""".strip()


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #
def _format_param(value: float, unit: str, kind: str | None) -> str:
    if kind == "discrete" and float(value).is_integer():
        body = f"{int(value)}"
    else:
        body = f"{value:.4g}"
    return f"{body} {unit}".rstrip()


def _csv_data_url(rows: list[dict], columns: list[str]) -> str:
    header = ",".join(columns)
    lines = [header]
    for r in rows:
        lines.append(",".join(_csv_cell(r.get(c, "")) for c in columns))
    csv_text = "\n".join(lines) + "\n"
    b64 = base64.b64encode(csv_text.encode("utf-8")).decode("ascii")
    return f"data:text/csv;base64,{b64}"


def _csv_cell(v) -> str:
    s = "" if v is None else str(v)
    if any(ch in s for ch in (",", "\"", "\n")):
        s = "\"" + s.replace("\"", "\"\"") + "\""
    return s


def _knowledge_human_list(items: list[dict]) -> list[str]:
    out: list[str] = []
    for it in items:
        kind = it.get("kind")
        param = it.get("param", "")
        if kind == "arrhenius":
            out.append(f"Arrhenius prior on '{param}' (frozen={it.get('frozen', True)})")
        elif kind == "quadratic_peak":
            out.append(
                f"Quadratic {it.get('direction', 'peak')} on '{param}' "
                f"centred at {it.get('center')}"
            )
        elif kind == "monotone":
            out.append(
                f"Monotone: '{param}' {it.get('effect', '?')} "
                f"(ε={it.get('epsilon')}, n_pairs={it.get('n_pairs_per_dim', '?')})"
            )
        elif kind == "random_augment":
            out.append(f"Random augment (n={it.get('n', '?')})  — Category ② safety net")
        elif kind == "gp_prior":
            out.append(f"GP hyperparameter prior (strength={it.get('lengthscale', '?')})")
        else:
            out.append(json.dumps(it, ensure_ascii=False))
    return out


# ---------------------------------------------------------------------- #
# Cumulative-best computation (in physical / user frame)
# ---------------------------------------------------------------------- #
def _cumulative_best(values: list[float], maximize: bool) -> list[float]:
    cur = values[0]
    out = [cur]
    op = max if maximize else min
    for v in values[1:]:
        cur = op(cur, v)
        out.append(cur)
    return out


def _directional_report(result: "Result") -> str:
    specs = [spec for spec in (result.knowledge or {}).get("specs", [])
             if (spec["pattern"], spec["version"]) == ("monotone", "2.0")]
    if not specs:
        return ""

    def metadata(value):
        return '<pre style="white-space:pre-wrap;overflow-wrap:anywhere">' + html.escape(
            json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)) + '</pre>'

    diagnostics = result.multiobjective_diagnostics
    model = diagnostics.get("directional")
    return ("<h2>Directional knowledge (declarations, not measurements)</h2>"
        "<p>Monotone 2.0 uses soft directional factors, not fabricated function-value observations "
        "or measurement variances. Confidence is a fractional evidence weight, not a calibrated probability "
        "that the rule is true. Zero confidence is inactive.</p>"
        "<p>Tempered-probit Laplace approximation: each accepted pair has weight confidence / accepted pair count. "
        "Requested pairs per dimension: 1 to 32; at most 64 active pairs per output and 4096 generated centers "
        "per specification. Finite local pairs do not guarantee global or partial monotonicity.</p>"
        + metadata(specs)
        + "<h2>Directional model diagnostics</h2>"
        + ("<p>No ask-time fit diagnostics are available. No model is fitted by reporting or checkpoint restore.</p>"
           if model is None else
           "<p>Last successful ask-time fit, not a post-tell refit or a historical sequence of model fits. "
           "Physical endpoints and model-coordinate endpoints are labeled separately. "
           "Declared measurement variance and fitted likelihood noise may differ, including zero-noise clamping.</p>"
           + metadata(model))
        + "<h2>Measured trend checks</h2>"
        + "<p>Marginal measured trends can be confounded when other factors vary. They do not prove "
        "partial monotonicity or its violation. Conditioning results are not empirical validation. "
        "No automatic change to epsilon or confidence is made.</p>"
        + metadata(diagnostics.get("validation", [])))


# ---------------------------------------------------------------------- #
# Main entry point
# ---------------------------------------------------------------------- #
def _render_multiobjective(result: "Result", title: str) -> str:
    """Offline measured tables; all user text is escaped at the HTML boundary."""
    def escaped(value):
        return html.escape("" if value is None else str(value))

    def table(rows, columns):
        header = "".join(f"<th>{escaped(c)}</th>" for c in columns)
        body = "".join("<tr>" + "".join(f"<td>{escaped(row.get(c))}</td>" for c in columns) + "</tr>"
                       for row in rows)
        return f'<div style="overflow-x:auto"><table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table></div>'

    def metadata(value):
        return '<pre style="white-space:pre-wrap;overflow-wrap:anywhere">' + escaped(
            json.dumps(value, indent=2, ensure_ascii=False)) + '</pre>'

    d = result.multiobjective_diagnostics
    columns = result.history_columns
    # Use role mappings even when a response is literally named status or trial_id.
    pareto_columns = [columns["metadata"]["trial_id"], *columns["parameters"].values(),
                      *columns["objectives"].values(), columns["metadata"]["status"], columns["metadata"]["kind"]]
    rows = result.history_df.to_dict("records")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(result.history_df.columns))
    writer.writeheader()
    writer.writerows(rows)
    encoded = base64.b64encode(buffer.getvalue().encode("utf-8")).decode("ascii")
    content = [f"<h1>{escaped(title)}</h1>",
               "<p>Observed measured outcomes, not a true or predicted Pareto front. No scalar winner.</p>",
               "<h2>Objectives</h2>", table(d["objectives"], ["name", "direction", "unit", "target", "priority"]),
               "<h2>Observed Pareto recipes</h2>"]
    if result.pareto_rows:
        content.append(table(result.pareto_rows, pareto_columns))
    else:
        content.append("<p>No feasible measurements. Pareto set is empty; HV = 0; reference unresolved.</p>")
    content += ["<h2>Observed hypervolume progress</h2>",
                f'<p>HV = {escaped(d["hypervolume"])}. Fixed normalized Objective utilities; '
                'no preference weights or knowledge target-range penalties.</p>',
                '<p>Report-only geometry is not comparable across separately finalized reports. '
                'Only campaign-fixed geometry supports plateau advice.</p>',
                table([dict(update, hypervolume=value) for update, value in
                       zip(d["history_updates"], result.hypervolume_history)], ["batch_index", "row_count", "hypervolume"]),
                "<h2>Reference and scale</h2>", metadata({k: d[k] for k in
                    ("reference_status", "geometry_scope", "hypervolume_history_scope", "geometry")}),
                "<h2>Configured route (not a model fit)</h2>", metadata({"noise_mode": d["noise_mode"], "route": d["route"]}),
                "<h2>Soft preferences</h2>",
                '<p>Knowledge target ranges and tradeoff weights guide acquisition, not measured HV or hard safety.</p>',
                metadata(d["soft_preferences"]), "<h2>Advisory stop signals</h2>", metadata(result.stop_signals),
                "<h2>Trial history</h2>", table(rows, list(result.history_df.columns)),
                f'<a class="download" download="history.csv" href="data:text/csv;base64,{encoded}">Download CSV</a>',
                "<h2>Parameter units</h2>", metadata(result.param_units),
                "<h2>Knowledge (declarations, not measurements)</h2>", metadata(result.knowledge),
                _directional_report(result),
                "".join(f'<p class="note">{escaped(note)}</p>' for note in result.notes)]
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<title>{escaped(title)}</title><meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<style>{_CSS}</style></head><body><main>' + "".join(content) + '</main></body></html>')


def render_html(result: "Result", title: str | None = None) -> str:
    """Return a complete HTML document as a string."""
    if title is None:
        title = "expdoe-dk campaign report"
    if len(result.objectives) > 1:
        return _render_multiobjective(result, title)

    objective = result.objectives[0]
    maximize = bool(result.maximize[0])
    direction_label = "maximise" if maximize else "minimise"

    history = result.history_df.copy()
    param_names = [c for c in history.columns if c not in ("y", "kind")]
    unit_of: dict[str, str] = dict(result.param_units or {})
    kind_of: dict[str, str] = dict(result.param_kinds or {})

    y_values = history["y"].astype(float).tolist()
    cum_best = _cumulative_best(y_values, maximize=maximize) if y_values else []
    trial_index = list(range(1, len(y_values) + 1))

    # ----------------------------------------------------------------- #
    # Header
    # ----------------------------------------------------------------- #
    header_html = (
        f'<h1>{html.escape(title)}</h1>'
        f'<p class="meta">Objective: <code>{html.escape(objective)}</code>'
        f' ({direction_label}) · {result.n_doe} DoE + {result.n_bo} BO trials</p>'
    )

    # ----------------------------------------------------------------- #
    # Best point card
    # ----------------------------------------------------------------- #
    best_html = '<p>No measured observations; no best point.</p>'
    if y_values:
        best_idx = int(history["y"].idxmax() if maximize else history["y"].idxmin())
        best_row = history.iloc[best_idx]
        best_y_str = f"{result.best_y_physical:.4g}"
        best_params_rows = "".join(
            f"<dt>{html.escape(p)}</dt><dd>"
            f"{_format_param(float(best_row[p]), unit_of.get(p, ''), kind_of.get(p))}"
            f"</dd>"
            for p in param_names
        )
        best_html = (
            '<section class="best">'
            f'<div class="best-cap">Best {html.escape(objective)} ({direction_label})</div>'
            f'<div class="best-val">{best_y_str}</div>'
            f'<dl class="kv">{best_params_rows}</dl>'
            '</section>'
        )

    # ----------------------------------------------------------------- #
    # Convergence chart
    # ----------------------------------------------------------------- #
    chart_data_js = json.dumps({
        "labels": trial_index,
        "values": cum_best,
        "raw": y_values,
        "kinds": history["kind"].tolist() if "kind" in history.columns else [],
        "maximize": maximize,
        "objective": objective,
    }).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    chart_section = (
        '<h2>Convergence</h2>'
        '<div class="chart-wrap"><canvas id="conv-chart" '
        'role="img" aria-label="Best yield vs trial number"></canvas></div>'
    )

    # ----------------------------------------------------------------- #
    # History table + CSV
    # ----------------------------------------------------------------- #
    table_rows = []
    for i, (_, r) in enumerate(history.iterrows(), start=1):
        cells = [f"<td>{i}</td>"]
        for p in param_names:
            cells.append(
                f"<td>{_format_param(float(r[p]), unit_of.get(p, ''), kind_of.get(p))}</td>"
            )
        cells.append(f"<td>{float(r['y']):.4g}</td>")
        kind = str(r.get("kind", ""))
        cls = f"kind-{kind}" if kind in ("doe", "bo") else ""
        cells.append(f'<td class="{cls}">{html.escape(kind)}</td>')
        table_rows.append("<tr>" + "".join(cells) + "</tr>")

    table_header_cells = (
        "<th>trial</th>"
        + "".join(f"<th>{html.escape(p)}</th>" for p in param_names)
        + f"<th>{html.escape(objective)}</th>"
        + "<th>kind</th>"
    )
    csv_rows = [
        {**{p: float(r[p]) for p in param_names},
         "y": float(r["y"]),
         "kind": str(r.get("kind", ""))}
        for _, r in history.iterrows()
    ]
    csv_url = _csv_data_url(csv_rows, param_names + ["y", "kind"])

    history_html = (
        '<h2>History</h2>'
        f'<table><thead><tr>{table_header_cells}</tr></thead>'
        f'<tbody>{"".join(table_rows)}</tbody></table>'
        f'<a class="download" href="{csv_url}" download="campaign_history.csv">'
        '⬇ Download CSV</a>'
    )

    # ----------------------------------------------------------------- #
    # Knowledge spec listing
    # ----------------------------------------------------------------- #
    directional_html = _directional_report(result)
    items_human = _knowledge_human_list(result.knowledge_summary or [])
    if items_human:
        knowledge_html = (
            '<h2>Knowledge applied</h2>'
            '<ul class="knowledge">'
            + "".join(f"<li>{html.escape(line)}</li>" for line in items_human)
            + "</ul>"
        )
    elif directional_html:
        knowledge_html = ""
    else:
        knowledge_html = (
            '<h2>Knowledge applied</h2>'
            '<p class="meta">None. Campaign ran with the baseline GP only.</p>'
        )
    knowledge_html += directional_html

    # Notes (auto-default applied etc.)
    notes_html = ""
    if result.notes:
        notes_html = "".join(
            f'<div class="note">{html.escape(n)}</div>' for n in result.notes
        )

    # ----------------------------------------------------------------- #
    # Footer
    # ----------------------------------------------------------------- #
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    footer = (
        f'<footer>Generated by '
        f'<a href="https://github.com/517justin/expdoe-dk">expdoe-dk</a>'
        f' · {now}</footer>'
    )

    # ----------------------------------------------------------------- #
    # Final document
    # ----------------------------------------------------------------- #
    script = """
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.js"></script>
<script>
(function() {
  if (typeof Chart === "undefined") return;
  var data = __CHART_DATA__;
  var ctx = document.getElementById("conv-chart");
  if (!ctx) return;
  new Chart(ctx, {
    type: "line",
    data: {
      labels: data.labels,
      datasets: [
        {
          label: "Best so far (" + data.objective + ")",
          data: data.values,
          borderColor: "#185FA5",
          backgroundColor: "rgba(24,95,165,0.15)",
          fill: false,
          tension: 0.0,
          pointRadius: 0,
          borderWidth: 2,
          stepped: true,
        },
        {
          label: "Per-trial " + data.objective,
          data: data.raw,
          borderColor: "rgba(60,60,60,0.0)",
          backgroundColor: data.kinds.map(function(k){
            return k === "doe" ? "rgba(95,94,90,0.7)" : "rgba(24,95,165,0.7)";
          }),
          pointRadius: 4,
          showLine: false,
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { title: { display: true, text: "trial" } },
        y: { title: { display: true, text: data.objective } }
      },
      plugins: { legend: { position: "bottom", labels: { boxWidth: 12 } } }
    }
  });
})();
</script>
""".replace("__CHART_DATA__", chart_data_js)

    head = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f'<title>{html.escape(title)}</title>'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f'<style>{_CSS}</style></head>'
    )
    body = (
        f'<body><main>{header_html}{notes_html}{best_html}'
        f'{chart_section}{history_html}{knowledge_html}{footer}'
        f'</main>{script}</body></html>'
    )
    return head + body


def write_html_report(
    result: "Result",
    path: str | Path,
    title: str | None = None,
) -> Path:
    """Render the report and write it to disk; returns the resolved Path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(render_html(result, title=title), encoding="utf-8")
    return p
