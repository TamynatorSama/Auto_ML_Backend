"""
report_html.py
--------------
Render a RunReport as one self-contained page.

No scripts and no external assets beyond the webfont: a file that can be opened
from the run directory, attached to a message, or archived next to the model and
still read correctly years later.

The page is a measurement record, not a dashboard. Its job is to make a number
believable, so the protocol, the floor it beat, the attempt that produced it and
the places it fails are given the same weight as the score itself.

    render_html(report) -> str
"""

from __future__ import annotations

import html
import math
from typing import Optional

from models import RunReport, ScoreRow

_FONTS = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
    'family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap">'
)

_CSS = """
:root{
  --ground:#0c100f; --panel:#141918; --raised:#1a201e; --line:#242b29;
  --ink:#dde4e1; --ink-2:#a8b4b0; --dim:#76847f;
  --accent:#45c9a0; --accent-dim:#2a6b56;
  --over:#d98a5a; --under:#5a9fd9; --warn:#d9a441; --critical:#d9615a;
  --sans:"IBM Plex Sans",ui-sans-serif,system-ui,sans-serif;
  --mono:"IBM Plex Mono",ui-monospace,Menlo,Consolas,monospace;
}
:root[data-theme="light"]{
  --ground:#f4f6f5; --panel:#ffffff; --raised:#eef1f0; --line:#dde2e0;
  --ink:#16201d; --ink-2:#41504b; --dim:#6b7873;
  --accent:#1f7a5e; --accent-dim:#a9ded0;
  --over:#a85a25; --under:#2b5f8f; --warn:#8a6412; --critical:#a3382f;
}
@media (prefers-color-scheme: light){
  :root:not([data-theme="dark"]){
    --ground:#f4f6f5; --panel:#ffffff; --raised:#eef1f0; --line:#dde2e0;
    --ink:#16201d; --ink-2:#41504b; --dim:#6b7873;
    --accent:#1f7a5e; --accent-dim:#a9ded0;
    --over:#a85a25; --under:#2b5f8f; --warn:#8a6412; --critical:#a3382f;
  }
}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);
  font-family:var(--sans);font-size:14px;line-height:1.55;
  -webkit-font-smoothing:antialiased}
.wrap{max-width:1100px;margin:0 auto;padding-block:32px 72px;padding-left:20px;padding-right:20px;
  display:flex;flex-direction:column;gap:18px}
h1{font-size:26px;font-weight:600;margin:0;letter-spacing:-.015em;text-wrap:balance}
h2{font-family:var(--mono);font-size:11px;font-weight:500;letter-spacing:.13em;
  text-transform:uppercase;color:var(--dim);margin:0 0 15px}
p{margin:0}
.head .meta{font-family:var(--mono);font-size:12px;color:var(--dim);margin-top:8px;line-height:1.7}
.state{display:inline-block;font-family:var(--mono);font-size:10px;letter-spacing:.13em;
  text-transform:uppercase;color:var(--accent);border:1px solid var(--accent-dim);
  border-radius:2px;padding:3px 8px;vertical-align:7px;margin-left:12px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:22px}
.grid{display:grid;gap:18px}
@media(min-width:860px){.split{grid-template-columns:1.15fr 1fr}.halves{grid-template-columns:1fr 1fr}}
.selected{display:flex;flex-wrap:wrap;gap:30px;justify-content:space-between;align-items:flex-start}
.selected .who{flex:1 1 320px;max-width:520px}
.selected .name{font-size:21px;font-weight:600;margin:3px 0 8px;letter-spacing:-.01em}
.selected .why{color:var(--ink-2);font-size:13px}
.readout{display:flex;flex-wrap:wrap;gap:28px}
.readout .k{font-family:var(--mono);font-size:10px;letter-spacing:.13em;
  text-transform:uppercase;color:var(--dim)}
.readout .v{font-family:var(--mono);font-size:27px;font-weight:500;color:var(--accent);
  margin-top:4px;font-variant-numeric:tabular-nums}
.readout .n{font-family:var(--mono);font-size:11px;color:var(--dim);margin-top:3px}
.scroll{overflow-x:auto}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
th{font-family:var(--mono);font-size:10px;font-weight:500;letter-spacing:.1em;text-transform:uppercase;
  color:var(--dim);text-align:left;padding:0 12px 10px 0;border-bottom:1px solid var(--line);white-space:nowrap}
td{padding:12px 12px 12px 0;border-bottom:1px solid var(--line);vertical-align:top}
tbody tr:last-child td{border-bottom:0}
.num{text-align:right;font-family:var(--mono)}
tr.win td{background:color-mix(in srgb,var(--accent) 7%,transparent)}
tr.win td:first-child{box-shadow:inset 2px 0 0 var(--accent)}
.model{font-family:var(--mono);font-size:13px}
.sub{font-family:var(--mono);font-size:11px;color:var(--dim);margin-top:3px}
.dim{color:var(--dim)}
.bar{background:var(--raised);border-radius:2px;height:9px;overflow:hidden}
.bar>i{display:block;height:100%;background:var(--accent);border-radius:0 2px 2px 0}
.imp{display:grid;grid-template-columns:minmax(0,1fr) 3fr 52px;gap:12px;align-items:center;
  padding:6px 0;font-family:var(--mono);font-size:12px}
.imp .f{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.imp .n{text-align:right;color:var(--dim);font-variant-numeric:tabular-nums}
.axis{position:relative;height:9px;background:var(--raised);border-radius:2px}
.axis>i{position:absolute;top:0;height:100%;border-radius:2px}
.axis>i.pos{background:var(--over)}
.axis>i.neg{background:var(--under)}
.axis>s{position:absolute;top:-3px;bottom:-3px;left:50%;width:1px;background:var(--line);opacity:.9}
.step{display:grid;grid-template-columns:22px minmax(0,1fr) auto;gap:14px;
  padding:12px 0;border-bottom:1px solid var(--line)}
.step:last-child{border-bottom:0}
.step .g{font-family:var(--mono);color:var(--dim);font-size:12px}
.step .what{font-size:13px;overflow-wrap:anywhere}
.delta{font-family:var(--mono);font-size:12px;font-variant-numeric:tabular-nums;white-space:nowrap}
.better{color:var(--accent)}
.worse{color:var(--warn)}
.flag{display:flex;gap:11px;padding:11px 0;border-bottom:1px solid var(--line);font-size:13px}
.flag:last-child{border-bottom:0}
.flag .dot{flex:0 0 auto;width:6px;height:6px;border-radius:50%;margin-top:7px;background:var(--warn)}
.flag.severe .dot{background:var(--critical)}
.flag .who{font-family:var(--mono);font-size:12px;color:var(--ink-2);flex:0 0 auto}
.flag .msg{color:var(--ink-2);overflow-wrap:anywhere}
dl{display:grid;grid-template-columns:auto minmax(0,1fr);gap:9px 20px;margin:0;font-size:13px}
dt{font-family:var(--mono);font-size:11px;color:var(--dim);text-transform:uppercase;letter-spacing:.07em}
dd{margin:0;overflow-wrap:anywhere}
code{font-family:var(--mono);font-size:12px;background:var(--raised);
  padding:2px 6px;border-radius:3px;color:var(--ink-2)}
ul.reqs{margin:0;padding-left:18px;font-size:13px;color:var(--ink-2)}
ul.reqs li{margin-bottom:6px}
.empty{color:var(--dim);font-size:13px}
"""


def _e(value) -> str:
    return html.escape(str(value))


def _num(value: Optional[float], places: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float) and not math.isfinite(value):
        return "inf" if value > 0 else "−inf"
    if abs(value) >= 1e6 or (value != 0 and abs(value) < 1e-3):
        return f"{value:.3e}"
    return f"{value:,.{places}f}"


def _duration(seconds: float) -> str:
    minutes, rest = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m {rest}s" if minutes else f"{rest}s"


def _cell(row: ScoreRow, metric: str) -> Optional[float]:
    return row.test_scores.get(metric, row.cv_scores.get(metric))


def _selected(report: RunReport) -> str:
    chosen = report.selected
    if chosen is None:
        return ('<section class="panel"><h2>selected model</h2>'
                '<p class="empty">No model produced a usable score.</p></section>')

    readouts = []
    for metric in report.metrics:
        test = chosen.test_scores.get(metric)
        cv = chosen.cv_scores.get(metric)
        readouts.append(
            f'<div><div class="k">{_e(metric)}</div>'
            f'<div class="v">{_num(test if test is not None else cv)}</div>'
            f'<div class="n">cv {_num(cv)}</div></div>'
        )
    if report.improvement_over_baseline is not None:
        readouts.append(
            f'<div><div class="k">vs baseline</div>'
            f'<div class="v">{report.improvement_over_baseline * 100:.0f}%</div>'
            f'<div class="n">{_e(report.baseline_strategy)} floor</div></div>'
        )

    return f"""<section class="panel">
  <div class="selected">
    <div class="who">
      <h2>selected model</h2>
      <div class="name">{_e(chosen.model)}</div>
      <p class="why">{_e(report.selection_reason)}</p>
    </div>
    <div class="readout">{''.join(readouts)}</div>
  </div>
</section>"""


def _comparison(report: RunReport) -> str:
    heads = "".join(f'<th class="num">{_e(m)}</th>' for m in report.metrics)
    rows = []
    for row in report.comparison:
        won = report.selected is not None and row.model == report.selected.model
        scores = "".join(f'<td class="num">{_num(_cell(row, m))}</td>' for m in report.metrics)
        note = f'<div class="sub">{_e(row.note[:120])}</div>' if row.note else ""
        rows.append(
            f'<tr class="{"win" if won else ""}">'
            f'<td class="num dim">{row.rank}</td>'
            f'<td><div class="model">{_e(row.model)}</div>'
            f'<div class="sub">{row.generations} generations · {row.repairs} repairs · '
            f"{_duration(row.wall_seconds)}</div>{note}</td>"
            f'<td class="sub">{_e(row.status)}</td>{scores}</tr>'
        )
    return f"""<section class="panel scroll">
  <h2>model comparison · every model scored on the same held-out rows</h2>
  <table><thead><tr><th class="num">#</th><th>model</th><th>status</th>{heads}</tr></thead>
  <tbody>{''.join(rows)}</tbody></table>
</section>"""


def _importance(report: RunReport) -> str:
    who = report.selected.model if report.selected else ""
    if not report.importance:
        return ('<section class="panel"><h2>feature importance</h2>'
                '<p class="empty">The winning script wrote no <code>feature_importance.csv</code>.</p></section>')

    top = max(abs(f.importance) for f in report.importance) or 1.0
    bars = "".join(
        f'<div class="imp"><span class="f">{_e(f.feature)}</span>'
        f'<span class="bar"><i style="width:{max(abs(f.importance) / top * 100, 1.2):.1f}%"></i></span>'
        f'<span class="n">{_num(f.importance, 3)}</span></div>'
        for f in report.importance
    )
    return f"""<section class="panel">
  <h2>feature importance · {_e(who)}</h2>
  {bars}
  <p class="sub" style="margin-top:12px">Permutation importance, measured on held-back
  training rows. The test set is never touched for this.</p>
</section>"""


def _errors(report: RunReport) -> str:
    if report.error_bands:
        widest = max(abs(b.mean_signed_error) for b in report.error_bands) or 1.0
        rows = []
        for band in report.error_bands:
            share = abs(band.mean_signed_error) / widest * 50
            side = "pos" if band.mean_signed_error > 0 else "neg"
            offset = 50 if band.mean_signed_error > 0 else 50 - share
            rows.append(
                f"<tr><td class='model'>{_e(band.band)}</td>"
                f"<td class='num dim'>{band.rows:,}</td>"
                f"<td class='num'>{_num(band.mean_absolute_error, 2)}</td>"
                f"<td style='min-width:120px'><span class='axis'>"
                f"<i class='{side}' style='left:{offset:.1f}%;width:{max(share,0.8):.1f}%'></i><s></s></span></td>"
                f"<td class='num'>{band.mean_signed_error:+,.1f}</td></tr>"
            )
        return f"""<section class="panel scroll">
  <h2>where it errs · test rows by {_e(report.target)}</h2>
  <table><thead><tr><th>band</th><th class="num">rows</th><th class="num">mean abs error</th>
  <th>bias</th><th class="num">value</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
  <p class="sub" style="margin-top:12px">Bias is mean predicted minus mean actual:
  bars right of centre over-predict, left under-predict.</p>
</section>"""

    if report.confusion:
        rows = "".join(
            f"<tr><td class='model'>{_e(c.actual)}</td><td class='model'>{_e(c.predicted)}</td>"
            f"<td class='num'>{c.rows:,}</td></tr>"
            for c in report.confusion
        )
        return f"""<section class="panel scroll"><h2>confusion · test rows</h2>
  <table><thead><tr><th>actual</th><th>predicted</th><th class="num">rows</th></tr></thead>
  <tbody>{rows}</tbody></table></section>"""

    return ('<section class="panel"><h2>where it errs</h2>'
            '<p class="empty">The winning script wrote no <code>test_predictions.csv</code>.</p></section>')


def _trace(report: RunReport) -> str:
    blocks = []
    for model, steps in report.trace.items():
        if not steps:
            continue
        lines = []
        for step in steps:
            if step.delta is None:
                delta = '<span class="delta dim">first</span>'
            elif step.delta > 0:
                delta = f'<span class="delta better">−{_num(abs(step.delta), 4)}</span>'
            else:
                delta = f'<span class="delta worse">+{_num(abs(step.delta), 4)}</span>'
            detail = [step.status]
            if step.repairs:
                detail.append(f"{step.repairs} repairs")
            if step.score is not None:
                detail.append(f"{report.primary_metric} {_num(step.score)}")
            lines.append(
                f'<div class="step"><span class="g">{step.generation}</span>'
                f'<span><span class="what">{_e(step.changes[:160] or "initial implementation")}</span>'
                f'<div class="sub">{_e(" · ".join(detail))}</div></span>{delta}</div>'
            )
        blocks.append(
            f'<section class="panel"><h2>{_e(model)} · what each attempt changed</h2>{"".join(lines)}</section>'
        )
    return "".join(blocks)


def _protocol(report: RunReport) -> str:
    baseline = ", ".join(f"{k} {_num(v)}" for k, v in report.baseline_scores.items()) or "none"
    environment = ", ".join(f"{k} {v}" for k, v in sorted(report.environment.items()))
    requirements = "".join(f"<li>{_e(r)}</li>" for r in report.requirements)
    return f"""<section class="panel">
  <h2>how this was measured</h2>
  <dl>
    <dt>split</dt><dd>{_e(report.split_summary)}</dd>
    <dt>folds</dt><dd>{_e(report.cv_summary)}</dd>
    <dt>test set</dt><dd>{report.test_rows:,} rows, unseen during the loop, scored once per model</dd>
    <dt>baseline</dt><dd>{_e(report.baseline_strategy)} — {_e(baseline)}</dd>
    <dt>libraries</dt><dd>{_e(environment)}</dd>
  </dl>
  {'<h2 style="margin:22px 0 12px">preprocessing required of every script</h2><ul class="reqs">' + requirements + "</ul>" if requirements else ""}
</section>"""


def _artifacts(report: RunReport) -> str:
    if not report.artifacts:
        return ""
    rows = "".join(f"<dt>{_e(k)}</dt><dd><code>{_e(v)}</code></dd>" for k, v in report.artifacts.items())
    return f'<section class="panel"><h2>artifacts · winning attempt</h2><dl>{rows}</dl></section>'


def _warnings(report: RunReport) -> str:
    if not report.warnings:
        return ""
    severe = ("cannot be loaded", "not measuring", "is not a usable score", "leakage")
    flags = []
    for model, messages in report.warnings.items():
        for message in messages:
            bad = any(mark in message for mark in severe)
            flags.append(
                f'<div class="flag{" severe" if bad else ""}"><span class="dot"></span>'
                f'<span class="who">{_e(model)}</span>'
                f'<span class="msg">{_e(message[:420])}</span></div>'
            )
    return f'<section class="panel"><h2>raised about these results</h2>{"".join(flags)}</section>'


def render_html(report: RunReport, standalone: bool = True) -> str:
    narrative = (
        f'<section class="panel"><h2>summary</h2><p>{_e(report.narrative)}</p></section>'
        if report.narrative else ""
    )
    body = f"""<div class="wrap">
  <header class="head">
    <h1>Run {report.run_id} · {_e(report.target)}<span class="state">{_e(report.status)}</span></h1>
    <div class="meta">
      {_e(report.task_type)} · {report.models_scored} of {report.models_planned} models scored ·
      {report.generations} generations · {report.repairs} repairs · {report.executions} executions ·
      {_duration(report.wall_seconds)} fitting · {report.train_rows:,} train / {report.test_rows:,} test rows
    </div>
  </header>
  {narrative}
  {_selected(report)}
  {_comparison(report)}
  <div class="grid split">{_importance(report)}{_errors(report)}</div>
  {_trace(report)}
  <div class="grid halves">{_protocol(report)}{_artifacts(report)}</div>
  {_warnings(report)}
</div>"""

    head = f'<title>Run {report.run_id} · {_e(report.target)}</title>{_FONTS}<style>{_CSS}</style>'
    if not standalone:
        # the artifact host supplies the document shell
        return head + body
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"{head}</head><body>{body}</body></html>"
    )
