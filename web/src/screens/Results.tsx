import { useState, type ReactNode } from "react";

import {
  api, fileName, fileUrl, useReport,
  type ConfusionCell, type ErrorBand, type FeatureWeight, type Job, type MetricSpec, type Report,
  type ScoreRow,
} from "../api";
import { count, when } from "../format";

/**
 * The stored RunReport, laid out as the Console canvas lays it out: the chosen
 * model across the top with its metrics beside it, the comparison next to the
 * importances, and three cards under them — where the gain came from, how wrong
 * it was, and what you can take away (docs/PHASE5.md §7, §16).
 *
 * The report was written for exactly this — every metric states its direction,
 * every importance its share, every generation its delta — so nothing here
 * recomputes what the run already decided, and nothing ranks a metric by
 * guessing which way is better.
 *
 * Regression and classification come out of the same code: the report carries
 * error bands for one and a confusion matrix for the other, and whichever is
 * empty simply doesn't render (G37).
 */

const STATUS_NOTE: Record<string, string> = {
  max_tries: "used every try", no_improvement: "stopped improving", failed: "never produced a score",
  unavailable: "not installed in this environment", running: "still going",
};

function Card({ title, meta, children }: { title: string; meta?: ReactNode; children: ReactNode }) {
  return (
    <div style={{ border: "1px solid var(--line)", borderRadius: 6, background: "var(--card)",
                  overflow: "hidden", display: "flex", flexDirection: "column", minWidth: 0 }}>
      <div className="chead">
        <div className="label" style={{ color: "var(--muted)" }}>{title}</div>
        {meta}
      </div>
      <div style={{ flex: 1, minHeight: 0 }}>{children}</div>
    </div>
  );
}

function num(value: number | null | undefined, places = 4): string {
  return value === null || value === undefined || Number.isNaN(value) ? "—" : Number(value).toFixed(places);
}

/** Enough places to tell two scores apart, without a wall of zeros on a big one. */
function score(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  const size = Math.abs(value);
  return value.toFixed(size >= 100 ? 2 : size >= 1 ? 3 : 4);
}

function minutes(seconds: number): string {
  if (seconds >= 3600) return `${Math.floor(seconds / 3600)} h ${Math.round((seconds % 3600) / 60)} m`;
  return seconds >= 60 ? `${(seconds / 60).toFixed(1)} min` : `${Math.round(seconds)} s`;
}

/** The chosen model across the top, with the run's metrics beside it. */
function Chosen({ report, selected, primary }: {
  report: Report; selected: ScoreRow | undefined; primary: MetricSpec | undefined;
}) {
  return (
    <div style={{ border: "1px solid var(--line)", borderRadius: 6, background: "var(--card)",
                  padding: "20px 24px", marginBottom: 16, display: "grid", gap: "20px 32px",
                  gridTemplateColumns: "minmax(280px, 1.2fr) minmax(0, 1fr)", alignItems: "start" }}>
      <div style={{ minWidth: 0 }}>
        <div className="label" style={{ color: "var(--accent-bright)", marginBottom: 9 }}>Selected model</div>
        <div style={{ fontSize: 23, letterSpacing: "-0.015em", color: "var(--bright)", marginBottom: 10 }}>
          {report.selected_model ?? "no model was selected"}
        </div>
        <div style={{ fontSize: 13, lineHeight: 1.65, color: "var(--muted)", maxWidth: 560 }}>
          {report.selection_reason || report.narrative || "The run finished."}
        </div>
      </div>
      <div style={{ display: "grid", gap: "16px 20px",
                    gridTemplateColumns: "repeat(auto-fit, minmax(112px, 1fr))" }}>
        {report.metrics.map((metric) => (
          <div key={metric.name} style={{ minWidth: 0 }}>
            {/* balanced_accuracy is long enough to wrap and shove its own value out of
                line with the others, so the label is kept to one line */}
            <div className="label" title={metric.name}
                 style={{ color: "var(--faint)", whiteSpace: "nowrap", overflow: "hidden",
                          textOverflow: "ellipsis" }}>
              {metric.name} {metric.direction === "higher" ? "↑" : "↓"}
            </div>
            <div className="mono" style={{ fontSize: 19, marginTop: 7,
                                           color: metric.name === primary?.name ? "var(--accent-bright)"
                                                                                : "var(--text)" }}>
              {score(selected?.test_scores?.[metric.name])}
            </div>
            <div className="mono" style={{ fontSize: 10, color: "var(--faint)", marginTop: 4 }}>
              cv {score(selected?.cv_scores?.[metric.name])}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

/** Every model that ran, ranked as the report ranked them. */
function Comparison({ ws, jobId, rows, metrics, open, onOpen }: {
  ws: string;
  jobId: number;
  rows: ScoreRow[];
  metrics: MetricSpec[];
  open: string | null;
  onOpen: (model: string | null) => void;
}) {
  const columns = metrics.map((metric) => metric.name);
  const grid = `26px minmax(110px, 1.3fr) ${columns.map(() => "minmax(66px, 1fr)").join(" ")} 60px`;
  return (
    <div style={{ overflowX: "auto" }}>
      <div style={{ minWidth: 420 }}>
        <div className="srow head" style={{ gridTemplateColumns: grid, minWidth: 0, padding: "10px 22px" }}>
          <div>#</div>
          <div>model</div>
          {metrics.map((metric) => (
            <div key={metric.name} style={{ textAlign: "right" }}>
              {metric.name}
              <span style={{ color: "var(--faint)" }}> {metric.direction === "higher" ? "↑" : "↓"}</span>
            </div>
          ))}
          <div style={{ textAlign: "right" }}>tries</div>
        </div>
        {rows.map((row) => (
          <div key={row.model}>
            <div className="srow" style={{ gridTemplateColumns: grid, minWidth: 0, padding: "12px 22px",
                                           cursor: "pointer",
                                           borderLeft: `2px solid ${row.selected ? "var(--accent)"
                                                                                 : "transparent"}`,
                                           background: row.selected ? "#0F1D19" : undefined }}
                 onClick={() => onOpen(open === row.model ? null : row.model)}>
              <div className="mono" style={{ fontSize: 12, color: "var(--faint)" }}>{row.rank ?? "—"}</div>
              <div style={{ minWidth: 0 }}>
                <div style={{ fontSize: 13.5, color: row.selected ? "var(--accent-bright)" : "var(--text)" }}>
                  {row.model}
                </div>
                <div className="mono" style={{ fontSize: 10.5, color: "var(--faint)", marginTop: 3 }}>
                  {row.selected ? "selected · " : ""}
                  {row.note || STATUS_NOTE[row.status] || row.status.replace(/_/g, " ")}
                  {row.eligibility !== "clean" ? ` · ${row.eligibility}` : ""}
                </div>
              </div>
              {columns.map((metric) => (
                <div key={metric} className="mono" style={{ fontSize: 12, textAlign: "right",
                                                            color: "var(--text)" }}>
                  {score(row.test_scores?.[metric])}
                  <div style={{ fontSize: 10, color: "var(--faint)", marginTop: 3 }}>
                    cv {score(row.cv_scores?.[metric])}
                  </div>
                </div>
              ))}
              <div className="mono" style={{ fontSize: 12, textAlign: "right", color: "var(--muted)" }}>
                {row.generations}
                <div style={{ fontSize: 10, color: "var(--faint)", marginTop: 3 }}>
                  {row.repairs ? `${row.repairs} fixed` : `${row.executions} runs`}
                </div>
              </div>
            </div>
            {open === row.model && <Artifacts ws={ws} jobId={jobId} row={row} />}
          </div>
        ))}
      </div>
    </div>
  );
}

/** What this model left behind. Downloads only: nothing here is opened or unpickled. */
function Artifacts({ ws, jobId, row }: { ws: string; jobId: number; row: ScoreRow }) {
  const names = Object.entries(row.artifacts ?? {});
  if (!names.length) {
    return <div style={{ padding: "12px 22px 16px", color: "var(--faint)", fontSize: 12.5 }}>
      This model left no files.
    </div>;
  }
  return (
    <div style={{ padding: "12px 22px 16px", background: "var(--bg)",
                  borderBottom: "1px solid var(--line-soft)" }}>
      <div className="mono" style={{ fontSize: 10.5, color: "var(--faint)", marginBottom: 9 }}>
        from attempt {row.best_attempt ?? "—"} · {row.wall_seconds.toFixed(1)}s of training
      </div>
      <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
        {names.map(([label, path]) => (
          <a key={label} className="mono" download
             href={fileUrl(ws, jobId, row.model, row.best_attempt ?? 1, fileName(path))}
             style={{ fontSize: 11.5, borderRadius: 4, padding: "6px 10px",
                      border: "1px solid var(--line)", color: "var(--muted)" }}>
            {label.replace(/_/g, " ")}
          </a>
        ))}
      </div>
    </div>
  );
}

function Importance({ rows, note }: { rows: FeatureWeight[]; note: string }) {
  if (!rows.length) {
    return <div style={{ padding: "26px 22px", color: "var(--faint)", fontSize: 13 }}>
      This model reported no feature importances.
    </div>;
  }
  return (
    <div style={{ padding: "14px 22px 16px" }}>
      {rows.map((row) => (
        <div key={row.feature} style={{ display: "flex", alignItems: "center", gap: 14, padding: "6px 0" }}>
          <div className="mono" style={{ flex: "0 0 38%", minWidth: 0, fontSize: 12, color: "var(--muted)",
                                         overflow: "hidden", textOverflow: "ellipsis",
                                         whiteSpace: "nowrap" }}>
            {row.feature}
          </div>
          <div style={{ flex: 1, height: 6, borderRadius: 3, background: "#1B2220", overflow: "hidden" }}>
            <div style={{ height: "100%", width: `${Math.max(row.share * 100, 0.5)}%`,
                          background: "var(--accent)" }} />
          </div>
          <div className="mono" style={{ flex: "0 0 46px", textAlign: "right", fontSize: 11,
                                         color: "var(--faint)" }}>
            {(row.share * 100).toFixed(1)}%
          </div>
        </div>
      ))}
      {note && (
        <div style={{ fontSize: 12.5, lineHeight: 1.6, color: "var(--dim)", marginTop: 14,
                      borderTop: "1px solid var(--line-soft)", paddingTop: 12 }}>
          {note}
        </div>
      )}
    </div>
  );
}

/**
 * Where the score came from: the baseline the run had to beat, then what each
 * generation of the chosen model added — the canvas's pass contribution, over
 * generations, which is what this engine counts.
 */
function Gain({ report, model, metric }: {
  report: Report; model: string | undefined; metric: MetricSpec | undefined;
}) {
  const steps = (model && report.trace?.[model]) || [];
  const base = metric ? report.protocol.baseline_scores?.[metric.name] : undefined;
  if (!steps.length) {
    return <div style={{ padding: "26px 22px", color: "var(--faint)", fontSize: 13 }}>
      No attempt of the chosen model was recorded.
    </div>;
  }
  return (
    <div style={{ padding: "8px 22px 16px" }}>
      {base !== undefined && (
        <div style={{ display: "flex", alignItems: "baseline", gap: 12, padding: "11px 0",
                      borderBottom: "1px solid var(--line-soft)" }}>
          <div className="mono" style={{ flex: "0 0 18px", fontSize: 11, color: "var(--faint)" }}>—</div>
          <div style={{ minWidth: 0, flex: 1 }}>
            <div style={{ fontSize: 13, color: "var(--muted)" }}>Baseline</div>
            <div className="mono" style={{ fontSize: 10.5, color: "var(--faint)", marginTop: 3 }}>
              {report.protocol.baseline_strategy}
              {report.protocol.baseline_applicable ? "" : " · not applicable"}
            </div>
          </div>
          <div className="mono" style={{ fontSize: 12.5, color: "var(--dim)" }}>{score(base)}</div>
        </div>
      )}
      {steps.map((step) => (
        <div key={step.attempt} style={{ display: "flex", alignItems: "baseline", gap: 12,
                                         padding: "11px 0", borderBottom: "1px solid var(--line-soft)" }}>
          <div className="mono" style={{ flex: "0 0 18px", fontSize: 11, color: "var(--faint)" }}>
            {step.generation}
          </div>
          <div style={{ minWidth: 0, flex: 1 }}>
            <div style={{ fontSize: 13, lineHeight: 1.5, color: step.status === "ok" ? "var(--muted)"
                                                                                     : "#C9673F",
                          overflowWrap: "anywhere" }}>
              {step.changes || (step.status === "ok" ? "no note" : "failed to run")}
            </div>
            {step.judge_notes && (
              <div style={{ fontSize: 12, color: "var(--dim)", marginTop: 5, lineHeight: 1.55 }}>
                {step.judge_notes}
              </div>
            )}
          </div>
          <div style={{ textAlign: "right", flex: "0 0 auto" }}>
            <div className="mono" style={{ fontSize: 12.5, color: "var(--text)" }}>{score(step.score)}</div>
            {step.delta !== null && step.delta !== undefined && (
              <div className="mono" style={{ fontSize: 10.5, marginTop: 3,
                                             color: step.delta > 0 ? "var(--accent-bright)"
                                                                   : "var(--faint)" }}>
                {step.delta > 0 ? "+" : ""}{num(step.delta, 4)}
              </div>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}

/** Classification: actual against predicted, as a square. */
function Confusion({ cells }: { cells: ConfusionCell[] }) {
  const actual = [...new Set(cells.map((cell) => cell.actual))];
  const predicted = [...new Set(cells.map((cell) => cell.predicted))];
  const at = (a: string, p: string) => cells.find((cell) => cell.actual === a && cell.predicted === p)?.rows ?? 0;
  const most = Math.max(...cells.map((cell) => cell.rows), 1);

  return (
    <div style={{ padding: "16px 22px", overflowX: "auto" }}>
      <div style={{ display: "grid", gap: 4,
                    gridTemplateColumns: `76px repeat(${predicted.length}, minmax(64px, 1fr))` }}>
        <div />
        {predicted.map((column) => (
          <div key={column} className="mono" style={{ fontSize: 10, color: "var(--faint)",
                                                      textAlign: "center", paddingBottom: 4 }}>
            said {column}
          </div>
        ))}
        {actual.map((row) => (
          <div key={row} style={{ display: "contents" }}>
            <div className="mono" style={{ fontSize: 10, color: "var(--faint)", display: "flex",
                                           alignItems: "center" }}>
              was {row}
            </div>
            {predicted.map((column) => {
              const rows = at(row, column);
              const right = row === column;
              return (
                <div key={column} className="mono"
                     style={{ fontSize: 12.5, textAlign: "center", padding: "14px 6px", borderRadius: 4,
                              color: right ? "#9FE6D4" : "#E09B86",
                              border: `1px solid ${right ? "#1B3F36" : "#3E2A26"}`,
                              background: right ? `rgba(18,135,111,${0.10 + 0.45 * (rows / most)})`
                                                : `rgba(201,103,63,${0.06 + 0.3 * (rows / most)})` }}>
                  {count(rows)}
                </div>
              );
            })}
          </div>
        ))}
      </div>
    </div>
  );
}

/** Regression: how wrong it was, by where in the target range the row sat. */
function Bands({ bands }: { bands: ErrorBand[] }) {
  const worst = Math.max(...bands.map((band) => band.mean_absolute_error), 1e-9);
  return (
    <div style={{ padding: "12px 22px 16px" }}>
      {bands.map((band) => (
        <div key={band.band} style={{ padding: "9px 0", borderBottom: "1px solid var(--line-soft)" }}>
          <div style={{ display: "flex", alignItems: "baseline", gap: 10, flexWrap: "wrap" }}>
            <div className="mono" style={{ fontSize: 11.5, color: "var(--text)" }}>{band.band}</div>
            <div className="mono" style={{ fontSize: 10, color: "var(--faint)" }}>
              {count(band.rows)} rows
            </div>
            <div className="mono" style={{ marginLeft: "auto", fontSize: 11.5, color: "var(--muted)" }}>
              ±{num(band.mean_absolute_error, 2)}
            </div>
          </div>
          <div style={{ height: 5, borderRadius: 3, background: "#1B2220", marginTop: 7,
                        overflow: "hidden" }}>
            <div style={{ height: "100%", width: `${(band.mean_absolute_error / worst) * 100}%`,
                          background: "var(--accent)" }} />
          </div>
          <div className="mono" style={{ fontSize: 10, marginTop: 5,
                                         color: band.mean_signed_error > 0 ? "#C9873F" : "var(--faint)" }}>
            {band.mean_signed_error > 0 ? "over" : "under"}-predicts by{" "}
            {num(Math.abs(band.mean_signed_error), 2)}
          </div>
        </div>
      ))}
    </div>
  );
}

/**
 * What you can do with the run — the canvas's handoff. Serving and scheduling
 * are not built; they sit here disabled rather than being left out, so the screen
 * doesn't quietly pretend they were never promised.
 */
function Handoff({ ws, job, selected, onTraining }: {
  ws: string; job: Job; selected: ScoreRow | undefined; onTraining: () => void;
}) {
  const artifacts = Object.entries(selected?.artifacts ?? {});
  const pick = (want: string) => artifacts.find(([label]) => label === want);
  const model = pick("model");
  const code = pick("candidate");

  const row = (label: string, meta: string, href?: string, onClick?: () => void) => {
    const live = Boolean(href || onClick);
    const inner = (
      <>
        <div style={{ fontSize: 13, color: live ? "var(--text)" : "var(--faint)" }}>{label}</div>
        <div className="mono" style={{ marginLeft: "auto", fontSize: 10.5, color: "var(--faint)" }}>
          {meta}
        </div>
      </>
    );
    const style = { display: "flex", alignItems: "center", gap: 12, padding: "12px 15px",
                    border: "1px solid var(--line)", borderRadius: 5, background: "var(--bg)",
                    cursor: live ? "pointer" : "default", textDecoration: "none" } as const;
    return href
      ? <a key={label} href={href} download style={style}>{inner}</a>
      : <div key={label} style={style} onClick={onClick}>{inner}</div>;
  };

  return (
    <div style={{ padding: "14px 22px 18px", display: "flex", flexDirection: "column", gap: 8 }}>
      {model && row("Download the model", `joblib · attempt ${selected?.best_attempt ?? 1}`,
                    fileUrl(ws, job.id, selected!.model, selected!.best_attempt ?? 1,
                            fileName(model[1])))}
      {code && row("Download the script", "candidate.py",
                   fileUrl(ws, job.id, selected!.model, selected!.best_attempt ?? 1,
                           fileName(code[1])))}
      {row("Open the run log", "events · attempts · code", undefined, onTraining)}
      {row("Deploy as an endpoint", "after the beta")}
      {row("Schedule batch scoring", "after the beta")}
      <div className="mono" style={{ fontSize: 10, color: "var(--faint)", marginTop: 4, lineHeight: 1.6 }}>
        served as bytes · the model file is a pickle, so load it only where you trust this run
        {job.files_expire_at
          ? ` · files kept until ${new Date(job.files_expire_at).toLocaleDateString("en-GB")}` : ""}
      </div>
    </div>
  );
}

/** Everything raised about the run, in one place (G40). */
function Checks({ report }: { report: Report }) {
  const { warnings, exclusions, leakage_screen: leaks, protocol } = report;
  const nothing = !warnings.length && !exclusions.length && !leaks.length && !protocol.split_warnings.length;
  return (
    <div style={{ padding: "14px 22px 18px", display: "flex", flexDirection: "column", gap: 12 }}>
      {nothing && (
        <div style={{ fontSize: 13, color: "var(--muted)" }}>
          No warnings, no excluded columns, and the leakage screen found nothing.
        </div>
      )}
      {warnings.map((warning, index) => (
        <div key={index} style={{ fontSize: 13, lineHeight: 1.55 }}>
          <span className="mono" style={{ fontSize: 11,
                                          color: warning.severity === "critical" ? "#C9673F" : "#C9873F" }}>
            {warning.model} · {warning.stage}
          </span>
          <div style={{ color: "var(--muted)", marginTop: 3 }}>{warning.message}</div>
        </div>
      ))}
      {protocol.split_warnings.map((warning) => (
        <div key={warning} style={{ fontSize: 13, lineHeight: 1.55, color: "var(--muted)" }}>
          <span className="mono" style={{ fontSize: 11, color: "#C9873F" }}>split</span>
          <div style={{ marginTop: 3 }}>{warning}</div>
        </div>
      ))}
      {leaks.map((leak) => (
        <div key={leak.columns.join(",")} style={{ fontSize: 13, lineHeight: 1.55, color: "var(--muted)" }}>
          <span className="mono" style={{ fontSize: 11, color: "#E0B679" }}>{leak.columns.join(", ")}</span>
          <div style={{ marginTop: 3 }}>
            {leak.formula} ({leak.measure} {leak.score.toFixed(4)})
            {leak.outcome ? ` → ${leak.outcome}` : ""}
          </div>
        </div>
      ))}
      {exclusions.map((exclusion, index) => (
        <div key={index} className="mono" style={{ fontSize: 12, color: "var(--dim)",
                                                   overflowWrap: "anywhere" }}>
          excluded: {JSON.stringify(exclusion).slice(0, 220)}
        </div>
      ))}
    </div>
  );
}

/** The report as a file, built in the browser from what the API already returned. */
function exportReport(ws: string, job: Job) {
  api<Report>(`/api/w/${encodeURIComponent(ws)}/jobs/${job.id}/report`).then((report) => {
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `run-${job.id}-report.json`;
    link.click();
    URL.revokeObjectURL(link.href);
  });
}

export function Results({ ws, job, onTraining }: { ws: string; job: Job; onTraining: () => void }) {
  const ready = ["succeeded", "stopped", "failed"].includes(job.status);
  const query = useReport(ws, job.id, ready);
  const [open, setOpen] = useState<string | null>(null);

  if (!ready) {
    return (
      <div style={{ padding: 40, color: "var(--muted)" }}>
        This run hasn't finished yet.{" "}
        <span className="mono" style={{ color: "var(--accent-bright)", cursor: "pointer" }}
              onClick={onTraining}>Watch it training</span>.
      </div>
    );
  }
  if (query.isPending) {
    return (
      <div style={{ padding: "36px 40px", display: "flex", flexDirection: "column", gap: 14 }}>
        <div className="skel" style={{ width: 300, height: 26 }} />
        <div className="skel" style={{ width: 460, height: 13 }} />
        <div className="skel" style={{ width: "100%", maxWidth: 1240, height: 130, marginTop: 8 }} />
        <div className="skel" style={{ width: "100%", maxWidth: 1240, height: 300 }} />
      </div>
    );
  }
  if (query.error || !query.data) {
    return <div style={{ padding: 40, color: "var(--muted)" }}>
      {(query.error as Error)?.message ?? "This run has no report."}
    </div>;
  }

  const report = query.data;
  const primary = report.metrics.find((metric) => metric.primary) ?? report.metrics[0];
  const selected = report.comparison.find((row) => row.selected);
  const regression = report.dataset.task_type === "regression";
  const gain = report.improvement_over_baseline;
  const spend = job.usage_totals?.cost_usd;
  const model = selected?.artifacts?.model;

  const facts = [
    `${report.totals.models_scored} of ${report.totals.models_planned} models`,
    `${report.totals.generations} attempts`,
    minutes(report.totals.wall_seconds),
    spend === undefined ? null : `$${spend.toFixed(4)}`,
    `held out ${count(report.dataset.test_rows)} rows`,
    gain === null ? null : `${(gain * 100).toFixed(1)}% over the baseline`,
  ].filter(Boolean).join(" · ");

  return (
    <div style={{ padding: "32px 40px 84px", maxWidth: 1340 }}>
      <div style={{ display: "flex", alignItems: "flex-end", justifyContent: "space-between", gap: 24,
                    flexWrap: "wrap", marginBottom: 18 }}>
        <div style={{ minWidth: 0 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap", marginBottom: 7 }}>
            <div style={{ fontSize: 23, letterSpacing: "-0.015em", color: "var(--bright)" }}>
              {job.name} · results
            </div>
            <div className="mono" style={{ fontSize: 10.5, letterSpacing: "0.06em", borderRadius: 3,
                                           padding: "3px 8px", border: "1px solid var(--edge)",
                                           color: job.status === "succeeded" ? "var(--accent-bright)"
                                                                             : "var(--dim)" }}>
              {job.status}
            </div>
          </div>
          <div className="mono" style={{ fontSize: 11, color: "var(--faint)" }}>{facts}</div>
        </div>
        <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
          <button className="btn ghost" onClick={() => exportReport(ws, job)}>Export report</button>
          {model && selected && (
            <a className="btn ghost" style={{ textDecoration: "none", padding: "7px 14px" }} download
               href={fileUrl(ws, job.id, selected.model, selected.best_attempt ?? 1, fileName(model))}>
              Download artifact
            </a>
          )}
          <button className="btn" disabled title="Serving a model arrives after the beta">
            Deploy endpoint
          </button>
        </div>
      </div>

      {report.status !== "complete" && (
        <div className="note" style={{ marginBottom: 16 }}>{report.status}</div>
      )}

      <Chosen report={report} selected={selected} primary={primary} />

      <div style={{ display: "grid", gridTemplateColumns: "minmax(0, 1.15fr) minmax(0, 1fr)", gap: 16,
                    alignItems: "stretch", marginBottom: 16 }}>
        <Card title="Model comparison"
              meta={<div className="mono" style={{ fontSize: 11, color: "var(--faint)" }}>
                {report.comparison.length} model{report.comparison.length === 1 ? "" : "s"} ·
                {" "}click a row for its files
              </div>}>
          <Comparison ws={ws} jobId={job.id} rows={report.comparison} metrics={report.metrics}
                      open={open} onOpen={setOpen} />
        </Card>
        <Card title="Feature importance"
              meta={<div className="mono" style={{ fontSize: 11, color: "var(--faint)" }}>
                {report.selected_model ?? "—"}
              </div>}>
          <Importance rows={report.importance} note="" />
        </Card>
      </div>

      <div style={{ display: "grid", gap: 16, alignItems: "stretch", marginBottom: 16,
                    gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))" }}>
        <Card title="Where the gain came from"
              meta={<div className="mono" style={{ fontSize: 11, color: "var(--faint)" }}>
                {primary?.name ?? ""} by generation
              </div>}>
          <Gain report={report} model={report.selected_model ?? undefined} metric={primary} />
        </Card>
        <Card title={regression ? "Where the error sits" : "Where it was right and wrong"}
              meta={<div className="mono" style={{ fontSize: 11, color: "var(--faint)" }}>
                held-out rows
              </div>}>
          {regression
            ? (report.error_bands.length
              ? <Bands bands={report.error_bands} />
              : <div style={{ padding: "26px 22px", color: "var(--faint)", fontSize: 13 }}>
                  No error bands were recorded.
                </div>)
            : (report.confusion.length
              ? <Confusion cells={report.confusion} />
              : <div style={{ padding: "26px 22px", color: "var(--faint)", fontSize: 13 }}>
                  No confusion matrix was recorded.
                </div>)}
        </Card>
        <Card title="Take it away">
          <Handoff ws={ws} job={job} selected={selected} onTraining={onTraining} />
        </Card>
      </div>

      <Card title="Checks"
            meta={<div className="mono" style={{ fontSize: 11, color: "var(--faint)" }}>
              read these before trusting the number
            </div>}>
        <Checks report={report} />
      </Card>
    </div>
  );
}
