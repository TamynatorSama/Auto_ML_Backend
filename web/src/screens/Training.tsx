import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";

import {
  attemptCode, fileUrl, resumeJob, running as approvedPlan, stopJob, TRAINING, useLive,
  type Attempt, type Job, type JobEvent, type JobModel, type Usage,
} from "../api";
import { count } from "../format";

/**
 * A run while it trains, laid out as the Console canvas lays it out: the stat
 * strip, the timeline, the leaderboard beside the gain chart and this job's
 * usage, the run log beside the queue, and the script of whichever attempt you
 * click (docs/PHASE5.md §7).
 *
 * The canvas counts passes, candidates, workers and cluster load. This engine has
 * none of those: it has model families, generations within each, attempts, and one
 * sandbox host. Every number below is the engine's own — nothing is invented to
 * fill a tile the canvas happens to have.
 *
 * One poll every two seconds carries the id of the last event seen, so a page
 * opened mid-run catches up and then follows (D6). Every gauge is derived —
 * job_models for what is running, the ledger for the rest (§8.2).
 */

const LANE_TONE: Record<string, string> = {
  running: "var(--accent-bright)", waiting: "var(--faint)", max_tries: "var(--accent)",
  no_improvement: "var(--accent)", failed: "#C9673F", unavailable: "var(--faint)",
};
const MARK: Record<string, string> = {
  running: "◆", waiting: "○", max_tries: "✓", no_improvement: "✓", failed: "✕", unavailable: "—",
};

/** What each event says, in a sentence. Anything unlisted prints its own kind and data. */
const SAYS: Record<string, (event: JobEvent) => string> = {
  stage: (event) => `${event.data.stage}`,
  plan_ready: () => "plan drafted, waiting for review",
  report_ready: () => "report written",
  model_waiting: (event) => `${event.model} is waiting for a sandbox`,
  model_started: (event) => `${event.model} started`,
  model_resumed: (event) => `${event.model} resumed`,
  model_finished: (event) => `${event.model} finished — ${event.data.status}`,
  generation_started: (event) => `${event.model} generation ${event.data.generation}`,
  attempt_finished: (event) =>
    `${event.model} attempt ${event.data.attempt} ${event.data.status}`
    + (event.data.wall_seconds ? ` in ${Number(event.data.wall_seconds).toFixed(1)}s` : ""),
  repair: (event) => `${event.model} repaired a script that wouldn't run`,
  judge_note: (event) => `${event.model}: ${String(event.data.note ?? event.data.notes ?? "")}`,
  lesson: (event) => `${event.model}: ${String(event.data.lesson ?? "")}`,
  sandbox_waiting: (event) => `waiting for the host — ${event.data.problem}`,
};
const LEVEL: Record<string, string> = {
  sandbox_waiting: "warn", repair: "warn", model_finished: "metric", attempt_finished: "metric",
  report_ready: "metric", plan_ready: "metric",
};
const LEVEL_TONE: Record<string, string> = { warn: "#C9873F", metric: "var(--accent-bright)" };
const QUIET = new Set(["stage", "generation_started"]);

function say(event: JobEvent): string {
  const line = SAYS[event.kind];
  return line ? line(event) : `${event.kind} ${JSON.stringify(event.data).slice(0, 120)}`;
}

function num(value: number | null | undefined, places = 4): string {
  return value === null || value === undefined || Number.isNaN(value) ? "—" : Number(value).toFixed(places);
}

function clock(seconds: number): string {
  const whole = Math.max(Math.floor(seconds), 0);
  const parts = [Math.floor(whole / 3600), Math.floor((whole % 3600) / 60), whole % 60];
  return (parts[0] ? [parts[0], parts[1], parts[2]] : [parts[1], parts[2]])
    .map((part, index) => (index ? String(part).padStart(2, "0") : String(part))).join(":");
}

function Card({ title, meta, children, grow }: {
  title: string; meta?: ReactNode; children: ReactNode; grow?: boolean;
}) {
  return (
    <div style={{ border: "1px solid var(--line)", borderRadius: 6, background: "var(--card)",
                  overflow: "hidden", display: "flex", flexDirection: "column",
                  ...(grow ? { flex: 1, minHeight: 0 } : {}) }}>
      <div className="chead">
        <div className="label" style={{ color: "var(--muted)" }}>{title}</div>
        {meta}
      </div>
      {children}
    </div>
  );
}

/** The canvas's five tiles across the top. Every one is a number this engine keeps. */
function Stats({ tiles }: { tiles: { k: string; v: string; sub: string; fg?: string }[] }) {
  return (
    <div style={{ display: "grid", gap: 12, marginBottom: 16,
                  gridTemplateColumns: "repeat(auto-fit, minmax(170px, 1fr))" }}>
      {tiles.map((tile) => (
        <div key={tile.k} style={{ border: "1px solid var(--line)", borderRadius: 6,
                                   background: "var(--card)", padding: "14px 16px" }}>
          <div className="label" style={{ color: "var(--faint)" }}>{tile.k}</div>
          <div className="mono" style={{ fontSize: 21, marginTop: 8, color: tile.fg ?? "var(--text)" }}>
            {tile.v}
          </div>
          <div className="mono" style={{ fontSize: 10.5, color: "var(--faint)", marginTop: 5 }}>{tile.sub}</div>
        </div>
      ))}
    </div>
  );
}

/** One model family's progress: the canvas's pass timeline, over what this engine actually has. */
function Timeline({ models, metric, tries, best }: {
  models: JobModel[]; metric: string; tries: number; best: (row: JobModel) => number | undefined;
}) {
  return (
    <div style={{ padding: "6px 0", overflowX: "auto" }}>
      {models.map((row) => {
        const done = row.status === "running" ? Math.max((row.generation ?? 1) - 1, 0)
          : row.status === "waiting" ? 0 : tries;
        const score = best(row);
        return (
          <div key={row.model} style={{ display: "grid", alignItems: "center", gap: 14, minWidth: 460,
                                        gridTemplateColumns: "76px minmax(110px, 1.2fr) minmax(90px, 1.6fr) 104px",
                                        padding: "11px 22px", borderBottom: "1px solid var(--line-soft)" }}>
            <div style={{ display: "flex", alignItems: "center", gap: 7 }}>
              <div className="mono" style={{ fontSize: 11, color: LANE_TONE[row.status] ?? "var(--faint)" }}>
                {MARK[row.status] ?? "○"}
              </div>
              <div className="mono" style={{ fontSize: 10.5, color: LANE_TONE[row.status] ?? "var(--faint)" }}>
                {row.status === "no_improvement" ? "settled" : row.status.replace(/_/g, " ")}
              </div>
            </div>
            <div style={{ fontSize: 13.5, color: "var(--text)", minWidth: 0, overflow: "hidden",
                          textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
              {row.model}
            </div>
            <div style={{ height: 5, borderRadius: 3, background: "#1B2220", overflow: "hidden" }}>
              <div style={{ height: "100%", width: `${tries ? (done / tries) * 100 : 0}%`,
                            background: row.status === "running" ? "var(--accent-bright)" : "#2E5C51" }} />
            </div>
            <div className="mono" style={{ fontSize: 12, textAlign: "right",
                                           color: score === undefined ? "var(--faint)" : "var(--text)" }}>
              {score === undefined ? "—" : num(score)}
              <div style={{ fontSize: 10, color: "var(--faint)", marginTop: 3 }}>
                {row.status === "waiting" ? "not started" : `gen ${row.generation ?? 1} of ${tries}`}
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

/** Every attempt, ranked — the canvas's live leaderboard, whose candidates are our attempts. */
function Leaderboard({ rows, metric, ranked, onPick }: {
  rows: (Attempt & { score?: number })[];
  metric: string;
  /** false when the plan carries no direction for the metric: listed, not ranked */
  ranked: boolean;
  onPick: (model: string, attempt: number) => void;
}) {
  const grid = "30px minmax(120px, 1.5fr) 48px minmax(74px, 1fr) 72px";
  return (
    <div style={{ overflowX: "auto" }}>
      <div style={{ minWidth: 420 }}>
        <div className="srow head" style={{ gridTemplateColumns: grid, minWidth: 0, padding: "10px 22px" }}>
          <div>{ranked ? "#" : ""}</div><div>attempt</div><div>gen</div>
          <div style={{ textAlign: "right" }}>{metric || "score"}</div>
          <div style={{ textAlign: "right" }}>time</div>
        </div>
        {rows.length ? rows.map((row, index) => (
          <div key={row.id} className="srow" style={{ gridTemplateColumns: grid, minWidth: 0,
                                                      padding: "10px 22px", cursor: "pointer",
                                                      background: ranked && index === 0 ? "#0F1D19"
                                                        : undefined }}
               onClick={() => onPick(row.model, row.attempt)}>
            <div className="mono" style={{ fontSize: 11,
                                           color: ranked && index === 0 ? "var(--accent-bright)"
                                                                        : "var(--faint)" }}>
              {!ranked ? "·" : row.score === undefined ? "·" : index + 1}
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 8, minWidth: 0 }}>
              <div style={{ width: 5, height: 5, borderRadius: "50%", flex: "0 0 5px",
                            background: row.status === "ok"
                              ? (ranked && index === 0 ? "var(--accent-bright)" : "#3D4744")
                              : "#C9673F" }} />
              <div className="mono" style={{ fontSize: 12, color: "var(--muted)", overflow: "hidden",
                                             textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                {row.model} · {row.kind === "repair" ? "fix " : ""}{row.attempt}
              </div>
            </div>
            <div className="mono" style={{ fontSize: 11, color: "var(--faint)" }}>{row.generation ?? "—"}</div>
            <div className="mono" style={{ fontSize: 12, textAlign: "right",
                                           color: ranked && index === 0 ? "var(--accent-bright)"
                                                                       : "var(--text)" }}>
              {row.score === undefined ? "—" : num(row.score)}
            </div>
            <div className="mono" style={{ fontSize: 11, textAlign: "right", color: "var(--faint)" }}>
              {row.wall_seconds ? `${row.wall_seconds.toFixed(0)}s` : "—"}
            </div>
          </div>
        )) : (
          <div style={{ padding: "24px 22px", color: "var(--faint)", fontSize: 13 }}>
            No attempt has finished yet.
          </div>
        )}
      </div>
    </div>
  );
}

/**
 * One line per model family, tracking its best score at each generation — the
 * canvas's "best roc auc by pass", over what this engine actually has.
 *
 * The hues are the eight validated categorical slots, assigned in fixed order and
 * never cycled; a ninth family would need folding, not a ninth hue. Identity is
 * never colour alone: every line is labelled at its end and in the legend, which
 * is also what makes the 6–8 colour-vision band legal.
 *
 * The axis carries the metric's own direction. For a lower-is-better metric an
 * improving line descends, which is how a loss curve reads — the note says which
 * way is better so nothing rests on the reader's assumption.
 */
const SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500",
                "#d55181", "#008300", "#9085e9", "#e66767"];
const MAX_SERIES = SERIES.length;

function Gain({ series, metric, higher }: {
  series: { model: string; points: { generation: number; score: number }[] }[];
  metric: string;
  higher: boolean;
}) {
  const drawn = series.filter((line) => line.points.length > 0).slice(0, MAX_SERIES);
  const hidden = series.length - drawn.length;
  const every = drawn.flatMap((line) => line.points);

  if (!metric) {
    return (
      <div style={{ padding: "26px 22px", fontSize: 12.5, color: "var(--faint)", lineHeight: 1.6 }}>
        This plan predates the metric's direction, so nothing here is ranked. Draft again.
      </div>
    );
  }
  if (every.length < 2) {
    return (
      <div style={{ padding: "26px 22px", fontSize: 12.5, color: "var(--faint)", lineHeight: 1.6 }}>
        A second generation will draw the gain here.
      </div>
    );
  }

  const width = 340;
  const height = 152;
  const padLeft = 40;
  const padRight = 54;
  const padTop = 10;
  const padBottom = 20;
  const plotW = width - padLeft - padRight;
  const plotH = height - padTop - padBottom;

  const lastGeneration = Math.max(...every.map((point) => point.generation));
  const firstGeneration = Math.min(...every.map((point) => point.generation));
  const scores = every.map((point) => point.score);
  const high = Math.max(...scores);
  const low = Math.min(...scores);
  const pad = (high - low) * 0.12 || Math.abs(high) * 0.05 || 1;
  const ceiling = high + pad;
  const floor = low - pad;

  const x = (generation: number) => padLeft + (lastGeneration === firstGeneration ? plotW / 2
    : ((generation - firstGeneration) / (lastGeneration - firstGeneration)) * plotW);
  const y = (score: number) => padTop + (1 - (score - floor) / (ceiling - floor)) * plotH;
  const tick = (value: number) => (Math.abs(value) >= 100 ? value.toFixed(0)
    : Math.abs(value) >= 1 ? value.toFixed(2) : value.toFixed(3));

  const generations = Array.from({ length: lastGeneration - firstGeneration + 1 },
                                 (unused, index) => firstGeneration + index);

  return (
    <div style={{ padding: "12px 14px 10px" }}>
      <div style={{ display: "flex", gap: 10, flexWrap: "wrap", padding: "0 6px 8px" }}>
        {drawn.map((line, index) => (
          <div key={line.model} style={{ display: "flex", alignItems: "center", gap: 5 }}>
            <div style={{ width: 8, height: 8, borderRadius: 2, flex: "0 0 8px",
                          background: SERIES[index] }} />
            <div className="mono" style={{ fontSize: 10, color: "var(--muted)" }}>{line.model}</div>
          </div>
        ))}
        {hidden > 0 && (
          <div className="mono" style={{ fontSize: 10, color: "var(--faint)" }}>+{hidden} more</div>
        )}
      </div>

      <svg viewBox={`0 0 ${width} ${height}`} role="img"
           aria-label={`best ${metric} by generation, one line per model family`}
           style={{ width: "100%", height: "auto", display: "block" }}>
        {[0, 0.5, 1].map((at) => (
          <line key={at} x1={padLeft} x2={padLeft + plotW} y1={padTop + at * plotH} y2={padTop + at * plotH}
                stroke="#1E2523" strokeWidth={1} />
        ))}
        {[ceiling, (ceiling + floor) / 2, floor].map((value, index) => (
          <text key={value} x={padLeft - 6} y={padTop + (index / 2) * plotH + 3} textAnchor="end"
                fill="#4E5A57" fontSize={8} fontFamily="var(--mono)">
            {tick(value)}
          </text>
        ))}
        {generations.map((generation) => (
          <text key={generation} x={x(generation)} y={height - 6} textAnchor="middle"
                fill="#4E5A57" fontSize={8} fontFamily="var(--mono)">
            gen {generation}
          </text>
        ))}

        {drawn.map((line, index) => {
          const colour = SERIES[index];
          const ordered = [...line.points].sort((a, b) => a.generation - b.generation);
          const path = ordered.map((point, at) =>
            `${at ? "L" : "M"}${x(point.generation).toFixed(1)} ${y(point.score).toFixed(1)}`).join(" ");
          const last = ordered[ordered.length - 1];
          return (
            <g key={line.model}>
              <path d={path} fill="none" stroke={colour} strokeWidth={2}
                    strokeLinejoin="round" strokeLinecap="round" />
              {ordered.map((point) => (
                <circle key={point.generation} cx={x(point.generation)} cy={y(point.score)} r={3}
                        fill={colour} stroke="var(--card)" strokeWidth={1.5}>
                  <title>{`${line.model} · generation ${point.generation} · ${metric} ${point.score}`}</title>
                </circle>
              ))}
              <text x={x(last.generation) + 8} y={y(last.score) + 3} fill={colour} fontSize={8.5}
                    fontFamily="var(--mono)">
                {line.model.length > 8 ? `${line.model.slice(0, 7)}…` : line.model}
              </text>
            </g>
          );
        })}
      </svg>

      <div className="mono" style={{ fontSize: 10, color: "var(--faint)", padding: "6px 6px 0",
                                     lineHeight: 1.6 }}>
        {metric} {higher ? "↑ higher is better" : "↓ lower is better"}
      </div>
    </div>
  );
}

/** The ledger's totals for this job. */
function Meters({ usage, cap }: { usage: Usage; cap: number | null }) {
  const meter = usage.meters;
  const spend = meter.cost_usd ?? 0;
  const rows: [string, string][] = [
    ["models training", String(usage.running)],
    ["tokens", count(Math.round((meter.input_tokens ?? 0) + (meter.output_tokens ?? 0)))],
    ["cpu time", meter.cpu_seconds ? `${(meter.cpu_seconds / 60).toFixed(1)} min` : "0"],
    ["reserved memory", meter.reserved_gb_seconds
      ? `${(meter.reserved_gb_seconds / 3600).toFixed(2)} GB-h` : "0"],
  ];
  return (
    <div style={{ padding: "10px 20px 16px" }}>
      {rows.map(([label, value]) => (
        <div key={label} style={{ display: "flex", justifyContent: "space-between", gap: 12,
                                  padding: "8px 0", borderBottom: "1px solid var(--line-soft)" }}>
          <div className="mono" style={{ fontSize: 11.5, color: "var(--faint)" }}>{label}</div>
          <div className="mono" style={{ fontSize: 12, color: "var(--text)" }}>{value}</div>
        </div>
      ))}
      {cap !== null && (
        <div style={{ marginTop: 12 }}>
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 7 }}>
            <div className="mono" style={{ fontSize: 11.5, color: "var(--faint)" }}>spend</div>
            <div className="mono" style={{ fontSize: 12, color: "var(--text)" }}>
              ${spend.toFixed(4)} of ${cap.toFixed(2)}
            </div>
          </div>
          <div style={{ height: 4, borderRadius: 2, background: "#1B2220", overflow: "hidden" }}>
            <div style={{ height: "100%", width: `${Math.min(100, (spend / cap) * 100)}%`,
                          background: spend >= cap ? "#C9673F" : "var(--accent)" }} />
          </div>
          <div className="mono" style={{ fontSize: 10, color: "var(--faint)", marginTop: 7, lineHeight: 1.6 }}>
            only model spend counts against the cap · sandbox time is metered, not capped
          </div>
        </div>
      )}
    </div>
  );
}

/** Longest common subsequence over lines, so a diff shows what actually moved. */
function diffLines(before: string[], after: string[]) {
  const rows = before.length;
  const columns = after.length;
  const table: number[][] = Array.from({ length: rows + 1 }, () => new Array(columns + 1).fill(0));
  for (let i = rows - 1; i >= 0; i -= 1) {
    for (let j = columns - 1; j >= 0; j -= 1) {
      table[i][j] = before[i] === after[j]
        ? table[i + 1][j + 1] + 1 : Math.max(table[i + 1][j], table[i][j + 1]);
    }
  }
  const out: { sign: " " | "-" | "+"; text: string; n: number | null }[] = [];
  let i = 0;
  let j = 0;
  while (i < rows && j < columns) {
    if (before[i] === after[j]) { out.push({ sign: " ", text: after[j], n: j + 1 }); i += 1; j += 1; }
    else if (table[i + 1][j] >= table[i][j + 1]) { out.push({ sign: "-", text: before[i], n: null }); i += 1; }
    else { out.push({ sign: "+", text: after[j], n: j + 1 }); j += 1; }
  }
  while (i < rows) { out.push({ sign: "-", text: before[i], n: null }); i += 1; }
  while (j < columns) { out.push({ sign: "+", text: after[j], n: j + 1 }); j += 1; }
  return out;
}

type DiffLine = { sign: " " | "-" | "+"; text: string; n: number | null };

/** Only the changed stretches, with a few lines either side, the way a diff reads. */
function hunks(lines: DiffLine[], context = 3): (DiffLine | null)[] {
  const keep = new Set<number>();
  lines.forEach((line, index) => {
    if (line.sign === " ") return;
    for (let at = index - context; at <= index + context; at += 1) {
      if (at >= 0 && at < lines.length) keep.add(at);
    }
  });
  const out: (DiffLine | null)[] = [];
  let gap = false;
  lines.forEach((line, index) => {
    if (keep.has(index)) { out.push(line); gap = false; }
    else if (!gap) { out.push(null); gap = true; }        // null renders as the skipped marker
  });
  return out;
}

const SIGN_TONE: Record<string, [string, string]> = {
  "+": ["#9FE6D4", "rgba(18,135,111,0.14)"],
  "-": ["#E09B86", "rgba(201,103,63,0.12)"],
  " ": ["#7C8985", "transparent"],
};

/**
 * The canvas's Pipeline code block: every attempt of one model down the left, and
 * on the right what that attempt changed from the one before it. The scripts are
 * the real ones the sandbox ran, fetched as text and diffed here — the pipeline
 * keeps no diff of its own, so this is computed, with the attempt's own `changes`
 * note beside it as the author's account of the same edit.
 */
function PipelineCode({ ws, job, pick, attempts, onPick }: {
  ws: string;
  job: number;
  pick: { model: string; attempt: number };
  attempts: Attempt[];
  onPick: (model: string, attempt: number) => void;
}) {
  const [text, setText] = useState("");
  const [was, setWas] = useState<string | null>(null);
  const [state, setState] = useState<"loading" | "ready" | "gone">("loading");
  const [whole, setWhole] = useState(false);

  const mine = attempts.filter((attempt) => attempt.model === pick.model);
  const index = mine.findIndex((attempt) => attempt.attempt === pick.attempt);
  const previous = index > 0 ? mine[index - 1] : null;

  useEffect(() => {
    let alive = true;
    setState("loading");
    Promise.all([
      attemptCode(ws, job, pick.model, pick.attempt),
      previous ? attemptCode(ws, job, pick.model, previous.attempt).catch(() => null)
               : Promise.resolve(null),
    ]).then(([now, before]) => {
      if (!alive) return;
      setText(now);
      setWas(before);
      setState("ready");
    }).catch(() => alive && setState("gone"));
    return () => { alive = false; };
  }, [ws, job, pick.model, pick.attempt, previous?.attempt]);

  const lines = text ? text.replace(/\n+$/, "").split("\n") : [];
  const diff = was === null ? null : diffLines(was.replace(/\n+$/, "").split("\n"), lines);
  const added = diff ? diff.filter((line) => line.sign === "+").length : 0;
  const removed = diff ? diff.filter((line) => line.sign === "-").length : 0;
  const showing: (DiffLine | null)[] = !whole && diff
    ? hunks(diff)
    : lines.map((line, at) => ({ sign: " " as const, text: line, n: at + 1 }));

  return (
    <Card title="Pipeline code"
          meta={
            <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
              {diff && (
                <div className="mono" style={{ fontSize: 11 }}>
                  <span style={{ color: "#9FE6D4" }}>+{added}</span>{" "}
                  <span style={{ color: "#E09B86" }}>&minus;{removed}</span>
                </div>
              )}
              {diff && (
                <button className="pager" onClick={() => setWhole(!whole)}>
                  {whole ? "just the changes" : "whole file"}
                </button>
              )}
              <a className="pager" style={{ textDecoration: "none" }} download
                 href={fileUrl(ws, job, pick.model, pick.attempt, "candidate.py")}>
                export file
              </a>
            </div>
          }>
      <div className="script-layout" style={{ display: "grid", gridTemplateColumns: "minmax(190px, 250px) minmax(0, 1fr)" }}>
        <div style={{ borderRight: "1px solid var(--line)", height: 320, overflowY: "auto",
                      overflowX: "hidden" }}>
          {mine.map((attempt) => {
            const on = attempt.attempt === pick.attempt;
            return (
              <div key={attempt.id} onClick={() => onPick(attempt.model, attempt.attempt)}
                   style={{ padding: "11px 16px", cursor: "pointer",
                            borderBottom: "1px solid var(--line-soft)",
                            background: on ? "#131817" : undefined,
                            borderLeft: `2px solid ${on ? "var(--accent)" : "transparent"}` }}>
                <div style={{ display: "flex", justifyContent: "space-between", gap: 8 }}>
                  <div className="mono" style={{ fontSize: 11,
                                                 color: attempt.status === "ok" ? "var(--muted)" : "#C9673F" }}>
                    {attempt.kind === "repair" ? "fix" : "gen"} {attempt.attempt}
                  </div>
                  <div className="mono" style={{ fontSize: 10.5, color: "var(--faint)" }}>
                    {new Date(attempt.at).toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit" })}
                  </div>
                </div>
                <div style={{ fontSize: 12, color: "var(--dim)", marginTop: 4, lineHeight: 1.5,
                              overflowWrap: "anywhere", wordBreak: "break-word" }}>
                  {attempt.changes || "no note"}
                </div>
              </div>
            );
          })}
        </div>
        {/* a fixed height, so a revision that is longer or shorter never moves the page */}
        <div style={{ minWidth: 0, height: 320, overflow: "auto",
                      opacity: state === "loading" ? 0.45 : 1, transition: "opacity 120ms" }}>
          <div style={{ padding: "9px 18px", borderBottom: "1px solid var(--line-soft)",
                        position: "sticky", top: 0, background: "var(--card)" }}>
            <div className="mono" style={{ fontSize: 11, color: "var(--faint)" }}>
              candidate.py · {pick.model} ·{" "}
              {previous && !whole ? `attempt ${pick.attempt} vs ${previous.attempt}`
                                  : `attempt ${pick.attempt}`}
            </div>
          </div>
          {state === "loading" && !text && (
            <div style={{ padding: 18, display: "flex", flexDirection: "column", gap: 8 }}>
              {[0, 1, 2, 3, 4, 5].map((row) => (
                <div className="skel" key={row} style={{ height: 11, width: `${90 - row * 9}%` }} />
              ))}
            </div>
          )}
          {state === "gone" && (
            <div style={{ padding: "20px 18px", fontSize: 13, color: "var(--faint)", lineHeight: 1.6 }}>
              This script isn't here any more — a finished run's files are kept for seven days.
            </div>
          )}
          {(state === "ready" || (state === "loading" && text)) && (
            <pre className="mono" style={{ margin: 0, padding: "10px 0 16px", fontSize: 11.5,
                                           lineHeight: 1.7 }}>
              {showing.map((line, at) => {
                if (line === null) {
                  return <div key={`gap-${at}`} style={{ padding: "4px 18px", color: "#3D4744" }}>&#8943;</div>;
                }
                const tone = SIGN_TONE[line.sign];
                return (
                  <div key={at} style={{ display: "grid", gridTemplateColumns: "44px 14px minmax(0, 1fr)",
                                         background: tone[1], padding: "0 18px" }}>
                    <span style={{ color: "#3D4744", textAlign: "right", paddingRight: 12 }}>
                      {line.n ?? ""}
                    </span>
                    <span style={{ color: tone[0] }}>{line.sign === " " ? "" : line.sign}</span>
                    <span style={{ color: tone[0], whiteSpace: "pre-wrap", overflowWrap: "anywhere" }}>
                      {line.text || " "}
                    </span>
                  </div>
                );
              })}
            </pre>
          )}
        </div>
      </div>
    </Card>
  );
}

export function Training({ ws, job, onResults }: { ws: string; job: Job; onResults: () => void }) {
  const queries = useQueryClient();
  const [cursor, setCursor] = useState(0);
  const [log, setLog] = useState<JobEvent[]>([]);
  const [pick, setPick] = useState<{ model: string; attempt: number } | null>(null);
  const [problem, setProblem] = useState("");
  const [quiet, setQuiet] = useState(true);
  const [now, setNow] = useState(() => Date.now());
  const bottom = useRef<HTMLDivElement>(null);

  // the job comes from useJob, which polls while the worker may still be moving it;
  // /live is for the usage, the models, the attempts and the events. An earlier cut
  // preferred /live's copy of the job and got stuck twice over: it has no plan, and
  // once polling stopped its stale status outvoted the fresh one, so a resumed run
  // stayed "stopped" on screen while the worker trained.
  const running = TRAINING.includes(job.status);
  const live = useLive(ws, job.id, cursor, running);

  useEffect(() => {
    const page = live.data;
    if (!page) return;
    if (page.events.length) setLog((seen) => [...seen, ...page.events]);
    if (page.cursor > cursor) setCursor(page.cursor);
  }, [live.data?.cursor, live.data?.events.length]);

  useEffect(() => { bottom.current?.scrollIntoView({ block: "nearest" }); }, [log.length]);
  // the rail and the step bar read the listing, which has no poll of its own
  useEffect(() => { queries.invalidateQueries({ queryKey: ["jobs", ws] }); }, [job.status]);
  useEffect(() => {
    if (!running) return;
    const tick = setInterval(() => setNow(Date.now()), 1000);   // the elapsed tile
    return () => clearInterval(tick);
  }, [running]);

  const refresh = () => {
    queries.invalidateQueries({ queryKey: ["job", ws, job.id] });
    queries.invalidateQueries({ queryKey: ["jobs", ws] });
  };
  const fail = (error: Error) => setProblem(error.message);
  const stop = useMutation({ mutationFn: () => stopJob(ws, job.id),
                             onSuccess: () => { setProblem(""); refresh(); }, onError: fail });
  const again = useMutation({ mutationFn: () => resumeJob(ws, job.id),
                              onSuccess: () => { setProblem(""); refresh(); }, onError: fail });

  const page = live.data;
  const models = page?.models ?? [];
  // a resumed run records an attempt it had already done a second time — random_forest
  // came back with six rows for three attempts — so each is shown once, as last written
  const attempts = useMemo(() => {
    const latest = new Map<string, Attempt>();
    for (const attempt of page?.attempts ?? []) {
      latest.set(`${attempt.model}#${attempt.kind}#${attempt.attempt}`, attempt);
    }
    return [...latest.values()].sort((a, b) => a.id - b.id);
  }, [page?.attempts]);
  // what the run is actually training against, not the draft it was offered
  const config = approvedPlan(job)?.config;
  const metric = config?.improvement_metric ?? config?.eval_matrics?.[0] ?? "";
  const tries = config?.max_tries ?? 0;
  const planned = (config?.models.length ?? models.length) * tries;
  // which way is better comes from the pipeline's own table, never from a list here.
  // A plan drafted before that table was carried has no answer, and guessing it
  // would rank a lower-is-better metric backwards — so nothing is ranked at all.
  const direction = job.plan?.choices?.directions?.[metric];
  const higher = direction === "higher";
  const ranked = useMemo(() => {
    const scored = attempts.map((attempt) => ({ ...attempt, score: attempt.cv_scores?.[metric] }));
    if (!direction) return scored;
    return scored.sort((a, b) => {
      if (a.score === undefined) return b.score === undefined ? a.id - b.id : 1;
      if (b.score === undefined) return -1;
      return higher ? b.score - a.score : a.score - b.score;
    });
  }, [attempts, metric, direction, higher]);

  const series = useMemo(() => {
    if (!direction) return [];
    const byModel = new Map<string, Map<number, number>>();
    for (const attempt of attempts) {
      const score = attempt.cv_scores?.[metric];
      if (score === undefined || attempt.generation === null) continue;
      if (!byModel.has(attempt.model)) byModel.set(attempt.model, new Map());
      const points = byModel.get(attempt.model)!;
      const seen = points.get(attempt.generation);
      if (seen === undefined || (higher ? score > seen : score < seen)) {
        points.set(attempt.generation, score);
      }
    }
    // the order the plan lists them, so a colour follows the family and not its rank
    const order = config?.models ?? [...byModel.keys()];
    return order.filter((model) => byModel.has(model)).map((model) => ({
      model,
      points: [...byModel.get(model)!.entries()].sort((a, b) => a[0] - b[0])
        .map(([generation, score]) => ({ generation, score })),
    }));
  }, [attempts, metric, direction, higher, config?.models]);

  const bestOf = (row: JobModel) => row.best_cv?.[metric];
  const leader = direction ? ranked.find((row) => row.score !== undefined) : undefined;
  const failures = attempts.filter((attempt) => attempt.status !== "ok").length;
  const generations = attempts.filter((attempt) => attempt.kind !== "repair").length;
  // a finished run's clock stops at finished_at; only a live one counts up to now
  const until = job.finished_at ? new Date(job.finished_at).getTime() : now;
  const elapsed = job.started_at ? (until - new Date(job.started_at).getTime()) / 1000 : 0;
  const spend = page?.usage.meters.cost_usd ?? 0;
  const waiting = models.filter((row) => row.status === "waiting");
  const ended = ["succeeded", "stopped", "failed"].includes(job.status);
  // the code card is always open, on whatever you last clicked, else the best attempt
  const shownPick = pick ?? (leader ? { model: leader.model, attempt: leader.attempt }
    : attempts.length ? { model: attempts[0].model, attempt: attempts[0].attempt } : null);
  const shown = quiet ? log.filter((event) => !QUIET.has(event.kind)) : log;

  const tiles = [
    { k: `best ${metric || "score"}`, v: leader?.score === undefined ? "—" : num(leader.score),
      sub: !direction ? "this plan predates the metric's direction"
        : leader ? `${leader.model} · attempt ${leader.attempt}` : "nothing scored yet",
      fg: "var(--accent-bright)" },
    { k: "attempts", v: planned ? `${generations} / ${planned}` : String(generations),
      sub: `${page?.usage.running ?? 0} model${(page?.usage.running ?? 0) === 1 ? "" : "s"} going · ${
        waiting.length} waiting` },
    { k: "elapsed", v: job.started_at ? clock(elapsed) : "—",
      sub: job.deadline_seconds ? `cap ${clock(job.deadline_seconds)}` : "no time cap" },
    { k: "spend", v: `$${spend.toFixed(4)}`,
      sub: job.budget_usd ? `cap $${job.budget_usd.toFixed(2)}` : "no cap" },
    { k: "failures", v: String(failures),
      sub: failures ? "scripts that wouldn't run" : "none", fg: failures ? "#C9873F" : undefined },
  ];

  return (
    <div className="page-body" style={{ padding: "32px 40px 84px", maxWidth: 1340 }}>
      <div style={{ display: "flex", alignItems: "flex-end", justifyContent: "space-between", gap: 24,
                    flexWrap: "wrap", marginBottom: 20 }}>
        <div style={{ minWidth: 0 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap", marginBottom: 7 }}>
            {running && <div className="pulse" />}
            <div className="responsive-title" style={{ fontSize: 23, letterSpacing: "-0.015em", color: "var(--bright)" }}>{job.name}</div>
            <div className="mono" style={{ fontSize: 10.5, letterSpacing: "0.06em", borderRadius: 3,
                                           padding: "3px 8px", border: "1px solid var(--edge)",
                                           color: running ? "var(--accent-bright)" : "var(--dim)" }}>
              {job.status}
            </div>
            {job.status === "queued" && job.queue_position !== null && (
              <div className="mono" style={{ fontSize: 11, color: "#C9873F" }}>
                {job.queue_position === 0 ? "next to run" : `${job.queue_position} ahead in the queue`}
              </div>
            )}
          </div>
          <div className="mono" style={{ fontSize: 11, color: "var(--faint)" }}>
            {job.plan?.task_type ?? "—"} · target {job.plan?.target ?? "—"} · best by {metric || "—"} ·
            {" "}schema pinned · {job.started_at
              ? `started ${new Date(job.started_at).toLocaleTimeString("en-GB")}` : "not started"}
          </div>
        </div>
        <div style={{ display: "flex", gap: 10 }}>
          {!ended && (
            <button className="btn ghost" disabled={stop.isPending || job.status === "stopping"}
                    onClick={() => stop.mutate()}>
              {stop.isPending ? "Stopping…" : job.status === "stopping" ? "Stopping…" : "Stop job"}
            </button>
          )}
          {ended && job.status !== "succeeded" && !job.files_expired && (
            <button className="btn ghost" disabled={again.isPending} onClick={() => again.mutate()}>
              {again.isPending ? "Resuming…" : "Resume"}
            </button>
          )}
          <button className="btn ghost" onClick={onResults}>
            {ended ? "Results" : "Results so far"}
          </button>
        </div>
      </div>

      {problem && <div className="note" style={{ marginBottom: 16 }}>{problem}</div>}
      {job.error && <div className="note" style={{ marginBottom: 16 }}>{job.error}</div>}

      <Stats tiles={tiles} />

      <div style={{ marginBottom: 16 }}>
        <Card title="Model timeline"
              meta={<div className="mono" style={{ fontSize: 11, color: "var(--faint)" }}>
                {job.started_at ? `elapsed ${clock(elapsed)}` : "not started"} · up to {tries} tries each
              </div>}>
          {models.length
            ? <Timeline models={models} metric={metric} tries={tries} best={bestOf} />
            : <div style={{ padding: "26px 22px", color: "var(--faint)", fontSize: 13 }}>
                No model has started yet.
              </div>}
        </Card>
      </div>

      <div className="training-layout" style={{ display: "grid", gridTemplateColumns: "minmax(0, 1fr) 380px", gap: 16,
                    alignItems: "stretch", marginBottom: 16 }}>
        <Card title="Live leaderboard"
              meta={<div className="mono" style={{ fontSize: 11, color: "var(--faint)" }}>
                every attempt · click one for its script
              </div>}>
          <Leaderboard rows={ranked} metric={metric} ranked={Boolean(direction)}
                       onPick={(model, attempt) => setPick({ model, attempt })} />
        </Card>
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <Card title={`Best ${metric || "score"} by generation`}>
            <Gain series={series} metric={metric} higher={higher} />
          </Card>
          <Card title="This job">
            <Meters usage={page?.usage ?? { running: 0, meters: {} }} cap={job.budget_usd} />
          </Card>
        </div>
      </div>

      <div className="training-layout" style={{ display: "grid", gridTemplateColumns: "minmax(0, 1fr) 380px", gap: 16,
                    alignItems: "stretch", marginBottom: 16 }}>
        <Card title="Run log"
              meta={<button className="pager" onClick={() => setQuiet(!quiet)}>
                {quiet ? "all levels" : "fewer"}
              </button>}>
          <div style={{ maxHeight: 300, overflowY: "auto", padding: "10px 22px" }}>
            {shown.length ? shown.map((event) => {
              const level = LEVEL[event.kind] ?? "info";
              return (
                <div key={event.id} style={{ display: "grid", gap: 12, padding: "5px 0",
                                             gridTemplateColumns: "62px 46px minmax(0, 1fr)",
                                             fontSize: 12.5 }}>
                  <div className="mono" style={{ fontSize: 11, color: "var(--faint)" }}>
                    {new Date(event.at).toLocaleTimeString("en-GB")}
                  </div>
                  <div className="mono" style={{ fontSize: 10.5,
                                                 color: LEVEL_TONE[level] ?? "var(--faint)" }}>
                    {level}
                  </div>
                  <div style={{ color: "var(--muted)", minWidth: 0 }}>{say(event)}</div>
                </div>
              );
            }) : <div style={{ padding: "16px 0", color: "var(--faint)", fontSize: 13 }}>Nothing yet.</div>}
            <div ref={bottom} />
          </div>
        </Card>

        <Card title="Queue"
              meta={<div className="mono" style={{ fontSize: 11, color: "var(--faint)" }}>
                {waiting.length} waiting · {page?.usage.running ?? 0} running
              </div>}>
          <div style={{ maxHeight: 300, overflowY: "auto", padding: "8px 0" }}>
            {models.filter((row) => ["running", "waiting"].includes(row.status)).map((row) => (
              <div key={row.model} style={{ display: "flex", alignItems: "center", gap: 10,
                                            padding: "9px 20px" }}>
                <div className="mono" style={{ fontSize: 10.5, flex: "0 0 58px",
                                               color: LANE_TONE[row.status] }}>
                  {row.status}
                </div>
                <div className="mono" style={{ fontSize: 12, color: "var(--muted)", minWidth: 0,
                                               overflow: "hidden", textOverflow: "ellipsis",
                                               whiteSpace: "nowrap" }}>
                  {row.model}
                </div>
                <div className="mono" style={{ marginLeft: "auto", fontSize: 10.5, color: "var(--faint)" }}>
                  {row.status === "running" ? `gen ${row.generation ?? 1}` : "—"}
                </div>
              </div>
            ))}
            {!models.some((row) => ["running", "waiting"].includes(row.status)) && (
              <div style={{ padding: "16px 20px", color: "var(--faint)", fontSize: 13 }}>
                {ended ? "Nothing left to run." : "Nothing queued yet."}
              </div>
            )}
            <div className="mono" style={{ fontSize: 10, color: "var(--faint)", padding: "10px 20px 4px",
                                           lineHeight: 1.6, borderTop: "1px solid var(--line-soft)" }}>
              a model here is training or waiting for room on the host · one sandbox per
              attempt · up to {job.sandbox_parallel} job{job.sandbox_parallel === 1 ? "" : "s"} at once
            </div>
          </div>
        </Card>
      </div>

      {shownPick && (
        <PipelineCode ws={ws} job={job.id} pick={shownPick} attempts={attempts}
                      onPick={(model, attempt) => setPick({ model, attempt })} />
      )}
    </div>
  );
}
