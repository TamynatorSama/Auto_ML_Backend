import { useEffect, useState, type ReactNode } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";

import {
  deleteJob, redraftJob, savePlan, startJob, useJob,
  type Choices, type Configs, type Job, type Leak, type ModelChoice, type PlanEdit, type SplitPlan,
} from "../api";
import { when } from "../format";

/**
 * The plan the pipeline drafted, with the reason it chose each field, and your
 * edits on top (docs/PHASE5.md §7). Laid out as the Console canvas lays it out:
 * the sections on the left, and a rail on the right carrying the planned work,
 * the limits and the one button that starts the run.
 *
 * Nothing here invents a field. The canvas's class weighting, feature-engineering
 * pass and ensembling are absent rather than greyed, because the pipeline has no
 * such setting to send them to.
 *
 * Edits are partial and go to the pipeline as they are — review_plan merges them
 * over its own draft and validates them by building SplitPlan and Configs.
 */

const STATUS_TONE: Record<string, string> = {
  planning: "#C9873F", review: "var(--accent-bright)", queued: "#C9873F", running: "var(--accent-bright)",
  stopping: "#C9873F", succeeded: "var(--accent-bright)", stopped: "var(--dim)", failed: "#C9673F",
};

// the canvas's split: the rail is nearly as wide as the sections beside it
const RAIL = 440;

const input = { fontSize: 12.5, fontFamily: "var(--mono)", color: "var(--text)", background: "var(--bg)",
                border: "1px solid var(--edge)", borderRadius: 4, padding: "7px 12px", outline: "none",
                width: "100%", maxWidth: 460 } as const;

function hours(seconds: number): string {
  if (!seconds) return "none";
  const whole = Math.floor(seconds / 3600);
  const minutes = Math.round((seconds % 3600) / 60);
  return whole ? `${whole} h${minutes ? ` ${minutes} m` : ""}` : `${minutes} m`;
}

function Card({ children }: { children: ReactNode }) {
  return <div style={{ border: "1px solid var(--line)", borderRadius: 6, background: "var(--card)",
                       overflow: "hidden" }}>{children}</div>;
}

function Section({ title, meta, children }: { title: string; meta: string; children: ReactNode }) {
  return (
    <div style={{ marginBottom: 16 }}>
      <Card>
        <div className="chead">
          <div className="label" style={{ color: "var(--muted)" }}>{title}</div>
          <div className="mono" style={{ fontSize: 11, color: "var(--faint)" }}>{meta}</div>
        </div>
        {children}
      </Card>
    </div>
  );
}

/** One field: its name, the key it goes back to the pipeline as, its value, and the reason for it. */
function Row({ label, field, why, edited, children }: {
  label: string;
  field: string;
  why?: string;
  edited?: boolean;
  children: ReactNode;
}) {
  return (
    <div className="crow">
      <div>
        <div style={{ fontSize: 13.5, color: "var(--text)" }}>{label}</div>
        <div className="ckey">{field}</div>
      </div>
      <div style={{ minWidth: 0 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
          {children}
          {edited && <div className="mono" style={{ fontSize: 10, letterSpacing: "0.06em", color: "#C9873F" }}>
            EDITED
          </div>}
        </div>
        {why && <div className="why">{why}</div>}
      </div>
    </div>
  );
}

function Fixed({ value }: { value: string }) {
  return <div className="pill off">{value || "—"}</div>;
}

function Text({ value, onChange, disabled }: {
  value: string; onChange: (next: string) => void; disabled: boolean;
}) {
  return <input style={{ ...input, color: disabled ? "var(--dim)" : "var(--text)" }}
                value={value} disabled={disabled} onChange={(event) => onChange(event.target.value)} />;
}

function Num({ value, step, onChange, disabled, width = 150 }: {
  value: number; step: number; onChange: (next: number) => void; disabled: boolean; width?: number;
}) {
  return (
    <input type="number" step={step} value={value} disabled={disabled}
           style={{ ...input, width, color: disabled ? "var(--dim)" : "var(--text)" }}
           onChange={(event) => {
             const next = Number(event.target.value);
             if (!Number.isNaN(next)) onChange(next);
           }} />
  );
}

/** One chip: picked, pickable, or there but not usable. */
function Chip({ label, tone, title, onClick, sign }: {
  label: string;
  tone: "on" | "off" | "dead";
  title?: string;
  onClick?: () => void;
  sign?: string;
}) {
  const colour = { on: ["#9FE6D4", "#1B3F36", "#0F1D19"], off: ["var(--muted)", "var(--line)", "transparent"],
                   dead: ["var(--faint)", "var(--line)", "transparent"] }[tone];
  return (
    <div className="mono" title={title}
         style={{ display: "inline-flex", alignItems: "center", gap: 7, fontSize: 12, borderRadius: 4,
                  padding: "6px 10px", color: colour[0], border: `1px solid ${colour[1]}`,
                  background: colour[2], cursor: onClick ? "pointer" : "default",
                  textDecoration: tone === "dead" ? "line-through" : undefined }}
         onClick={onClick}>
      {label}
      {sign && <span style={{ fontSize: 13, lineHeight: 1, opacity: 0.75 }}>{sign}</span>}
    </div>
  );
}

/**
 * Pick from what the pipeline actually supports: chosen chips carry an x, the rest
 * sit under them to be clicked in. Options it cannot run are shown struck through
 * with the reason, not hidden — a family dropped quietly is how a run trains four
 * of the five you asked for.
 */
function Pick({ value, options, onChange, disabled, empty }: {
  value: string[];
  options: ModelChoice[];
  onChange: (next: string[]) => void;
  disabled: boolean;
  empty: string;
}) {
  const chosen = options.filter((option) => value.includes(option.name));
  const rest = options.filter((option) => !value.includes(option.name));
  const remove = (name: string) => onChange(value.filter((each) => each !== name));
  const add = (name: string) => onChange([...value, name]);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
      <div style={{ display: "flex", gap: 7, flexWrap: "wrap" }}>
        {chosen.length
          ? chosen.map((option) => (
            <Chip key={option.name} label={option.name} sign={disabled ? undefined : "×"}
                  tone={option.available ? "on" : "dead"}
                  title={option.why ?? undefined}
                  onClick={disabled ? undefined : () => remove(option.name)} />
          ))
          : <div className="mono" style={{ fontSize: 12, color: "#C9673F" }}>{empty}</div>}
      </div>
      {!disabled && rest.length > 0 && (
        <div style={{ display: "flex", gap: 7, flexWrap: "wrap", alignItems: "center" }}>
          <div className="mono" style={{ fontSize: 10.5, color: "var(--faint)" }}>add</div>
          {rest.map((option) => (
            <Chip key={option.name} label={option.name} sign={option.available ? "+" : undefined}
                  tone={option.available ? "off" : "dead"}
                  title={option.why ?? undefined}
                  onClick={option.available ? () => add(option.name) : undefined} />
          ))}
        </div>
      )}
    </div>
  );
}

/** One of a fixed set — the split method, or which metric the run improves. */
function One({ value, options, onChange, disabled }: {
  value: string; options: string[]; onChange: (next: string) => void; disabled: boolean;
}) {
  const known = options.includes(value) ? options : [value, ...options];
  return (
    <div style={{ display: "flex", gap: 7, flexWrap: "wrap" }}>
      {known.map((option) => (
        <Chip key={option} label={option} tone={option === value ? "on" : "off"}
              onClick={disabled || option === value ? undefined : () => onChange(option)} />
      ))}
    </div>
  );
}

/** Model families and metrics as plain names, when a plan predates the choices block. */
function named(names: string[]): ModelChoice[] {
  return names.map((name) => ({ name, available: true, why: null }));
}

function Leaks({ hits }: { hits: Leak[] }) {
  const named = new Set(hits.flatMap((hit) => hit.columns)).size;
  if (!hits.length) return null;
  return (
    <div style={{ border: "1px solid #3E3122", borderRadius: 6, background: "#17130D", padding: "16px 20px",
                  marginBottom: 16 }}>
      <div style={{ fontSize: 14, color: "#E0B679", marginBottom: 8 }}>
        {/* the screen counts columns, not findings: one finding can name a pair that
            only reproduces the target together, and calling that "one column" is wrong */}
        {named === 1 ? "One column reproduces the target"
          : hits.length === 1 ? `${named} columns together reproduce the target`
            : `${named} columns reproduce the target`}
      </div>
      <div style={{ fontSize: 13, lineHeight: 1.6, color: "var(--muted)", marginBottom: 12, maxWidth: 680 }}>
        Found on held-out rows, before any model ran. The run excludes these unless you said they are known at
        prediction time; that answer lives on the schema, not here.
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        {hits.map((hit) => (
          <div key={hit.columns.join(",")} className="mono" style={{ fontSize: 12, color: "var(--dim)" }}>
            <span style={{ color: "#E0B679" }}>{hit.columns.join(", ")}</span>
            {" · "}{hit.formula} ({hit.measure} {hit.score.toFixed(4)})
            {hit.outcome ? ` → ${hit.outcome}` : ""}
          </div>
        ))}
      </div>
    </div>
  );
}

/** One line of the compute card: a name, and a value or a small input. */
function Stat({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12,
                  padding: "9px 0", borderBottom: "1px solid var(--line-soft)" }}>
      <div style={{ fontSize: 13, color: "var(--muted)" }}>{label}</div>
      {children}
    </div>
  );
}

/**
 * The right rail's first card: what the run will actually do. The pipeline works
 * in models and tries, not the canvas's passes and candidates, so this counts the
 * real thing — one lane per model family, up to max_tries attempts each.
 */
function PlannedWork({ config }: { config: Configs }) {
  return (
    <Card>
      <div className="chead"><div className="label" style={{ color: "var(--muted)" }}>Planned work</div></div>
      <div style={{ padding: "6px 20px 0" }}>
        {config.models.map((model, index) => (
          <div key={model} style={{ display: "flex", gap: 14, padding: "12px 0",
                                    borderBottom: "1px solid var(--line-soft)" }}>
            <div className="mono" style={{ fontSize: 11, color: "var(--accent-bright)", width: 10 }}>{index + 1}</div>
            <div style={{ minWidth: 0 }}>
              <div style={{ fontSize: 13.5, color: "var(--text)" }}>{model}</div>
              <div className="mono" style={{ fontSize: 11, color: "var(--faint)", marginTop: 3 }}>
                up to {config.max_tries} {config.max_tries === 1 ? "try" : "tries"} · stops after{" "}
                {config.early_stopping_patience} with no gain
              </div>
            </div>
          </div>
        ))}
      </div>
      <div style={{ display: "flex", justifyContent: "space-between", gap: 12, padding: "12px 20px" }}>
        <div className="mono" style={{ fontSize: 11, color: "var(--dim)" }}>
          {config.models.length} model{config.models.length === 1 ? "" : "s"} ·{" "}
          {config.models.length * config.max_tries} attempts at most
        </div>
        <div className="mono" style={{ fontSize: 11, color: "var(--faint)" }}>
          best by {config.improvement_metric}
        </div>
      </div>
    </Card>
  );
}

/** The page's own shape while it loads, rather than an empty screen. */
function Skeleton({ note }: { note?: string }) {
  const rows = (count: number) => Array.from({ length: count }, (unused, index) => index);
  return (
    <div className="page-body" style={{ padding: "36px 40px 84px", maxWidth: RAIL + 800 }}>
      <div className="config-layout" style={{ display: "grid", gridTemplateColumns: `minmax(0, 1fr) ${RAIL}px`, gap: 22,
                    alignItems: "start" }}>
        <div>
          <div className="skel" style={{ width: 260, height: 27, marginBottom: 12 }} />
          <div className="skel" style={{ width: 440, height: 14, marginBottom: 7 }} />
          <div className="skel" style={{ width: 360, height: 14, marginBottom: 26 }} />
          {note && <div style={{ display: "flex", alignItems: "center", gap: 9, marginBottom: 22 }}>
            <div className="pulse" />
            <div style={{ fontSize: 13.5, color: "var(--muted)" }}>{note}</div>
          </div>}
          {[4, 5, 4].map((count, section) => (
            <div key={section} style={{ marginBottom: 16 }}>
              <Card>
                <div className="chead">
                  <div className="skel" style={{ width: 110, height: 11 }} />
                  <div className="skel" style={{ width: 80, height: 11 }} />
                </div>
                {rows(count).map((row) => (
                  <div className="crow" key={row}>
                    <div>
                      <div className="skel" style={{ width: 96, height: 13 }} />
                      <div className="skel" style={{ width: 120, height: 10, marginTop: 6 }} />
                    </div>
                    <div>
                      <div className="skel" style={{ width: 200, height: 31 }} />
                      <div className="skel" style={{ width: "72%", height: 12, marginTop: 12 }} />
                    </div>
                  </div>
                ))}
              </Card>
            </div>
          ))}
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <Card>
            <div className="chead"><div className="skel" style={{ width: 96, height: 11 }} /></div>
            <div style={{ padding: "14px 20px", display: "flex", flexDirection: "column", gap: 16 }}>
              {rows(4).map((row) => <div className="skel" key={row} style={{ width: "86%", height: 30 }} />)}
            </div>
          </Card>
          <Card>
            <div className="chead"><div className="skel" style={{ width: 70, height: 11 }} /></div>
            <div style={{ padding: "14px 20px", display: "flex", flexDirection: "column", gap: 14 }}>
              {rows(3).map((row) => <div className="skel" key={row} style={{ width: "100%", height: 18 }} />)}
            </div>
          </Card>
          <div className="skel" style={{ width: "100%", height: 42 }} />
        </div>
      </div>
    </div>
  );
}

export function Config({ ws, jobId, onBack, onGone, onStarted }: {
  ws: string; jobId: number; onBack: () => void; onGone: () => void; onStarted: () => void;
}) {
  const job = useJob(ws, jobId);
  const queries = useQueryClient();
  const [split, setSplit] = useState<SplitPlan | null>(null);
  const [config, setConfig] = useState<Configs | null>(null);
  const [limits, setLimits] = useState<{ budget_usd: number; deadline_seconds: number } | null>(null);
  const [problem, setProblem] = useState("");
  const [confirming, setConfirming] = useState(false);

  // the server's copy is the truth; edits already stored are laid over it
  useEffect(() => {
    const body = job.data;
    if (!body?.plan) return;
    setSplit({ ...body.plan.split_plan, ...(body.edits?.split_plan ?? {}) });
    setConfig({ ...body.plan.config, ...(body.edits?.config ?? {}) });
    setLimits({ budget_usd: body.budget_usd ?? 0, deadline_seconds: body.deadline_seconds ?? 0 });
  }, [job.data?.id, job.data?.status, job.data?.plan !== null]);

  const refresh = () => {
    queries.invalidateQueries({ queryKey: ["job", ws, jobId] });
    queries.invalidateQueries({ queryKey: ["jobs", ws] });
  };
  const fail = (error: Error) => setProblem(error.message);
  const body = (): PlanEdit => ({
    split_plan: changed(job.data?.plan?.split_plan, split),
    config: changed(job.data?.plan?.config, config),
    budget_usd: limits?.budget_usd || null,
    deadline_seconds: limits?.deadline_seconds || null,
  });

  const save = useMutation({ mutationFn: () => savePlan(ws, jobId, body()),
                             onSuccess: () => { setProblem(""); refresh(); }, onError: fail });
  // approving hands the run to the worker, so the screen goes with it
  const start = useMutation({ mutationFn: () => startJob(ws, jobId, body()),
                              onSuccess: () => { setProblem(""); refresh(); onStarted(); }, onError: fail });
  const again = useMutation({ mutationFn: () => redraftJob(ws, jobId),
                              onSuccess: () => { setProblem(""); refresh(); }, onError: fail });
  const drop = useMutation({
    mutationFn: () => deleteJob(ws, jobId),
    onSuccess: () => { queries.invalidateQueries({ queryKey: ["jobs", ws] }); onGone(); },
    onError: fail,
  });

  if (job.isPending) return <Skeleton />;
  if (job.error || !job.data) {
    return <div style={{ padding: 40, color: "var(--muted)" }}>
      {(job.error as Error)?.message ?? "No such run."}
    </div>;
  }

  const run = job.data;
  const reviewing = run.status === "review";
  const busy = save.isPending || start.isPending || again.isPending || drop.isPending;
  const over = ["succeeded", "stopped", "failed", "review"].includes(run.status);
  const plan = run.plan;

  if (!plan || !split || !config || !limits) {
    return run.status === "planning"
      ? <Skeleton note={`Reading every row and drafting a plan — queued ${when(run.created_at)}.`} />
      : <div style={{ padding: 40, color: "var(--muted)" }}>This run has no drafted plan.</div>;
  }

  const choices: Choices | null = plan.choices ?? null;
  const locked = !reviewing || !choices;   // an older plan carries no choices to pick from
  const why = (field: string) => config.reasons?.[field];
  const wasSplit = (field: keyof SplitPlan) =>
    JSON.stringify(split[field]) !== JSON.stringify(plan.split_plan[field]);
  const wasConfig = (field: keyof Configs) =>
    JSON.stringify(config[field]) !== JSON.stringify(plan.config[field]);
  const edits = Object.keys(changed(plan.split_plan, split)).length
              + Object.keys(changed(plan.config, config)).length;

  return (
    <div className="page-body" style={{ padding: "36px 40px 84px", maxWidth: RAIL + 800 }}>
      <div className="config-layout" style={{ display: "grid", gridTemplateColumns: `minmax(0, 1fr) ${RAIL}px`, gap: 22,
                    alignItems: "start" }}>
        <div>
      <div style={{ display: "flex", alignItems: "center", gap: 14, flexWrap: "wrap", marginBottom: 10 }}>
        <div style={{ fontSize: 27, letterSpacing: "-0.015em", color: "var(--bright)" }}>Job configuration</div>
        <div className="mono" style={{ fontSize: 10, letterSpacing: "0.06em", borderRadius: 3, padding: "3px 8px",
                                       color: "var(--accent-bright)", border: "1px solid #1B3F36",
                                       background: "#0F1D19" }}>
          AI DRAFTED
        </div>
        <div className="mono" style={{ fontSize: 10.5, letterSpacing: "0.06em", borderRadius: 3, padding: "3px 8px",
                                       color: STATUS_TONE[run.status] ?? "var(--dim)",
                                       border: "1px solid var(--edge)" }}>
          {run.status}
        </div>
      </div>
      <div style={{ fontSize: 13.5, lineHeight: 1.6, color: "var(--muted)", maxWidth: 620, marginBottom: 6 }}>
        {reviewing
          ? "Derived from the locked schema and a profile of every row. Every field is editable; edited fields "
            + "are marked and the reasoning is kept with the run."
          : "This run has been approved. The plan below is what it is training against."}
      </div>
      <div className="mono" style={{ fontSize: 11, color: "var(--faint)", marginBottom: 24 }}>
        run {run.id} · {run.source.name} · schema v-pinned{edits ? ` · ${edits} field${
          edits === 1 ? "" : "s"} edited` : ""}
      </div>

      {reviewing && !choices && (
        <div className="note" style={{ marginBottom: 16 }}>
          This plan was drafted before the console knew what the pipeline supports, so the metrics, model
          families and split method are locked. Draft again to edit them.
        </div>
      )}
      {problem && <div className="note" style={{ marginBottom: 16 }}>{problem}</div>}
      {run.error && <div className="note" style={{ marginBottom: 16 }}>{run.error}</div>}
      <Leaks hits={plan.leakage_screen} />

          <Section title="Objective" meta="inferred from the target">
            <Row label="Task" field="task_type"><Fixed value={plan.task_type} /></Row>
            <Row label="Target column" field="target"><Fixed value={plan.target} /></Row>
            <Row label="Metrics" field="config.eval_matrics" why={why("eval_matrics")}
                 edited={wasConfig("eval_matrics")}>
              <Pick value={config.eval_matrics} disabled={locked} empty="pick at least one"
                    options={named(choices?.metrics ?? config.eval_matrics)}
                    onChange={(eval_matrics) => setConfig({
                      ...config, eval_matrics,
                      // improving a metric you no longer measure is incoherent
                      improvement_metric: eval_matrics.includes(config.improvement_metric)
                        ? config.improvement_metric : eval_matrics[0] ?? config.improvement_metric,
                    })} />
            </Row>
            <Row label="Metric to improve" field="config.improvement_metric" why={why("improvement_metric")}
                 edited={wasConfig("improvement_metric")}>
              <One value={config.improvement_metric} options={config.eval_matrics} disabled={locked}
                   onChange={(improvement_metric) => setConfig({ ...config, improvement_metric })} />
            </Row>
            <Row label="Baseline" field="config.baseline" why={why("baseline")}>
              <Fixed value={`${config.baseline.strategy}${config.baseline.applicable ? "" : " · not applicable"}`} />
            </Row>
          </Section>

          <Section title="Validation" meta="leakage-aware">
            <div style={{ padding: "14px 22px 2px" }}>
              <div style={{ fontSize: 13, lineHeight: 1.6, color: "var(--dim)", maxWidth: 620 }}>
                {split.reason}
              </div>
              {split.warnings.map((warning) => (
                <div key={warning} style={{ fontSize: 12.5, lineHeight: 1.55, color: "#C9873F", marginTop: 8 }}>
                  {warning}
                </div>
              ))}
            </div>
            <Row label="Split method" field="split_plan.method" edited={wasSplit("method")}>
              <One value={split.method} options={choices?.split_methods ?? [split.method]} disabled={locked}
                   onChange={(method) => setSplit({ ...split, method })} />
            </Row>
            <Row label="Test size" field="split_plan.test_size" edited={wasSplit("test_size")}>
              <Num value={split.test_size} step={0.05} disabled={!reviewing}
                   onChange={(test_size) => setSplit({ ...split, test_size })} />
            </Row>
            <Row label="Cross-validation" field="split_plan.cv_strategy"><Fixed value={split.cv_strategy} /></Row>
            <Row label="Folds" field="split_plan.cv_folds" edited={wasSplit("cv_folds")}>
              <Num value={split.cv_folds} step={1} disabled={!reviewing}
                   onChange={(cv_folds) => setSplit({ ...split, cv_folds })} />
            </Row>
            <Row label="Seed" field="split_plan.random_seed" edited={wasSplit("random_seed")}>
              <Num value={split.random_seed} step={1} disabled={!reviewing}
                   onChange={(random_seed) => setSplit({ ...split, random_seed })} />
            </Row>
            <Row label="Shuffle · duplicates" field="split_plan.shuffle · drop_duplicates">
              <Fixed value={`${split.shuffle ? "shuffled" : "kept in order"} · ${
                split.drop_duplicates ? "duplicates dropped" : "duplicates kept"}`} />
            </Row>
            {(split.time_column || split.group_column || split.stratify_column) && (
              <Row label="Split by" field="split_plan.time · group · stratify_column">
                <Fixed value={[split.time_column && `time: ${split.time_column}`,
                               split.group_column && `group: ${split.group_column}`,
                               split.stratify_column && `stratify: ${split.stratify_column}`]
                  .filter(Boolean).join(" · ")} />
              </Row>
            )}
          </Section>

          <Section title="Search space" meta={`${config.models.length} model${
            config.models.length === 1 ? "" : "s"} · ${config.models.length * config.max_tries} attempts at most`}>
            <Row label="Model families" field="config.models" why={why("models")} edited={wasConfig("models")}>
              <Pick value={config.models} disabled={locked} empty="pick at least one"
                    options={choices?.models ?? named(config.models)}
                    onChange={(models) => setConfig({ ...config, models })} />
            </Row>
            <Row label="Tries per model" field="config.max_tries" why={why("max_tries")}
                 edited={wasConfig("max_tries")}>
              <Num value={config.max_tries} step={1} disabled={!reviewing}
                   onChange={(max_tries) => setConfig({ ...config, max_tries })} />
            </Row>
            <Row label="Stop after no gain in" field="config.early_stopping_patience"
                 why={why("early_stopping_patience")} edited={wasConfig("early_stopping_patience")}>
              <Num value={config.early_stopping_patience} step={1} disabled={!reviewing}
                   onChange={(early_stopping_patience) => setConfig({ ...config, early_stopping_patience })} />
            </Row>
            <Row label="Counts as a gain" field="config.improvement_delta" why={why("improvement_delta")}
                 edited={wasConfig("improvement_delta")}>
              <Num value={config.improvement_delta} step={0.001} disabled={!reviewing}
                   onChange={(improvement_delta) => setConfig({ ...config, improvement_delta })} />
            </Row>
            <Row label="Measured" field="config.improvement_mode" why={why("improvement_mode")}>
              <Fixed value={config.improvement_mode} />
            </Row>
          </Section>
        </div>

        <div className="config-rail" style={{ display: "flex", flexDirection: "column", gap: 16, position: "sticky", top: 0 }}>
          <PlannedWork config={config} />

          <Card>
            <div className="chead"><div className="label" style={{ color: "var(--muted)" }}>Compute</div></div>
            <div style={{ padding: "4px 20px 14px" }}>
              <Stat label="Sandboxes">
                <div className="mono" style={{ fontSize: 12, color: "var(--faint)" }}>
                  up to {run.sandbox_parallel} job{run.sandbox_parallel === 1 ? "" : "s"} on this host
                </div>
              </Stat>
              <Stat label="Spend cap">
                <Num value={limits.budget_usd} step={0.5} width={110} disabled={!reviewing}
                     onChange={(budget_usd) => setLimits({ ...limits, budget_usd })} />
              </Stat>
              <Stat label="Time cap">
                <Num value={limits.deadline_seconds} step={600} width={110} disabled={!reviewing}
                     onChange={(deadline_seconds) => setLimits({ ...limits, deadline_seconds })} />
              </Stat>
              <div className="mono" style={{ fontSize: 10.5, color: "var(--faint)", marginTop: 10,
                                             lineHeight: 1.6 }}>
                spend in USD, counted from the run's own ledger · time in seconds, {hours(limits.deadline_seconds)},
                wall-clock from when training starts
              </div>
            </div>
          </Card>

          <div>
            <button className="button" style={{ width: "100%" }} disabled={!reviewing || busy}
                    onClick={() => start.mutate()}>
              {start.isPending ? "Starting…" : reviewing ? "Start AutoML job" : `Run is ${run.status}`}
            </button>
            <div className="mono" style={{ fontSize: 10.5, color: "var(--faint)", textAlign: "center",
                                           marginTop: 10, lineHeight: 1.6 }}>
              runs as {ws} · {run.source.name} · one run per dataset at a time
            </div>
            <div style={{ display: "flex", gap: 8, marginTop: 14 }}>
              <button className="btn ghost" style={{ flex: 1 }} disabled={!reviewing || busy}
                      onClick={() => save.mutate()}>
                {save.isPending ? "Saving…" : "Save edits"}
              </button>
              <button className="btn ghost" style={{ flex: 1 }} disabled={!reviewing || busy}
                      title="Throw this draft away and plan again from nothing"
                      onClick={() => again.mutate()}>
                {again.isPending ? "Drafting…" : "Draft again"}
              </button>
            </div>
            <button className="btn ghost" style={{ width: "100%", marginTop: 8 }} onClick={onBack}>
              Back to the schema
            </button>
            {/* deleting takes the events, attempts, scores and usage with it, so it asks first */}
            <button className="btn ghost" style={{ width: "100%", marginTop: 8, borderColor: "#3E2A26",
                                                   color: confirming ? "var(--danger)" : "var(--dim)" }}
                    disabled={!over || busy}
                    title={over ? undefined : "Stop the run before deleting it"}
                    onClick={() => (confirming ? drop.mutate() : setConfirming(true))}>
              {drop.isPending ? "Deleting…"
                : confirming ? "Delete run " + run.id + " and everything recorded against it?"
                  : "Delete this run"}
            </button>
            {confirming && !drop.isPending && (
              <button className="btn ghost" style={{ width: "100%", marginTop: 8 }}
                      onClick={() => setConfirming(false)}>
                Keep it
              </button>
            )}
            {/* also beside the button: the banner at the top of the page is off-screen from here */}
            {drop.error && (
              <div className="note" style={{ marginTop: 8 }}>{(drop.error as Error).message}</div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

/** Only what differs from the draft: review_plan merges a partial over its own values. */
function changed<T extends object>(draft: T | undefined | null, edited: T | null): Partial<T> {
  if (!draft || !edited) return {};
  const out: Partial<T> = {};
  for (const key of Object.keys(edited) as (keyof T)[]) {
    if (JSON.stringify(edited[key]) !== JSON.stringify(draft[key])) out[key] = edited[key];
  }
  return out;
}

export type { Job };
