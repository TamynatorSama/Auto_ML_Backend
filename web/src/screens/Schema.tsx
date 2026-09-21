import { useEffect, useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";

import {
  analyzeSchema, DATA_TYPES, lockSchema, ROLES, saveSchema, useSchema, useVersions,
  type ColumnChange, type SchemaColumn, type Source,
} from "../api";
import { count, percent, when } from "../format";

const GRID = "minmax(170px, 1.3fr) 118px 132px minmax(230px, 1.6fr) 96px 88px";
const SEVERITY: Record<string, string> = {
  critical: "#C9673F", high: "#C9873F", medium: "#C9873F", low: "var(--dim)",
};
const ROLE_TONE: Record<string, string> = {
  target: "var(--accent-bright)", ignore: "var(--faint)", identifier: "#8A7BB8", feature: "#C8D2CE",
};
/** Only a person can answer this, so "not sure" is a real answer and the default (G13). */
const KNOWN = [["unset", "not sure"], ["yes", "yes"], ["no", "no"]] as const;

function known(value: boolean | null) {
  return value === null ? "unset" : value ? "yes" : "no";
}

function Versions({ ws, source }: { ws: string; source: Source }) {
  const [diff, setDiff] = useState<string | null>(null);
  const versions = useVersions(ws, source.id, diff);
  const rows = versions.data?.versions ?? [];
  if (rows.length < 1) return null;

  return (
    <div style={{ marginTop: 22, border: "1px solid var(--line)", borderRadius: 6, background: "var(--card)",
                  padding: "18px 22px" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 14, marginBottom: 14, flexWrap: "wrap" }}>
        <div className="label" style={{ color: "var(--faint)" }}>Versions</div>
        {rows.length > 1 && (
          <button className="pager" onClick={() => setDiff(diff ? null : `${rows[1].version},${rows[0].version}`)}>
            {diff ? "hide the diff" : `compare v${rows[1].version} with v${rows[0].version}`}
          </button>
        )}
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 9 }}>
        {rows.map((row) => (
          <div key={row.id} style={{ display: "flex", alignItems: "center", gap: 12, fontSize: 13,
                                     color: "var(--muted)" }}>
            <div className="mono" style={{ color: "#D8E0DD", width: 34 }}>v{row.version}</div>
            <div className="mono" style={{ fontSize: 11, borderRadius: 3, padding: "1px 6px",
                                           color: row.status === "locked" ? "var(--accent)" : "#C9873F",
                                           border: `1px solid ${row.status === "locked" ? "#1B3F36" : "#3E3122"}` }}>
              {row.status}
            </div>
            <div style={{ fontSize: 12.5, color: "var(--dim)" }}>
              {row.locked_at ? `locked ${when(row.locked_at)}` : `edited ${when(row.updated_at)}`}
              {row.locked_by ? ` by ${row.locked_by}` : ""}
            </div>
          </div>
        ))}
      </div>
      {versions.data?.diff && <Diff changes={versions.data.diff.columns} />}
    </div>
  );
}

function Diff({ changes }: { changes: ColumnChange[] }) {
  if (!changes.length) {
    return <div className="mono" style={{ marginTop: 14, fontSize: 12, color: "var(--faint)" }}>
      nothing changed between these two
    </div>;
  }
  return (
    <div style={{ marginTop: 14, borderTop: "1px solid var(--line-soft)", paddingTop: 14,
                  display: "flex", flexDirection: "column", gap: 10 }}>
      {changes.map((change) => (
        <div key={change.name} style={{ fontSize: 13 }}>
          <div className="mono" style={{ color: "#D8E0DD", marginBottom: 3 }}>
            {change.name} <span style={{ color: "var(--faint)" }}>{change.change}</span>
          </div>
          {change.fields.map((field) => (
            <div key={field.field} className="mono" style={{ fontSize: 12, color: "var(--dim)", paddingLeft: 14 }}>
              {field.field}: <span style={{ color: "#C9673F" }}>{String(field.from) || "—"}</span>
              {" → "}<span style={{ color: "var(--accent-bright)" }}>{String(field.to) || "—"}</span>
            </div>
          ))}
        </div>
      ))}
    </div>
  );
}

/** The table's own shape while the schema arrives, rather than an empty screen. */
function SchemaSkeleton() {
  const rows = Array.from({ length: 9 }, (unused, index) => index);
  return (
    <div style={{ padding: "40px 40px 84px", maxWidth: 1240 }}>
      <div className="skel" style={{ width: 160, height: 28, marginBottom: 12 }} />
      <div className="skel" style={{ width: 540, height: 14, marginBottom: 7 }} />
      <div className="skel" style={{ width: 480, height: 14, marginBottom: 26 }} />
      <div style={{ border: "1px solid var(--line)", borderRadius: 6, background: "var(--card)",
                    overflow: "hidden" }}>
        <div style={{ display: "flex", gap: 14, padding: "13px 24px", background: "#151b19",
                      borderBottom: "1px solid var(--edge)" }}>
          {[150, 100, 110, 200, 80, 70].map((width) => (
            <div className="skel" key={width} style={{ width, height: 11 }} />
          ))}
        </div>
        {rows.map((row) => (
          <div key={row} style={{ display: "flex", gap: 14, padding: "13px 24px",
                                  borderBottom: "1px solid #1a201e" }}>
            {[150, 100, 110, 200, 80, 70].map((width) => (
              <div className="skel" key={width} style={{ width, height: 22 }} />
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}

export function Schema({ ws, source }: { ws: string; source: Source }) {
  const schema = useSchema(ws, source.id);
  const queries = useQueryClient();
  const [columns, setColumns] = useState<SchemaColumn[] | null>(null);
  const [problem, setProblem] = useState("");
  // a locked schema is read-only until you say you want the next version
  const [editing, setEditing] = useState(false);
  const [showNotes, setShowNotes] = useState(false);

  // the server's copy is the truth; local edits sit on top until saved
  useEffect(() => {
    if (schema.data) {
      setColumns(schema.data.columns);
      if (schema.data.status === "draft") setEditing(false);
    }
  }, [schema.data]);

  const refresh = () => {
    queries.invalidateQueries({ queryKey: ["schema", ws, source.id] });
    queries.invalidateQueries({ queryKey: ["versions", ws, source.id] });
    queries.invalidateQueries({ queryKey: ["sources", ws] });
    queries.invalidateQueries({ queryKey: ["source", ws, source.id] });
  };
  const fail = (error: Error) => setProblem(error.message);

  const save = useMutation({
    mutationFn: (next: SchemaColumn[]) =>
      saveSchema(ws, source.id, { columns: next, description: schema.data?.description ?? "" }),
    onSuccess: () => { setProblem(""); refresh(); },
    onError: fail,
  });
  const analyze = useMutation({ mutationFn: () => analyzeSchema(ws, source.id),
                                onSuccess: () => { setProblem(""); refresh(); }, onError: fail });
  // lock saves first, so the server locks against what you are looking at
  const lock = useMutation({
    mutationFn: async (next: SchemaColumn[]) => {
      await saveSchema(ws, source.id, { columns: next, description: schema.data?.description ?? "" });
      return lockSchema(ws, source.id);
    },
    onSuccess: () => { setProblem(""); refresh(); },
    onError: fail,
  });

  const body = schema.data;
  const locked = body?.status === "locked";
  const frozen = locked && !editing;
  const target = (columns ?? []).find((column) => column.role === "target");
  const findings = body?.findings ?? [];
  const leaks = findings.filter((finding) => finding.id.startsWith("leak:") && finding.columns.length > 0);
  const notes = findings.filter((finding) => !leaks.includes(finding));
  const noteNames = [...new Set(notes.flatMap((finding) => finding.columns))].join(", ");
  const running = body?.task?.kind === "analyze";

  if (schema.isPending) return <SchemaSkeleton />;
  if (schema.error || !body || !columns) {
    return (
      <div style={{ padding: 40, color: "var(--muted)" }}>
        {(schema.error as Error)?.message ?? "This dataset has no schema yet."}
      </div>
    );
  }

  const edit = (name: string, change: Partial<SchemaColumn>) =>
    setColumns(columns.map((column) => {
      if (column.name !== name) {
        // one target at a time: naming a new one releases the old
        return change.role === "target" && column.role === "target" ? { ...column, role: "feature" } : column;
      }
      return { ...column, ...change };
    }));

  /** Answer "is it there when you predict?" for every column a leak names, in one go. */
  const declare = (named: string[], value: boolean | null) =>
    setColumns(columns.map((column) =>
      named.includes(column.name) ? { ...column, available_at_prediction: value } : column));

  return (
    <div style={{ padding: "40px 40px 84px", maxWidth: 1240 }}>
      <div style={{ display: "flex", flexWrap: "wrap", alignItems: "flex-end", justifyContent: "space-between",
                    gap: "16px 24px", marginBottom: 22 }}>
        <div>
          <div style={{ fontSize: 30, letterSpacing: "-0.015em", color: "var(--bright)", marginBottom: 5 }}>
            Schema
          </div>
          <div style={{ fontSize: 14, lineHeight: 1.6, color: "var(--muted)", maxWidth: 560 }}>
            Types and roles were read off the profile, over every row. Set one target, mark identifiers, and
            describe any column whose meaning isn't obvious from its name — the description reaches every
            prompt the pipeline writes. Then lock the schema so jobs reference it by version.
          </div>
        </div>
        <div style={{ display: "flex", gap: 12 }}>
          {/* a locked schema has no draft to analyse: save the next version first */}
          <button className="btn ghost" disabled={!target || running || analyze.isPending || locked}
                  title={locked ? `Save v${body.version + 1} first`
                    : target ? "Look for columns that reproduce the target" : "Choose a target first"}
                  onClick={() => analyze.mutate()}>
            {running ? "Analysing…" : body.findings.length ? "Analyse again" : "Analyse"}
          </button>
          <button className="btn ghost" disabled={save.isPending || frozen}
                  onClick={() => save.mutate(columns)}>
            {save.isPending ? "Saving…" : "Save"}
          </button>
          <button className="btn" disabled={lock.isPending || frozen || !target}
                  title={target ? undefined : "Mark one column as the target"}
                  onClick={() => lock.mutate(columns)}>
            {lock.isPending ? "Locking…" : locked ? `Locked · v${body.version}` : "Save and lock"}
          </button>
        </div>
      </div>

      {problem && <div className="note" style={{ marginBottom: 16 }}>{problem}</div>}

      {locked && (
        <div style={{ border: "1px solid #1B3F36", borderLeft: "2px solid var(--accent)", borderRadius: 4,
                      background: "#0F1D19", padding: "12px 16px", display: "flex", flexWrap: "wrap",
                      alignItems: "center", gap: 16, marginBottom: 18 }}>
          <div className="label" style={{ color: "var(--accent-bright)", flex: "0 0 auto" }}>
            locked · v{body.version}
          </div>
          <div style={{ flex: "1 1 320px", minWidth: 0, fontSize: 13.5, color: "#B9CFC8" }}>
            Types, roles and descriptions are frozen. Editing any of them starts v{body.version + 1} as a draft,
            so a job that ran against this version keeps it.
          </div>
          <button className="pager" style={{ marginLeft: "auto" }}
                  onClick={() => setEditing(!editing)}>
            {editing ? "stop editing" : `edit as v${body.version + 1}`}
          </button>
        </div>
      )}

      {/* A leak is the only warning with a decision in it, so it stays open. The rest are notes the
          planners already have, folded into one line so they never push the table off the screen. */}
      {leaks.map((finding) => {
        const settled = finding.columns.every((name) =>
          columns.find((column) => column.name === name)?.available_at_prediction !== null);
        return (
          <div key={finding.id}
               style={{ border: "1px solid var(--edge)",
                        borderLeft: `2px solid ${settled ? "var(--accent)" : SEVERITY[finding.severity]}`,
                        borderRadius: 4, background: settled ? "#0F1413" : "#14100B", padding: "12px 16px",
                        display: "flex", flexWrap: "wrap", alignItems: "center", gap: 14, marginBottom: 10 }}>
            <div className="label" style={{ flex: "0 0 auto",
                                            color: settled ? "var(--accent)" : SEVERITY[finding.severity] }}>
              {settled ? "answered" : "possible leak"}
            </div>
            <div style={{ flex: "1 1 340px", minWidth: 0, fontSize: 13.5, lineHeight: 1.5, color: "#D8CFC0" }}>
              <span className="mono" style={{ color: "#E6ECE9" }}>{finding.columns.join(", ")}{" — "}</span>
              {finding.issue}
              <div style={{ fontSize: 12.5, color: "var(--dim)", marginTop: 3 }}>
                Left out of the run unless you say it's known at prediction time.
              </div>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 7, flex: "0 0 auto" }}>
              <div style={{ fontSize: 12.5, color: "var(--dim)" }}>known at prediction?</div>
              {KNOWN.map(([value, label]) => (
                <button key={value} className="pager" disabled={frozen}
                        onClick={() => declare(finding.columns, value === "unset" ? null : value === "yes")}>
                  {label}
                </button>
              ))}
            </div>
          </div>
        );
      })}

      {notes.length > 0 && (
        <div style={{ border: "1px solid var(--line)", borderRadius: 4, background: "var(--panel)",
                      marginBottom: 18 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 12, padding: "10px 16px" }}>
            <div className="label" style={{ color: "#C9873F", flex: "0 0 auto" }}>
              {notes.length} note{notes.length === 1 ? "" : "s"}
            </div>
            <div style={{ flex: 1, minWidth: 0, fontSize: 13, color: "var(--dim)", overflow: "hidden",
                          textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
              {noteNames || "about the data itself"} · the planners read these too
            </div>
            <button className="pager" onClick={() => setShowNotes(!showNotes)}>
              {showNotes ? "hide" : "show"}
            </button>
          </div>
          {showNotes && (
            <div style={{ borderTop: "1px solid var(--line-soft)", padding: "6px 16px 12px" }}>
              {notes.map((finding) => (
                <div key={finding.id} style={{ display: "flex", gap: 10, padding: "7px 0", fontSize: 13,
                                               lineHeight: 1.5, color: "var(--muted)" }}>
                  <div className="mono" style={{ flex: "0 0 auto", fontSize: 11,
                                                 color: SEVERITY[finding.severity] }}>
                    {finding.severity}
                  </div>
                  <div style={{ minWidth: 0 }}>
                    {finding.columns.length > 0 && (
                      <span className="mono" style={{ color: "#D8E0DD" }}>
                        {finding.columns.join(", ")}{" — "}
                      </span>
                    )}
                    {finding.issue}
                    {finding.action && <div style={{ fontSize: 12, color: "var(--faint)", marginTop: 2 }}>
                      {finding.action}
                    </div>}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      <div style={{ border: "1px solid var(--line)", borderRadius: 6, background: "var(--card)",
                    overflowX: "auto" }}>
        <div className="srow head" style={{ gridTemplateColumns: GRID }}>
          <div>column</div><div>type</div><div>role</div><div>description</div>
          <div>null %</div><div>distinct</div>
        </div>
        {columns.map((column) => {
          const stats = body.profile[column.name];
          const nulls = stats?.missing_pct ?? 0;
          return (
            <div key={column.name} className="srow" style={{ gridTemplateColumns: GRID }}>
              <div style={{ minWidth: 0 }}>
                <div className="mono" style={{ fontSize: 13.5, color: "var(--text)", overflow: "hidden",
                                               textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                  {column.name}
                </div>
                {/* the storage dtype is a fact about the file, so it is a hint, not a field */}
                <div className="mono" style={{ fontSize: 11.5, color: "var(--faint)", marginTop: 2 }}>
                  {stats?.dtype ?? "—"}{stats?.semantic_type ? ` · ${stats.semantic_type}` : ""}
                </div>
              </div>
              <select className="cell" disabled={frozen} value={column.data_type}
                      onChange={(event) => edit(column.name, { data_type: event.target.value })}>
                {DATA_TYPES.map((type) => <option key={type} value={type}>{type}</option>)}
              </select>
              <select className="cell" disabled={frozen} value={column.role}
                      style={{ color: ROLE_TONE[column.role] }}
                      onChange={(event) => edit(column.name, { role: event.target.value as SchemaColumn["role"] })}>
                {ROLES.map((role) => <option key={role} value={role}>{role}</option>)}
              </select>
              <input className="cell" disabled={frozen} value={column.description}
                     placeholder="Describe this column"
                     onChange={(event) => edit(column.name, { description: event.target.value })} />
              <div style={{ display: "flex", alignItems: "center", gap: 7 }}>
                <div style={{ flex: 1, height: 3, background: "var(--line)", borderRadius: 2, overflow: "hidden" }}>
                  <div style={{ height: 3, width: `${nulls}%`,
                                background: nulls > 50 ? "#C9673F" : nulls > 0 ? "#C9873F" : "#2E5C51" }} />
                </div>
                <div className="mono" style={{ fontSize: 11.5, color: "var(--dim)", width: 34,
                                               textAlign: "right" }}>{percent(nulls, 0)}</div>
              </div>
              <div className="mono" style={{ fontSize: 12.5, color: "var(--dim)", textAlign: "right" }}>
                {stats ? count(stats.n_unique) : "—"}
              </div>
            </div>
          );
        })}
      </div>

      <div className="mono" style={{ marginTop: 12, fontSize: 11.5, color: "var(--faint)" }}>
        {columns.length} columns · {columns.filter((c) => c.description).length} described ·{" "}
        {target ? `target ${target.name}` : "no target yet"}
        {body.analyzed_target && body.analyzed_target !== target?.name
          ? ` · analysed against ${body.analyzed_target}, run it again` : ""}
      </div>

      <Versions ws={ws} source={source} />
    </div>
  );
}
