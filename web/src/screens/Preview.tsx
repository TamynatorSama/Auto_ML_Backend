import { useState } from "react";

import { useRows, type ProfileColumn, type Source } from "../api";
import { compact, count, percent } from "../format";

const SIZE = 50;

/** The profile's role, shortened for the column head, with the canvas's colour per kind. */
const KIND: Record<string, { short: string; colour: string }> = {
  identifier: { short: "id", colour: "#8A7BB8" },
  numeric: { short: "num", colour: "#6E8FD8" },
  discrete_numeric: { short: "num", colour: "#6E8FD8" },
  datetime: { short: "date", colour: "#C9873F" },
  binary: { short: "bool", colour: "#1FB894" },
  categorical: { short: "cat", colour: "#7E8A86" },
  text: { short: "text", colour: "#7E8A86" },
  constant: { short: "const", colour: "#667370" },
  empty: { short: "empty", colour: "#667370" },
};

function kindOf(column: ProfileColumn | undefined) {
  if (!column) return { short: "—", colour: "#667370" };
  return KIND[column.role] ?? { short: column.role.slice(0, 5), colour: "#7E8A86" };
}

/** The strip above the table: five facts about the file, all from the profile. */
function stats(source: Source): [string, string][] {
  const dataset = source.profile?.dataset;
  const columns = Object.values(source.profile?.columns ?? {});
  const withNulls = columns.filter((column) => column.missing_pct > 0).length;
  const identifiers = columns.filter((column) => column.role === "identifier").length;
  return [
    ["rows", count(source.rows)],
    ["columns", count(source.columns)],
    ["missing values", dataset ? percent(dataset.total_missing_pct) : "—"],
    ["columns w/ nulls", columns.length ? `${withNulls} of ${columns.length}` : "—"],
    ["identifier-like", columns.length ? String(identifiers) : "—"],
  ];
}

export function Preview({ ws, source, onSchema }: { ws: string; source: Source; onSchema?: () => void }) {
  const [page, setPage] = useState(1);
  const ready = source.status === "ready" || source.status === "profiling";
  const rows = useRows(ws, source.id, page, SIZE, ready);

  const total = source.rows ?? 0;
  const pages = Math.max(Math.ceil(total / SIZE), 1);
  const first = (page - 1) * SIZE + 1;
  const last = Math.min(page * SIZE, total || page * SIZE);
  const profile = source.profile?.columns ?? {};

  return (
    <div style={{ padding: "34px 40px 0", display: "flex", flexDirection: "column", height: "100%" }}>
      <div style={{ display: "flex", flexWrap: "wrap", alignItems: "flex-end", justifyContent: "space-between",
                    gap: "16px 24px", marginBottom: 18, flex: "0 0 auto" }}>
        <div style={{ minWidth: 0 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 9, marginBottom: 5 }}>
            <div style={{ fontSize: 27, letterSpacing: "-0.015em", color: "var(--bright)" }}>{source.name}</div>
            <div className="mono" style={{ fontSize: 11, color: "var(--accent)", border: "1px solid #1B3F36",
                                           borderRadius: 3, padding: "2px 6px" }}>csv</div>
          </div>
          <div className="mono" style={{ fontSize: 12, color: "var(--dim)" }}>
            {count(source.rows)} rows · {count(source.columns)} columns ·{" "}
            {source.profile ? "profiled on every row" : "profiling…"}
          </div>
        </div>
        <div style={{ display: "flex", gap: 12 }}>
          <button className="btn ghost" disabled title="Filtering arrives with the schema screen">Filter</button>
          <button className="btn ghost" disabled={!onSchema} onClick={onSchema}
                  title={onSchema ? undefined : "The schema screen arrives with 4c"}>Edit schema</button>
        </div>
      </div>

      <div style={{ display: "flex", border: "1px solid var(--line)", borderRadius: "6px 6px 0 0",
                    borderBottom: "none", background: "var(--panel)", overflowX: "auto", flex: "0 0 auto" }}>
        {stats(source).map(([label, value]) => (
          <div key={label} style={{ padding: "15px 24px", borderRight: "1px solid #1E2523", flex: "0 0 auto" }}>
            <div className="label" style={{ color: "var(--faint)", marginBottom: 4, whiteSpace: "nowrap" }}>{label}</div>
            <div className="mono" style={{ fontSize: 14, color: "#D8E0DD", whiteSpace: "nowrap" }}>{value}</div>
          </div>
        ))}
        <div className="mono" style={{ flex: "1 0 auto", display: "flex", alignItems: "center",
                                       justifyContent: "flex-end", padding: "0 18px", fontSize: 11.5,
                                       color: "var(--dim)", whiteSpace: "nowrap" }}>
          {total ? `showing rows ${count(first)}–${count(last)} of ${count(total)}` : "counting rows…"}
        </div>
      </div>

      <div style={{ flex: 1, minHeight: 0, border: "1px solid var(--line)", borderRadius: "0 0 6px 6px",
                    background: "var(--card)", overflow: "auto" }}>
        {rows.error && <div className="note" style={{ margin: 20 }}>{(rows.error as Error).message}</div>}
        {!rows.data && !rows.error && (
          <div className="mono" style={{ padding: 24, fontSize: 12.5, color: "var(--faint)" }}>
            {ready ? "reading the file…" : "the rows appear once the file has been checked"}
          </div>
        )}
        {rows.data && (
          <table className="grid">
            <thead>
              <tr>
                <th className="num-head">#</th>
                {rows.data.columns.map((name) => {
                  const column = profile[name];
                  const kind = kindOf(column);
                  const filled = column ? 100 - column.missing_pct : 100;
                  return (
                    <th key={name}>
                      <div style={{ display: "flex", alignItems: "center", gap: 7 }}>
                        <div style={{ color: "#D8E0DD" }}>{name}</div>
                        <div style={{ fontSize: 10, letterSpacing: "0.1em", color: kind.colour,
                                      textTransform: "uppercase" }}>{kind.short}</div>
                      </div>
                      {/* how much of the column is filled in, the canvas's bar under each head */}
                      <div style={{ height: 3, marginTop: 5, background: "var(--line)", borderRadius: 2,
                                    overflow: "hidden" }}>
                        <div style={{ height: 3, background: "#2E5C51", width: `${filled}%` }} />
                      </div>
                    </th>
                  );
                })}
              </tr>
            </thead>
            <tbody>
              {rows.data.rows.map((row, index) => (
                <tr key={first + index}>
                  <td className="num-cell">{count(first + index)}</td>
                  {row.map((cell, column) => (
                    <td key={column} style={{ color: cell === null ? "var(--faint)" : "#C8D2CE" }}>
                      {cell === null ? "—" : cell}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      <div className="mono" style={{ flex: "0 0 auto", padding: "10px 0 14px", display: "flex", alignItems: "center",
                                     gap: 14, fontSize: 11.5, color: "var(--dim)" }}>
        <button className="pager" disabled={page <= 1 || rows.isFetching} onClick={() => setPage(page - 1)}>
          ← prev
        </button>
        <button className="pager" disabled={page >= pages || rows.isFetching} onClick={() => setPage(page + 1)}>
          next →
        </button>
        <div>page {count(page)} of {count(pages)}</div>
        <div style={{ marginLeft: "auto" }}>
          {rows.isFetching ? "reading…" : `${compact(SIZE)} rows a page · read straight from the CSV`}
        </div>
      </div>
    </div>
  );
}
