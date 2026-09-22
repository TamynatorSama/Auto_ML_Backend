import { useRef, useState, type ReactNode } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";

import { deleteSource, loadSample, uploadCsv, WORKING, type Job, type Source } from "../api";
import { bytes, compact, count, percent, until, when } from "../format";

/** What the card says while a source is still being read (§5). */
const WORKING_ON: Record<string, string> = {
  uploading: "waiting for the file",
  checking: "checking it parses, counting rows",
  profiling: "profiling every column",
};

/** The sources that follow the file upload (D3, G6). */
const LATER = [["Snowflake", "warehouse"], ["BigQuery", "warehouse"], ["S3", "object storage"],
               ["Postgres", "database"]];

const card = { border: "1px solid var(--line)", borderRadius: 6, background: "var(--card)" } as const;

function Tile({ label, value, note, tone }: { label: string; value: string; note?: string; tone?: string }) {
  return (
    <div style={{ ...card, padding: "20px 22px" }}>
      <div className="label" style={{ color: "var(--faint)", marginBottom: 8 }}>{label}</div>
      <div className="mono" style={{ fontSize: 20, color: tone ?? "var(--text)" }}>{value}</div>
      {note && <div style={{ fontSize: 12.5, color: "var(--dim)", marginTop: 4 }}>{note}</div>}
    </div>
  );
}

function Panel({ title, chip, children, footer }: {
  title: string; chip?: ReactNode; children: ReactNode; footer?: ReactNode;
}) {
  return (
    <div style={{ ...card, flex: "1 1 380px", minWidth: 0, padding: "22px 24px" }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 16 }}>
        <div className="label" style={{ color: "var(--faint)" }}>{title}</div>
        {chip}
      </div>
      {children}
      {footer}
    </div>
  );
}

function Row({ left, right }: { left: string; right: string }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", gap: 16, fontSize: 13.5, color: "var(--muted)" }}>
      <div>{left}</div>
      <div className="mono" style={{ color: "#D8E0DD", textAlign: "right" }}>{right}</div>
    </div>
  );
}

/** The drop zone, the four sources that aren't here yet, and the sample. */
function Drop({ ws, busy, onDone, replacing }: {
  ws: string; busy: boolean; onDone: (source: Source) => void; replacing: boolean;
}) {
  const input = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);
  const [error, setError] = useState("");
  const queries = useQueryClient();

  const done = (source: Source) => {
    queries.invalidateQueries({ queryKey: ["sources", ws] });
    onDone(source);
  };
  const send = useMutation({
    mutationFn: (file: File) => uploadCsv(ws, file),
    onSuccess: done,
    onError: (problem: Error) => setError(problem.message),
  });
  const sample = useMutation({
    mutationFn: () => loadSample(ws),
    onSuccess: done,
    onError: (problem: Error) => setError(problem.message),
  });

  const take = (file: File | undefined) => {
    setError("");
    if (!file) return;
    if (!/\.csv$/i.test(file.name)) return setError("Only .csv files for now");
    send.mutate(file);
  };
  const working = busy || send.isPending || sample.isPending;

  return (
    <>
      <div onDragOver={(event) => { event.preventDefault(); setOver(true); }}
           onDragLeave={() => setOver(false)}
           onDrop={(event) => { event.preventDefault(); setOver(false); take(event.dataTransfer.files[0]); }}
           style={{ border: `1px dashed ${over ? "var(--accent)" : "var(--edge)"}`, borderRadius: 6,
                    background: over ? "#101a18" : "var(--panel)", padding: "24px 26px", display: "flex",
                    flexWrap: "wrap", alignItems: "center", gap: 20 }}>
        <div className="mono" style={{ width: 40, height: 40, flex: "0 0 40px", border: "1px solid var(--edge)",
                                       borderRadius: 4, display: "flex", alignItems: "center",
                                       justifyContent: "center", fontSize: 15, color: "var(--accent)" }}>↑</div>
        <div style={{ flex: "1 1 300px", minWidth: 0 }}>
          <div style={{ fontSize: 14.5, color: "var(--text)", marginBottom: 4 }}>
            {send.isPending ? "Sending the file…"
              : sample.isPending ? "Copying the sample…"
                : replacing ? "Add another source, or load the sample"
                  : "Drop a .csv here, or browse for one"}
          </div>
          <div className="mono" style={{ fontSize: 11.5, color: "var(--dim)" }}>
            csv · up to 100 MB · profiled on every row, no sampling
          </div>
        </div>
        <div style={{ display: "flex", gap: 12 }}>
          <button className="btn ghost" disabled={working} onClick={() => sample.mutate()}>
            Load the housing sample
          </button>
          <button className="btn" disabled={working} onClick={() => input.current?.click()}>Browse</button>
        </div>
        <input ref={input} type="file" accept=".csv,text/csv" hidden
               onChange={(event) => { take(event.target.files?.[0]); event.target.value = ""; }} />
      </div>

      {error && <div className="note" style={{ marginTop: 12 }}>{error}</div>}

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(190px, 1fr))", gap: 12,
                    marginTop: 14 }}>
        {LATER.map(([name, kind]) => (
          <div key={name} style={{ ...card, background: "var(--panel)", padding: "14px 16px", opacity: 0.55,
                                   display: "flex", alignItems: "center", justifyContent: "space-between", gap: 10 }}
               title="Warehouses and object storage follow the file upload">
            <div>
              <div className="mono" style={{ fontSize: 13, color: "var(--dim)" }}>{name}</div>
              <div style={{ fontSize: 12, color: "var(--faint)", marginTop: 3 }}>{kind}</div>
            </div>
            <div className="mono" style={{ fontSize: 10.5, color: "var(--faint)", border: "1px solid var(--edge)",
                                           borderRadius: 3, padding: "2px 6px" }}>soon</div>
          </div>
        ))}
      </div>
    </>
  );
}

export function Data({ ws, source, job, first, onPreview, onPick, onSchema, onJob }: {
  ws: string;
  source: Source | null;
  /** this source's newest run, if it has one */
  job: Job | null;
  /** the workspace has no datasets at all, so this is the first one */
  first: boolean;
  onPreview: (id: number) => void;
  onPick: (id: number | null) => void;
  onSchema: (id: number) => void;
  onJob: (id: number) => void;
}) {
  const queries = useQueryClient();
  const remove = useMutation({
    mutationFn: (id: number) => deleteSource(ws, id),
    // the refusal is the useful part: a dataset with runs says to delete those first,
    // and saying nothing reads as the button being broken
    onSuccess: () => { queries.invalidateQueries({ queryKey: ["sources", ws] }); onPick(null); },
  });

  if (!source) {
    return (
      <div className="page-body" style={{ padding: "48px 40px 84px", maxWidth: 1100 }}>
        <div style={{ maxWidth: 560, marginBottom: 30 }}>
          <div className="label" style={{ color: "var(--faint)", marginBottom: 12 }}>Step 1 · data source</div>
          <div style={{ fontSize: 27, letterSpacing: "-0.015em", color: "var(--bright)", marginBottom: 12 }}>
            {first ? "No data connected yet" : "Add a dataset"}
          </div>
          <div style={{ fontSize: 14.5, lineHeight: 1.6, color: "var(--muted)" }}>
            Every job starts from one table. Drop a CSV in and it is checked, counted and profiled column by
            column — then you correct its schema and lock a version.
            {first ? "" : " The datasets you already have are in the rail; this one joins them."}
          </div>
        </div>
        <Drop ws={ws} busy={false} replacing={false} onDone={(made) => onPick(made.id)} />
      </div>
    );
  }

  const working = WORKING.includes(source.status);
  const dataset = source.profile?.dataset;
  const failed = source.status === "failed";
  const expired = source.status === "expired";

  return (
    <div className="page-body" style={{ padding: "40px 40px 84px", maxWidth: 1100 }}>
      <div style={{ display: "flex", flexWrap: "wrap", alignItems: "flex-end", justifyContent: "space-between",
                    gap: "16px 24px", marginBottom: 28 }}>
        <div style={{ minWidth: 0 }}>
          <div className="label" style={{ color: "var(--faint)", marginBottom: 9 }}>Active data source</div>
          <div style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap", marginBottom: 7 }}>
            <div className="mono responsive-title" style={{ fontSize: 26, letterSpacing: "-0.01em", color: "var(--bright)" }}>
              {source.name}
            </div>
            <div className="mono" style={{ fontSize: 11, color: "var(--accent)", border: "1px solid #1B3F36",
                                           borderRadius: 3, padding: "3px 7px" }}>csv</div>
          </div>
          <div style={{ fontSize: 14, color: "var(--muted)" }}>
            {source.original_name ?? "uploaded"} · added {when(source.created_at)}
            {source.files_expire_at && !expired ? ` · file kept ${until(source.files_expire_at)}` : ""}
          </div>
        </div>
        <div style={{ display: "flex", gap: 12 }}>
          <button className="btn ghost" disabled={working || remove.isPending}
                  onClick={() => remove.mutate(source.id)}>
            {remove.isPending ? "Deleting…" : "Delete"}
          </button>
          <button className="btn" disabled={working || expired || failed} onClick={() => onPreview(source.id)}>
            View all rows
          </button>
        </div>
      </div>

      {remove.error && (
        <div className="note" style={{ marginBottom: 14 }}>{(remove.error as Error).message}</div>
      )}

      {working && (
        <div style={{ ...card, background: "var(--panel)", padding: "16px 20px", marginBottom: 14,
                      display: "flex", alignItems: "center", gap: 12 }}>
          <div className="pulse" />
          <div style={{ fontSize: 13.5, color: "var(--muted)" }}>{WORKING_ON[source.status] ?? "working"}</div>
          <div className="mono" style={{ marginLeft: "auto", fontSize: 11.5, color: "var(--faint)" }}>
            {source.status}
          </div>
        </div>
      )}
      {failed && (
        <div className="note" style={{ marginBottom: 14 }}>
          {source.error ?? "This file couldn't be read."} Delete it and try another.
        </div>
      )}
      {expired && (
        <div className="note" style={{ marginBottom: 14 }}>
          The file was removed seven days after it was last used. Its profile and schemas are still here; upload
          the CSV again to run anything new against it.
        </div>
      )}

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: 14,
                    marginBottom: 14 }}>
        <Tile label="rows" value={compact(source.rows)} note={`${count(source.rows)} exact`} />
        <Tile label="columns" value={count(source.columns)}
              note={dataset ? `${percent(dataset.total_missing_pct)} of values missing` : "counting"} />
        <Tile label="on disk" value={bytes(source.bytes)} note="as uploaded" />
        <Tile label="duplicate rows" value={dataset ? count(dataset.duplicate_rows) : "—"}
              note={dataset ? "identical across every column" : "after profiling"} />
        <Tile label="profiled" value={dataset ? "every row" : "—"}
              tone={dataset ? "var(--accent-bright)" : undefined}
              note={dataset ? "no sampling" : "after profiling"} />
      </div>

      <div style={{ display: "flex", flexWrap: "wrap", gap: 14, alignItems: "stretch" }}>
        <Panel title="Schema"
               chip={<div className="mono" style={{ fontSize: 11, borderRadius: 3, padding: "2px 7px",
                                                    color: source.schema?.status === "locked" ? "var(--accent)" : "#C9873F",
                                                    border: `1px solid ${source.schema?.status === "locked" ? "#1B3F36" : "#3E3122"}` }}>
                       {source.schema ? source.schema.status : "none yet"}
                     </div>}
               footer={<button className="btn ghost" disabled={!source.schema}
                               onClick={() => onSchema(source.id)}
                               style={{ width: "100%", marginTop: 18 }}>Open schema</button>}>
          <div style={{ display: "flex", flexDirection: "column", gap: 11 }}>
            <Row left="Version" right={source.schema ? `v${source.schema.version}` : "—"} />
            <Row left="Columns described"
                 right={source.schema && source.columns ? `${source.schema.described} of ${source.columns}` : "—"} />
            <Row left="Target" right={source.schema?.target ?? "not set"} />
            <Row left="Locked" right={source.schema?.status === "locked" ? "yes" : "not yet"} />
          </div>
        </Panel>

        <Panel title="Job" chip={<div className="mono" style={{ fontSize: 11, color: "var(--dim)" }}>one per source</div>}
               footer={<button className="btn ghost" disabled={!job} onClick={() => job && onJob(job.id)}
                               title={job ? undefined : "Lock a schema, then press New job"}
                               style={{ width: "100%", marginTop: 18 }}>
                         {job ? "Open this run" : "Open job configuration"}
                       </button>}>
          {job ? (
            <div style={{ display: "flex", flexDirection: "column", gap: 11 }}>
              <Row left="Latest run" right={`#${job.id}`} />
              <Row left="Status" right={job.status} />
              <Row left="Started" right={job.started_at ? when(job.started_at) : "not yet"} />
              <Row left="Spend" right={job.usage_totals?.cost_usd
                ? `$${job.usage_totals.cost_usd.toFixed(4)}` : "—"} />
            </div>
          ) : (
            <div style={{ border: "1px dashed var(--edge)", borderRadius: 4, padding: "13px 15px",
                          fontSize: 13, color: "var(--faint)", lineHeight: 1.55 }}>
              No run yet. Lock the schema, then press <span style={{ color: "var(--muted)" }}>New job</span>
              {" "}up in the corner.
            </div>
          )}
        </Panel>
      </div>

      <div style={{ marginTop: 14 }}>
        <Drop ws={ws} busy={working} replacing onDone={(made) => onPick(made.id)} />
      </div>
    </div>
  );
}
