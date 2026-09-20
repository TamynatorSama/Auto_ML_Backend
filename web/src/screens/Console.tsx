import { useQuery } from "@tanstack/react-query";
import { useNavigate, useParams } from "react-router-dom";

import { api, ApiError, useSource, useSources, type Me, type Source, type Workspace } from "../api";
import { Shell, type Source as Branch, type Step, type StepName } from "../shell/Shell";
import { Data } from "./Data";
import { Preview } from "./Preview";
import { Schema } from "./Schema";

/** The setup checklist (G46), against what the workspace actually has. */
function setup(sources: Source[]): Step[] {
  const locked = sources.some((source) => source.schema?.status === "locked");
  return [
    { label: "Workspace created", meta: "done", done: true },
    { label: "Add a model key", meta: "soon", done: false },
    { label: "Connect a source", meta: sources.length ? "done" : "now", done: sources.length > 0 },
    { label: "Lock a schema", meta: locked ? "done" : sources.length ? "next" : "—", done: locked },
    { label: "Run your first job", meta: "—", done: false },
  ];
}

const CHIP = { locked: "locked", draft: "draft" } as const;

/** A source as the sidebar's tree draws it. */
function branch(source: Source, active: boolean, go: () => void): Branch {
  const status = source.schema?.status;
  return {
    name: source.name,
    active,
    go,
    schema: {
      label: status ? `${CHIP[status]} · v${source.schema!.version}` : "no schema",
      tone: status ?? "none",
    },
  };
}

export function Console({ me, at }: { me: Me; at: "Data" | "Preview" | "Schema" }) {
  const { ws = "", sourceId } = useParams();
  const navigate = useNavigate();

  const workspace = useQuery({
    queryKey: ["workspace", ws],
    retry: false,
    queryFn: () => api<Workspace>(`/api/w/${encodeURIComponent(ws)}`),
  });
  const sources = useSources(ws);

  const listing = sources.data?.sources ?? [];
  // the source in the address, else the newest — the canvas's "active data source"
  const picked = sourceId ? Number(sourceId) : listing[0]?.id ?? null;
  const one = useSource(ws, picked);
  const source = one.data ?? listing.find((row) => row.id === picked) ?? null;

  const go = (id: number | null, screen: "" | "preview" | "schema" = "") =>
    navigate(id === null ? `/${ws}` : `/${ws}/s/${id}${screen ? `/${screen}` : ""}`);

  if (workspace.isPending) return <div className="mono" style={{ padding: 40, color: "var(--faint)" }}>loading…</div>;
  if (workspace.error) {
    const missing = workspace.error instanceof ApiError && workspace.error.status === 404;
    return (
      <div style={{ padding: 40, color: "var(--muted)" }}>
        {missing ? "No such workspace, or you're not a member of it." : (workspace.error as Error).message}
      </div>
    );
  }

  // Preview and Schema need a source; without one the address falls back to Data
  const showing: StepName = at !== "Data" && source ? at : "Data";
  const reached: StepName[] = source && source.status === "ready"
    ? ["Data", "Preview", "Schema"] : ["Data"];
  const step = (name: StepName) => (name === "Preview" ? "preview" : name === "Schema" ? "schema" : "");

  return (
    <Shell me={me} workspace={workspace.data} steps={setup(listing)} at={showing} reached={reached}
           sources={listing.map((row) => branch(row, row.id === picked, () => go(row.id)))}
           onStep={(name) => go(picked, step(name))}
           action={<button className="btn" disabled title="A job needs a locked schema first">New job</button>}>
      {showing === "Schema" && source ? <Schema ws={ws} source={source} />
        : showing === "Preview" && source
          ? <Preview ws={ws} source={source} onSchema={() => go(source.id, "schema")} />
          : <Data ws={ws} source={source} onPreview={(id) => go(id, "preview")} onPick={(id) => go(id)}
                  onSchema={(id) => go(id, "schema")} />}
    </Shell>
  );
}
