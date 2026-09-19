import { useQuery } from "@tanstack/react-query";
import { useParams } from "react-router-dom";

import { api, ApiError, type Me, type Workspace } from "../api";
import { Shell, type Step } from "../shell/Shell";

/** The setup checklist from the design (G46); later phases light the rest up. */
const SETUP: Step[] = [
  { label: "Workspace created", meta: "done", done: true },
  { label: "Add a model key", meta: "soon", done: false },
  { label: "Connect a source", meta: "soon", done: false },
  { label: "Lock a schema", meta: "—", done: false },
  { label: "Run your first job", meta: "—", done: false },
];

function Empty() {
  const card = { border: "1px solid var(--line)", borderRadius: 6, background: "var(--card)", padding: "20px 22px",
                 display: "flex", flexDirection: "column", gap: 8 } as const;
  return (
    <div style={{ padding: "56px 40px 84px", maxWidth: 980 }}>
      <div style={{ maxWidth: 560, marginBottom: 34 }}>
        <div className="label" style={{ color: "var(--faint)", marginBottom: 12 }}>Step 1 · data source</div>
        <div style={{ fontSize: 27, letterSpacing: "-0.015em", color: "var(--bright)", marginBottom: 12 }}>
          No data connected yet
        </div>
        <div style={{ fontSize: 14.5, lineHeight: 1.6, color: "var(--muted)" }}>
          Every job starts from one table. Uploading a file, locking its schema and running a job arrive in the next
          releases; for now jobs are started from the command line.
        </div>
      </div>

      <div style={{ border: "1px dashed var(--edge)", borderRadius: 8, background: "var(--panel)", padding: 40,
                    textAlign: "center", marginBottom: 14 }}>
        <div className="mono" style={{ fontSize: 14, color: "var(--dim)", marginBottom: 8 }}>drop a .csv here</div>
        <div style={{ fontSize: 13.5, color: "var(--faint)" }}>up to 100 MB · schema inferred on upload, then locked</div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(230px, 1fr))", gap: 14,
                    marginBottom: 40 }}>
        {[["Model key", "yours", "Paste a key for the models the pipeline writes code with."],
          ["Warehouse", "later", "Snowflake, BigQuery and object storage follow the file upload."],
          ["Sandbox host", "later", "Point a workspace at a machine of your own and jobs run there."]].map(
          ([name, tag, text]) => (
            <div key={name} style={card}>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12 }}>
                <div className="mono" style={{ fontSize: 14, color: "var(--dim)" }}>{name}</div>
                <div className="mono" style={{ fontSize: 10.5, color: "var(--faint)", border: "1px solid var(--edge)",
                                               borderRadius: 3, padding: "2px 6px" }}>{tag}</div>
              </div>
              <div style={{ fontSize: 13, lineHeight: 1.5, color: "var(--faint)" }}>{text}</div>
            </div>
          ))}
      </div>

      <div className="rule" style={{ marginBottom: 28 }} />

      <div style={{ border: "1px solid var(--line)", borderRadius: 6, background: "var(--panel)", padding: "18px 20px",
                    maxWidth: 420 }}>
        <div className="label" style={{ color: "var(--faint)", marginBottom: 12 }}>What happens next</div>
        <div style={{ display: "flex", flexDirection: "column", gap: 9 }}>
          {[["01", "Types and roles are inferred, then you lock the schema."],
            ["02", "The plan is proposed for review before any model trains."],
            ["03", "One job per source, with the leaderboard and holdout scores."]].map(([num, text]) => (
            <div key={num} style={{ display: "flex", gap: 11, fontSize: 13, color: "var(--muted)" }}>
              <div className="mono" style={{ color: "var(--accent)", flex: "0 0 auto" }}>{num}</div>
              <div>{text}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export function Home({ me }: { me: Me }) {
  const { ws = "" } = useParams();
  const workspace = useQuery({
    queryKey: ["workspace", ws],
    retry: false,
    queryFn: () => api<Workspace>(`/api/w/${encodeURIComponent(ws)}`),
  });

  if (workspace.isPending) return <div className="mono" style={{ padding: 40, color: "var(--faint)" }}>loading…</div>;
  if (workspace.error) {
    const missing = workspace.error instanceof ApiError && workspace.error.status === 404;
    return (
      <div style={{ padding: 40, color: "var(--muted)" }}>
        {missing ? "No such workspace, or you're not a member of it." : (workspace.error as Error).message}
      </div>
    );
  }

  return (
    <Shell me={me} workspace={workspace.data} steps={SETUP} at="Data"
           action={<button className="btn" disabled title="A job needs a source first">New job</button>}>
      <Empty />
    </Shell>
  );
}
