import { useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useNavigate, useParams } from "react-router-dom";

import { api, ApiError, type Me, type Workspace } from "../api";
import { Logo } from "../ui";
import { Settings } from "./Settings";

const RAIL = ["Data", "Preview", "Schema", "Configure", "Train", "Results"];

/** The setup checklist from the design (G46); later phases light the rest up. */
function checklist(workspace: Workspace) {
  return [
    { label: "Workspace created", meta: "done", done: true },
    { label: "Add a model key", meta: "soon", done: false },
    { label: "Connect a source", meta: "soon", done: false },
    { label: "Lock a schema", meta: "—", done: false },
    { label: "Run your first job", meta: "—", done: false },
  ].map((item) => ({ ...item, key: `${workspace.slug}:${item.label}` }));
}

function initials(name: string) {
  return name.split(/\s+/).filter(Boolean).slice(0, 2).map((part) => part[0]?.toUpperCase()).join("") || "?";
}

function Sidebar({ me, workspace, onSettings }: { me: Me; workspace: Workspace; onSettings: () => void }) {
  const [menu, setMenu] = useState(false);
  const [leaving, setLeaving] = useState(false);
  const navigate = useNavigate();
  const queries = useQueryClient();

  const logOut = async () => {
    setMenu(false);          // the menu goes at once; the round trip to the database is behind it
    setLeaving(true);
    try {
      await api("/api/auth/logout", { method: "POST" });
    } finally {
      queries.setQueryData(["me"], null);
      navigate("/login", { replace: true });
    }
  };

  return (
    <div style={{ width: 292, flex: "0 0 292px", borderRight: "1px solid var(--line)", background: "var(--panel)",
                  display: "flex", flexDirection: "column", overflow: "hidden" }}>
      <div style={{ padding: "20px 20px 18px", borderBottom: "1px solid var(--line)" }}>
        <Logo />
      </div>

      <div style={{ flex: 1, padding: "14px 0 20px", overflowY: "auto" }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "0 18px 12px" }}>
          <div className="label">Workspaces</div>
          <div className="mono" style={{ fontSize: 12, color: "var(--faint)" }} title="More workspaces come with invites">+</div>
        </div>

        <div style={{ margin: "0 18px", border: "1px dashed var(--edge)", borderRadius: 5, padding: "16px 15px",
                      display: "flex", flexDirection: "column", gap: 10 }}>
          <div className="mono" style={{ fontSize: 12, color: "var(--dim)" }}>no sources yet</div>
          <div style={{ fontSize: 12.5, lineHeight: 1.55, color: "var(--faint)" }}>
            Connect one and its schema and job slots appear here.
          </div>
        </div>

        <div className="rule" style={{ margin: "18px 18px 14px" }} />
        <div className="label" style={{ padding: "0 18px 8px" }}>Setup</div>
        {checklist(workspace).map((item) => (
          <div key={item.key} style={{ display: "flex", alignItems: "center", gap: 11, padding: "8px 20px" }}>
            <div style={{ width: 14, height: 14, borderRadius: "50%", flex: "0 0 14px",
                          border: `1px solid ${item.done ? "var(--accent)" : "var(--edge)"}`,
                          background: item.done ? "var(--accent)" : "transparent" }} />
            <div style={{ fontSize: 13, color: item.done ? "var(--muted)" : "var(--faint)" }}>{item.label}</div>
            <div className="mono" style={{ marginLeft: "auto", fontSize: 10.5, color: "var(--faint)" }}>{item.meta}</div>
          </div>
        ))}
      </div>

      <div style={{ borderTop: "1px solid var(--line)", position: "relative", flex: "0 0 auto" }}>
        {menu && (
          <>
            <div style={{ position: "fixed", inset: 0, zIndex: 40 }} onClick={() => setMenu(false)} />
            <div style={{ position: "absolute", left: 16, right: 16, bottom: 62, zIndex: 41, padding: 6,
                          border: "1px solid var(--edge)", borderRadius: 6, background: "#141918",
                          boxShadow: "0 18px 40px rgba(0,0,0,0.55)", display: "flex", flexDirection: "column", gap: 1 }}>
              <div className="menu-item" style={{ display: "flex", justifyContent: "space-between", gap: 12,
                                                 fontSize: 13, color: "#D8E0DD" }}
                   onClick={() => { setMenu(false); onSettings(); }}>
                <div>Settings</div>
                <div className="mono" style={{ fontSize: 10.5, color: "var(--faint)" }}>profile · workspace</div>
              </div>
              <div className="rule" style={{ margin: "5px 4px 6px", background: "var(--line)" }} />
              <div className="menu-item" style={{ fontSize: 13, color: "var(--danger)" }} onClick={logOut}>
                Log out
              </div>
            </div>
          </>
        )}
        <div style={{ padding: "16px 24px", display: "flex", alignItems: "center", gap: 9, cursor: "pointer" }}
             onClick={() => setMenu(!menu)}>
          <div className="mono" style={{ width: 24, height: 24, borderRadius: "50%", background: "#1B3F36",
                                         color: "var(--accent-bright)", fontSize: 11, display: "flex",
                                         alignItems: "center", justifyContent: "center", flex: "0 0 24px" }}>
            {initials(me.user.name)}
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 1, minWidth: 0 }}>
            <div style={{ fontSize: 13, color: "#C8D2CE" }}>{me.user.name}</div>
            <div className="mono" style={{ fontSize: 10.5, color: "var(--dim)" }}>
              {leaving ? "logging out…" : `${workspace.slug} · new workspace`}
            </div>
          </div>
          <div className="mono" style={{ marginLeft: "auto", fontSize: 11, color: "var(--faint)" }}>{menu ? "▾" : "▴"}</div>
        </div>
      </div>
    </div>
  );
}

function Empty() {
  const soon = { border: "1px solid var(--line)", borderRadius: 6, background: "var(--card)", padding: "20px 22px",
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
            <div key={name} style={soon}>
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
  const [settings, setSettings] = useState(false);
  const workspace = useQuery({
    queryKey: ["workspace", ws],
    retry: false,
    queryFn: () => api<Workspace>(`/api/w/${encodeURIComponent(ws)}`),
  });

  if (workspace.isPending) return <div style={{ padding: 40, color: "var(--faint)" }} className="mono">loading…</div>;
  if (workspace.error) {
    const missing = workspace.error instanceof ApiError && workspace.error.status === 404;
    return (
      <div style={{ padding: 40, color: "var(--muted)" }}>
        {missing ? "No such workspace, or you're not a member of it." : (workspace.error as Error).message}
      </div>
    );
  }

  return (
    <div style={{ display: "flex", height: "100%" }}>
      <Sidebar me={me} workspace={workspace.data} onSettings={() => setSettings(true)} />
      <div style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column", overflow: "hidden" }}>
        <div style={{ borderBottom: "1px solid var(--line)", background: "var(--panel)", padding: "0 40px", height: 64,
                      display: "flex", alignItems: "center", justifyContent: "space-between", gap: 20, flex: "0 0 auto" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 3, overflowX: "auto" }}>
            {RAIL.map((label, index) => (
              <div key={label} style={{ display: "flex", alignItems: "center", gap: 7, padding: "7px 10px",
                                        borderRadius: 4, whiteSpace: "nowrap" }}>
                <div className="mono" style={{ width: 16, height: 16, borderRadius: "50%", fontSize: 10, display: "flex",
                                               alignItems: "center", justifyContent: "center",
                                               background: index === 0 ? "var(--accent)" : "transparent",
                                               color: index === 0 ? "#04140F" : "var(--faint)",
                                               border: `1px solid ${index === 0 ? "var(--accent)" : "var(--edge)"}` }}>
                  {index + 1}
                </div>
                <div style={{ fontSize: 13, color: index === 0 ? "var(--text)" : "var(--faint)" }}>{label}</div>
              </div>
            ))}
          </div>
          <div style={{ padding: "6px 14px", borderRadius: 4, border: "1px solid #2E3634", color: "var(--faint)",
                        fontSize: 13, flex: "0 0 auto" }}>New job</div>
        </div>
        <div style={{ flex: 1, overflowY: "auto" }}><Empty /></div>
      </div>
      {settings && <Settings me={me} workspace={workspace.data} onClose={() => setSettings(false)} />}
    </div>
  );
}
