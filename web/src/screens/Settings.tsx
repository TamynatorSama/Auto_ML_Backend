import { useState } from "react";
import { useQueryClient } from "@tanstack/react-query";

import { api, type Me, type Workspace } from "../api";
import { Note } from "../ui";

const TABS = [
  { id: "account", label: "Account", meta: "name · email" },
  { id: "workspace", label: "Workspace", meta: "name · address" },
  { id: "members", label: "Members & access", meta: "invites later" },
  { id: "compute", label: "Compute", meta: "your own host later" },
  { id: "data", label: "Data & retention", meta: "7 days" },
] as const;

const LATER = new Set(["members", "compute", "data"]);

function Row({ label, hint, children }: { label: string; hint: string; children: React.ReactNode }) {
  return (
    <div>
      <div className="label" style={{ color: "var(--faint)", marginBottom: 7 }}>{label}</div>
      {children}
      <div className="mono" style={{ fontSize: 10.5, color: "var(--faint)", marginTop: 6 }}>{hint}</div>
    </div>
  );
}

const box = { width: "100%", fontSize: 13, color: "var(--text)", background: "var(--bg)",
              border: "1px solid var(--line)", borderRadius: 4, padding: "9px 11px", outline: "none",
              fontFamily: "inherit" } as const;

/** A name the user can change, saved on its own button. */
function Name({ value, save, hint, label }: {
  value: string;
  save: (name: string) => Promise<unknown>;
  hint: string;
  label: string;
}) {
  const [name, setName] = useState(value);
  const [state, setState] = useState({ error: "", saved: false, busy: false });

  const submit = async () => {
    setState({ error: "", saved: false, busy: true });
    try {
      await save(name);
      setState({ error: "", saved: true, busy: false });
    } catch (problem) {
      setState({ error: (problem as Error).message, saved: false, busy: false });
    }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 10, maxWidth: 520 }}>
      <Row label={label} hint={hint}>
        <input style={box} value={name} onChange={(event) => setName(event.target.value)} />
      </Row>
      <Note>{state.error}</Note>
      <div>
        <button className="quiet" onClick={submit} disabled={state.busy || name === value}>
          {state.busy ? "Saving…" : state.saved ? "Saved" : "Save"}
        </button>
      </div>
    </div>
  );
}

function Locked({ text }: { text: string }) {
  return (
    <div style={{ border: "1px solid var(--line)", borderRadius: 6, background: "var(--card)", padding: "16px 18px",
                  display: "flex", alignItems: "center", gap: 14 }}>
      <div style={{ fontSize: 13.5, color: "var(--faint)" }}>{text}</div>
      <div className="mono" style={{ marginLeft: "auto", fontSize: 10.5, color: "var(--faint)",
                                     border: "1px solid var(--edge)", borderRadius: 3, padding: "2px 6px" }}>
        coming soon
      </div>
    </div>
  );
}

export function Settings({ me, workspace, onClose }: { me: Me; workspace: Workspace; onClose: () => void }) {
  const [tab, setTab] = useState<string>("account");
  const queries = useQueryClient();

  const body = {
    account: (
      <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
        <Name label="Full name" hint="shown to anyone you work with" value={me.user.name}
              save={async (name) => {
                queries.setQueryData(["me"], await api<Me>("/api/me", { method: "PATCH", body: { name } }));
              }} />
        <div className="rule" />
        <Row label="Email" hint={me.user.has_password ? "sign in with this address or a provider" : "signed in through Google or GitHub"}>
          <input style={{ ...box, color: "var(--faint)" }} value={me.user.email} readOnly />
        </Row>
      </div>
    ),
    workspace: (
      <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
        <Name label="Workspace name" hint="visible to everyone in the workspace" value={workspace.name}
              save={async (name) => {
                await api(`/api/w/${workspace.slug}`, { method: "PATCH", body: { name } });
                await queries.invalidateQueries({ queryKey: ["workspace", workspace.slug] });
                await queries.invalidateQueries({ queryKey: ["me"] });
              }} />
        <div className="rule" />
        <Row label="Address" hint="the address can't change while links point at it">
          <input style={{ ...box, color: "var(--faint)", fontFamily: "var(--mono)" }}
                 value={`${workspace.slug}.automl.io`} readOnly />
        </Row>
        <div className="rule" />
        <div className="label" style={{ color: "var(--faint)" }}>Model providers</div>
        <Locked text="Your own model key, used for every job in this workspace." />
      </div>
    ),
    members: (
      <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
        <div style={{ fontSize: 13.5, color: "var(--muted)", maxWidth: 460 }}>
          One person per workspace for now. Invitations, roles and shared access come after the beta.
        </div>
        <div style={{ border: "1px solid var(--line)", borderRadius: 6, background: "var(--card)", padding: "13px 16px",
                      display: "flex", alignItems: "center", gap: 14 }}>
          <div>
            <div style={{ fontSize: 13.5, color: "var(--text)" }}>{me.user.name}</div>
            <div className="mono" style={{ fontSize: 10.5, color: "var(--faint)", marginTop: 3 }}>{me.user.email}</div>
          </div>
          <div className="mono" style={{ marginLeft: "auto", fontSize: 10.5, color: "var(--muted)",
                                         border: "1px solid var(--edge)", borderRadius: 3, padding: "3px 8px" }}>
            {workspace.role}
          </div>
        </div>
        <Locked text="Invite someone to this workspace." />
      </div>
    ),
    compute: (
      <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
        <div style={{ fontSize: 13.5, color: "var(--muted)", maxWidth: 620 }}>
          Jobs run in the platform's sandbox for now. Later you can start the sandbox on a machine of your own and
          paste its link here; data and artifacts then stay on that machine.
        </div>
        <Locked text="Link a sandbox host of your own." />
      </div>
    ),
    data: (
      <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
        <div style={{ fontSize: 13.5, color: "var(--muted)", maxWidth: 620 }}>
          Uploads and a job's files are kept for 7 days after it finishes, then swept. Results and scores stay.
        </div>
        <Locked text="Choose how long files are kept." />
      </div>
    ),
  }[tab];

  return (
    <div style={{ position: "fixed", inset: 0, zIndex: 60, background: "rgba(6,9,8,0.72)", display: "flex",
                  alignItems: "center", justifyContent: "center", padding: 40 }}>
      <div style={{ position: "absolute", inset: 0 }} onClick={onClose} />
      <div style={{ position: "relative", width: "100%", maxWidth: 940, height: 640, maxHeight: "100%", display: "flex",
                    flexDirection: "column", border: "1px solid var(--edge)", borderRadius: 8, background: "#0F1413",
                    boxShadow: "0 30px 80px rgba(0,0,0,0.6)", overflow: "hidden" }}>
        <div style={{ padding: "22px 28px 18px", borderBottom: "1px solid var(--line)", display: "flex",
                      alignItems: "flex-start", justifyContent: "space-between", gap: 20 }}>
          <div>
            <div className="label" style={{ color: "var(--faint)", marginBottom: 6 }}>
              Settings · {workspace.slug} workspace
            </div>
            <div style={{ fontSize: 19, letterSpacing: "-0.01em", color: "var(--bright)" }}>
              {TABS.find((entry) => entry.id === tab)?.label}
            </div>
          </div>
          <div className="mono" style={{ fontSize: 14, color: "var(--dim)", cursor: "pointer", padding: "2px 6px" }}
               onClick={onClose}>×</div>
        </div>

        <div style={{ display: "flex", alignItems: "stretch", minHeight: 0, flex: 1 }}>
          <div style={{ width: 208, flex: "0 0 208px", borderRight: "1px solid var(--line)", background: "#0D1110",
                        padding: "14px 10px", display: "flex", flexDirection: "column", gap: 2 }}>
            {TABS.map((entry) => {
              const on = entry.id === tab;
              const later = LATER.has(entry.id);
              return (
                <div key={entry.id} onClick={() => setTab(entry.id)}
                     style={{ padding: "10px 12px", borderRadius: 5, cursor: "pointer",
                              background: on ? "#161D1B" : "transparent" }}>
                  <div style={{ fontSize: 13.5, color: on ? "var(--text)" : later ? "var(--faint)" : "var(--muted)" }}>
                    {entry.label}
                  </div>
                  <div className="mono" style={{ fontSize: 10.5, color: "var(--faint)", marginTop: 3 }}>{entry.meta}</div>
                </div>
              );
            })}
          </div>
          <div style={{ flex: 1, minWidth: 0, padding: "20px 28px 26px", overflowY: "auto" }}>{body}</div>
        </div>

        <div style={{ borderTop: "1px solid var(--line)", padding: "14px 28px", display: "flex", justifyContent: "flex-end" }}>
          <button className="button" style={{ padding: "9px 20px" }} onClick={onClose}>Done</button>
        </div>
      </div>
    </div>
  );
}
