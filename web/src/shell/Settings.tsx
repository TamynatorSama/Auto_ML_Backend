import { useState, type ReactNode } from "react";
import { useQueryClient } from "@tanstack/react-query";

import { api, type Me, type Workspace } from "../api";
import { Note } from "../ui";

/**
 * The Settings modal from the Console canvas: five tabs on a rail, a footer that
 * sums up the tab. Anything a later phase owns is shown in its real place and
 * disabled, so the screen doesn't promise what the backend can't do yet.
 */

const TABS = [
  { id: "account", label: "Account", meta: "profile · sign-in" },
  { id: "workspace", label: "Workspace config", meta: "providers · defaults" },
  { id: "compute", label: "Compute", meta: "runners · limits" },
  { id: "members", label: "Members & access", meta: "1 person" },
  { id: "data", label: "Data & retention", meta: "privacy · storage" },
] as const;

type TabId = (typeof TABS)[number]["id"];

const box = { width: "100%", fontSize: 13, color: "var(--text)", background: "var(--bg)",
              border: "1px solid var(--line)", borderRadius: 4, padding: "9px 11px", outline: "none",
              fontFamily: "inherit" } as const;

const cardStyle = { border: "1px solid var(--line)", borderRadius: 6, background: "var(--card)",
                    padding: "16px 18px" } as const;

function Label({ children }: { children: ReactNode }) {
  return <div className="label" style={{ color: "var(--faint)", marginBottom: 8 }}>{children}</div>;
}

function Hint({ children }: { children: ReactNode }) {
  return <div className="mono" style={{ fontSize: 10.5, color: "var(--faint)", marginTop: 6 }}>{children}</div>;
}

function Lead({ children }: { children: ReactNode }) {
  return <div style={{ fontSize: 13.5, lineHeight: 1.6, color: "var(--muted)", maxWidth: 620,
                       marginBottom: 16 }}>{children}</div>;
}

function Chips({ options, chosen }: { options: { label: string; locked?: boolean }[]; chosen: string }) {
  return (
    <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
      {options.map((option) => {
        const on = option.label === chosen;
        return (
          <div key={option.label} className="mono"
               style={{ fontSize: 12, borderRadius: 4, padding: "8px 12px",
                        color: on ? "var(--accent-bright)" : "var(--faint)",
                        border: `1px solid ${on ? "#1B3F36" : "var(--line)"}`,
                        background: on ? "#0F1D19" : "var(--card)" }}>
            {option.label}{option.locked ? " · locked" : ""}
          </div>
        );
      })}
    </div>
  );
}

function Toggle({ on }: { on: boolean }) {
  return (
    <div style={{ marginLeft: "auto", flex: "0 0 auto", width: 34, height: 20, borderRadius: 10,
                  background: on ? "var(--accent)" : "#232A28", display: "flex", alignItems: "center" }}>
      <div style={{ width: 16, height: 16, borderRadius: 8, background: on ? "#04140F" : "var(--faint)",
                    marginLeft: on ? 16 : 2 }} />
    </div>
  );
}

function Switch({ title, meta, on }: { title: string; meta: string; on: boolean }) {
  return (
    <div style={{ ...cardStyle, padding: "14px 16px", display: "flex", alignItems: "center", gap: 14 }}>
      <div style={{ minWidth: 0 }}>
        <div style={{ fontSize: 13.5, color: "var(--faint)" }}>{title}</div>
        <div className="mono" style={{ fontSize: 10.5, color: "var(--faint)", marginTop: 3 }}>{meta}</div>
      </div>
      <Toggle on={on} />
    </div>
  );
}

/** A name the user can change, saved on its own button. */
function Name({ label, hint, value, save }: {
  label: string;
  hint: string;
  value: string;
  save: (name: string) => Promise<unknown>;
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
    <div style={{ marginBottom: 18 }}>
      <Label>{label}</Label>
      <div style={{ display: "flex", gap: 8, alignItems: "flex-start" }}>
        <input id={`settings-${label}`} style={box} value={name} onChange={(event) => setName(event.target.value)} />
        <button className="btn ghost" style={{ padding: "9px 15px" }} onClick={submit}
                disabled={state.busy || name.trim() === value}>
          {state.busy ? "Saving…" : state.saved ? "Saved" : "Save"}
        </button>
      </div>
      <Hint>{hint}</Hint>
      <Note>{state.error}</Note>
    </div>
  );
}

function ReadOnly({ label, value, hint }: { label: string; value: string; hint: string }) {
  return (
    <div style={{ marginBottom: 18 }}>
      <Label>{label}</Label>
      <input style={{ ...box, color: "var(--faint)" }} value={value} readOnly />
      <Hint>{hint}</Hint>
    </div>
  );
}

/** One model provider, in the canvas's shape; keys themselves arrive with Phase 5. */
function Provider({ name, models, status }: { name: string; models: string; status: "ready" | "later" }) {
  const ready = status === "ready";
  return (
    <div style={cardStyle}>
      <div style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap", marginBottom: 12 }}>
        <div style={{ fontSize: 14, color: ready ? "var(--text)" : "var(--dim)" }}>{name}</div>
        <div className="mono" style={{ fontSize: 10.5, borderRadius: 3, padding: "2px 6px",
                                       color: ready ? "var(--accent-bright)" : "var(--faint)",
                                       border: `1px solid ${ready ? "#1B3F36" : "var(--edge)"}` }}>
          {ready ? "no key" : "not yet"}
        </div>
        <div className="mono" style={{ marginLeft: "auto", fontSize: 11, color: "var(--faint)" }}>{models}</div>
      </div>
      <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
        <input style={{ ...box, flex: "1 1 260px", fontFamily: "var(--mono)", fontSize: 12, color: "var(--faint)" }}
               placeholder={ready ? "AIza…" : "greyed out until a provider is proven (A2)"} disabled />
        <button className="btn ghost" disabled>Verify &amp; save</button>
      </div>
    </div>
  );
}

function Runner({ name, where, meta, status, action }: {
  name: string;
  where: string;
  meta: string;
  status: "active" | "later";
  action: string;
}) {
  const live = status === "active";
  return (
    <div style={{ ...cardStyle, padding: "15px 17px", display: "flex", alignItems: "center", gap: 14,
                  flexWrap: "wrap" }}>
      <div style={{ minWidth: 0 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
          <div style={{ fontSize: 14, color: live ? "var(--text)" : "var(--dim)" }}>{name}</div>
          <div className="mono" style={{ fontSize: 10.5, borderRadius: 3, padding: "2px 6px",
                                         color: live ? "var(--accent-bright)" : "var(--faint)",
                                         border: `1px solid ${live ? "#1B3F36" : "var(--edge)"}` }}>
            {live ? "running" : "coming soon"}
          </div>
        </div>
        <div className="mono" style={{ fontSize: 11, color: "var(--dim)", marginTop: 5, overflowWrap: "anywhere" }}>
          {where}
        </div>
        <div className="mono" style={{ fontSize: 11, color: "var(--faint)", marginTop: 3 }}>{meta}</div>
      </div>
      <button className="btn ghost" style={{ marginLeft: "auto" }} disabled>{action}</button>
    </div>
  );
}

export function Settings({ me, workspace, onClose }: { me: Me; workspace: Workspace; onClose: () => void }) {
  const [tab, setTab] = useState<TabId>("account");
  const queries = useQueryClient();

  const bodies: Record<TabId, ReactNode> = {
    account: (
      <div>
        <Name label="Full name" hint="shown to anyone you work with" value={me.user.name}
              save={async (name) => {
                queries.setQueryData(["me"], await api<Me>("/api/me", { method: "PATCH", body: { name } }));
              }} />
        <ReadOnly label="Email" value={me.user.email}
                  hint={me.user.has_password ? "sign in with this address, Google or GitHub"
                                             : "signed in through Google or GitHub"} />
        <div className="rule" style={{ margin: "22px 0 18px" }} />
        <Label>Sign-in</Label>
        <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          <Switch title="Password" meta={me.user.has_password ? "set · change it from the log-in screen"
                                                              : "none · you sign in with a provider"}
                  on={me.user.has_password} />
          <Switch title="Two-step sign-in" meta="after the beta" on={false} />
        </div>
      </div>
    ),
    workspace: (
      <div>
        <Name label="Workspace name" hint="visible to everyone in the workspace" value={workspace.name}
              save={async (name) => {
                await api(`/api/w/${workspace.slug}`, { method: "PATCH", body: { name } });
                await queries.invalidateQueries({ queryKey: ["workspace", workspace.slug] });
                await queries.invalidateQueries({ queryKey: ["me"] });
              }} />
        <ReadOnly label="Slug" value={workspace.slug}
                  hint={`${workspace.slug}.automl.io · used in API paths`} />

        <div className="rule" style={{ margin: "22px 0 18px" }} />
        <Label>Model providers</Label>
        <Lead>
          Bring your own key for the models that infer schemas, draft a plan and write each candidate script.
          Keys are stored per workspace, encrypted, and never reach a sandbox. Saving one arrives with jobs.
        </Lead>
        <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          <Provider name="Google" models="gemini-2.5-flash · pro" status="ready" />
          <Provider name="Anthropic" models="claude-opus-5 · sonnet-5" status="later" />
          <Provider name="OpenAI" models="gpt-5 · o4-mini" status="later" />
          <Provider name="Mistral" models="mistral-large-2 · codestral" status="later" />
          <Provider name="xAI" models="grok-4" status="later" />
        </div>

        <div className="rule" style={{ margin: "22px 0 18px" }} />
        <Label>Default model for assistive steps</Label>
        <Chips chosen="gemini-2.5-flash"
               options={[{ label: "gemini-2.5-flash" }, { label: "claude-sonnet-5", locked: true },
                         { label: "gpt-5", locked: true }, { label: "mistral-large-2", locked: true },
                         { label: "grok-4", locked: true }]} />
        <Hint>
          assistive steps: schema inference, planning, candidate code · training runs no model of its own
        </Hint>
      </div>
    ),
    compute: (
      <div>
        <Lead>
          Jobs run in sandboxes on the machine that hosts AutoML. Later you can run the same sandbox server on a
          machine of your own and paste its link here; the data and the artifacts then stay on that machine.
        </Lead>
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 20 }}>
          <input style={{ ...box, flex: "1 1 320px", fontFamily: "var(--mono)", fontSize: 12, color: "var(--faint)" }}
                 placeholder="https://my-machine.local:7443" disabled />
          <button className="btn ghost" disabled>Link runner</button>
        </div>
        <Label>Runners</Label>
        <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          <Runner name="AutoML sandbox" status="active" where="the machine that hosts AutoML"
                  meta="gVisor · up to 6 jobs at once, one per workspace" action="Unlink" />
          <Runner name="Your own machine" status="later" where="paste the link the sandbox server prints"
                  meta="needs a public HTTPS address · a tunnel is the easy way" action="Set active" />
        </div>
        <div className="rule" style={{ margin: "22px 0 18px" }} />
        <Label>Sandboxes in parallel</Label>
        <Chips chosen="set by the host" options={[{ label: "set by the host" }, { label: "2", locked: true },
                                                  { label: "4", locked: true }, { label: "8", locked: true }]} />
        <Hint>each sandbox waits for the host's memory and CPUs · a freed place goes to the job with the fewest</Hint>
      </div>
    ),
    members: (
      <div>
        <div style={{ display: "flex", alignItems: "center", gap: 14, marginBottom: 16, flexWrap: "wrap" }}>
          <div style={{ fontSize: 13.5, color: "var(--muted)", maxWidth: 460 }}>
            One person per workspace during the beta. Roles, invitations and shared access come after it.
          </div>
          <button className="btn ghost" style={{ marginLeft: "auto" }} disabled>Invite member</button>
        </div>
        <div style={{ ...cardStyle, padding: "13px 16px", display: "flex", alignItems: "center", gap: 14 }}>
          <div style={{ minWidth: 0 }}>
            <div style={{ fontSize: 13.5, color: "var(--text)" }}>{me.user.name}</div>
            <div className="mono" style={{ fontSize: 10.5, color: "var(--faint)", marginTop: 3 }}>
              {me.user.email} · joined at sign-up
            </div>
          </div>
          <div className="mono" style={{ marginLeft: "auto", fontSize: 10.5, color: "var(--muted)",
                                         border: "1px solid var(--edge)", borderRadius: 3, padding: "3px 8px" }}>
            {workspace.role}
          </div>
        </div>
      </div>
    ),
    data: (
      <div>
        <Label>Keep uploaded files for</Label>
        <Chips chosen="7 days" options={[{ label: "7 days" }, { label: "30 days", locked: true },
                                         { label: "90 days", locked: true }, { label: "indefinite", locked: true }]} />
        <Hint>reports, scores and usage are kept · the files behind them are swept</Hint>

        <div className="rule" style={{ margin: "22px 0 18px" }} />
        <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          <Switch title="Redact detected personal data in previews"
                  meta="emails, phone numbers and ids masked in the data view · after the beta" on={false} />
          <Switch title="Share anonymised diagnostics" meta="run timings only · never your data" on={false} />
          <div style={{ border: "1px solid #3A2A26", borderRadius: 6, background: "#14100F", padding: "15px 17px",
                        display: "flex", alignItems: "center", gap: 14 }}>
            <div style={{ minWidth: 0 }}>
              <div style={{ fontSize: 13.5, color: "var(--faint)" }}>Delete workspace</div>
              <div className="mono" style={{ fontSize: 10.5, color: "#8A6A5E", marginTop: 4 }}>
                removes sources, jobs and artifacts · cannot be undone
              </div>
            </div>
            <button className="btn ghost" style={{ marginLeft: "auto" }} disabled>Delete</button>
          </div>
        </div>
      </div>
    ),
  };

  const footer: Record<TabId, string> = {
    account: me.user.has_password ? "signed in with a password · sessions last 30 days from last use"
                                  : "signed in with a provider · sessions last 30 days from last use",
    workspace: "0 of 5 providers connected · keys are encrypted at rest",
    compute: "active runner: the AutoML sandbox · your own machine comes later",
    data: "files kept 7 days · reports, scores and usage stay",
    members: "1 person · invitations come after the beta",
  };

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
              return (
                <div key={entry.id} onClick={() => setTab(entry.id)} className="menu-item"
                     style={{ padding: "10px 12px", background: on ? "#161D1B" : "transparent" }}>
                  <div style={{ fontSize: 13.5, color: on ? "var(--text)" : "var(--muted)" }}>{entry.label}</div>
                  <div className="mono" style={{ fontSize: 10.5, marginTop: 3,
                                                 color: on ? "var(--accent-bright)" : "var(--faint)" }}>
                    {entry.meta}
                  </div>
                </div>
              );
            })}
          </div>
          <div style={{ flex: 1, minWidth: 0, padding: "20px 28px 26px", overflowY: "auto" }}>{bodies[tab]}</div>
        </div>

        <div style={{ borderTop: "1px solid var(--line)", background: "var(--panel)", padding: "16px 28px",
                      display: "flex", alignItems: "center", justifyContent: "space-between", gap: 16 }}>
          <div className="mono" style={{ fontSize: 11, color: "var(--faint)" }}>{footer[tab]}</div>
          <button className="button" style={{ padding: "9px 20px" }} onClick={onClose}>Done</button>
        </div>
      </div>
    </div>
  );
}
