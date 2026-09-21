import { useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";

import { api, type Me, type Workspace } from "../api";
import { Logo } from "../ui";

/** One dataset in the rail, as the Console canvas draws it. */
export type Source = {
  name: string;
  schema?: { status: "locked" | "draft"; version: number; go?: () => void };
  job?: { name: string; note: string; tone: "live" | "waiting" | "done" | "bad"; go?: () => void };
  /** this is the dataset being worked on */
  active?: boolean;
  /** which of its rows the screen you are looking at belongs to */
  on?: "name" | "schema" | "job";
  go?: () => void;
};

export type Step = { label: string; meta: string; done: boolean };

function initials(name: string) {
  return name.split(/\s+/).filter(Boolean).slice(0, 2).map((part) => part[0]?.toUpperCase()).join("") || "?";
}

const CHIP = {
  locked: ["var(--accent)", "#1B3F36"],
  draft: ["#C9873F", "#3E3122"],
} as const;

const JOB_TONE = {
  live: "var(--accent-bright)", waiting: "#C9873F", done: "var(--accent)", bad: "#C9673F",
} as const;

/** The teal edge, and the lift behind the row the screen actually belongs to. */
function mark(edge: boolean | undefined, here: boolean | undefined) {
  return {
    borderLeft: `2px solid ${edge ? "var(--accent)" : "transparent"}`,
    background: here ? "#131817" : undefined,
  };
}

function Sources({ sources }: { sources: Source[] }) {
  if (!sources.length) {
    return (
      <div style={{ margin: "0 18px", border: "1px dashed var(--edge)", borderRadius: 5, padding: "16px 15px",
                    display: "flex", flexDirection: "column", gap: 10 }}>
        <div className="mono" style={{ fontSize: 12, color: "var(--dim)" }}>no datasets yet</div>
        <div style={{ fontSize: 12.5, lineHeight: 1.55, color: "var(--faint)" }}>
          Connect one with the + above and its schema and runs appear here.
        </div>
      </div>
    );
  }
  return (
    <div style={{ display: "flex", flexDirection: "column" }}>
      {sources.map((source, index) => (
        <div key={source.name}
             style={{ borderTop: index ? "1px solid var(--line-soft)" : undefined,
                      padding: "4px 0 8px" }}>
          {/* the marker sits on a row, never on the block: the name says which dataset
              you are in, and a sub-row says which of its steps you are looking at */}
          <div className="rail-item" onClick={source.go} style={{ paddingLeft: 16, ...mark(source.active,
               source.on === "name") }}>
            <div style={{ width: 5, height: 5, borderRadius: "50%", flex: "0 0 5px",
                          background: source.active ? "var(--accent-bright)" : "#3D4744" }} />
            <div className="mono" style={{ fontSize: 12.5, minWidth: 0,
                                           color: source.active ? "var(--text)" : "var(--muted)",
                                           overflow: "hidden", textOverflow: "ellipsis",
                                           whiteSpace: "nowrap" }}>
              {source.name}
            </div>
            {!source.schema && (
              <div className="mono" style={{ marginLeft: "auto", fontSize: 11, color: "var(--faint)" }}>
                no schema
              </div>
            )}
          </div>

          {source.schema && (
            <div className="rail-sub" onClick={source.schema.go}
                 style={{ display: "flex", alignItems: "center", justifyContent: "space-between",
                          gap: 10, padding: "6px 20px 6px 31px",
                          cursor: source.schema.go ? "pointer" : undefined,
                          ...mark(source.on === "schema", source.on === "schema") }}>
              <div style={{ fontSize: 13, color: "var(--muted)" }}>Schema</div>
              <div className="mono" title={`version ${source.schema.version}`}
                   style={{ fontSize: 11, borderRadius: 3, padding: "1px 5px",
                            color: CHIP[source.schema.status][0],
                            border: `1px solid ${CHIP[source.schema.status][1]}` }}>
                {source.schema.status}
              </div>
            </div>
          )}

          {source.job && (
            <>
              <div className="label" style={{ padding: "8px 20px 4px 33px", color: "var(--faint)" }}>Job</div>
              <div className="rail-sub" onClick={source.job.go}
                   style={{ display: "flex", alignItems: "center", gap: 8, padding: "5px 20px 5px 31px",
                            cursor: source.job.go ? "pointer" : undefined,
                            ...mark(source.on === "job", source.on === "job") }}>
                <div style={{ width: 5, height: 5, borderRadius: "50%", flex: "0 0 5px",
                              background: JOB_TONE[source.job.tone] }} />
                <div className="mono" style={{ fontSize: 12.5, minWidth: 0, color: "var(--muted)",
                                               overflow: "hidden", textOverflow: "ellipsis",
                                               whiteSpace: "nowrap" }}>
                  {source.job.name}
                </div>
                <div className="mono" style={{ marginLeft: "auto", fontSize: 11, flex: "0 0 auto",
                                               color: JOB_TONE[source.job.tone] }}>
                  {source.job.note}
                </div>
              </div>
            </>
          )}
        </div>
      ))}
    </div>
  );
}

export function Sidebar({ me, workspace, sources = [], steps, onSettings, onAddSource }: {
  me: Me;
  workspace: Workspace;
  sources?: Source[];
  steps: Step[];
  onSettings: () => void;
  onAddSource?: () => void;
}) {
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
    <div className="rail">
      <div style={{ padding: "20px 20px 18px", borderBottom: "1px solid var(--line)" }}>
        <Logo />
      </div>

      <div style={{ flex: 1, padding: "14px 0 20px", overflowY: "auto" }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between",
                      padding: "0 18px 12px" }}>
          <div className="label">Datasets</div>
          <div className="mono add-source" onClick={onAddSource} title="Add a dataset"
               style={{ fontSize: 14, lineHeight: 1, color: onAddSource ? "var(--accent-bright)"
                                                                        : "var(--faint)",
                        cursor: onAddSource ? "pointer" : "default", padding: "0 3px" }}>
            +
          </div>
        </div>

        <Sources sources={sources} />

        {steps.some((step) => !step.done) && (
          <>
            <div className="rule" style={{ margin: "18px 18px 14px" }} />
            <div className="label" style={{ padding: "0 18px 8px" }}>Setup</div>
            {steps.map((step) => (
              <div className="rail-row" key={step.label}>
                <div style={{ width: 14, height: 14, borderRadius: "50%", flex: "0 0 14px",
                              border: `1px solid ${step.done ? "var(--accent)" : "var(--edge)"}`,
                              background: step.done ? "var(--accent)" : "transparent" }} />
                <div style={{ fontSize: 13, color: step.done ? "var(--muted)" : "var(--faint)" }}>
                  {step.label}
                </div>
                <div className="mono" style={{ marginLeft: "auto", fontSize: 10.5,
                                               color: "var(--faint)" }}>
                  {step.meta}
                </div>
              </div>
            ))}
          </>
        )}
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
                <div className="mono" style={{ fontSize: 10.5, color: "var(--faint)" }}>keys · providers</div>
              </div>
              {[["Language", "English (UK)"], ["Get help", "docs · support"]].map(([label, meta]) => (
                <div key={label} style={{ display: "flex", justifyContent: "space-between", gap: 12,
                                          padding: "9px 11px", fontSize: 13, color: "var(--faint)" }}>
                  <div>{label}</div>
                  <div className="mono" style={{ fontSize: 10.5, color: "var(--faint)" }}>{meta}</div>
                </div>
              ))}
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
              {leaving ? "logging out…"
                : `${workspace.slug} · ${sources.length ? `${sources.length} dataset${
                sources.length === 1 ? "" : "s"}` : "new workspace"}`}
            </div>
          </div>
          <div className="mono" style={{ marginLeft: "auto", fontSize: 11, color: "var(--faint)" }}>{menu ? "▾" : "▴"}</div>
        </div>
      </div>
    </div>
  );
}
