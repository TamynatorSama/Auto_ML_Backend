import type { ReactNode } from "react";

/** The six steps of a job, across the top of every console screen. */
export const STEPS = ["Data", "Preview", "Schema", "Configure", "Train", "Results"] as const;
export type StepName = (typeof STEPS)[number];

export function TopBar({ at, reached = [], onStep, action }: {
  at: StepName;
  reached?: StepName[];
  onStep?: (step: StepName) => void;
  action?: ReactNode;
}) {
  return (
    <div className="top">
      <div style={{ display: "flex", alignItems: "center", gap: 3, minWidth: 0, flex: "1 1 auto",
                    overflowX: "auto" }}>
        {STEPS.map((label, index) => {
          const here = label === at;
          const open = reached.includes(label);
          const go = onStep && open && !here ? () => onStep(label) : undefined;
          return (
            <div key={label} style={{ display: "flex", alignItems: "center", flex: "0 0 auto" }}>
              <div className={`step${here ? " on" : open ? " done" : ""}${go ? " can" : ""}`} onClick={go}>
                <div className="num">{index + 1}</div>
                <div style={{ fontSize: 13, letterSpacing: "0.01em",
                              color: here ? "var(--text)" : open ? "var(--muted)" : "var(--faint)" }}>
                  {label}
                </div>
              </div>
              {index < STEPS.length - 1 && <div style={{ width: 10, height: 1, background: "#262E2C" }} />}
            </div>
          );
        })}
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 14, flex: "0 0 auto" }}>{action}</div>
    </div>
  );
}
