import { useState, type ReactNode } from "react";

import type { Me, Workspace } from "../api";
import { Settings } from "./Settings";
import { Sidebar, type Source, type Step } from "./Sidebar";
import { TopBar, type StepName } from "./TopBar";

/**
 * The console: the sidebar and the step bar every screen from Phase 4 on shares,
 * with the screen's own body between them. Settings opens over all of it.
 */
export function Shell({ me, workspace, sources, steps, at, reached, onStep, action, children }: {
  me: Me;
  workspace: Workspace;
  sources?: Source[];
  steps: Step[];
  at: StepName;
  reached?: StepName[];
  onStep?: (step: StepName) => void;
  action?: ReactNode;
  children: ReactNode;
}) {
  const [settings, setSettings] = useState(false);
  return (
    <div className="shell">
      <Sidebar me={me} workspace={workspace} sources={sources} steps={steps}
               onSettings={() => setSettings(true)} />
      <div className="shell-main">
        <TopBar at={at} reached={reached} onStep={onStep} action={action} />
        <div style={{ flex: 1, overflowY: "auto" }}>{children}</div>
      </div>
      {settings && <Settings me={me} workspace={workspace} onClose={() => setSettings(false)} />}
    </div>
  );
}

export type { Source, Step, StepName };
