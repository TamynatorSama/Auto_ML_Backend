import { useEffect, useState, type ReactNode } from "react";
import { useLocation } from "react-router-dom";

import type { Me, Workspace } from "../api";
import { Settings } from "./Settings";
import { Sidebar, type Source, type Step } from "./Sidebar";
import { TopBar, type StepName } from "./TopBar";

/**
 * The console: the sidebar and the step bar every screen from Phase 4 on shares,
 * with the screen's own body between them. Settings opens over all of it.
 */
export function Shell({ me, workspace, sources, steps, at, reached, onStep, action, onAddSource,
                       children }: {
  me: Me;
  workspace: Workspace;
  sources?: Source[];
  steps: Step[];
  at: StepName;
  reached?: StepName[];
  onStep?: (step: StepName) => void;
  action?: ReactNode;
  onAddSource?: () => void;
  children: ReactNode;
}) {
  const [settings, setSettings] = useState(false);
  const [navigationOpen, setNavigationOpen] = useState(false);
  const { pathname } = useLocation();
  useEffect(() => setNavigationOpen(false), [pathname]);
  useEffect(() => {
    if (!navigationOpen) return;
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setNavigationOpen(false);
    };
    document.addEventListener("keydown", closeOnEscape);
    return () => document.removeEventListener("keydown", closeOnEscape);
  }, [navigationOpen]);
  return (
    <div className={`shell${navigationOpen ? " navigation-open" : ""}`}>
      {navigationOpen && <button className="navigation-backdrop" aria-label="Close navigation"
                                 onClick={() => setNavigationOpen(false)} />}
      <Sidebar me={me} workspace={workspace} sources={sources} steps={steps}
               onSettings={() => { setNavigationOpen(false); setSettings(true); }}
               onAddSource={onAddSource} onClose={() => setNavigationOpen(false)} />
      <div className="shell-main">
        <TopBar at={at} reached={reached} onStep={onStep} action={action}
                onNavigation={() => setNavigationOpen(true)} navigationOpen={navigationOpen} />
        <div className="shell-content">{children}</div>
      </div>
      {settings && <Settings me={me} workspace={workspace} onClose={() => setSettings(false)} />}
    </div>
  );
}

export type { Source, Step, StepName };
