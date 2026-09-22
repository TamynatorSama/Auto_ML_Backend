import { useState, type ReactNode } from "react";

/**
 * The AutoML mark: public/automl-mark.png from the design canvas when it's there,
 * otherwise a drawn stand-in (a teal tile with a rising step).
 */
export function Mark({ size = 26 }: { size?: number }) {
  const [missing, setMissing] = useState(false);
  if (missing) {
    return (
      <svg width={size} height={size} viewBox="0 0 32 32" aria-label="AutoML">
        <rect width="32" height="32" rx="7" fill="#12876F" />
        <path d="M7 22.5h4.5v-6H7zM13.75 22.5h4.5V12h-4.5zM20.5 22.5H25V7h-4.5z" fill="#04140F" />
      </svg>
    );
  }
  return <img src="/automl-mark.png" alt="AutoML" width={size} height={size} style={{ display: "block" }}
              onError={() => setMissing(true)} />;
}

export function Logo({ size = 26, text = 19 }: { size?: number; text?: number }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
      <Mark size={size} />
      <div style={{ fontSize: text, fontWeight: 500, letterSpacing: "-0.01em", color: "var(--bright)" }}>AutoML</div>
    </div>
  );
}

const STEPS = ["Sign up your account", "Connect a data source", "Lock a schema"];

/** The onboarding canvas: a panel that explains, beside the form. */
export function AuthShell({ panelTitle, panelSub, step = 1, title, sub, fine, children }: {
  panelTitle: string;
  panelSub: string;
  step?: number;
  title: string;
  sub?: ReactNode;
  fine?: string;
  children: ReactNode;
}) {
  return (
    <div className="auth-outer" style={{ height: "100%", background: "var(--bg-deep)", display: "flex", justifyContent: "center",
                  padding: "40px 32px", overflowY: "auto" }}>
      {/* a floor, not a fixed height: log in, forgot, reset, verify and welcome all sit at it */}
      <div className="auth-card" style={{ width: "100%", maxWidth: 1140, minHeight: "min(600px, 100%)", margin: "auto", display: "flex",
                    flexWrap: "wrap", border: "1px solid var(--line-soft)", borderRadius: 10, background: "var(--bg)",
                    boxShadow: "0 40px 90px rgba(0,0,0,0.55)", overflow: "hidden" }}>

        <div className="auth-intro" style={{ flex: "1 1 420px", minWidth: 0, padding: "44px 44px 40px", display: "flex",
                      flexDirection: "column", gap: 26, background: "var(--panel)", borderRight: "1px solid var(--line-soft)",
                      backgroundImage: "repeating-linear-gradient(to right, rgba(255,255,255,0.022) 0 1px, transparent 1px 44px), " +
                                       "repeating-linear-gradient(to bottom, rgba(255,255,255,0.022) 0 1px, transparent 1px 44px)" }}>
          <Logo size={30} />
          <div style={{ maxWidth: 420, marginTop: "auto" }}>
            <div style={{ fontSize: 33, lineHeight: 1.2, letterSpacing: "-0.02em", color: "#fff", marginBottom: 14 }}>
              {panelTitle}
            </div>
            <div style={{ fontSize: 14.5, lineHeight: 1.6, color: "var(--muted)" }}>{panelSub}</div>
          </div>
          <div className="auth-steps" style={{ display: "grid", gridTemplateColumns: "repeat(3, minmax(0, 1fr))", gap: 12 }}>
            {STEPS.map((label, index) => {
              const on = index + 1 === step;
              return (
                <div key={label} style={{ borderRadius: 8, padding: "16px 15px", minHeight: 108, display: "flex",
                                          flexDirection: "column", justifyContent: "space-between", gap: 18,
                                          background: on ? "var(--accent)" : "var(--card)",
                                          border: `1px solid ${on ? "var(--accent)" : "var(--line)"}` }}>
                  <div className="mono" style={{ width: 22, height: 22, borderRadius: "50%", display: "flex",
                                                 alignItems: "center", justifyContent: "center", fontSize: 11,
                                                 background: on ? "#04140F" : "#1B2220", color: on ? "#9FE6D4" : "var(--dim)" }}>
                    {index + 1}
                  </div>
                  <div style={{ fontSize: 13, lineHeight: 1.4, color: on ? "#04140F" : "var(--muted)" }}>{label}</div>
                </div>
              );
            })}
          </div>
        </div>

        <div className="auth-form-panel" style={{ flex: "1 1 420px", minWidth: 0, display: "flex", flexDirection: "column",
                      padding: "44px 48px 26px" }}>
          <div style={{ flex: 1, display: "flex", flexDirection: "column", justifyContent: "center",
                        overflowY: "auto" }}>
            <div style={{ width: "100%", maxWidth: 400, margin: "0 auto" }}>
              <div style={{ textAlign: "center", marginBottom: 24 }}>
                <div style={{ fontSize: 23, letterSpacing: "-0.015em", color: "var(--bright)", marginBottom: 7 }}>{title}</div>
                {sub && <div style={{ fontSize: 13.5, lineHeight: 1.55, color: "var(--muted)" }}>{sub}</div>}
              </div>
              {children}
              {fine && (
                <div style={{ fontSize: 12, lineHeight: 1.6, color: "var(--dim)", textAlign: "center", marginTop: 14 }}>
                  {fine}
                </div>
              )}
            </div>
          </div>

          {/* the card's own footer, along its bottom edge */}
          <div style={{ width: "100%", maxWidth: 400, margin: "0 auto", display: "flex", alignItems: "center",
                        justifyContent: "space-between", flexWrap: "wrap", gap: 12,
                        borderTop: "1px solid var(--line-soft)", marginTop: 20, paddingTop: 16 }}>
            <div style={{ display: "flex", gap: 20 }}>
              {["About", "Docs", "Support"].map((text) => (
                <div key={text} style={{ fontSize: 12.5, color: "var(--dim)" }}>{text}</div>
              ))}
            </div>
            <div className="mono" style={{ fontSize: 10.5, color: "#3D4744" }}>AutoML</div>
          </div>
        </div>
      </div>
    </div>
  );
}

export function Field({ label, aside, children }: { label: string; aside?: ReactNode; children: ReactNode }) {
  return (
    <div className="field">
      <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", gap: 12 }}>
        <div className="label">{label}</div>
        {aside}
      </div>
      {children}
    </div>
  );
}

export function Divider({ text }: { text: string }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 14, margin: "20px 0" }}>
      <div style={{ flex: 1, height: 1, background: "#1E2523" }} />
      <div className="mono" style={{ fontSize: 10.5, color: "var(--faint)" }}>{text}</div>
      <div style={{ flex: 1, height: 1, background: "#1E2523" }} />
    </div>
  );
}

/** Google and GitHub: a plain link each, since the browser must leave the app for the provider. */
export function Providers() {
  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(2, minmax(0, 1fr))", gap: 10 }}>
      {[["google", "Google"], ["github", "GitHub"]].map(([id, label]) => (
        <a key={id} className="quiet" href={`/api/auth/oauth/${id}`}
           style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 9, color: "#c8d2ce" }}>
          <img src={`/${id}-mark.svg`} alt="" width={16} height={16} style={{ display: "block", flex: "0 0 16px" }} />
          {label}
        </a>
      ))}
    </div>
  );
}

export function Note({ children, ok = false }: { children: ReactNode; ok?: boolean }) {
  if (!children) return null;
  return <div className={ok ? "note ok" : "note"}>{children}</div>;
}

/**
 * The console's own shape while the workspace is still arriving: the rail, the
 * step bar and a body, greyed. A blank screen reads as broken; this reads as busy.
 */
export function ConsoleSkeleton() {
  const rows = (count: number) => Array.from({ length: count }, (unused, index) => index);
  return (
    <div className="shell">
      <div className="rail">
        <div style={{ padding: "20px 20px 18px", borderBottom: "1px solid var(--line)" }}>
          <Logo />
        </div>
        <div style={{ padding: "22px 18px", display: "flex", flexDirection: "column", gap: 14 }}>
          <div className="skel" style={{ width: 90, height: 11 }} />
          {rows(3).map((row) => <div className="skel" key={row} style={{ width: "100%", height: 30 }} />)}
          <div className="rule" style={{ margin: "8px 0" }} />
          <div className="skel" style={{ width: 60, height: 11 }} />
          {rows(5).map((row) => <div className="skel" key={row} style={{ width: "86%", height: 14 }} />)}
        </div>
      </div>
      <div className="shell-main">
        <div className="top">
          <div style={{ display: "flex", gap: 14 }}>
            {rows(6).map((row) => <div className="skel" key={row} style={{ width: 86, height: 15 }} />)}
          </div>
          <div className="skel" style={{ width: 88, height: 27 }} />
        </div>
        <div style={{ padding: "36px 40px", display: "flex", flexDirection: "column", gap: 16 }}>
          <div className="skel" style={{ width: 240, height: 26 }} />
          <div className="skel" style={{ width: 460, height: 14 }} />
          <div className="skel" style={{ width: "100%", maxWidth: 1000, height: 190, marginTop: 10 }} />
          <div className="skel" style={{ width: "100%", maxWidth: 1000, height: 120 }} />
        </div>
      </div>
    </div>
  );
}

/** The sign-in card's shape, for the moment before /api/me answers on a public screen. */
export function AuthSkeleton() {
  return (
    <div className="auth-outer" style={{ height: "100%", background: "var(--bg-deep)", display: "flex", justifyContent: "center",
                  padding: "40px 32px" }}>
      <div className="auth-card" style={{ width: "100%", maxWidth: 1140, minHeight: "min(600px, 100%)", margin: "auto", display: "flex",
                    flexWrap: "wrap", border: "1px solid var(--line-soft)", borderRadius: 10,
                    background: "var(--bg)", overflow: "hidden" }}>
        <div className="auth-intro" style={{ flex: "1 1 420px", minWidth: 0, padding: "44px", background: "var(--panel)",
                      borderRight: "1px solid var(--line-soft)", display: "flex", flexDirection: "column",
                      gap: 26 }}>
          <Logo size={30} />
          <div style={{ marginTop: "auto", display: "flex", flexDirection: "column", gap: 12 }}>
            <div className="skel" style={{ width: "80%", height: 30 }} />
            <div className="skel" style={{ width: "95%", height: 14 }} />
            <div className="skel" style={{ width: "70%", height: 14 }} />
          </div>
          <div className="auth-steps" style={{ display: "grid", gridTemplateColumns: "repeat(3, minmax(0, 1fr))", gap: 12 }}>
            {[0, 1, 2].map((tile) => <div className="skel" key={tile} style={{ height: 108 }} />)}
          </div>
        </div>
        <div className="auth-form-panel" style={{ flex: "1 1 420px", minWidth: 0, display: "flex", alignItems: "center",
                      justifyContent: "center", padding: "44px 48px" }}>
          <div style={{ width: "100%", maxWidth: 400, display: "flex", flexDirection: "column", gap: 14 }}>
            <div className="skel" style={{ width: "60%", height: 22, margin: "0 auto 10px" }} />
            {[0, 1, 2].map((row) => <div className="skel" key={row} style={{ height: 42 }} />)}
          </div>
        </div>
      </div>
    </div>
  );
}
