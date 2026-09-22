import { useEffect, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { Link, useNavigate, useSearchParams } from "react-router-dom";

import { api, useLinkToken, type Me } from "../api";
import { AuthShell, Field, Note, Providers } from "../ui";

/** After signing in: the workspace, or Welcome to pick a slug. */
function useLandAfterSignIn() {
  const navigate = useNavigate();
  const queries = useQueryClient();
  return (me: Me) => {
    queries.setQueryData(["me"], me);
    navigate(me.workspace ? `/${me.workspace.slug}` : "/welcome", { replace: true });
  };
}

function useSubmit<T>(run: () => Promise<T>, then: (result: T) => void) {
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    setError("");
    setBusy(true);
    try {
      then(await run());
    } catch (problem) {
      setError((problem as Error).message);
    } finally {
      setBusy(false);
    }
  };
  return { submit, error, busy };
}

function Password({ value, onChange, autoComplete }: {
  value: string;
  onChange: (value: string) => void;
  autoComplete: string;
}) {
  const [shown, setShown] = useState(false);
  return (
    <div className="suffix">
      <input className="input" type={shown ? "text" : "password"} value={value} autoComplete={autoComplete}
             placeholder="••••••••••••" onChange={(event) => onChange(event.target.value)} />
      <span className="mono" style={{ cursor: "pointer" }} onClick={() => setShown(!shown)}>
        {shown ? "hide" : "show"}
      </span>
    </div>
  );
}

function Strength({ password }: { password: string }) {
  const [pct, label, color] = !password
    ? ["0%", "empty", "var(--faint)"]
    : password.length < 10
      ? ["28%", "too short", "#B5533C"]
      : password.length < 14
        ? ["62%", "fair", "#C9873F"]
        : ["100%", "strong", "var(--accent-bright)"];
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 9, marginTop: 2 }}>
      <div style={{ flex: 1, height: 3, borderRadius: 2, background: "#1E2523", overflow: "hidden" }}>
        <div style={{ height: "100%", width: pct, background: color }} />
      </div>
      <div className="mono" style={{ fontSize: 10.5, color }}>{label}</div>
    </div>
  );
}

const form = { display: "flex", flexDirection: "column", gap: 14 } as const;
const switcher = { textAlign: "center", fontSize: 13, color: "var(--muted)", marginTop: 4 } as const;

export function SignUp() {
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [workspace, setWorkspace] = useState("");
  const [sent, setSent] = useState("");
  const { submit, error, busy } = useSubmit(
    () => api<{ email: string }>("/api/auth/signup", { body: { name, email, password, workspace } }),
    (result) => setSent(result.email),
  );

  if (sent) {
    return (
      <AuthShell panelTitle="Get started with us" panelSub="Three steps and your first table is training."
                 title="Check your email" sub={<>We've sent a link to <span className="mono">{sent}</span>. Open it to confirm the address and finish signing up.</>}>
        <div style={switcher}>Wrong address? <Link to="/signup" onClick={() => setSent("")}>Sign up again</Link></div>
      </AuthShell>
    );
  }

  return (
    <AuthShell panelTitle="Get started with us" panelSub="Three steps and your first table is training."
               title="Sign up account"
               sub="Your workspace holds your sources, schemas and job history. The beta is invitation only."
               fine="Only invited addresses can sign up while the beta runs.">
      <Providers />
      <form style={form} onSubmit={submit}>
        <Field label="Full name">
          <input className="input" value={name} placeholder="Rhea Kapoor" autoComplete="name"
                 onChange={(event) => setName(event.target.value)} />
        </Field>
        <Field label="Work email">
          <input className="input" type="email" value={email} placeholder="rhea@acme.com" autoComplete="email"
                 onChange={(event) => setEmail(event.target.value)} />
        </Field>
        <Field label="Password">
          <Password value={password} onChange={setPassword} autoComplete="new-password" />
          <Strength password={password} />
        </Field>
        <Field label="Workspace">
          <div className="suffix">
            <input className="input" value={workspace} placeholder="acme"
                   onChange={(event) => setWorkspace(event.target.value.toLowerCase())} />
            <span>.automl.io</span>
          </div>
        </Field>
        <Note>{error}</Note>
        <button className="button" disabled={busy}>{busy ? "Creating…" : "Create workspace"}</button>
        <div style={switcher}>Already have an account? <Link to="/login">Log in</Link></div>
      </form>
    </AuthShell>
  );
}

export function LogIn() {
  const [params] = useSearchParams();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const land = useLandAfterSignIn();
  const { submit, error, busy } = useSubmit(
    () => api<Me>("/api/auth/login", { body: { email, password } }),
    land,
  );

  return (
    <AuthShell panelTitle="Pick up where the last run left off"
               panelSub="Jobs keep running while you are away, and every schema revision stays diffable."
               title="Log in to AutoML" sub="Use the email address on your account."
               fine="Only invited addresses can create an account during the beta.">
      <Providers />
      <form style={form} onSubmit={submit}>
        <Field label="Work email">
          <input className="input" type="email" value={email} placeholder="rhea@acme.com" autoComplete="email"
                 onChange={(event) => setEmail(event.target.value)} />
        </Field>
        <Field label="Password" aside={<Link to="/forgot" style={{ fontSize: 12 }}>Forgot?</Link>}>
          <Password value={password} onChange={setPassword} autoComplete="current-password" />
        </Field>
        <Note>{error || params.get("error") || ""}</Note>
        <button className="button" disabled={busy}>{busy ? "Logging in…" : "Log in"}</button>
        <div style={switcher}>No account yet? <Link to="/signup">Create one</Link></div>
      </form>
    </AuthShell>
  );
}

export function Forgot() {
  const [email, setEmail] = useState("");
  const [sent, setSent] = useState("");
  const { submit, error, busy } = useSubmit(
    () => api<{ detail: string }>("/api/auth/password/forgot", { body: { email } }),
    (result) => setSent(result.detail),
  );

  return (
    <AuthShell panelTitle="Locked out?" panelSub="A reset link works once, and for an hour."
               title="Reset your password" sub="Tell us the address you signed up with.">
      <form style={form} onSubmit={submit}>
        <Field label="Work email">
          <input className="input" type="email" value={email} placeholder="rhea@acme.com" autoComplete="email"
                 onChange={(event) => setEmail(event.target.value)} />
        </Field>
        <Note>{error}</Note>
        <Note ok>{sent}</Note>
        <button className="button" disabled={busy}>{busy ? "Sending…" : "Email me a link"}</button>
        <div style={switcher}>Remembered it? <Link to="/login">Log in</Link></div>
      </form>
    </AuthShell>
  );
}

export function Reset() {
  const [password, setPassword] = useState("");
  const token = useLinkToken();
  const land = useLandAfterSignIn();
  const { submit, error, busy } = useSubmit(
    () => api<Me>("/api/auth/password/reset", { body: { token, password } }),
    land,
  );

  return (
    <AuthShell panelTitle="Locked out?" panelSub="A reset link works once, and for an hour."
               title="Choose a new password"
               sub={token ? "Ten characters or more." : "This link has no token in it. Ask for a new one."}>
      <form style={form} onSubmit={submit}>
        <Field label="New password">
          <Password value={password} onChange={setPassword} autoComplete="new-password" />
          <Strength password={password} />
        </Field>
        <Note>{error}</Note>
        <button className="button" disabled={busy || !token}>{busy ? "Saving…" : "Save and log in"}</button>
        <div style={switcher}>Need another link? <Link to="/forgot">Send one</Link></div>
      </form>
    </AuthShell>
  );
}

export function Verify() {
  const token = useLinkToken();
  const land = useLandAfterSignIn();
  const [error, setError] = useState("");

  useEffect(() => {
    if (!token) {
      setError("This link has no token in it.");
      return;
    }
    api<Me>("/api/auth/verify", { body: { token } }).then(land, (problem) => setError((problem as Error).message));
  }, [token]);

  return (
    <AuthShell panelTitle="Almost there" panelSub="One click and the workspace is yours." step={1}
               title={error ? "That link didn't work" : "Confirming your email…"}
               sub={error ? "" : "One moment."}>
      <Note>{error}</Note>
      {error && <div style={switcher}><Link to="/login">Log in</Link> to have a new link sent.</div>}
    </AuthShell>
  );
}

export function Welcome() {
  const [slug, setSlug] = useState("");
  const land = useLandAfterSignIn();
  const { submit, error, busy } = useSubmit(
    () => api<Me>("/api/workspaces", { body: { slug } }),
    land,
  );

  return (
    <AuthShell panelTitle="Name your workspace" step={2}
               panelSub="It holds your sources, schemas and job history. You can rename it later; the address stays."
               title="One last thing" sub="Pick the address your workspace lives at.">
      <form style={form} onSubmit={submit}>
        <Field label="Workspace">
          <div className="suffix">
            <input className="input" value={slug} placeholder="acme"
                   onChange={(event) => setSlug(event.target.value.toLowerCase())} />
            <span>.automl.io</span>
          </div>
        </Field>
        <Note>{error}</Note>
        <button className="button" disabled={busy}>{busy ? "Creating…" : "Create workspace"}</button>
      </form>
    </AuthShell>
  );
}
