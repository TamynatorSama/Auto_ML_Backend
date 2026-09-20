import type { ReactElement } from "react";
import { Navigate, Route, Routes } from "react-router-dom";

import { useMe } from "./api";
import { Forgot, LogIn, Reset, SignUp, Verify, Welcome } from "./screens/Auth";
import { Console } from "./screens/Console";

export function App() {
  const me = useMe();

  if (me.isPending) return <div className="mono" style={{ padding: 40, color: "var(--faint)" }}>loading…</div>;
  if (me.error) {
    return (
      <div style={{ padding: 40, color: "var(--muted)" }}>
        Can't reach the API. Start it with <span className="mono">python -m uvicorn api.app:app --reload</span>.
      </div>
    );
  }

  // signed in: the sign-up screens redirect on; signed out: everything but them does
  const signedIn = me.data !== null;
  const gate = (screen: ReactElement) =>
    signedIn ? <Navigate to={me.data!.workspace ? `/${me.data!.workspace.slug}` : "/welcome"} replace /> : screen;

  return (
    <Routes>
      <Route path="/login" element={gate(<LogIn />)} />
      <Route path="/signup" element={gate(<SignUp />)} />
      <Route path="/forgot" element={gate(<Forgot />)} />
      <Route path="/reset" element={<Reset />} />
      <Route path="/verify" element={<Verify />} />
      <Route path="/welcome" element={
        !signedIn ? <Navigate to="/login" replace />
          : me.data!.workspace ? <Navigate to={`/${me.data!.workspace.slug}`} replace />
            : <Welcome />} />
      <Route path="/:ws" element={signedIn ? <Console me={me.data!} at="Data" /> : <Navigate to="/login" replace />} />
      <Route path="/:ws/s/:sourceId" element={
        signedIn ? <Console me={me.data!} at="Data" /> : <Navigate to="/login" replace />} />
      <Route path="/:ws/s/:sourceId/preview" element={
        signedIn ? <Console me={me.data!} at="Preview" /> : <Navigate to="/login" replace />} />
      <Route path="/:ws/s/:sourceId/schema" element={
        signedIn ? <Console me={me.data!} at="Schema" /> : <Navigate to="/login" replace />} />
      <Route path="*" element={
        <Navigate to={signedIn ? (me.data!.workspace ? `/${me.data!.workspace.slug}` : "/welcome") : "/login"} replace />} />
    </Routes>
  );
}
