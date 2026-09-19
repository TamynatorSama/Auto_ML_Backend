import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";

export type User = { id: number; email: string; name: string; has_password: boolean };
export type Workspace = { slug: string; name: string; role: string; created_at?: string };
export type Me = { user: User; workspace: Workspace | null };

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

/** One message out of FastAPI's answer: a string detail, or the first validation error. */
function message(body: unknown, status: number): string {
  const detail = (body as { detail?: unknown })?.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail) && detail.length) {
    const first = detail[0] as { loc?: string[]; msg?: string };
    const field = first.loc?.[first.loc.length - 1];
    return field ? `${field}: ${first.msg}` : String(first.msg);
  }
  return `Something went wrong (${status})`;
}

export async function api<T>(path: string, init?: { method?: string; body?: unknown }): Promise<T> {
  const response = await fetch(path, {
    method: init?.method ?? (init?.body ? "POST" : "GET"),
    headers: init?.body ? { "Content-Type": "application/json" } : undefined,
    body: init?.body ? JSON.stringify(init.body) : undefined,
  });
  const text = await response.text();
  const body = text ? JSON.parse(text) : null;
  if (!response.ok) throw new ApiError(response.status, message(body, response.status));
  return body as T;
}

/** The signed-in user, or null. 401 is an answer, not a failure. */
export function useMe() {
  return useQuery({
    queryKey: ["me"],
    retry: false,
    queryFn: async () => {
      try {
        return await api<Me>("/api/me");
      } catch (error) {
        if (error instanceof ApiError && error.status === 401) return null;
        throw error;
      }
    },
  });
}

/**
 * The token an emailed link carries after the '#', which never reaches a server.
 * Read once, then wiped from the address bar so it isn't left in history or a shared screen.
 */
export function useLinkToken(): string {
  const [token] = useState(() => new URLSearchParams(window.location.hash.slice(1)).get("token") ?? "");
  useEffect(() => {
    if (window.location.hash) window.history.replaceState(null, "", window.location.pathname);
  }, []);
  return token;
}
