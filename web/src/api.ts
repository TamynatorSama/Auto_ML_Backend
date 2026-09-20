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

// ---- sources (Phase 4)

export type SchemaRef = { id: number; version: number; status: "draft" | "locked";
                          described: number; target: string | null };
export type SourceTask = { kind: "ingest" | "profile" | "analyze"; status: string };
export type SourceStatus = "uploading" | "checking" | "profiling" | "ready" | "failed" | "expired";

export type Source = {
  id: number;
  name: string;
  original_name: string | null;
  status: SourceStatus;
  rows: number | null;
  columns: number | null;
  bytes: number | null;
  error: string | null;
  created_at: string;
  files_expire_at: string | null;
  schema: SchemaRef | null;
  task: SourceTask | null;
  profile?: Profile | null;
};

/** Only the parts of the pipeline's profile the console reads; it holds a great deal more. */
export type Profile = {
  dataset: { n_rows: number; n_columns: number; duplicate_rows: number; total_missing_pct: number };
  columns: Record<string, ProfileColumn>;
};

export type ProfileColumn = {
  dtype: string;
  role: string;
  missing_pct: number;
  n_unique: number;
  semantic_type?: string;
  stats?: Record<string, number | string | null>;
};

export type Page = { page: number; size: number; total: number | null; columns: string[]; rows: string[][] };

/** Still being read: the Data screen polls while one of these is true. */
export const WORKING: SourceStatus[] = ["uploading", "checking", "profiling"];

export function useSources(ws: string) {
  return useQuery({
    queryKey: ["sources", ws],
    queryFn: () => api<{ sources: Source[] }>(`/api/w/${encodeURIComponent(ws)}/sources`),
    // while anything is being read, ask again; otherwise leave it alone
    refetchInterval: ({ state }) =>
      state.data?.sources.some((source) => WORKING.includes(source.status)) ? 2000 : false,
  });
}

export function useSource(ws: string, id: number | null) {
  return useQuery({
    enabled: id !== null,
    queryKey: ["source", ws, id],
    queryFn: () => api<Source>(`/api/w/${encodeURIComponent(ws)}/sources/${id}`),
    refetchInterval: ({ state }) => (state.data && WORKING.includes(state.data.status) ? 2000 : false),
  });
}

export function useRows(ws: string, id: number | null, page: number, size: number, ready: boolean) {
  return useQuery({
    enabled: id !== null && ready,
    queryKey: ["rows", ws, id, page, size],
    placeholderData: (previous) => previous,   // the table stays put while the next page arrives
    queryFn: () => api<Page>(`/api/w/${encodeURIComponent(ws)}/sources/${id}/rows?page=${page}&size=${size}`),
  });
}

/** Make the row, then send the file to the address it gives back. */
export async function uploadCsv(ws: string, file: File): Promise<Source> {
  const created = await api<Source & { upload_url: string }>(
    `/api/w/${encodeURIComponent(ws)}/sources`, { body: { name: file.name.replace(/\.csv$/i, "") } });
  const response = await fetch(created.upload_url, { method: "PUT", body: file });
  const text = await response.text();
  const body = text ? JSON.parse(text) : null;
  if (!response.ok) throw new ApiError(response.status, message(body, response.status));
  return body as Source;
}

export function loadSample(ws: string) {
  return api<Source>(`/api/w/${encodeURIComponent(ws)}/sources/sample`, { method: "POST" });
}

export function deleteSource(ws: string, id: number) {
  return api<null>(`/api/w/${encodeURIComponent(ws)}/sources/${id}`, { method: "DELETE" });
}

// ---- the schema (Phase 4c)

export type SchemaColumn = {
  name: string;
  data_type: string;
  role: "feature" | "identifier" | "ignore" | "target";
  description: string;
  is_target?: boolean;
  available_at_prediction: boolean | null;
};

export type Finding = {
  id: string;
  severity: "critical" | "high" | "medium" | "low";
  columns: string[];
  issue: string;
  action: string;
};

export type SchemaBody = {
  id: number;
  version: number;
  status: "draft" | "locked";
  columns: SchemaColumn[];
  description: string;
  locked_at: string | null;
  updated_at: string;
  findings: Finding[];
  analyzed_target: string | null;
  profile: Record<string, ProfileColumn>;
  task: SourceTask | null;
};

export type VersionRow = {
  id: number; version: number; status: "draft" | "locked";
  locked_at: string | null; locked_by: string | null; updated_at: string;
};
export type FieldChange = { field: string; from: unknown; to: unknown };
export type ColumnChange = { name: string; change: "added" | "removed" | "changed"; fields: FieldChange[] };

export const DATA_TYPES = ["numeric", "categorical", "text", "ordinal", "json", "latitude", "long"] as const;
export const ROLES = ["feature", "identifier", "ignore", "target"] as const;

function schemaPath(ws: string, id: number) {
  return `/api/w/${encodeURIComponent(ws)}/sources/${id}/schema`;
}

export function useSchema(ws: string, id: number) {
  return useQuery({
    queryKey: ["schema", ws, id],
    retry: false,
    queryFn: () => api<SchemaBody>(schemaPath(ws, id)),
    // while analyze runs, ask again until its findings land
    refetchInterval: ({ state }) => (state.data?.task ? 2000 : false),
  });
}

export function saveSchema(ws: string, id: number, body: {
  columns: SchemaColumn[]; description: string;
}) {
  return api<SchemaBody>(schemaPath(ws, id), { method: "PUT", body });
}

export function analyzeSchema(ws: string, id: number) {
  return api<SchemaBody>(`${schemaPath(ws, id)}/analyze`, { method: "POST" });
}

export function lockSchema(ws: string, id: number) {
  return api<SchemaBody>(`${schemaPath(ws, id)}/lock`, { method: "POST" });
}

export function useVersions(ws: string, id: number, diff: string | null) {
  return useQuery({
    queryKey: ["versions", ws, id, diff],
    queryFn: () => api<{ versions: VersionRow[]; diff: { from: number; to: number; columns: ColumnChange[] } | null }>(
      `${schemaPath(ws, id)}/versions${diff ? `?diff=${diff}` : ""}`),
  });
}
