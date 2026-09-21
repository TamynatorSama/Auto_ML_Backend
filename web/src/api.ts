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

// ---- model keys (Phase 5a)

export type ProviderKey = { provider: string; available: boolean; last4: string | null; verified_at: string | null };

/** The canvas's card names, against the ids the API uses. */
export const PROVIDER_NAMES: Record<string, { name: string; models: string; placeholder: string }> = {
  gemini: { name: "Google", models: "gemini-2.5-flash", placeholder: "AIza…" },
  anthropic: { name: "Anthropic", models: "claude-opus-5 · sonnet-5", placeholder: "sk-ant-api03-…" },
  openai: { name: "OpenAI", models: "gpt-5 · o4-mini", placeholder: "sk-proj-…" },
  mistral: { name: "Mistral", models: "mistral-large-2 · codestral", placeholder: "mist-…" },
  xai: { name: "xAI", models: "grok-4", placeholder: "xai-…" },
};

export function useProviders(ws: string) {
  return useQuery({
    queryKey: ["providers", ws],
    queryFn: () => api<{ providers: ProviderKey[] }>(`/api/w/${encodeURIComponent(ws)}/providers`),
  });
}

export function saveKey(ws: string, provider: string, key: string) {
  return api<ProviderKey>(`/api/w/${encodeURIComponent(ws)}/providers/${provider}`,
                          { method: "PUT", body: { key } });
}

export function removeKey(ws: string, provider: string) {
  return api<ProviderKey>(`/api/w/${encodeURIComponent(ws)}/providers/${provider}`, { method: "DELETE" });
}

// ---- jobs (Phase 5)

export type JobStatus = "planning" | "review" | "queued" | "running" | "stopping"
                      | "succeeded" | "stopped" | "failed";

/** Unfinished: the console keeps asking while a job is one of these. */
export const LIVE: JobStatus[] = ["planning", "review", "queued", "running", "stopping"];

export type SplitPlan = {
  method: string;
  test_size: number;
  random_seed: number;
  shuffle: boolean;
  time_column: string | null;
  group_column: string | null;
  stratify_column: string | null;
  drop_duplicates: boolean;
  cv_strategy: string;
  cv_folds: number;
  reason: string;
  warnings: string[];
};

export type Configs = {
  models: string[];
  baseline: { strategy: string; applicable: boolean };
  eval_matrics: string[];
  early_stopping_patience: number;
  max_tries: number;
  improvement_delta: number;
  improvement_mode: string;
  improvement_metric: string;
  reasons: Record<string, string>;
};

/** A model family the pipeline knows, and whether this run's environment can import it. */
export type ModelChoice = { name: string; available: boolean; why: string | null };

/**
 * What the reviewer may pick from, out of the pipeline's own registries
 * (ALL_MODELS, METRIC_DIRECTION, the split methods apply_split_plan accepts).
 * Absent on a plan drafted before choices existed; the screen then locks those fields.
 */
export type Choices = {
  models: ModelChoice[];
  metrics: string[];
  /** metric -> which way is better; absent on a plan drafted before this existed */
  directions?: Record<string, "higher" | "lower">;
  split_methods: string[];
};

/** A column, or a pair, that reproduced the target on held-out rows. */
export type Leak = { columns: string[]; relation: string; score: number; measure: string;
                     formula: string; outcome?: string };

/** What review_plan paused with: the draft, and the two facts it isn't yours to change. */
export type Plan = {
  task_type: string;
  target: string;
  split_plan: SplitPlan;
  config: Configs;
  leakage_screen: Leak[];
  choices?: Choices;
};

export type Edits = { split_plan?: Partial<SplitPlan>; config?: Partial<Configs> };

/**
 * The plan a run is actually using: the draft with the approved edits laid over
 * it, the same merge review_plan does before training. Reading plan.config alone
 * describes a plan that never ran — it was still showing 4 models and 4 tries for
 * a run the person had cut to 2 and 2.
 */
export function running(job: Job | null | undefined):
  { split: SplitPlan; config: Configs } | null {
  if (!job?.plan) return null;
  return {
    split: { ...job.plan.split_plan, ...(job.edits?.split_plan ?? {}) },
    config: { ...job.plan.config, ...(job.edits?.config ?? {}) },
  };
}

export type Job = {
  id: number;
  name: string;
  status: JobStatus;
  error: string | null;
  source: { id: number; name: string };
  schema_id: number;
  budget_usd: number | null;
  deadline_seconds: number | null;
  usage_totals: Record<string, number> | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  files_expire_at: string | null;
  files_expired: boolean;
  queue_position: number | null;
  sandbox_parallel: number;
  plan?: Plan | null;
  edits?: Edits | null;
};

function jobsPath(ws: string) {
  return `/api/w/${encodeURIComponent(ws)}/jobs`;
}

export function useJobs(ws: string) {
  return useQuery({
    queryKey: ["jobs", ws],
    queryFn: () => api<{ jobs: Job[] }>(jobsPath(ws)),
    refetchInterval: ({ state }) =>
      state.data?.jobs.some((job) => LIVE.includes(job.status)) ? 2000 : false,
  });
}

export function useJob(ws: string, id: number | null) {
  return useQuery({
    enabled: id !== null,
    queryKey: ["job", ws, id],
    retry: false,
    queryFn: () => api<Job>(`${jobsPath(ws)}/${id}`),
    // the worker moves a job through planning, queued, running and stopping without
    // telling anyone; only review and the end states sit still
    refetchInterval: ({ state }) => (state.data && MOVING.includes(state.data.status) ? 2000 : false),
  });
}

export function createJob(ws: string, sourceId: number, name?: string) {
  return api<Job>(`/api/w/${encodeURIComponent(ws)}/sources/${sourceId}/jobs`, { body: { name: name ?? null } });
}

export type PlanEdit = Edits & { budget_usd?: number | null; deadline_seconds?: number | null };

export function savePlan(ws: string, id: number, body: PlanEdit) {
  return api<Job>(`${jobsPath(ws)}/${id}/plan`, { method: "PATCH", body });
}

export function startJob(ws: string, id: number, body: PlanEdit) {
  return api<Job>(`${jobsPath(ws)}/${id}/start`, { method: "POST", body });
}

export function redraftJob(ws: string, id: number) {
  return api<Job>(`${jobsPath(ws)}/${id}/redraft`, { method: "POST" });
}

export function deleteJob(ws: string, id: number) {
  return api<null>(`${jobsPath(ws)}/${id}`, { method: "DELETE" });
}

/**
 * The Training screen's one poll (D6): every two seconds while the run is live,
 * carrying the id of the last event seen, so a page opened mid-run catches up
 * and then follows.
 */
export function useLive(ws: string, id: number, cursor: number, live: boolean) {
  return useQuery({
    queryKey: ["live", ws, id, cursor],
    queryFn: () => api<Live>(`${jobsPath(ws)}/${id}/live?after=${cursor}`),
    refetchInterval: live ? 2000 : false,
    placeholderData: (previous) => previous,
  });
}

// ---- the live view (Phase 5c)

export type JobModel = {
  model: string;
  status: string;            // waiting · running · max_tries · no_improvement · failed · …
  generation: number | null;
  best_attempt: number | null;
  best_cv: Record<string, number> | null;
  test_scores: Record<string, number> | null;
};

export type Attempt = {
  id: number;
  model: string;
  attempt: number;
  generation: number | null;
  kind: string | null;       // generate · repair
  status: string;            // ok · error
  cv_scores: Record<string, number> | null;
  wall_seconds: number | null;
  changes: string | null;
  at: string;
};

export type JobEvent = {
  id: number;
  at: string;
  kind: string;
  model: string | null;
  data: Record<string, unknown>;
};

/** Every gauge is a sum over the ledger; nothing is stored per job and overwritten (§8.2). */
export type Usage = { running: number; meters: Record<string, number> };

export type Live = {
  job: Job;
  usage: Usage;
  models: JobModel[];
  attempts: Attempt[];
  events: JobEvent[];
  cursor: number;
};

/** A job the worker may still be writing to. */
export const TRAINING: JobStatus[] = ["queued", "running", "stopping"];

/** The worker may change any of these underneath the screen, so they are polled. */
export const MOVING: JobStatus[] = ["planning", "queued", "running", "stopping"];

export function stopJob(ws: string, id: number) {
  return api<Job>(`${jobsPath(ws)}/${id}/stop`, { method: "POST" });
}

export function resumeJob(ws: string, id: number) {
  return api<Job>(`${jobsPath(ws)}/${id}/resume`, { method: "POST" });
}

export function attemptCode(ws: string, id: number, model: string, attempt: number) {
  return fetch(`${jobsPath(ws)}/${id}/models/${encodeURIComponent(model)}/attempts/${attempt}/code`)
    .then(async (response) => {
      const text = await response.text();
      if (!response.ok) throw new ApiError(response.status, "That script isn't here any more");
      return text;
    });
}

// ---- the report (Phase 5d)

export type MetricSpec = { name: string; direction: "higher" | "lower"; primary: boolean;
                           drives_improvement: boolean };
export type ScoreRow = {
  rank: number | null; model: string; status: string; selected: boolean;
  test_scores: Record<string, number>; cv_scores: Record<string, number>;
  fold_scores: (number | null)[]; generations: number; repairs: number; executions: number;
  best_attempt: number | null; wall_seconds: number; artifacts: Record<string, string>;
  note: string; eligibility: "clean" | "unverified" | "blocked";
};
export type GenerationStep = { generation: number; attempt: number; status: string; changes: string;
                               score: number | null; delta: number | null; repairs: number;
                               judge_notes: string };
export type FeatureWeight = { feature: string; importance: number; share: number };
export type ErrorBand = { band: string; lower: number | null; upper: number | null; rows: number;
                          mean_absolute_error: number; mean_signed_error: number };
export type ConfusionCell = { actual: string; predicted: string; rows: number };
export type RunWarning = { model: string; severity: "critical" | "warning"; stage: string; message: string };

export type Report = {
  schema_version: string;
  run_id: number;
  generated_at: string;
  status: string;
  dataset: { target: string; task_type: string; train_rows: number; test_rows: number;
             target_stats: Record<string, number> };
  protocol: {
    split_method: string; test_size: number; seed: number; drop_duplicates: boolean;
    cv_strategy: string; cv_folds: number; split_reason: string; split_warnings: string[];
    baseline_strategy: string; baseline_applicable: boolean; baseline_scores: Record<string, number>;
    max_tries: number; early_stopping_patience: number; improvement_delta: number;
    improvement_mode: string; environment: Record<string, string>;
    requirements: Record<string, string>[];
  };
  totals: { models_planned: number; models_scored: number; executions: number; generations: number;
            repairs: number; failures: number; wall_seconds: number };
  metrics: MetricSpec[];
  selected_model: string | null;
  selection_reason: string;
  improvement_over_baseline: number | null;
  comparison: ScoreRow[];
  trace: Record<string, GenerationStep[]>;
  importance: FeatureWeight[];
  error_bands: ErrorBand[];
  confusion: ConfusionCell[];
  warnings: RunWarning[];
  exclusions: Record<string, unknown>[];
  leakage_screen: Leak[];
  narrative: string;
};

export function useReport(ws: string, id: number, ready: boolean) {
  return useQuery({
    enabled: ready,
    queryKey: ["report", ws, id],
    retry: false,
    queryFn: () => api<Report>(`${jobsPath(ws)}/${id}/report`),
  });
}

/** Where a file the run produced is served from; the browser downloads it, nothing parses it here. */
export function fileUrl(ws: string, id: number, model: string, attempt: number, name: string) {
  return `${jobsPath(ws)}/${id}/files/${encodeURIComponent(model)}/${attempt}/${encodeURIComponent(name)}`;
}

/** "runs\1\6\lightgbm\attempt_2\model.joblib" -> "model.joblib"; the report writes host paths. */
export function fileName(path: string): string {
  return path.split(/[\/]/).pop() ?? path;
}
