import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useNavigate, useParams } from "react-router-dom";

import {
  api, ApiError, createJob, LIVE, TRAINING, useJob, useJobs, useProviders, useSource, useSources,
  type Job, type Me, type Source, type Workspace,
} from "../api";
import { Shell, type Source as Branch, type Step, type StepName } from "../shell/Shell";
import { ConsoleSkeleton } from "../ui";
import { Config } from "./Config";
import { Results } from "./Results";
import { Training } from "./Training";
import { Data } from "./Data";
import { Preview } from "./Preview";
import { Schema } from "./Schema";

/** The setup checklist (G46), against what the workspace actually has. */
function setup(sources: Source[], jobs: Job[], keyed: boolean): Step[] {
  const locked = sources.some((source) => source.schema?.status === "locked");
  const ran = jobs.some((job) => job.status === "succeeded");
  return [
    { label: "Workspace created", meta: "done", done: true },
    { label: "Add a model key", meta: keyed ? "done" : "now", done: keyed },
    { label: "Connect a source", meta: sources.length ? "done" : keyed ? "now" : "next", done: sources.length > 0 },
    { label: "Lock a schema", meta: locked ? "done" : sources.length ? "next" : "—", done: locked },
    { label: "Run your first job", meta: ran ? "done" : jobs.length ? `${jobs.length} started` : locked ? "next" : "—",
      done: ran },
  ];
}

// what the rail says a run is doing, in a word
const JOB_NOTE: Record<string, { note: string; tone: "live" | "waiting" | "done" | "bad" }> = {
  planning: { note: "planning", tone: "waiting" },
  review: { note: "config", tone: "waiting" },
  queued: { note: "queued", tone: "waiting" },
  running: { note: "training", tone: "live" },
  stopping: { note: "stopping", tone: "waiting" },
  succeeded: { note: "done", tone: "done" },
  stopped: { note: "stopped", tone: "bad" },
  failed: { note: "failed", tone: "bad" },
};

/** Which of a dataset's rows the screen you are looking at belongs to. */
function rowFor(step: StepName): "name" | "schema" | "job" {
  if (step === "Schema") return "schema";
  return step === "Data" || step === "Preview" ? "name" : "job";
}

/** A dataset as the rail draws it, with its schema and newest run under it. */
function branch(source: Source, job: Job | undefined, active: boolean, on: Branch["on"],
                go: () => void, goSchema: () => void, goJob: () => void): Branch {
  const said = job && (JOB_NOTE[job.status] ?? { note: job.status, tone: "waiting" as const });
  return {
    name: source.name,
    active,
    on,
    go,
    schema: source.schema
      ? { status: source.schema.status, version: source.schema.version, go: goSchema }
      : undefined,
    // a run defaults to its dataset's name, and the rail already says that above it;
    // when it hasn't been named something of its own, its number identifies it better
    job: job && said && {
      name: job.name === source.name ? `run ${job.id}` : job.name,
      note: said.note, tone: said.tone, go: goJob,
    },
  };
}

export function Console({ me, at, adding = false }: {
  me: Me;
  at: StepName;
  /** /:ws/new — the Data screen with nothing picked, so a new dataset can be dropped in */
  adding?: boolean;
}) {
  const { ws = "", sourceId, jobId } = useParams();
  const navigate = useNavigate();

  const workspace = useQuery({
    queryKey: ["workspace", ws],
    retry: false,
    queryFn: () => api<Workspace>(`/api/w/${encodeURIComponent(ws)}`),
  });
  const sources = useSources(ws);
  const jobs = useJobs(ws);
  const providers = useProviders(ws);

  const listing = sources.data?.sources ?? [];
  const runs = jobs.data?.jobs ?? [];
  const keyed = (providers.data?.providers ?? []).some((provider) => provider.last4);

  // the job in the address, then the source it belongs to, else the source in the address, else the newest
  const openJob = jobId ? Number(jobId) : null;
  const one = useJob(ws, openJob);
  const picked = one.data?.source.id ?? (sourceId ? Number(sourceId) : listing[0]?.id ?? null);
  // the hook is always called, whatever the route: React counts hooks, not branches
  const chosen = useSource(ws, picked);
  const source = adding ? null
    : chosen.data ?? listing.find((row) => row.id === picked) ?? null;
  // the newest run of the source on screen: its sidebar row, and what Configure opens
  const latest = (id: number | null) => runs.find((job) => job.source.id === id);
  const current = openJob ?? latest(picked)?.id ?? null;

  const go = (id: number | null, screen: "" | "preview" | "schema" = "") =>
    navigate(id === null ? `/${ws}` : `/${ws}/s/${id}${screen ? `/${screen}` : ""}`);
  const goJob = (id: number) => navigate(`/${ws}/j/${id}`);

  /**
   * Picking a source in the rail lands on how far it actually got, not back at
   * Data: a finished run opens its results, a live one its training, a drafted
   * plan its config, and a source with no run its schema.
   */
  const goFurthest = (row: Source) => {
    const run = latest(row.id);
    if (run) {
      // a failed run wants its error, which is on the job screen; a report it may
      // never have written is not the place to land
      if (["succeeded", "stopped"].includes(run.status)) {
        return navigate(`/${ws}/j/${run.id}/results`);
      }
      if (TRAINING.includes(run.status)) return navigate(`/${ws}/j/${run.id}/train`);
      return goJob(run.id);           // planning, waiting for review, or failed
    }
    return go(row.id, row.schema ? "schema" : row.status === "ready" ? "preview" : "");
  };

  if (workspace.isPending) return <ConsoleSkeleton />;
  if (workspace.error) {
    const missing = workspace.error instanceof ApiError && workspace.error.status === 404;
    return (
      <div style={{ padding: 40, color: "var(--muted)" }}>
        {missing ? "No such workspace, or you're not a member of it." : (workspace.error as Error).message}
      </div>
    );
  }

  // Preview and Schema need a source, Configure and Train need a job; without one the address falls back
  const run = one.data ?? runs.find((job) => job.id === current) ?? null;
  const trained = run !== null && !["planning", "review"].includes(run.status);
  const ended = run !== null && ["succeeded", "stopped", "failed"].includes(run.status);
  // nothing downstream is offered while a new dataset is being added: those steps
  // belong to a run on a source you are not looking at
  const forJob: StepName[] = !current || adding ? []
    : ended ? ["Configure", "Train", "Results"] : trained ? ["Configure", "Train"] : ["Configure"];
  const wanted: StepName = (at === "Train" && !trained) || (at === "Results" && !ended) ? "Configure" : at;
  const showing: StepName = adding ? "Data"
    : forJob.includes(wanted) ? wanted
      : ["Preview", "Schema"].includes(wanted) && source ? wanted : "Data";
  const reached: StepName[] = [
    "Data",
    ...(source && source.status === "ready" ? ["Preview", "Schema"] as StepName[] : []),
    ...forJob,
  ];
  const step = (name: StepName) => (name === "Preview" ? "preview" : name === "Schema" ? "schema" : "");

  return (
    <Shell me={me} workspace={workspace.data} steps={setup(listing, runs, keyed)} at={showing} reached={reached}
           sources={listing.map((row) => branch(row, latest(row.id), row.id === picked,
                                                 row.id === picked && !adding ? rowFor(showing)
                                                                              : undefined,
                                                 () => goFurthest(row),
                                                 () => go(row.id, "schema"),
                                                 () => goJob(latest(row.id)!.id)))}
           onStep={(name) => (name === "Configure" ? goJob(current!)
             : name === "Train" ? navigate(`/${ws}/j/${current}/train`)
               : name === "Results" ? navigate(`/${ws}/j/${current}/results`)
                 : go(picked, step(name)))}
           onAddSource={() => navigate(`/${ws}/new`)}
           action={<NewJob ws={ws} source={source} jobs={runs} keyed={keyed} onStarted={goJob} />}>
      {showing === "Results" && run
        ? <Results ws={ws} job={run} onTraining={() => navigate(`/${ws}/j/${run.id}/train`)} />
        : showing === "Train" && run
        ? <Training ws={ws} job={run} onResults={() => navigate(`/${ws}/j/${run.id}/results`)} />
        : showing === "Configure" && current
        ? <Config ws={ws} jobId={current} onBack={() => go(picked, "schema")}
                  onGone={() => go(picked, "schema")}
                  onStarted={() => navigate(`/${ws}/j/${current}/train`)} />
        : showing === "Schema" && source ? <Schema ws={ws} source={source} />
          : showing === "Preview" && source
            ? <Preview ws={ws} source={source} onSchema={() => go(source.id, "schema")} />
            : <Data ws={ws} source={source} job={source ? latest(source.id) ?? null : null}
                    first={listing.length === 0}
                    onPreview={(id) => go(id, "preview")} onPick={(id) => go(id)}
                    onSchema={(id) => go(id, "schema")} onJob={goJob} />}
    </Shell>
  );
}

/**
 * Starting a run needs a locked schema, a model key, and no unfinished run on
 * that dataset (D4). The button says which of those is missing rather than
 * sitting there disabled with no reason.
 */
function NewJob({ ws, source, jobs, keyed, onStarted }: {
  ws: string;
  source: Source | null;
  jobs: Job[];
  keyed: boolean;
  onStarted: (id: number) => void;
}) {
  const queries = useQueryClient();
  const start = useMutation({
    mutationFn: () => createJob(ws, source!.id),
    onSuccess: (job) => {
      queries.invalidateQueries({ queryKey: ["jobs", ws] });
      onStarted(job.id);
    },
  });

  const unfinished = jobs.find((job) => job.source.id === source?.id && LIVE.includes(job.status));
  const locked = source?.schema?.status === "locked";
  const why = !source ? "Connect a dataset first"
    : !keyed ? "Add a model key under Settings first"
      : !locked ? "Lock the schema first"
        : unfinished ? "This dataset already has a run that hasn't finished"
          : undefined;

  return (
    <button className="btn" disabled={Boolean(why) || start.isPending}
            title={why ?? (start.error as Error | null)?.message}
            onClick={() => start.mutate()}>
      {start.isPending ? "Starting…" : start.error ? "Couldn't start" : "New job"}
    </button>
  );
}
