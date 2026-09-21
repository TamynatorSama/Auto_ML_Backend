-- Phase 5b: what a deleted run leaves for the worker to collect (docs/PHASE5.md §11).
--
-- The API deletes a job's rows, but LangGraph's checkpoint tables and the runs
-- folder are the worker's. Rather than have the sweep guess which directories are
-- orphaned -- runs/ also holds the command line's own history -- the delete records
-- exactly what it left behind, and the sweep deletes that and nothing else.

CREATE TABLE deleted_runs (
    job_id     bigint PRIMARY KEY,     -- no longer a jobs row: also the graph's thread id
    run_dir    text,
    deleted_at timestamptz NOT NULL DEFAULT now()
);
