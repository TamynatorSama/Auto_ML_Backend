-- Phase 4: what a source needs between upload and a locked schema (docs/PHASE4.md §3).

ALTER TABLE sources
    ADD COLUMN status          text NOT NULL DEFAULT 'ready',   -- uploading · checking · profiling · ready · failed · expired
    ADD COLUMN profile         jsonb,
    ADD COLUMN error           text,
    ADD COLUMN created_by      bigint REFERENCES users,
    ADD COLUMN original_name   text,
    ADD COLUMN files_expire_at timestamptz,
    ADD COLUMN last_used_at    timestamptz NOT NULL DEFAULT now();

ALTER TABLE sources ALTER COLUMN path DROP NOT NULL;   -- null once the file has expired
CREATE INDEX ON sources (workspace_id, id DESC);

ALTER TABLE schemas
    ADD COLUMN analysis   jsonb,                        -- the leakage screen's findings
    ADD COLUMN locked_by  bigint REFERENCES users,
    ADD COLUMN created_at timestamptz NOT NULL DEFAULT now(),
    ADD COLUMN updated_at timestamptz NOT NULL DEFAULT now();

-- ingest, profile and analyze belong to a source, not a job, so a task now has one or the other
ALTER TABLE tasks
    ADD COLUMN source_id bigint REFERENCES sources,
    ALTER COLUMN job_id DROP NOT NULL,
    ADD CONSTRAINT tasks_one_owner CHECK ((job_id IS NULL) <> (source_id IS NULL));
CREATE INDEX ON tasks (source_id);
