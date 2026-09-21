-- Phase 5: the workspace's own model key, and a job's two limits (docs/PHASE5.md §3).

CREATE TABLE provider_keys (
    id            bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    workspace_id  bigint NOT NULL REFERENCES workspaces,
    provider      text NOT NULL,                    -- gemini only for now (A2)
    ciphertext    text NOT NULL,                    -- Fernet, AUTOML_SECRET_KEY
    last4         text NOT NULL,                    -- all the screen ever shows
    verified_at   timestamptz NOT NULL DEFAULT now(),
    created_by    bigint REFERENCES users,
    created_at    timestamptz NOT NULL DEFAULT now(),
    UNIQUE (workspace_id, provider)
);

-- the two arguments hooks.set_limits takes (A6, P8); null means no limit
ALTER TABLE jobs
    ADD COLUMN budget_usd       double precision,
    ADD COLUMN deadline_seconds int;

CREATE INDEX ON jobs (workspace_id, id DESC);
CREATE INDEX ON jobs (source_id);
