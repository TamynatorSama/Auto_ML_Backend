-- Phase 2: jobs, the queue and the usage ledger (docs/PHASE2.md §3).
-- Users, sessions, memberships and model keys arrive in later phases.

CREATE TABLE workspaces (
    id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    slug        text NOT NULL UNIQUE,
    name        text NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE sandbox_hosts (
    id                  bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    owner_workspace_id  bigint REFERENCES workspaces,     -- null: the platform's own
    url                 text NOT NULL,
    token_ciphertext    text NOT NULL,                    -- Fernet, AUTOML_SECRET_KEY
    token_last4         text NOT NULL,
    runtime_hash        text,
    max_running_jobs    int NOT NULL DEFAULT 1,
    status              text NOT NULL DEFAULT 'active',   -- active · unreachable · incompatible · disabled
    last_seen_at        timestamptz
);

CREATE TABLE sources (
    id            bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    workspace_id  bigint NOT NULL REFERENCES workspaces,
    name          text NOT NULL,
    path          text NOT NULL,
    rows          bigint,
    columns       int,
    bytes         bigint,
    created_at    timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE schemas (
    id         bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    source_id  bigint NOT NULL REFERENCES sources,
    version    int NOT NULL,
    status     text NOT NULL DEFAULT 'draft',   -- draft · locked
    columns    jsonb NOT NULL,                  -- the pipeline's DataSchema
    locked_at  timestamptz,
    UNIQUE (source_id, version)
);

-- the job id is also the graph's thread id and the pipeline's run_id
CREATE TABLE jobs (
    id                bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    workspace_id      bigint NOT NULL REFERENCES workspaces,
    source_id         bigint NOT NULL REFERENCES sources,
    schema_id         bigint NOT NULL REFERENCES schemas,
    sandbox_host_id   bigint NOT NULL REFERENCES sandbox_hosts,
    name              text NOT NULL,
    -- planning · review · queued · running · stopping · succeeded · stopped · failed
    status            text NOT NULL DEFAULT 'planning',
    plan              jsonb,
    edits             jsonb,
    report            jsonb,
    run_dir           text,
    cancel_requested  boolean NOT NULL DEFAULT false,
    error             text,
    started_by        bigint,                   -- a user, from Phase 3
    created_at        timestamptz NOT NULL DEFAULT now(),
    started_at        timestamptz,
    finished_at       timestamptz,
    usage_totals      jsonb,
    files_expire_at   timestamptz,
    files_expired     boolean NOT NULL DEFAULT false
);

CREATE TABLE job_models (
    job_id        bigint NOT NULL REFERENCES jobs,
    model         text NOT NULL,
    status        text NOT NULL,
    generation    int,
    best_attempt  int,
    best_cv       jsonb,
    test_scores   jsonb,
    PRIMARY KEY (job_id, model)
);

-- append-only: a resumed model numbers its attempts from 1 again
CREATE TABLE job_attempts (
    id            bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    job_id        bigint NOT NULL REFERENCES jobs,
    model         text NOT NULL,
    attempt       int NOT NULL,
    generation    int,
    kind          text,
    status        text NOT NULL,
    cv_scores     jsonb,
    wall_seconds  double precision,
    changes       text,
    usage         jsonb,
    at            timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX ON job_attempts (job_id);

CREATE TABLE job_events (
    id      bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,   -- the live view's cursor
    job_id  bigint NOT NULL REFERENCES jobs,
    at      timestamptz NOT NULL DEFAULT now(),
    kind    text NOT NULL,
    model   text,
    data    jsonb NOT NULL
);
CREATE INDEX ON job_events (job_id, id);

-- the ledger, append-only (D10)
CREATE TABLE usage_records (
    id               bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    workspace_id     bigint NOT NULL REFERENCES workspaces,
    job_id           bigint REFERENCES jobs,
    sandbox_host_id  bigint REFERENCES sandbox_hosts,
    source           text NOT NULL,             -- sandbox · llm
    meter            text NOT NULL,
    quantity         double precision NOT NULL,
    unit             text NOT NULL,
    model            text,                      -- sandbox: the model trained; llm: the language model
    attempt          int,
    role             text,
    occurred_at      timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX ON usage_records (job_id);
CREATE INDEX ON usage_records (workspace_id, occurred_at);

CREATE TABLE tasks (
    id            bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    kind          text NOT NULL,                     -- plan · train
    job_id        bigint NOT NULL REFERENCES jobs,
    status        text NOT NULL DEFAULT 'queued',    -- queued · running · done · failed · cancelled
    payload       jsonb NOT NULL DEFAULT '{}',
    tries         int NOT NULL DEFAULT 0,
    claimed_at    timestamptz,
    heartbeat_at  timestamptz,
    error         text
);
CREATE INDEX ON tasks (status, id);
