-- Phase 3: accounts and tenancy (docs/PHASE3.md §3).
-- Emails are stored lower-cased; tokens only as their SHA-256.

CREATE TABLE users (
    id                 bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    email              text NOT NULL UNIQUE CHECK (email = lower(email)),
    name               text NOT NULL,
    password_hash      text,                     -- argon2id; null for social-only
    email_verified_at  timestamptz,
    created_at         timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE oauth_identities (
    id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    user_id     bigint NOT NULL REFERENCES users ON DELETE CASCADE,
    provider    text NOT NULL,                   -- google · github
    subject     text NOT NULL,                   -- the provider's id for the person
    email       text,
    created_at  timestamptz NOT NULL DEFAULT now(),
    UNIQUE (provider, subject)
);

CREATE TABLE sessions (
    id            text PRIMARY KEY,              -- SHA-256 of the cookie's token
    user_id       bigint NOT NULL REFERENCES users ON DELETE CASCADE,
    created_at    timestamptz NOT NULL DEFAULT now(),
    expires_at    timestamptz NOT NULL,
    last_seen_at  timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX ON sessions (user_id);

CREATE TABLE memberships (
    user_id       bigint NOT NULL REFERENCES users ON DELETE CASCADE,
    workspace_id  bigint NOT NULL REFERENCES workspaces ON DELETE CASCADE,
    role          text NOT NULL DEFAULT 'owner',
    created_at    timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, workspace_id)
);
CREATE INDEX ON memberships (workspace_id);

CREATE TABLE email_tokens (
    id          text PRIMARY KEY,                -- SHA-256 of the token
    user_id     bigint NOT NULL REFERENCES users ON DELETE CASCADE,
    purpose     text NOT NULL,                   -- verify · reset
    created_at  timestamptz NOT NULL DEFAULT now(),
    expires_at  timestamptz NOT NULL,
    used_at     timestamptz
);
CREATE INDEX ON email_tokens (user_id);

CREATE TABLE signup_allowlist (
    email     text PRIMARY KEY CHECK (email = lower(email)),
    added_at  timestamptz NOT NULL DEFAULT now()
);

-- failed log-ins, for throttling; kept 15 minutes
CREATE TABLE login_failures (
    email  text NOT NULL,
    at     timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX ON login_failures (email, at);

ALTER TABLE jobs ADD FOREIGN KEY (started_by) REFERENCES users;
