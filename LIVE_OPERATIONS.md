# Live fixes and operations runbook

This runbook describes the fixes currently in the working tree and the routine
operations for the live beta at `https://tamynator.tailb90c04.ts.net/`.

The repository on the server is `/home/samuel/automl`. The live database is
`automl_live`. Application secrets remain outside the repository in
`/etc/automl/app.env` and `/etc/automl/postgres.env`.

## Changes in this update

### Training screen (`web/src/screens/Training.tsx`)

- **Correct attempt count.** The main Attempt tile now counts every recorded
  attempt, including repairs. Its smaller text separately reports generation
  slots, excluding repairs and ablations, so the two different counts are not
  confused.
- **No page jump when the run log updates.** New log messages scroll only the
  log panel. They no longer use `scrollIntoView`, which could move the entire
  page. If a person scrolls up inside the log, new messages do not pull them
  back down; automatic following resumes when they return to the bottom.
- **Live model-timeline scores.** The timeline reads completed attempts as well
  as the final `job_models` summary. A score such as MAE therefore appears as
  soon as the attempt reaches the live leaderboard, even while that model is
  still running.
- **Contained Live Leaderboard.** The results area is capped at the smaller of
  420 px or 60% of the viewport. It has its own vertical and horizontal scroll,
  contains scroll chaining, reserves scrollbar space to avoid layout shifts,
  and keeps its column headings visible.

### Sandbox OOM handling (`code_gen_eval/runner.py`)

When a sandbox is killed by its memory limit, the runner no longer tries to
download files from the stopped container. That download previously returned a
409 response and hid the real `out_of_memory` result behind an infrastructure
error. The real OOM status now reaches the evaluation flow, allowing it to
record the attempt and continue to the next generation instead of reporting
zero attempts.

`code_gen_eval/tests/test_runner.py` is the regression test for this behavior.
It verifies that normal exited sandboxes are downloaded, OOM-killed sandboxes
are not, and cleanup and usage reporting still occur.

### Fresh-job run-directory safety (`worker/jobs.py`)

The earlier stale-run fix in the working tree removes an old run directory when
a genuinely fresh job starts. This matters after a database reset because job
numbers begin again at 1 while the application-data volume may still contain a
directory belonging to the old job 1. The deletion is restricted to a child of
the configured runs directory.

## Verification already performed

The frontend production build completed successfully:

```powershell
cd D:\code\ML\AutoML\web
npm run build
```

The focused runner regression test also passed (`2 passed`):

```powershell
cd D:\code\ML\AutoML
python -m pytest -p no:cacheprovider code_gen_eval/tests/test_runner.py -q
```

## Deploy all current fixes

Do this when no training job is running, because recreating the worker interrupts
active work.

From PowerShell on the development computer:

```powershell
scp D:\code\ML\AutoML\web\src\screens\Training.tsx samuel@100.68.11.36:/home/samuel/automl/web/src/screens/Training.tsx
scp D:\code\ML\AutoML\code_gen_eval\runner.py samuel@100.68.11.36:/home/samuel/automl/code_gen_eval/runner.py
scp D:\code\ML\AutoML\worker\jobs.py samuel@100.68.11.36:/home/samuel/automl/worker/jobs.py
ssh samuel@100.68.11.36
```

On the server:

```bash
cd /home/samuel/automl
docker compose -f deploy/compose.yaml up -d --build --force-recreate api worker web
docker compose -f deploy/compose.yaml ps
curl --fail http://127.0.0.1:8000/api/health
curl --fail http://127.0.0.1:8080/api/health
```

The regression-test file does not need to be copied to production.

## Completely reset the live beta

This is destructive. It permanently deletes all live accounts, sessions,
workspaces, API-key records, datasets, jobs, attempts, events, reports, uploads,
and run artifacts. It preserves the server configuration and SMTP/OAuth secrets
in `/etc/automl`.

First stop the services that connect to the database:

```bash
cd /home/samuel/automl
docker compose -f deploy/compose.yaml stop api worker
```

Remove the persistent uploads and run artifacts without deleting either Docker
volume:

```bash
docker compose -f deploy/compose.yaml run --rm --no-deps api \
  python -c "import shutil; shutil.rmtree('/app/state/runs', ignore_errors=True); shutil.rmtree('/app/state/uploads', ignore_errors=True)"
```

Drop and recreate only the live database:

```bash
docker exec automl-postgres-1 \
  psql -U automl -d postgres \
  -c "DROP DATABASE IF EXISTS automl_live WITH (FORCE);"

docker exec automl-postgres-1 \
  createdb -U automl automl_live
```

Rebuild and start the application. The API applies the schema migrations to the
new database:

```bash
docker compose -f deploy/compose.yaml up -d --build --force-recreate api worker web
docker compose -f deploy/compose.yaml ps
curl --fail http://127.0.0.1:8000/api/health
curl --fail http://127.0.0.1:8080/api/health
```

The reset deletes the sandbox-host row, so register it again:

```bash
docker compose -f deploy/compose.yaml exec -T worker \
  python -m worker.cli add-host http://127.0.0.1:8765 \
  --token-env AUTOML_SANDBOX_TOKEN \
  --max-running-jobs 6
```

Finally, add the tester emails again as described below.

Do **not** use `docker compose down -v`. It removes the named Postgres and
application-data volumes, including databases that are not part of this reset.

## Manage the test signup list

Add one or several email addresses:

```bash
cd /home/samuel/automl
docker compose -f deploy/compose.yaml exec -T worker \
  python -m worker.cli allow first@example.com second@example.com
```

Addresses are normalized to lowercase, duplicates are ignored, and the command
prints the resulting allowlist.

Remove addresses from the signup list:

```bash
docker compose -f deploy/compose.yaml exec -T worker \
  python -m worker.cli disallow first@example.com
```

Disallowing an address prevents a new signup only. It does not disable or delete
an account that has already been created.

To inspect the list directly:

```bash
docker exec automl-postgres-1 \
  psql -U automl -d automl_live \
  -c "SELECT email, added_at FROM signup_allowlist ORDER BY email;"
```

## Create temporary SSH access

The person or tool that will connect should generate the key pair. Only the
public key is added to the server; never send or paste the private key.

On the Windows computer, create a uniquely named temporary key in the user's
temporary directory:

```powershell
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$keyComment = "codex-temp-$stamp"
$keyPath = Join-Path $env:TEMP $keyComment
ssh-keygen -t ed25519 -f $keyPath -C $keyComment
```

Press Enter twice when asked for a passphrase if the key must be usable by an
unattended diagnostic session. Keep the access window short and revoke it as
soon as the work is finished.

Display the public key:

```powershell
Get-Content "$keyPath.pub"
```

Sign in to the server with the normal administrator access, then append that
single public-key line. Prefixing it with `restrict` disables forwarding, PTY,
and agent/X11 features while still allowing commands and file copies:

```bash
umask 077
mkdir -p ~/.ssh
touch ~/.ssh/authorized_keys
chmod 700 ~/.ssh
chmod 600 ~/.ssh/authorized_keys
printf '%s\n' 'restrict ssh-ed25519 PASTE_THE_PUBLIC_KEY_BODY_AND_COMMENT_HERE' >> ~/.ssh/authorized_keys
```

Do not literally keep `PASTE_THE_PUBLIC_KEY_BODY_AND_COMMENT_HERE`; replace the
entire example key with the one printed by PowerShell. Do not add `restrict` a
second time if it is already present.

Test the temporary key from PowerShell:

```powershell
ssh -i $keyPath -o IdentitiesOnly=yes samuel@100.68.11.36 "hostname; whoami"
```

When the work is complete, revoke the key on the server using its unique comment:

```bash
sed -i '/ codex-temp-YYYYMMDD-HHMMSS$/d' ~/.ssh/authorized_keys
```

Replace the timestamp with the comment used when the key was created. Verify
that the temporary key is refused:

```powershell
ssh -i $keyPath -o IdentitiesOnly=yes -o BatchMode=yes samuel@100.68.11.36 "true"
```

The verification must end with `Permission denied (publickey)`. Only after that
confirmation, delete both local key files:

```powershell
Remove-Item -LiteralPath $keyPath, "$keyPath.pub" -Force
```
