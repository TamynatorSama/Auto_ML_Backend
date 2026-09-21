#!/usr/bin/env bash
# Ship the sandbox server change (engine.py: cpu_shares + a burst ceiling) and rebuild.
# Run from the repo root in Git Bash:   bash deploy_sandbox.sh
# One password prompt: the connection is shared between every step below.
set -euo pipefail

SERVER="${SERVER:-samuel@100.68.11.36}"
REMOTE="${REMOTE:-}"                  # the repo on the server; found below if not set

CTL="$(mktemp -d)/ssh-%C"
trap 'ssh -O exit -o ControlPath="$CTL" "$SERVER" 2>/dev/null || true' EXIT
SSH=(ssh -o ControlMaster=auto -o ControlPath="$CTL" -o ControlPersist=120)

echo "==> connecting to $SERVER (one password prompt)"
"${SSH[@]}" "$SERVER" true

if [ -z "$REMOTE" ]; then
  echo "==> finding the repo on the server"
  REMOTE=$("${SSH[@]}" "$SERVER" '
    dir=$(docker inspect $(docker ps -aq --filter name=sandbox | head -1) \
          --format "{{index .Config.Labels \"com.docker.compose.project.working_dir\"}}" 2>/dev/null)
    [ -n "$dir" ] && echo "${dir%/sandbox/deploy}" && exit
    find ~ /opt /srv -maxdepth 5 -path "*/sandbox/deploy/compose.yaml" 2>/dev/null |
      head -1 | sed "s#/sandbox/deploy/compose.yaml##"
  ')
  [ -n "$REMOTE" ] || { echo "Could not find it. Set it: REMOTE=/path bash deploy_sandbox.sh" >&2; exit 1; }
  echo "    $REMOTE"
fi

"${SSH[@]}" "$SERVER" "test -f '$REMOTE/sandbox/deploy/compose.yaml'" || {
  echo "No sandbox/deploy/compose.yaml under $REMOTE on $SERVER." >&2
  echo "Set the right path:  REMOTE=/your/path bash deploy_sandbox.sh" >&2
  exit 1
}

echo "==> is anything training right now?"
running=$("${SSH[@]}" "$SERVER" "docker ps -q --filter name=automl-sbx- | wc -l")
if [ "$running" -gt 0 ]; then
  echo "$running sandbox(es) still running. A rebuild loses their exec records and the" >&2
  echo "worker will read that as the host being down. Stop the job first." >&2
  exit 1
fi

echo "==> copying engine.py"
scp -o ControlPath="$CTL" sandbox/automl_sandbox/engine.py "$SERVER:$REMOTE/sandbox/automl_sandbox/engine.py"

echo "==> rebuilding the sandbox server"
"${SSH[@]}" "$SERVER" "cd '$REMOTE' && docker compose -f sandbox/deploy/compose.yaml up -d --build"

echo "==> health"
"${SSH[@]}" "$SERVER" "curl -fsS 127.0.0.1:8765/v1/health" && echo

echo
echo "Done (repo: $REMOTE)."
cat <<'NOTE'

While the next run trains, check the weights took (one line, run it as-is):

  ssh SERVER 'docker inspect --format "{{.Name}} shares={{.HostConfig.CpuShares}} nano={{.HostConfig.NanoCpus}}" $(docker ps -q --filter name=automl-sbx-)'

  want: more than one line, shares non-zero (was 0), nano=4000000000 (was 3000000000)
NOTE
