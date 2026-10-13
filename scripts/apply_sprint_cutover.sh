#!/usr/bin/env bash
# Apply the backed-up Compose installation's ownership transfer, without tests.
set -euo pipefail
umask 077

if (( EUID == 0 )); then
    echo 'Run as your normal user; the script requests sudo for Docker.' >&2
    exit 1
fi
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
backup_dir="${1:-}"
if [[ -z "$backup_dir" ]]; then
    backup_dir="$(python - <<'PY'
from pathlib import Path
backups = sorted((Path.home() / 'llm-sprint-backups').glob('pre-cutover-*'))
if not backups:
    raise SystemExit('Run prepare_sprint_cutover.sh first')
print(backups[-1])
PY
)"
fi
printf 'Using backup: %s\n' "$backup_dir"
test -s "$backup_dir/databases.sql"
test -s "$backup_dir/artifacts.tar"
(cd -- "$backup_dir" && sha256sum --check SHA256SUMS)

sudo -v
running="$(sudo docker ps --format '{{.Names}}')"
while IFS= read -r container; do
    case "$container" in
        slate-backend-1|slate-frontend-1|slate-llm-engine-1|slate-bench-engine-1|slate-data-engine-1|slate-gyms-engine-1|slate-engine-1)
            printf 'Writer %s is running. Run prepare_sprint_cutover.sh again.\n' "$container" >&2
            exit 1
            ;;
    esac
done <<< "$running"

cd -- "$repo_dir"
# Override the service entrypoint so its normal startup migration cannot run
# before the copy. The existing LLM image already has psycopg/Alembic installed.
sudo docker compose run --rm --no-deps -T --entrypoint python \
    -v "$repo_dir:/workspace" llm-engine \
    /workspace/scripts/apply_sprint_cutover.py --apply

echo 'Transfer complete; application writers are still stopped.'
echo 'Next: recreate bench-engine with its new image, then start the application services.'
