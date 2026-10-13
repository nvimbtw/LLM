#!/usr/bin/env bash
# Stop writers and preserve the current installation before ownership transfer.
# Run as the ordinary repository user; only Docker and artifact reads use sudo.
set -euo pipefail
umask 077

if (( EUID == 0 )); then
    echo "Run with bash as your normal user; the script requests sudo when needed." >&2
    exit 1
fi

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
backup_root="${1:-$HOME/llm-sprint-backups}"
mkdir -p -- "$backup_root"
backup_dir="$(mktemp -d "$backup_root/pre-cutover-$(date +%Y%m%d-%H%M%S)-XXXXXX")"
printf 'Backup directory: %s\n' "$backup_dir"

sudo -v
running="$(sudo docker ps --format '{{.Names}}')"
while IFS= read -r container; do
    case "$container" in
        slate-backend-1|slate-frontend-1|slate-llm-engine-1|slate-bench-engine-1|slate-data-engine-1|slate-gyms-engine-1|slate-engine-1)
            sudo docker stop "$container"
            ;;
    esac
done <<< "$running"

# pg_dumpall includes roles and every database, including an older bench service
# database if one exists. It may contain password hashes: keep this file private.
sudo docker exec slate-postgres-1 sh -c 'exec pg_dumpall -U "$POSTGRES_USER"' \
    > "$backup_dir/databases.sql.partial"
test -s "$backup_dir/databases.sql.partial"
mv -- "$backup_dir/databases.sql.partial" "$backup_dir/databases.sql"

artifact_paths=()
for path in llm-engine/data engine/data bench-engine/data bench-engine/eval_data; do
    if [[ -d "$repo_dir/$path" ]]; then
        artifact_paths+=("$path")
    fi
done
if (( ${#artifact_paths[@]} )); then
    sudo tar -C "$repo_dir" -cf - -- "${artifact_paths[@]}" > "$backup_dir/artifacts.tar.partial"
    mv -- "$backup_dir/artifacts.tar.partial" "$backup_dir/artifacts.tar"
fi

sql() {
    sudo docker exec slate-postgres-1 sh -c \
        'exec psql -X -U "$POSTGRES_USER" -d "$1" -At -v ON_ERROR_STOP=1 -c "$2"' \
        sh "$1" "$2"
}

databases="$(sql postgres "SELECT datname FROM pg_database WHERE NOT datistemplate ORDER BY datname")"
{
    printf 'Databases:\n%s\n' "$databases"
    for database in llm slate_engine slate_bench; do
        if [[ $'\n'"$databases"$'\n' != *$'\n'"$database"$'\n'* ]]; then
            continue
        fi
        printf '\n%s tables:\n' "$database"
        sql "$database" "SELECT tablename FROM pg_tables WHERE schemaname='public' ORDER BY tablename"
        if [[ "$(sql "$database" "SELECT to_regclass('public.alembic_version') IS NOT NULL")" == t ]]; then
            printf 'Alembic revision: '
            sql "$database" 'SELECT version_num FROM alembic_version'
        fi
    done
} | tee "$backup_dir/inventory.txt"

(
    cd -- "$backup_dir"
    sha256sum databases.sql > SHA256SUMS
    if [[ -f artifacts.tar ]]; then
        sha256sum artifacts.tar >> SHA256SUMS
    fi
)
printf '\nBackup complete: %s\n' "$backup_dir"
du -h -- "$backup_dir/databases.sql" "$backup_dir"/artifacts.tar 2>/dev/null || true
echo 'Writers remain stopped. No migrations or transfers have been run.'
echo 'Share inventory.txt, not databases.sql or the artifact archive.'
