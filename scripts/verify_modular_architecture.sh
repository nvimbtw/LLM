#!/usr/bin/env bash
# Run the documented FEAT-4 automated gates and retain separate logs/statuses.
# Module groups may be selected explicitly; fixture browser checks are not live acceptance.
set -uo pipefail

repository_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd) || exit 2
engine_python=${SLATE_ENGINE_PYTHON:-"$repository_dir/llm-engine/.venv/bin/python"}
if [[ ! -x "$engine_python" && -z ${SLATE_ENGINE_PYTHON:-} ]]; then
    engine_python="$repository_dir/engine/.venv/bin/python"
fi
if [[ ${1:-} == --help ]]; then
    printf 'Usage: bash scripts/verify_modular_architecture.sh [contracts|engine|backend|frontend|browser ...]\nDefault: contracts engine backend frontend browser\nSee docs/llm-engine/modular-architecture-acceptance.md for prerequisites and live scenarios.\n'
    exit 0
fi
groups=("$@")
if [[ ${#groups[@]} -eq 0 ]]; then
    groups=(contracts engine backend frontend browser)
fi
for group in "${groups[@]}"; do
    case "$group" in contracts|engine|backend|frontend|browser) ;;
        *) printf 'Unknown verification group: %s\n' "$group" >&2; exit 2 ;;
    esac
done
report_dir=${FEAT4_REPORT_DIR:-$(mktemp -d /tmp/feat4-acceptance.XXXXXX)}
mkdir -p -- "$report_dir" || exit 2
report_dir=$(cd -- "$report_dir" && pwd) || exit 2
status_file="$report_dir/status.tsv"
if [[ -e "$status_file" ]]; then
    printf 'Refusing to overwrite an existing acceptance report: %s\n' "$status_file" >&2
    exit 2
fi
printf 'check\texit_code\n' > "$status_file" || exit 2
failed=0

# Run one named check in its module and record its exact exit status without hiding later failures.
run_check() {
    local name=$1 directory=$2 result
    shift 2
    printf 'Running %s\n' "$name"
    (cd -- "$repository_dir/$directory" && "$@") > "$report_dir/$name.log" 2>&1
    result=$?
    printf '%s\t%s\n' "$name" "$result" >> "$status_file"
    if [[ $result -ne 0 ]]; then failed=1; fi
    printf '%s exited %s (%s)\n' "$name" "$result" "$report_dir/$name.log"
}

for group in "${groups[@]}"; do
    case "$group" in
        contracts)
            run_check schemas llm-engine "$engine_python" -m llm.architecture.contracts --check
            run_check types frontend node scripts/generate-architecture-types.mjs --check
            ;;
        engine)
            run_check engine llm-engine "$engine_python" -m pytest tests --junitxml="$report_dir/engine.xml"
            ;;
        backend)
            run_check rust-build backend env SQLX_OFFLINE=true cargo build --workspace
            run_check rust-clippy backend env SQLX_OFFLINE=true cargo clippy --workspace --all-targets -- -D warnings
            run_check rust-tests backend env SQLX_OFFLINE=true cargo test --workspace
            run_check rust-python-privacy backend env SQLX_OFFLINE=true SLATE_ENGINE_PYTHON="$engine_python" cargo test -p api --test architecture_proxy live_python -- --ignored --nocapture
            ;;
        frontend)
            run_check frontend-types frontend bun run typecheck
            run_check frontend-lint frontend bun run lint
            run_check frontend-tests frontend bun test
            run_check frontend-build frontend bun run build
            ;;
        browser)
            run_check authoring-browser-fixture frontend node scripts/check-authoring-browser.mjs
            run_check graph-browser-fixture frontend node scripts/check-architecture-graph-browser.mjs
            ;;
    esac
done
printf 'Reports: %s\nRequired skips and live scenarios must still be audited; exit zero alone does not complete FEAT-4.\n' "$report_dir"
exit "$failed"
