#!/usr/bin/env python3
"""
Fetch every corpus in one command.

There are now eight fetch scripts. Running them by hand means remembering eight
budgets, noticing which of them died overnight, and knowing which are safe to
run at the same time. This drives all of them: it plans the whole fetch, checks
there is disk for it, runs them with bounded parallelism, records what finished,
and can be re-run to pick up exactly what did not.

    python3 scripts/prepare_all.py --dry-run     # the plan, and what it costs
    python3 scripts/prepare_all.py --profile smoke   # ~350 MB, validates the pipeline
    python3 scripts/prepare_all.py               # the real fetch, ~54 GB
    python3 scripts/prepare_all.py               # again: resumes, skipping what is done

## Resume

Each step's completion is recorded in `data/.prepare_state.json` together with a
hash of the arguments it ran with. A re-run skips a step that finished with the
*same* arguments and re-runs one whose budget or flags changed — so raising
`--profile` does the right thing without a manual clean. `--force` ignores the
state entirely.

This matters more than it sounds: a 54 GB fetch is hours of network, and losing
all of it to one failed step at hour six is the failure mode this exists to
prevent. Steps are independent, so one failure never invalidates another.

## Parallelism

Steps share nothing — different repos, different output folders — so they run
concurrently. The default of 3 is a compromise: these are network-bound, but
past a handful of streams the Hub throttles and the disk starts seeking between
writers. Each step's output goes to its own log under `data/.prepare_logs/`,
and a status line is printed as steps finish, because interleaving eight
progress streams onto one terminal is unreadable.

## Profiles

    smoke      ~350 MB total   every source, tiny — proves the pipeline works
    standard   ~54 GB total    the real corpus (default)
    max        ~105 GB total   as much as these sources cheaply give

Per-source budgets are in the table below; `--only` and `--skip` select subsets.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import paths

HERE = os.path.dirname(os.path.abspath(__file__))
STATE_PATH = os.path.join(paths.DATA, ".prepare_state.json")
LOG_DIR = os.path.join(paths.DATA, ".prepare_logs")

PROFILES = ("smoke", "standard", "max")

# Headroom over the raw text budget: corpus.bin is ~2 bytes per token (~0.9x the
# text), the HuggingFace cache holds the compressed shards, and a pack needs room
# for its .partial. 2.5x the text budget is comfortable and still honest.
DISK_FACTOR = 2.5


@dataclass
class Step:
    """One fetch script, with a budget per profile."""

    name: str
    script: str
    kind: str  # pretrain | sft | migrate
    # Megabytes of text per profile. 0 means the step takes no --mb.
    budgets: dict[str, float] = field(default_factory=dict)
    # Conversations per profile, for the SFT steps that cap by count instead.
    limits: dict[str, int] = field(default_factory=dict)
    heldout_mb: float = 0.0
    extra: list[str] = field(default_factory=list)
    note: str = ""

    def mb(self, profile: str) -> float:
        return self.budgets.get(profile, 0.0)

    def argv(self, profile: str) -> list[str]:
        argv = [sys.executable, os.path.join(HERE, self.script)]
        if self.budgets:
            argv += ["--mb", str(self.mb(profile))]
        if self.limits and self.limits.get(profile):
            argv += ["--limit", str(self.limits[profile])]
        # Held-out text is only worth taking on a real fetch; on a smoke run it
        # would be a larger fraction of the corpus than the training text.
        if self.heldout_mb and profile != "smoke":
            argv += ["--heldout-mb", str(self.heldout_mb)]
        return argv + self.extra


STEPS = [
    Step("migrate-sft", "migrate_sft_layout.py", "migrate",
         note="move any flat data/sft/*.jsonl into a named set"),

    # Wikipedia is the anchor: the largest single source and the cleanest prose
    # in the corpus.
    Step("wikipedia", "prepare_wikipedia.py", "pretrain",
         {"smoke": 50, "standard": 20_000, "max": 20_000}, heldout_mb=4,
         note="full English Wikipedia"),
    # Free — already on disk, no download at all.
    Step("pg19", "prepare_pg19.py", "pretrain",
         {"smoke": 50, "standard": 11_000, "max": 11_000}, heldout_mb=8,
         note="Project Gutenberg books, local"),
    Step("fineweb", "prepare_fineweb.py", "pretrain",
         {"smoke": 50, "standard": 5_000, "max": 20_000}, heldout_mb=4,
         note="filtered web text"),
    Step("webtext", "prepare_webtext.py", "pretrain",
         {"smoke": 50, "standard": 5_000, "max": 20_000}, heldout_mb=4,
         note="C4, a differently-filtered web pool"),
    Step("stackexchange", "prepare_stackexchange.py", "pretrain",
         {"smoke": 50, "standard": 5_000, "max": 10_000},
         note="technical Q&A prose"),
    Step("code", "prepare_code.py", "pretrain",
         {"smoke": 50, "standard": 5_000, "max": 10_000},
         note="source files"),
    Step("math", "prepare_math.py", "pretrain",
         {"smoke": 50, "standard": 3_000, "max": 8_000},
         note="math and worked reasoning"),

    # The two SFT sets the whole point of the refactor is to keep apart.
    Step("smoltalk2", "prepare_smoltalk2.py", "sft",
         limits={"smoke": 2_000, "standard": 0, "max": 0},
         note="thinking + non-thinking chat sets"),
]

BY_NAME = {s.name: s for s in STEPS}


def fingerprint(step: Step, profile: str) -> str:
    """Identifies the exact invocation, so a changed budget re-runs the step."""
    argv = [a for a in step.argv(profile) if a != sys.executable]
    return hashlib.sha1(" ".join(argv).encode()).hexdigest()[:12]


def load_state() -> dict:
    if not os.path.exists(STATE_PATH):
        return {}
    try:
        with open(STATE_PATH) as f:
            return json.load(f)
    except ValueError:
        # A truncated state file must not block a fetch; the worst case is
        # re-running steps that had finished.
        return {}


def save_state(state: dict) -> None:
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    tmp = STATE_PATH + ".partial"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, STATE_PATH)


def selected(args) -> list[Step]:
    """The steps to run, after --only / --skip."""
    names = [s.name for s in STEPS]
    if args.only:
        asked = [n.strip() for n in args.only.split(",") if n.strip()]
        unknown = set(asked) - set(names)
        if unknown:
            sys.exit(f"Unknown step(s): {', '.join(sorted(unknown))}\n"
                     f"Available: {', '.join(names)}")
        chosen = [BY_NAME[n] for n in asked]
    else:
        chosen = list(STEPS)

    skip = {n.strip() for n in (args.skip or "").split(",") if n.strip()}
    unknown = skip - set(names)
    if unknown:
        sys.exit(f"Unknown step(s) in --skip: {', '.join(sorted(unknown))}")
    return [s for s in chosen if s.name not in skip]


def print_plan(steps: list[Step], profile: str, state: dict, force: bool) -> float:
    """Print the plan and return the megabytes of text it will fetch."""
    print(f"Plan — profile {profile!r}\n")
    print(f"  {'step':<15} {'kind':<9} {'budget':>9}  status   note")
    print(f"  {'-' * 15} {'-' * 9} {'-' * 9}  {'-' * 7}  {'-' * 34}")

    total = 0.0
    for step in steps:
        done = (not force
                and state.get(step.name, {}).get("fingerprint") == fingerprint(step, profile))
        mb = step.mb(profile)
        if not done:
            total += mb
        budget = f"{mb:,.0f} MB" if mb else "-"
        print(f"  {step.name:<15} {step.kind:<9} {budget:>9}  "
              f"{'done' if done else 'fetch':<7}  {step.note}")

    tokens = total * 1e6 / 2.2
    print(f"\n  text to fetch   {total / 1000:,.1f} GB")
    print(f"  ~tokens         {tokens / 1e9:,.2f}B")
    print(f"  Chinchilla-optimal for the full corpus: "
          f"~{tokens / 20 / 1e6:,.0f}M non-embedding parameters")
    return total


def check_disk(mb_needed: float) -> None:
    """Refuse to start a fetch that cannot finish."""
    free = shutil.disk_usage(paths.DATA if os.path.exists(paths.DATA) else HERE).free
    needed = mb_needed * 1e6 * DISK_FACTOR
    print(f"  disk            {free / 1e9:,.0f} GB free, "
          f"~{needed / 1e9:,.0f} GB wanted (text x{DISK_FACTOR} for corpus.bin + cache)")
    if needed > free:
        sys.exit(f"\nNot enough disk: this plan wants ~{needed / 1e9:,.0f} GB and "
                 f"{free / 1e9:,.0f} GB is free.\nLower --profile, or --skip the "
                 f"largest steps.")


def run_step(step: Step, profile: str) -> dict:
    """Run one fetch to completion, capturing its output to a log."""
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"{step.name}.log")
    started = time.time()

    with open(log_path, "w", encoding="utf-8") as log:
        log.write(f"$ {' '.join(step.argv(profile))}\n\n")
        log.flush()
        result = subprocess.run(
            step.argv(profile), stdout=log, stderr=subprocess.STDOUT,
            cwd=HERE, text=True,
        )

    return {
        "name": step.name,
        "returncode": result.returncode,
        "seconds": round(time.time() - started, 1),
        "log": log_path,
        "fingerprint": fingerprint(step, profile),
    }


def tail(path: str, lines: int = 12) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return "".join(f.readlines()[-lines:])
    except OSError:
        return "(no log)"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[1],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Steps: " + ", ".join(s.name for s in STEPS),
    )
    ap.add_argument("--profile", default="standard", choices=PROFILES,
                    help="how much to fetch (default: standard, ~54 GB)")
    ap.add_argument("--only", help="comma-separated steps to run, in order")
    ap.add_argument("--skip", help="comma-separated steps to leave out")
    ap.add_argument("--jobs", type=int, default=3,
                    help="steps to run concurrently (default: 3)")
    ap.add_argument("--force", action="store_true",
                    help="re-run steps already recorded as done")
    ap.add_argument("--keep-going", action="store_true",
                    help="finish the remaining steps even if one fails")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and exit")
    args = ap.parse_args()

    steps = selected(args)
    state = load_state()
    total_mb = print_plan(steps, args.profile, state, args.force)

    pending = [
        s for s in steps
        if args.force
        or state.get(s.name, {}).get("fingerprint") != fingerprint(s, args.profile)
    ]

    if not pending:
        print("\nEverything in this plan is already done. --force to re-fetch.")
        return

    check_disk(total_mb)
    if args.dry_run:
        print(f"\n{len(pending)} step(s) would run. Drop --dry-run to start.")
        return

    print(f"\nRunning {len(pending)} step(s), {args.jobs} at a time. "
          f"Logs in {LOG_DIR}/\n")

    started = time.time()
    results: list[dict] = []
    failed: list[dict] = []

    # Submitted in plan order so the largest sources start first and the short
    # ones fill in around them, rather than the tail being one 20 GB straggler.
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = {pool.submit(run_step, s, args.profile): s for s in pending}
        try:
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                ok = result["returncode"] == 0
                mark = "ok  " if ok else "FAIL"
                print(f"  [{len(results)}/{len(pending)}] {mark} {result['name']:<15} "
                      f"{result['seconds']:>8,.0f}s", flush=True)

                if ok:
                    state[result["name"]] = {
                        "fingerprint": result["fingerprint"],
                        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "seconds": result["seconds"],
                    }
                    save_state(state)
                else:
                    failed.append(result)
                    print(f"        {result['log']}:")
                    for line in tail(result["log"]).splitlines():
                        print(f"        | {line}")
                    if not args.keep_going:
                        for f in futures:
                            f.cancel()
                        break
        except KeyboardInterrupt:
            # Completed steps are already recorded, so a re-run resumes rather
            # than starting the whole fetch again.
            print("\nInterrupted. Finished steps are recorded; re-run to resume.")
            for f in futures:
                f.cancel()
            sys.exit(130)

    elapsed = time.time() - started
    print(f"\n{len(results) - len(failed)}/{len(pending)} step(s) succeeded "
          f"in {elapsed / 60:,.0f}m")

    if failed:
        print("\nFailed: " + ", ".join(f["name"] for f in failed))
        print(f"Logs in {LOG_DIR}/. Fix and re-run — finished steps are skipped.")
        sys.exit(1)

    summarize()


def summarize() -> None:
    """What is now on disk, and what to do with it."""
    print("\nOn disk:")
    for label, base, pattern in (
        ("pretrain", os.path.join(paths.DATA, "input"), "*.txt"),
        ("sft", os.path.join(paths.DATA, "sft"), "*.jsonl"),
    ):
        if not os.path.isdir(base):
            continue
        print(f"  {label}/")
        for name in sorted(os.listdir(base)):
            folder = os.path.join(base, name)
            if not os.path.isdir(folder):
                continue
            import glob
            files = glob.glob(os.path.join(folder, pattern))
            size = sum(os.path.getsize(f) for f in files)
            print(f"    {name:<22} {len(files):>7,} files  {size / 1e9:>8,.2f} GB")

    print("\nNext:")
    print("  Pretrain on everything:")
    print('    curl -XPOST localhost:8001/train -d \'{"model_name":"base","datasets":[]}\'')
    print("  Then two separate fine-tunes from that base:")
    print('    curl -XPOST localhost:8001/sft -d \'{"model_name":"chat-nothink",'
          '"base_model":"base","datasets":["smoltalk2-nothink"]}\'')
    print('    curl -XPOST localhost:8001/sft -d \'{"model_name":"chat-think",'
          '"base_model":"base","datasets":["smoltalk2-think"]}\'')
    print("  What is selectable:  GET /datasets  and  GET /sft/datasets")


if __name__ == "__main__":
    main()
