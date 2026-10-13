"""Retired: conversations now belong to Rust and scores to bench-engine.

Use the repository's scripts/migrate_conversations.py and migrate_benchmarks.py
for the forward cutover; see docs/backend/ownership-cutover.md. This legacy
import moved data in the opposite direction and must not run on the new schema.
"""
raise SystemExit("Legacy backend-to-LLM import is retired; see docs/backend/ownership-cutover.md")
