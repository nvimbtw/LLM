"""The Python training/inference engine.

Layering, strictly one-directional (mirrors the Rust backend's crate order):

    storage · corpus · gym  ->  model  ->  train  ->  orchestrate  ->  main.py

Nothing may import from a package to its right. `storage` imports nothing from
its siblings at all.

`backend.py` sits under all of it: the one hop that runs *toward* the Rust
backend, which `corpus` reads documents through and `gym` drives episodes
through. Both of those are the same shape — text this engine does not own,
arriving over one authenticated seam — which is why they are siblings and why
neither of them mints its own token.
"""
