"""The Python training/inference engine.

Layering, strictly one-directional (mirrors the Rust backend's crate order):

    storage · corpus  ->  model  ->  train  ->  orchestrate  ->  main.py

Nothing may import from a package to its right. `storage` imports nothing from
its siblings at all.
"""
