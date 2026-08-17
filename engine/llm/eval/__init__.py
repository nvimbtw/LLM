"""The benchmark: what a trained model is actually worth.

Loss says a run converged; it does not say the model learned English. These
stages do — held-out compression against a counting-table floor, a grammar
forced choice, surface properties of generated text — combined into one auditable
composite.

Ported from backend/crates/llm/src/eval/.
"""

SCORE_VERSION = 2
