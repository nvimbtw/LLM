"""Version-pinned standard, independent-stack and shared-loop recipe templates."""

from typing import Literal

from llm.architecture.builtins.catalog import builtin_definitions
from llm.architecture.builtins.components import LAYER_PARAMETERS, LOGIT_PORT, MASK_PORT, TOKEN_PORT, integer
from llm.architecture.spec import Recipe

RecipeName = Literal["standard", "expanded", "looped", "nested"]


def recipe_template(name: RecipeName) -> Recipe:
    """Return an owned complete recipe pinned to the installed builtin implementation manifest.

    The standard composite preserves legacy initialization order. Expanded recipes
    use deterministic instance-path order, with independently initialized stack
    layers or one shared layer called repeatedly; they do not claim seed-identical
    initialization to the composite's historical two-pass initializer.
    """
    if name not in {"standard", "expanded", "looped", "nested"}:
        raise ValueError("unknown builtin recipe template")
    catalog = builtin_definitions()
    parameters = {key: declaration.model_dump(mode="json") for key, declaration in LAYER_PARAMETERS.items()
                  if key != "ctx_window"}
    parameters["context_window"] = integer(32).model_dump(mode="json")
    layer_values = {key: f"${{{'context_window' if key == 'ctx_window' else key}}}" for key in LAYER_PARAMETERS}
    recipe = {"schema_version": 1, "interface": "causal_lm", "parameters": parameters,
              "inputs": {"tokens": TOKEN_PORT, "positions": TOKEN_PORT, "key_mask": MASK_PORT},
              "outputs": {"logits": LOGIT_PORT}, "result": {"logits": "logits"},
              "context_window": "${context_window}", "modules": {}, "flow": []}
    if name == "standard":
        parameters["layers"] = integer(2).model_dump(mode="json")
        recipe["modules"] = {"model": {"reference": catalog["standard_transformer"].reference,
            "parameters": {"vocab": "${tokenizer.vocabulary_size}", **layer_values, "layers": "${layers}"}}}
        recipe["flow"] = [{"call": "model", "inputs": {"tokens": "tokens", "positions": "positions", "key_mask": "key_mask"},
                           "outputs": {"logits": "logits"}}]
        return Recipe.model_validate(recipe)
    recipe["modules"] = {
        "embedding": {"reference": catalog["token_embedding"].reference,
                      "parameters": {"vocab": "${tokenizer.vocabulary_size}", "width": "${dimensions}"}},
        "final_norm": {"reference": catalog["normalization"].reference,
                       "parameters": {"width": "${dimensions}", "epsilon": 1e-5}},
        "head": {"reference": catalog["language_model_head"].reference,
                 "parameters": {"vocab": "${tokenizer.vocabulary_size}", "width": "${dimensions}"}},
    }
    recipe["aliases"] = [{"owner": "embedding.weight", "target": "head.weight"}]
    layer = {"reference": catalog["transformer_layer"].reference, "parameters": layer_values}
    if name == "expanded":
        parameters["layers"] = integer(2).model_dump(mode="json")
        trunk = {"stack": "${layers}", "name": "layers", "template": layer,
                 "inputs": {"positions": "positions", "key_mask": "key_mask"},
                 "carry": {"hidden": {"initial": "embedded", "input": "hidden", "output": "hidden", "result": "trunk"}}}
    else:
        recipe["modules"]["core"] = layer
        parameters["iterations"] = integer(2).model_dump(mode="json")
        trunk = {"repeat": "${iterations}",
                 "carry": {"hidden": {"initial": "embedded", "input": "iteration_hidden", "output": "next_hidden", "result": "trunk"}},
                 "body": [{"call": "core", "inputs": {"hidden": "iteration_hidden", "positions": "positions", "key_mask": "key_mask"},
                           "outputs": {"hidden": "next_hidden"}}]}
        if name == "nested":
            parameters["outer_iterations"] = integer(2).model_dump(mode="json")
            trunk["carry"]["hidden"]["initial"] = "outer_hidden"
            trunk["carry"]["hidden"]["result"] = "inner_result"
            trunk = {"repeat": "${outer_iterations}",
                     "carry": {"hidden": {"initial": "embedded", "input": "outer_hidden", "output": "inner_result", "result": "trunk"}},
                     "body": [trunk]}
    recipe["flow"] = [
        {"call": "embedding", "inputs": {"tokens": "tokens"}, "outputs": {"hidden": "embedded"}}, trunk,
        {"call": "final_norm", "inputs": {"hidden": "trunk"}, "outputs": {"hidden": "normalized"}},
        {"call": "head", "inputs": {"hidden": "normalized"}, "outputs": {"logits": "logits"}},
    ]
    return Recipe.model_validate(recipe)
