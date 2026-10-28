# Copyable architecture templates

These templates use the production SDK and resolver. They are examples, not
accepted catalog revisions: source inspection, strict validation and acceptance
still run through the supervised engine workflow before training.

| File | Behavior |
| --- | --- |
| [standard.yaml](standard.yaml) | Standard composite, with legacy seeded initialization and tied output weights. |
| [expanded.yaml](expanded.yaml) | Distinct Transformer layer instances in an independent stack. |
| [looped.yaml](looped.yaml) | One Transformer layer with shared weights across fixed iterations. |
| [nested.yaml](nested.yaml) | Nested fixed repeats sharing one layer across all execution occurrences. |
| [custom.yaml](custom.yaml) | A token-local causal language model using the custom gated FFN below. |
| [custom_ffn.py](custom_ffn.py) | A SiLU gated residual FFN with editable width/expansion controls. |

The custom model deliberately demonstrates an administrator's own arithmetic;
its token-local computation has no attention. Its repeated FFN remains causal
and keeps gradients through every iteration.

Templates pin actual builtin revisions derived from the installed implementation
manifest. Copy an example into a draft, then select the installed accepted
revisions using engine metadata. The source in `custom_ffn.py` has its own exact
content/revision pin in `custom.yaml`. Editing that file requires a new source
revision and updated recipe dependency pin.

For local resolver use from `llm-engine/`:

```python
from uuid import uuid4
from llm.architecture.builtins.catalog import builtin_definitions
from llm.architecture.builtins.recipes import recipe_template
from llm.architecture.build import build_model
from llm.architecture.resolve import resolve_recipe
from llm.architecture.spec import FinalTokenizer

definitions = builtin_definitions()
dependencies = {block.reference.revision_id: block for block in definitions.values()}
tokenizer = FinalTokenizer(
    kind="final", tokenizer_id=uuid4(), fingerprint="a" * 64, vocabulary_size=256,
)
plan = resolve_recipe(recipe_template("looped"), {"iterations": 4}, dependencies, tokenizer)
model = build_model(plan)
```

To start from any builtin in a custom source editor:

```python
from llm.architecture.builtins.catalog import copyable_source

source = copyable_source("transformer_layer")
```

This wrapper uses the public builtin dependency and is eligible for `SourceLoader`.
Save it as a new custom resource/revision using the hash of its exact source
bytes, and retain the builtin compatibility manifest with the resolved package.
Copies expose full-prefix behavior; optional capabilities require explicit
delegation and validation. Internal implementation files are readable inspection
source and may import legacy primitives, so copying those files verbatim into a
custom editor is not supported.

Real callers supply the selected tokenizer's actual identity and fingerprint.
The worker owns device/memory admission, source lifetime and model cleanup.
The example snippet constructs only domain computation; it creates no jobs or
catalog records. See [builtin contracts](../../../docs/llm-engine/architecture-builtins.md).
