"""Trusted package integration backend executed only inside the production supervised child."""

import json
import os
from pathlib import Path
import random

import numpy as np
import torch

from llm.architecture import package
from llm.architecture.build import build_model
from llm.architecture.package_execution import load_dependencies, load_granted_package, load_in_worker, restore_optimizer
from llm.architecture.package_integrity import PackageIntegrity
from llm.architecture.resolve import resolve_recipe
from llm.architecture.source import SourceLoader
from llm.architecture.spec import DefinitionPackage
from llm.model import checkpoint
from llm.runtime.protocol import Completed, RuntimePins
from llm.architecture.spec import canonical_digest


def publish_fixture(config: dict, context) -> tuple[Path, list]:
    """Construct and checkpoint one test snapshot under actual device/process admission."""
    manifest = DefinitionPackage.model_validate(config["manifest"])
    directory = Path(config["destination"])
    os.environ["DATA_DIR"] = str(directory.parent / "data")
    sources = {entry.reference.revision_id:config["files"][entry.path].encode()
               for entry in manifest.definition.dependencies}
    provisional = package.VerifiedPackage(directory,manifest,
        PackageIntegrity(definition_id=manifest.definition_id,torch_version=torch.__version__,
                         files={"fixture": {"sha256":"a"*64,"size_bytes":0}}),(),sources)
    with SourceLoader() as loader:
        definitions = load_dependencies(provisional,loader)
        definition = manifest.definition
        model = build_model(resolve_recipe(definition.recipe,definition.parameters,definitions,definition.tokenizer))
        optimizer = torch.optim.AdamW(model.parameters(),lr=0.01)
        tokens = torch.tensor([[1,2,3]])
        model(tokens).square().mean().backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            expected = model(tokens).tolist()
        root = context.directory / f"save-{manifest.definition_id}"
        root.mkdir()
        checkpoint.save_modular_weights(model,root/"model.pt",manifest.definition_id)
        checkpoint.save_modular_state(root/"train_state.pt",model=model,optimizer=optimizer,
            definition_id=manifest.definition_id,step=1,epoch=2,stage=3,total_steps=10,
            rng=checkpoint.rng_state(),extra={"cursor":{"next_batch":7}})
        order = tuple(name for group in checkpoint.optimizer_layout(model,optimizer) for name in group)
        manifest = manifest.model_copy(update={"optimizer_parameter_order":order})
        files = {name:data.encode() for name,data in config["files"].items()}
        files.update({"model.pt":root/"model.pt","train_state.pt":root/"train_state.pt"})
        package.publish(directory,manifest,files)
    return directory,expected


class PackageBackend:
    """Own verified package models through full save/load/optimizer/RNG round trips."""

    def __init__(self, context):
        """Keep no parent service clients or source objects; numerical work begins on Load."""
        torch.set_num_threads(1)
        self.loaded = []

    def handle(self,payload,context) -> Completed:
        """Return diagnostic artifacts for test failures without changing production worker error policy."""
        import traceback

        try:
            if payload.package is not None:
                loaded = load_granted_package(payload,context)
                self.loaded.append(loaded)
                with torch.no_grad():
                    logits = loaded.model(torch.tensor([[1,2,3]])).tolist()
                artifact = context.artifacts.write_bytes(json.dumps({"logits":logits,"definition_id":loaded.checked.manifest.definition_id}).encode(),"training_result_v1")
                return Completed(type="completed",outcome="ok",result=artifact)
            return self.roundtrip(payload,context)
        except Exception:
            artifact = context.artifacts.write_bytes(json.dumps({"test_error":traceback.format_exc()}).encode(),"training_result_v1")
            return Completed(type="completed",outcome="ok",result=artifact)

    def roundtrip(self,payload,context) -> Completed:
        """Consume a trusted fixture descriptor and prove saved package invariants in the child."""
        config = json.loads(context.artifacts.read_bytes(payload.definition))
        reports = []
        for item in config["packages"]:
            torch.manual_seed(42)
            directory,expected = publish_fixture(item,context)
            manifest = package.read_manifest(directory)
            receipt = package.verify(directory).receipts[0]
            pins = RuntimePins(definition_id=manifest.definition_id,bindings_digest=receipt.identity.parameter_digest,
                profile_digest=canonical_digest(receipt.identity.profile),receipt_id=receipt.receipt_id)
            loaded = load_in_worker(directory,context,pins=pins)
            optimizer = torch.optim.AdamW(loaded.model.parameters(),lr=0.01)
            restore_optimizer(loaded,optimizer)
            checkpoint.restore_rng(loaded.state.rng)
            first = (torch.rand(2).tolist(),np.random.rand(2).tolist(),random.random())
            checkpoint.restore_rng(loaded.state.rng)
            second = (torch.rand(2).tolist(),np.random.rand(2).tolist(),random.random())
            with torch.no_grad():
                actual = loaded.model(torch.tensor([[1,2,3]])).tolist()
            aliases = loaded.model.parameter_paths()
            tied = aliases.get("embedding.weight") is aliases.get("head.weight") if "head.weight" in aliases else True
            reports.append({"same_logits":actual==expected,"tied":tied,"rng":first==second,
                "step":loaded.state.step,"cursor":loaded.state.extra["cursor"],
                "optimizer":len(optimizer.state_dict()["state"]),"logits":actual,
                "imports":list(loaded.sources.modules)})
            self.loaded.append(loaded)
        artifact = context.artifacts.write_bytes(json.dumps(reports).encode(),"training_result_v1")
        return Completed(type="completed",outcome="ok",result=artifact)

    def finish_request(self) -> None:
        """Retain resident models/import owners until explicit worker cleanup."""
        return None

    def close(self) -> None:
        """Unload every revision namespace before guardian-certified process cleanup."""
        for loaded in self.loaded:
            loaded.close()
        self.loaded.clear()
