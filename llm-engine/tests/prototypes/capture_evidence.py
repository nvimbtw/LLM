"""Write representative ISS-48 captures for local review without changing artifacts."""

import argparse
import json
import platform
import time
from pathlib import Path

import torch

from tests.prototypes.capture import RepeatedBlock, strict_capture, transformer


def write_capture(name: str, exported: torch.export.ExportedProgram, directory: Path) -> dict:
    """Save one inspection graph and portable export; return measured artifact sizes."""
    nodes = []
    for node in exported.graph.nodes:
        nodes.append({
            "occurrence": node.name, "kind": node.op, "target": str(node.target),
            "tensor": str(node.meta.get("tensor_meta", "")),
            "module": str(node.meta.get("nn_module_stack", "")),
            "source": node.meta.get("stack_trace", ""),
        })
    graph_path = directory / f"{name}.json"
    graph_path.write_text(json.dumps({
        "strict": True, "ranges": str(exported.range_constraints),
        "signature": str(exported.graph_signature), "nodes": nodes,
    }, indent=2))
    export_path = directory / f"{name}.pt2"
    torch.export.save(exported, export_path)
    return {"nodes": len(nodes), "graph_bytes": graph_path.stat().st_size,
            "export_bytes": export_path.stat().st_size}


def main() -> None:
    """Capture audited CPU prototypes into the selected directory and print evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    report = {"python": platform.python_version(), "torch": torch.__version__,
              "device": "cpu", "cuda_available": torch.cuda.is_available(),
              "cuda_runtime": torch.version.cuda, "captures": {}}
    with torch.random.fork_rng():
        torch.manual_seed(48)
        tokens = torch.ones(2, 4, dtype=torch.long)
        sequence = torch.export.Dim("sequence", min=2, max=8)
        batch = torch.export.Dim("batch", min=1, max=3)
        for shared in (False, True):
            name = "shared-transformer" if shared else "independent-transformer"
            start = time.monotonic()
            exported = strict_capture(transformer(shared).eval(), (tokens,),
                                      dynamic_shapes=({0: batch, 1: sequence},))
            report["captures"][name] = write_capture(name, exported, args.output)
            report["captures"][name]["seconds"] = round(time.monotonic() - start, 4)
        start = time.monotonic()
        exported = strict_capture(RepeatedBlock(calls=128).train(),
                                  (torch.randn(1, 1, 8), torch.zeros(1, 1, 8)))
        report["captures"]["custom-128"] = write_capture("custom-128", exported, args.output)
        report["captures"]["custom-128"]["seconds"] = round(time.monotonic() - start, 4)
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
