"""Pure packed-data/schedule identity shared by parent preparation, workers and legacy recovery."""

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
import numpy as np

HASH_BLOCK_BYTES = 1 << 20


@dataclass(frozen=True)
class IdentityPart:
    """One bounded numeric byte range or schedule header, retaining its array owner for optional mapped-page release."""

    data: bytes | memoryview
    array: np.ndarray | None = None


def parts(*arrays: np.ndarray, plan=None) -> Iterator[IdentityPart]:
    """Yield the original exact identity byte stream in bounded ranges, without I/O or mapping side effects."""
    if plan is not None:
        yield IdentityPart(json.dumps([plan.eot_id, plan.shuffle, plan.seed, plan.documents.names], sort_keys=True).encode())
        arrays = (*arrays, plan.documents.rows)
        for stage in plan.stages:
            yield IdentityPart(json.dumps([stage.name, stage.epochs, stage.documents.names]).encode())
            arrays = (*arrays, stage.documents.rows)
    for array in arrays:
        yield IdentityPart(str((array.shape, str(array.dtype))).encode())
        if array.nbytes == 0:
            continue
        view = memoryview(array).cast("B")
        for start in range(0, len(view), HASH_BLOCK_BYTES):
            yield IdentityPart(view[start:start + HASH_BLOCK_BYTES], array)


def fingerprint(*arrays: np.ndarray, plan=None) -> str:
    """Hash exact numeric bytes, document ordering and schedule facts without storage or model execution."""
    digest = hashlib.sha256()
    for part in parts(*arrays, plan=plan):
        digest.update(part.data)
    return digest.hexdigest()
