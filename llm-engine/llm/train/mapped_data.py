"""Readonly packed-input scan effects, separate from pure numeric/schedule identity."""

from collections.abc import Callable
import hashlib
import mmap
import numpy as np

from llm.train import data_identity
from llm.train.plan import DataPlan


def release_pages(array: np.ndarray) -> None:
    """Evict readonly file-mapped pages after a completed scan block; leave ordinary arrays and writable mappings intact."""
    owner = array
    while isinstance(owner, np.ndarray):
        if owner.flags.writeable:
            return
        owner = owner.base
    if isinstance(owner, mmap.mmap):
        owner.madvise(mmap.MADV_DONTNEED)


def fingerprint(*arrays: np.ndarray, plan: DataPlan, check_cancelled: Callable[[], None]) -> str:
    """Hash bounded exact input ranges with cancellation and readonly-page eviction before the next block."""
    digest = hashlib.sha256()
    for part in data_identity.parts(*arrays, plan=plan):
        check_cancelled()
        digest.update(part.data)
        if part.array is not None:
            release_pages(part.array)
    return digest.hexdigest()
