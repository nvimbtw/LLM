"""What corpora are on disk, for a caller composing a curriculum.

Read-only: the fetch scripts fill these folders, nothing is uploaded through
the API.
"""

from fastapi import APIRouter

router = APIRouter(tags=["catalogue"])


@router.get("/datasets")
def datasets():
    from llm.corpus import sources

    return {"datasets": sources.catalogue()}


@router.get("/sft/datasets")
def sft_datasets():
    """File count and bytes rather than a conversation count: the latter costs a
    full pass over hundreds of megabytes of JSONL per request, and tells sets
    apart no better."""
    from llm.corpus import sft as sft_corpus

    return {"datasets": sft_corpus.catalogue()}
