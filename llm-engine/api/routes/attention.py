"""Reading one head's attention over a piece of text.

Its own router rather than a fifth concern in `models.py`, which is the join of
the registry row and the weights on disk and knows nothing about running one.
The path still hangs off `/models/{model_name}` because that is whose attention
it is.
"""

from fastapi import APIRouter, HTTPException, Request

from api import interactive, serving
from llm.architecture.wire import AttentionMatrix, AttentionSelection, InspectionCatalog
from llm.model import device
from llm.runtime.supervisor import WorkerStopped

router = APIRouter(tags=["attention"])


@router.get("/models/{model_name}/attention-points", response_model=InspectionCatalog)
async def inspection_points(model_name: str, request: Request):
    """Read actual legacy dimensions or declared candidates; numerical proofs remain scoped to their resident worker lifetime."""
    try:
        return await interactive.run(request, lambda cancel: serving.inspection_points(model_name, cancel))
    except FileNotFoundError as error:
        raise HTTPException(404, str(error)) from error
    except device.Busy as error:
        raise HTTPException(409, str(error)) from error
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    except WorkerStopped as error:
        raise HTTPException(409, "model inspection stopped; retry after the active job finishes") from error


@router.post("/models/{model_name}/attention", response_model=AttentionMatrix)
async def inspect_attention(model_name: str, body: AttentionSelection, request: Request):
    """One `(layer, head)` matrix and the token strings on both axes.

    Nothing is stored and nothing is recorded outside this call; see
    `llm/orchestrate/attention.py` for why that is the whole point.
    """
    try:
        return await interactive.run(request, lambda cancel: serving.attention(model_name, body, cancel))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except device.Busy as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except WorkerStopped as error:
        raise HTTPException(409, "model inspection stopped; retry after the active job finishes") from error
