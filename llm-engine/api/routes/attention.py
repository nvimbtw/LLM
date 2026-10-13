"""Reading one head's attention over a piece of text.

Its own router rather than a fifth concern in `models.py`, which is the join of
the registry row and the weights on disk and knows nothing about running one.
The path still hangs off `/models/{model_name}` because that is whose attention
it is.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from llm.orchestrate import attention
from llm.model import device

router = APIRouter(tags=["attention"])


class AttentionRequest(BaseModel):
    text: str = Field(min_length=1)
    layer: int = Field(default=0, ge=0)
    head: int = Field(default=0, ge=0)


@router.post("/models/{model_name}/attention")
def inspect_attention(model_name: str, body: AttentionRequest):
    """One `(layer, head)` matrix and the token strings on both axes.

    Nothing is stored and nothing is recorded outside this call; see
    `llm/orchestrate/attention.py` for why that is the whole point.
    """
    try:
        with device.admit():
            return attention.matrix(model_name, body.text, body.layer, body.head)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except device.Busy as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
