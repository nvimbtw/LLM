"""Model measurements for bench-engine, reached only through Rust."""
import threading
import uuid

import torch
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

import store
from store import queries as q
from llm.config import EvalConfig
from llm.corpus import bpe
from llm.model import device, likelihood, continuation
from llm.eval import ablation
from llm.orchestrate import chat

router = APIRouter(tags=['inference'])
_active: dict[uuid.UUID, threading.Event] = {}
_guard = threading.Lock()


class Measure(BaseModel):
    config: EvalConfig
    text: str = Field(max_length=1048576)
    prompts: list[str] = Field(max_length=32)


@router.post('/measure/{request_id}/cancel')
def cancel_measure(request_id: uuid.UUID):
    with _guard:
        event = _active.get(request_id)
        if event:
            event.set()
    return {'cancelled': event is not None}


@router.post('/measure/{request_id}')
def measure(request_id: uuid.UUID, body: Measure):
    cancel = threading.Event()
    with _guard:
        if request_id in _active:
            raise HTTPException(409, 'measurement is already active')
        _active[request_id] = cancel
    try:
        with device.admit(wait=True):
            return _measure(body, cancel)
    finally:
        with _guard:
            _active.pop(request_id, None)


def _measure(body: Measure, cancel: threading.Event):
    model, tokenizer, device_name = chat.load(body.config.model_name)
    with store.session() as db:
        row = q.get_model(db, uuid.UUID(body.config.model_name))
        if row and row.checkpoint_index is not None:
            row.parameters = {**(row.parameters or {}), 'recovery_referenced': True}
        name = row.name if row else body.config.model_name
    info = dict(id=body.config.model_name, name=name, layers=model.layers,
                dimensions=model.dimensions, heads=model.heads, context=model.ctx_window,
                vocab=model.vocab, parameters=model.parameter_count())
    ids = bpe.encode(tokenizer, body.text)
    scores = likelihood.token_logprobs(model, ids, device_name, cancel=cancel)
    raw = dict(total=scores.total, count=scores.count,
               unscored_bytes=len(bpe.decode(tokenizer, ids[:1]).encode('utf-8')))
    drawn = []
    sampling = body.config.model_dump(exclude={'model_name','suite'})
    for index, prompt in enumerate(body.prompts):
        if cancel.is_set():
            break
        cfg = continuation.SamplingConfig(**{**sampling, 'seed': body.config.seed + index})
        drawn.append(dict(prompt=prompt, text=continuation.continue_tokens(model,tokenizer,prompt,cfg,device_name)))
    contribution = None
    if len(ids) > 1 and not cancel.is_set():
        window = torch.tensor([ids[:model.ctx_window]], dtype=torch.long, device=device_name)
        contribution = ablation.ablate(model,window,body.config.model_name).as_dict()
    return dict(model=info, likelihood=raw, samples=drawn, ablation=contribution,cancelled=cancel.is_set())
