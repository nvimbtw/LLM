"""The Python half of conversation ownership is stateless preparation."""
import uuid
import pytest
from api.chat_request import Ask, _new_turns
from fastapi import HTTPException
from llm.chat.template import Role, Turn, render_conversation
from tests.fixtures.tiny_env import TINY_MODEL


async def test_prepare_counts_without_persisting(engine_client):
    response = await engine_client.post('/chat/prepare',json={'model_name':TINY_MODEL,'text':'hello'})
    assert response.status_code == 200
    assert response.json()['turns'][0]['role'] == 'user'
    assert response.json()['turns'][0]['tokens'] > 0
    assert (await engine_client.get('/chats')).status_code == 404


def test_tool_and_interjection_keep_canonical_order():
    """Prepared continuation turns render one tool block before user steering."""
    body = Ask(
        model_name="m",
        tool_result={"tool": "bash", "argument": "pwd", "body": "/tmp"},
        interject="also list files",
    )
    turns = _new_turns(body)
    assert [role for role, _ in turns] == ["tool", "user"]
    rendered = render_conversation(None, [Turn(Role(role), text) for role, text in turns])
    assert rendered == (
        '<|tool-result|><|tool-call|>{"tool": "bash", "argument": "pwd"}'
        '<|/tool-call|>/tmp<|/tool-result|><|user|>also list files<|/user|>'
    )


@pytest.mark.parametrize('extra',[{}, {'text':'a','tool_result':{'tool':'bash','argument':'pwd','body':'x'}}, {'text':'a','interject':'b'}])
def test_invalid_turn_combinations_are_refused(extra):
    with pytest.raises(HTTPException):
        _new_turns(Ask(model_name='m',**extra))


def test_files_cannot_be_bound_to_a_tool_result():
    with pytest.raises(HTTPException):
        _new_turns(Ask(model_name='m',tool_result={'tool':'bash','argument':'pwd','body':'x'},attachment_ids=[uuid.uuid4()]))
