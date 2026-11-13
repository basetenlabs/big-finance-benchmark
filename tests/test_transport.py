"""The injected transport leaves grading semantics in the harness."""
from unittest.mock import AsyncMock

import pytest

from big_finance_harness import grading, grader
from .label_fixtures import labels_for
from .test_accuracy_judge import submitted_run, rubric, response
from .test_grader import _make_item


async def test_injected_transport_owns_both_calls(monkeypatch):
    native = AsyncMock(side_effect=AssertionError('default transport used'))
    monkeypatch.setattr(grader.litellm, 'acompletion', native)
    transport = AsyncMock(side_effect=[rubric(), response({
        'explanation': 'Correct', 'final_answer_correct': True})])
    item = _make_item()
    result = await grading.grade(run=submitted_run(), item=item,
        judge_model_id='openai:test', rubric_labels=labels_for(item), completion=transport)
    assert transport.await_count == 2
    native.assert_not_awaited()
    assert result.grading.setup == 'revised'
    assert result.final_answer_correct is True
    assert 'TRACE-ONLY' not in str(transport.call_args_list[1])


async def test_original_rejects_injected_transport():
    with pytest.raises(ValueError, match='only by revised'):
        await grading.grade(run=submitted_run(), item=_make_item(),
            judge_model_id='openai:test', grader='original', completion=AsyncMock())
