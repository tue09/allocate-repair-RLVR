"""Answer extraction (think block excluded), repair-prompt construction and absence of leakage."""

import inspect
import os

import pytest

from allocate_repair.repair import (
    DEFAULT_INSTRUCTION,
    TRUNCATION_MARKER,
    AnswerExtractor,
    RepairPromptBuilder,
    build_repair_requests,
)

QUESTION = [{"role": "user", "content": "Solve: what is 6*7? End with Answer: $Answer."}]


def ids(tok, text, eos=True):
    return tok.encode(text) + ([tok.eos_token_id] if eos else [])


def test_extracts_only_text_after_think(fake_tokenizer):
    ex = AnswerExtractor(fake_tokenizer, mode="auto")
    assert ex.mode == "think"
    res = ex.extract(ids(fake_tokenizer, "secret reasoning 41? </think>\n\nThe product is 41.\nAnswer: 41"))
    assert res.ok and res.reason == "ok"
    assert res.text == "The product is 41.\nAnswer: 41"
    assert "secret reasoning" not in res.text


def test_uses_last_think_close(fake_tokenizer):
    ex = AnswerExtractor(fake_tokenizer)
    res = ex.extract(ids(fake_tokenizer, "a </think> b </think> final Answer: 3"))
    assert res.text == "final Answer: 3"


def test_missing_think_close_is_a_failure_not_the_reasoning(fake_tokenizer):
    ex = AnswerExtractor(fake_tokenizer)
    res = ex.extract(ids(fake_tokenizer, "truncated reasoning that never closes", eos=False))
    assert not res.ok and res.reason == "no_think_close" and res.text == ""


def test_empty_answer_after_think_is_a_failure(fake_tokenizer):
    res = AnswerExtractor(fake_tokenizer).extract(ids(fake_tokenizer, "reasoning </think>   "))
    assert not res.ok and res.reason == "empty_after_think"


def test_long_answer_keeps_tail(fake_tokenizer):
    ex = AnswerExtractor(fake_tokenizer, max_answer_tokens=12)
    res = ex.extract(ids(fake_tokenizer, "r </think>" + "x" * 50 + "Answer: 9"))
    assert res.ok and res.truncated
    assert res.text.startswith(TRUNCATION_MARKER) and res.text.endswith("Answer: 9")


def test_full_mode_for_models_without_think(fake_tokenizer):
    res = AnswerExtractor(fake_tokenizer, mode="full").extract(ids(fake_tokenizer, "work. Answer: 2"))
    assert res.ok and res.text == "work. Answer: 2"


def test_single_turn_prompt_keeps_question_verbatim():
    msgs = RepairPromptBuilder().build(QUESTION, "Answer: 41")
    assert len(msgs) == 1 and msgs[0]["role"] == "user"
    content = msgs[0]["content"]
    assert content.startswith(QUESTION[0]["content"])
    assert "Answer: 41" in content and content.endswith(DEFAULT_INSTRUCTION)
    assert "incorrect" in DEFAULT_INSTRUCTION and "solve the problem again" in DEFAULT_INSTRUCTION
    assert QUESTION[0]["content"] == "Solve: what is 6*7? End with Answer: $Answer."  # input not mutated


def test_multi_turn_prompt_structure():
    msgs = RepairPromptBuilder(prompt_format="multi_turn").build(QUESTION, "Answer: 41")
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert msgs[0] == QUESTION[0] and msgs[1]["content"] == "Answer: 41"


def _requests(tok, policy="placeholder", max_prompt_tokens=10_000, gt="GT_SECRET_917"):
    responses = {
        0: ids(tok, "SIBLING_SECRET_REASONING </think> SIBLING_SECRET_ANSWER Answer: 42"),  # correct sibling
        1: ids(tok, "my reasoning </think> I think 41. Answer: 41"),
        2: ids(tok, "never finished thinking", eos=False),
    }
    reward_metadata = {0: {"ground_truth": gt}, 1: {"ground_truth": gt}, 2: {"ground_truth": gt}}  # noqa: F841
    return build_repair_requests(
        failed_rows=[1, 2],
        response_ids=responses,
        question_messages={i: QUESTION for i in responses},
        extractor=AnswerExtractor(tok),
        builder=RepairPromptBuilder(),
        tokenizer=tok,
        max_prompt_tokens=max_prompt_tokens,
        failure_policy=policy,
    )


def test_no_ground_truth_or_sibling_leakage(fake_tokenizer):
    # the construction API cannot receive the ground truth or other responses' rewards
    params = set(inspect.signature(build_repair_requests).parameters)
    assert not params & {"ground_truth", "reward_model", "rewards", "gt"}
    requests, stats = _requests(fake_tokenizer)
    assert [r.parent_index for r in requests] == [1, 2]
    for req in requests:
        text = req.messages[-1]["content"]
        assert "GT_SECRET_917" not in text and "SIBLING_SECRET" not in text
        assert "my reasoning" not in text  # reasoning inside the think block is excluded
    assert "I think 41. Answer: 41" in requests[0].messages[-1]["content"]
    assert stats["extraction_failures"] == 1 and stats["reasons"] == {"no_think_close": 1}
    # changing the ground truth changes nothing
    again, _ = _requests(fake_tokenizer, gt="OTHER_GT")
    assert [r.messages for r in again] == [r.messages for r in requests]


def test_extraction_failure_policies(fake_tokenizer):
    requests, stats = _requests(fake_tokenizer, policy="placeholder")
    assert len(requests) == 2 and RepairPromptBuilder().failure_placeholder in requests[1].messages[-1]["content"]
    requests, stats = _requests(fake_tokenizer, policy="skip")
    assert [r.parent_index for r in requests] == [1] and stats["skipped"] == 1


def test_prompt_budget_shortens_answer_not_question(fake_tokenizer):
    tok = fake_tokenizer
    long_answer = "detail " * 200 + "Answer: 41"
    base_len = len(tok.apply_chat_template(RepairPromptBuilder().build(QUESTION, ""), tokenize=False))
    requests, stats = build_repair_requests(
        failed_rows=[0],
        response_ids={0: ids(tok, "r </think>" + long_answer)},
        question_messages={0: QUESTION},
        extractor=AnswerExtractor(tok, max_answer_tokens=100_000),
        builder=RepairPromptBuilder(),
        tokenizer=tok,
        max_prompt_tokens=base_len + 200,
    )
    content = requests[0].messages[-1]["content"]
    assert stats["prompt_shortened"] == 1 and requests[0].prompt_tokens <= base_len + 200
    assert content.startswith(QUESTION[0]["content"]) and "Answer: 41" in content


@pytest.mark.skipif(not os.environ.get("AR_TEST_TOKENIZER"), reason="set AR_TEST_TOKENIZER=<model path> to run")
def test_real_tokenizer_think_extraction():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(os.environ["AR_TEST_TOKENIZER"])
    ex = AnswerExtractor(tok)
    assert ex.mode == "think" and ex.think_end_id is not None
    response = "Let me compute 6*7... maybe 41.\n</think>\n\nThe answer is 41.\n\nAnswer: 41"
    rids = tok(response, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    res = ex.extract(rids)
    assert res.ok and res.text == "The answer is 41.\n\nAnswer: 41"
    msgs = RepairPromptBuilder().build(QUESTION, res.text)
    rendered = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    assert "Let me compute" not in rendered and QUESTION[0]["content"] in rendered
