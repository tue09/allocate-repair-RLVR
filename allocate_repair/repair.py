"""Phase-2 prompt construction: model-aware answer extraction and the explain-and-repair prompt.

Nothing in this module receives the ground truth or any other response. The repair prompt for a failed
response y_qi is a function of (original question messages, the answer portion of y_qi) only, which is what
makes ground-truth / sibling leakage impossible by construction.
"""

import copy
from collections.abc import Mapping
from dataclasses import dataclass

DEFAULT_THINK_END = "</think>"

DEFAULT_INSTRUCTION = (
    "The previous answer to this question is incorrect. Explain why it is incorrect and solve the problem again. "
    'End with your corrected final answer on its own line in the form "Answer: $Answer".'
)
DEFAULT_ANSWER_HEADER = "Previous answer:"
DEFAULT_FAILURE_PLACEHOLDER = "(The previous response ended without a final answer after its reasoning.)"
TRUNCATION_MARKER = "[...] "

# Prompt v3 (allocate_repair.repair.prompt_version=v3): no "reason step by step"; a wrong final answer must be repaired
# in place (not re-solved) with an explicit reason section; a response without a final answer is asked to solve again,
# more concisely. Chosen after the inference pilot in /cm/shared/tue09_new/analysis (repair_ablation report).
V3_WRONG_ANSWER_INSTRUCTION = (
    "The final answer above is incorrect. Do not solve the problem from scratch. You must repair the previous answer "
    "itself: go through its steps in order, keep every step that is correct, find the step that is wrong, and correct "
    "the solution from that step onward. Write your final response in this format:\n"
    "# REASON WHY THE PREVIOUS ANSWER WAS WRONG:\n"
    "<which step is wrong and why, in a few lines>\n"
    "# CORRECTED SOLUTION:\n"
    "<the corrected steps>\n"
    'End with your corrected final answer on its own line in the form "Answer: $Answer".'
)
V3_NO_FINAL_ANSWER_INSTRUCTION = (
    "The previous attempt ended before giving a final answer. Solve the problem again more concisely. "
    'End with your final answer on its own line in the form "Answer: $Answer".'
)


@dataclass
class ExtractionResult:
    text: str
    """Answer portion shown in the repair prompt ("" when extraction failed)."""
    ok: bool
    reason: str
    """"ok", "no_think_close", "empty_after_think" or "empty_response"."""
    num_answer_tokens: int
    truncated: bool


class AnswerExtractor:
    """Extract the final-answer portion of a response, excluding the reasoning inside the think block.

    Modes:
        auto  -- "think" when the tokenizer knows the closing think delimiter, otherwise "full".
        think -- text after the last closing think delimiter; a response without it is an extraction failure
                 (it never falls back to the reasoning trace).
        full  -- the whole response (models without a think block).
    """

    def __init__(
        self, tokenizer, mode: str = "auto", think_end: str = DEFAULT_THINK_END, max_answer_tokens: int = 1024
    ):
        assert mode in ("auto", "think", "full"), f"unknown extraction mode {mode}"
        self.tokenizer = tokenizer
        self.think_end = think_end
        self.max_answer_tokens = max_answer_tokens
        vocab = tokenizer.get_vocab()
        self.think_end_id = vocab.get(think_end)
        chat_template = getattr(tokenizer, "chat_template", None) or ""
        if mode == "auto":
            mode = "think" if (self.think_end_id is not None or think_end in chat_template) else "full"
        self.mode = mode
        self._strip_ids = {i for i in (tokenizer.eos_token_id, tokenizer.pad_token_id) if i is not None}

    def describe(self) -> dict:
        return {"mode": self.mode, "think_end": self.think_end, "think_end_id": self.think_end_id}

    def _strip_trailing(self, ids: list[int]) -> list[int]:
        end = len(ids)
        while end > 0 and ids[end - 1] in self._strip_ids:
            end -= 1
        return ids[:end]

    def _finish(self, answer_ids: list[int]) -> ExtractionResult:
        answer_ids = self._strip_trailing(answer_ids)
        text = self.tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
        if not text:
            return ExtractionResult(
                "", False, "empty_after_think" if self.mode == "think" else "empty_response", 0, False
            )
        truncated = False
        if len(answer_ids) > self.max_answer_tokens:
            # keep the tail: the final answer statement is at the end of the answer portion
            answer_ids = answer_ids[-self.max_answer_tokens :]
            text = TRUNCATION_MARKER + self.tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
            truncated = True
        return ExtractionResult(text, True, "ok", len(answer_ids), truncated)

    def extract(self, response_ids: list[int]) -> ExtractionResult:
        response_ids = [int(i) for i in response_ids]
        if self.mode == "full":
            return self._finish(response_ids)
        if self.think_end_id is not None:
            positions = [k for k, tok in enumerate(response_ids) if tok == self.think_end_id]
            if not positions:
                return ExtractionResult("", False, "no_think_close", 0, False)
            return self._finish(response_ids[positions[-1] + 1 :])
        # multi-token delimiter: search the decoded text
        text = self.tokenizer.decode(self._strip_trailing(response_ids), skip_special_tokens=False)
        pos = text.rfind(self.think_end)
        if pos < 0:
            return ExtractionResult("", False, "no_think_close", 0, False)
        answer_ids = self.tokenizer(text[pos + len(self.think_end) :], add_special_tokens=False)["input_ids"]
        return self._finish(answer_ids)


class RepairPromptBuilder:
    """Build c_qi from the original question messages and the previous answer.

    single_turn: the original user message is kept verbatim and followed by the previous answer and the feedback.
    multi_turn:  [original messages..., assistant: previous answer, user: feedback].

    ``failure_instruction`` (optional) replaces ``instruction`` when no answer portion could be extracted from the
    previous response (the failure placeholder is shown instead of an answer).
    """

    def __init__(
        self,
        instruction: str = DEFAULT_INSTRUCTION,
        prompt_format: str = "single_turn",
        answer_header: str = DEFAULT_ANSWER_HEADER,
        failure_placeholder: str = DEFAULT_FAILURE_PLACEHOLDER,
        failure_instruction: str | None = None,
    ):
        assert prompt_format in ("single_turn", "multi_turn"), f"unknown repair prompt format {prompt_format}"
        self.instruction = instruction
        self.prompt_format = prompt_format
        self.answer_header = answer_header
        self.failure_placeholder = failure_placeholder
        self.failure_instruction = failure_instruction

    def build(self, question_messages: list[dict], previous_answer: str, extraction_ok: bool = True) -> list[dict]:
        messages = [dict(m) for m in copy.deepcopy(list(question_messages))]
        assert messages and messages[-1]["role"] == "user", "the question must end with a user message"
        instruction = self.instruction if extraction_ok else (self.failure_instruction or self.instruction)
        if self.prompt_format == "multi_turn":
            return messages + [
                {"role": "assistant", "content": previous_answer},
                {"role": "user", "content": instruction},
            ]
        messages[-1]["content"] = f"{messages[-1]['content']}\n\n{self.answer_header}\n{previous_answer}\n\n{instruction}"
        return messages


def builder_from_config(repair_cfg) -> RepairPromptBuilder:
    """RepairPromptBuilder for ``allocate_repair.repair``: prompt_version v1 (one instruction for every failed
    response) or v3 (per-case instructions, see V3_* above)."""
    version = str(repair_cfg.get("prompt_version", "v1"))
    common = {"prompt_format": repair_cfg.prompt_format, "answer_header": repair_cfg.answer_header,
              "failure_placeholder": repair_cfg.extraction_failure_placeholder}
    if version == "v1":
        return RepairPromptBuilder(instruction=repair_cfg.instruction, **common)
    if version == "v3":
        return RepairPromptBuilder(instruction=repair_cfg.v3.wrong_answer_instruction,
                                   failure_instruction=repair_cfg.v3.no_final_answer_instruction, **common)
    raise ValueError(f"unknown allocate_repair.repair.prompt_version {version!r} (v1 | v3)")


def count_prompt_tokens(tokenizer, messages: list[dict], chat_template_kwargs: dict | None = None) -> int:
    text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False, **(chat_template_kwargs or {})
    )
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


@dataclass
class RepairRequest:
    parent_index: int
    """Row of the failed phase-1 response in the phase-1 batch."""
    messages: list
    extraction: ExtractionResult
    prompt_tokens: int


def build_repair_requests(
    failed_rows: list[int],
    response_ids: Mapping[int, list[int]],
    question_messages: Mapping[int, list[dict]],
    extractor: AnswerExtractor,
    builder: RepairPromptBuilder,
    tokenizer,
    max_prompt_tokens: int,
    failure_policy: str = "placeholder",
    chat_template_kwargs: dict | None = None,
) -> tuple[list[RepairRequest], dict]:
    """Construct one repair prompt per failed phase-1 response.

    Args:
        failed_rows: indices of responses with r_qi = 0.
        response_ids: valid (unpadded) response token ids for every phase-1 row.
        question_messages: original chat messages for every phase-1 row.
        failure_policy: "placeholder" shows ``builder.failure_placeholder`` when no answer portion can be
            extracted; "skip" drops that parent from phase 2. Either way the failure is counted.

    Returns:
        (requests, stats) where stats holds raw counts for logging.
    """
    assert failure_policy in ("placeholder", "skip")
    stats = {
        "failed": len(failed_rows),
        "extraction_failures": 0,
        "skipped": 0,
        "answer_truncated": 0,
        "prompt_shortened": 0,
        "reasons": {},
    }
    requests = []
    for row in failed_rows:
        result = extractor.extract(response_ids[row])
        if not result.ok:
            stats["extraction_failures"] += 1
            stats["reasons"][result.reason] = stats["reasons"].get(result.reason, 0) + 1
            if failure_policy == "skip":
                stats["skipped"] += 1
                continue
            answer = builder.failure_placeholder
        else:
            answer = result.text
            stats["answer_truncated"] += int(result.truncated)
        messages = builder.build(question_messages[row], answer, extraction_ok=result.ok)
        n_tokens = count_prompt_tokens(tokenizer, messages, chat_template_kwargs)
        # Never let the rollout left-truncate the question: shorten the previous answer instead.
        if n_tokens > max_prompt_tokens and result.ok:
            stats["prompt_shortened"] += 1
            answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
            while n_tokens > max_prompt_tokens and len(answer_ids) > 16:
                answer_ids = answer_ids[len(answer_ids) // 4 :]
                answer = TRUNCATION_MARKER + tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
                messages = builder.build(question_messages[row], answer, extraction_ok=result.ok)
                n_tokens = count_prompt_tokens(tokenizer, messages, chat_template_kwargs)
        if n_tokens > max_prompt_tokens:
            raise ValueError(
                f"repair prompt needs {n_tokens} tokens > rollout.prompt_length={max_prompt_tokens}; "
                "increase actor_rollout_ref.rollout.prompt_length"
            )
        requests.append(RepairRequest(row, messages, result, n_tokens))
    return requests, stats
