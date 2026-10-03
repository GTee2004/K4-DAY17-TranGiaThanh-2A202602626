from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from uuid import uuid4

try:
    from tabulate import tabulate as _tabulate
except ImportError:  # The offline benchmark should still run before setup.
    _tabulate = None

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig, load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    """Read and minimally validate one UTF-8 benchmark dataset."""

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Benchmark dataset must be a JSON list: {path}")

    required = {"id", "user_id", "turns", "recall_questions"}
    for index, conversation in enumerate(data):
        if not isinstance(conversation, dict):
            raise ValueError(f"Conversation {index} in {path} must be an object")
        missing = required.difference(conversation)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"Conversation {index} in {path} is missing: {names}")
        if not isinstance(conversation["turns"], list):
            raise ValueError(f"Conversation {index} has a non-list 'turns' value")
        if not isinstance(conversation["recall_questions"], list):
            raise ValueError(
                f"Conversation {index} has a non-list 'recall_questions' value"
            )
    return data


def recall_points(answer: str, expected: list[str]) -> float:
    """Return 0 for no recall, 0.5 for partial, and 1 for full recall."""

    if not expected:
        return 0.0
    normalized_answer = answer.casefold()
    matches = sum(fact.casefold() in normalized_answer for fact in expected)
    if matches == len(expected):
        return 1.0
    return 0.5 if matches else 0.0


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Measure expected-fact coverage as a transparent offline proxy."""

    if not expected:
        return 0.0
    normalized_answer = answer.casefold()
    matches = sum(fact.casefold() in normalized_answer for fact in expected)
    return matches / len(expected)


def run_agent_benchmark(
    agent_name: str,
    agent: Any,
    conversations: list[dict[str, Any]],
    config: LabConfig,
) -> BenchmarkRow:
    """Run training turns and evaluate every recall question in a new thread.

    Recall is evaluated immediately after each conversation. This preserves the
    timeline of corrections in the dataset: later facts must not retroactively
    change the expected answer for an earlier conversation.
    """

    del config  # Kept in the shared API for compatibility with the lab scaffold.
    run_id = uuid4().hex
    users = {str(conversation["user_id"]) for conversation in conversations}
    memory_size = getattr(agent, "memory_file_size", None)
    initial_memory = (
        sum(int(memory_size(user_id)) for user_id in users)
        if callable(memory_size)
        else 0
    )

    measured_threads: list[str] = []
    recall_scores: list[float] = []
    quality_scores: list[float] = []

    for conversation_index, conversation in enumerate(conversations):
        user_id = str(conversation["user_id"])
        training_thread = f"benchmark-{run_id}-{conversation_index}-training"
        measured_threads.append(training_thread)

        for message in conversation["turns"]:
            agent.reply(user_id, training_thread, str(message))

        for question_index, question in enumerate(conversation["recall_questions"]):
            recall_thread = (
                f"benchmark-{run_id}-{conversation_index}-recall-{question_index}"
            )
            measured_threads.append(recall_thread)
            result = agent.reply(user_id, recall_thread, str(question["question"]))
            answer = _response_text(result)
            expected = [str(item) for item in question["expected_contains"]]
            recall_scores.append(recall_points(answer, expected))
            quality_scores.append(heuristic_quality(answer, expected))

    final_memory = (
        sum(int(memory_size(user_id)) for user_id in users)
        if callable(memory_size)
        else 0
    )
    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=sum(
            int(agent.token_usage(thread_id)) for thread_id in measured_threads
        ),
        prompt_tokens_processed=sum(
            int(agent.prompt_token_usage(thread_id)) for thread_id in measured_threads
        ),
        recall_score=_average(recall_scores),
        response_quality=_average(quality_scores),
        memory_growth_bytes=final_memory - initial_memory,
        compactions=sum(
            int(agent.compaction_count(thread_id)) for thread_id in measured_threads
        ),
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    """Render all six required metrics as a GitHub-style table."""

    values = [
        [
            row.agent_name,
            row.agent_tokens_only,
            row.prompt_tokens_processed,
            f"{row.recall_score:.1%}",
            f"{row.response_quality:.1%}",
            row.memory_growth_bytes,
            row.compactions,
        ]
        for row in rows
    ]
    headers = [
        "Agent",
        "Agent tokens only",
        "Prompt tokens processed",
        "Cross-session recall",
        "Response quality",
        "Memory growth (bytes)",
        "Compactions",
    ]
    if _tabulate is not None:
        return _tabulate(values, headers=headers, tablefmt="github")
    return _markdown_table(headers, values)


def main() -> None:
    """Run the standard and long-context suites in deterministic offline mode."""

    config = load_config(Path(__file__).resolve().parent.parent)
    print("Offline benchmark: token counts are deterministic estimates.")
    print("Response quality is expected-fact coverage, not an LLM judge score.")

    suites = (
        ("Standard Benchmark", "conversations.json"),
        ("Long-Context Stress Benchmark", "advanced_long_context.json"),
    )
    for title, filename in suites:
        conversations = load_conversations(config.data_dir / filename)
        with TemporaryDirectory(prefix="memory-benchmark-") as temporary_state:
            suite_config = replace(config, state_dir=Path(temporary_state))
            rows = [
                run_agent_benchmark(
                    agent_name,
                    agent_type(suite_config, force_offline=True),
                    conversations,
                    suite_config,
                )
                for agent_name, agent_type in (
                    ("Baseline", BaselineAgent),
                    ("Advanced", AdvancedAgent),
                )
            ]

        print(f"\n## {title}\n")
        print(format_rows(rows))


def _response_text(result: Any) -> str:
    """Read the common response keys used by the lab's agent variants."""

    if isinstance(result, dict):
        for key in ("answer", "content", "response"):
            value = result.get(key)
            if value is not None:
                return str(value)
    return str(result)


def _average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    """Small dependency-free fallback for GitHub-style benchmark tables."""

    string_rows = [[str(value) for value in row] for row in rows]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in string_rows))
        if string_rows
        else len(headers[index])
        for index in range(len(headers))
    ]

    def render(row: list[str]) -> str:
        cells = [value.ljust(widths[index]) for index, value in enumerate(row)]
        return "| " + " | ".join(cells) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join([render(headers), separator, *(render(row) for row in string_rows)])


if __name__ == "__main__":
    main()
