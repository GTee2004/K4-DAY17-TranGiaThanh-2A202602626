from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig, load_config
from memory_store import (
    CompactMemoryManager,
    UserProfileStore,
    extract_profile_candidates,
)


def make_config(tmp_path: Path) -> LabConfig:
    """Build a fast, isolated configuration for each test."""

    config = load_config(Path(__file__).resolve().parent.parent)
    return replace(
        config,
        state_dir=tmp_path / "state",
        compact_threshold_tokens=80,
        compact_keep_messages=2,
        memory_confidence_threshold=0.8,
    )


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    """Verify ``User.md`` creation, reading, editing, and byte size."""

    store = UserProfileStore(tmp_path / "profiles")
    path = store.write_text(
        "dungct",
        "# User Profile\n\n- name: DungCT\n",
    )

    assert path == tmp_path / "profiles" / "dungct" / "User.md"
    assert path.name == "User.md"
    assert "DungCT" in store.read_text("dungct")
    assert store.edit_text("dungct", "DungCT", "DũngCT") is True
    assert store.edit_text("dungct", "missing", "value") is False
    assert "DũngCT" in store.read_text("dungct")
    assert store.file_size("dungct") == path.stat().st_size
    assert store.file_size("unknown-user") == 0


def test_compact_trigger(tmp_path: Path) -> None:
    """Verify a long thread is summarized while recent messages are kept."""

    manager = CompactMemoryManager(threshold_tokens=40, keep_messages=2)
    for index in range(6):
        manager.append("thread-1", "user", f"message {index} " * 20)

    context = manager.context("thread-1")
    assert manager.compaction_count("thread-1") > 0
    assert context["summary"]
    assert len(context["messages"]) <= 2

    # Compact memory is in-memory only and must not create persistent files.
    assert list(tmp_path.iterdir()) == []


def test_cross_session_recall(tmp_path: Path) -> None:
    """Verify Advanced recalls persisted facts while Baseline forgets them."""

    config = make_config(tmp_path)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)

    fact_messages = (
        "Mình tên là DũngCT.",
        "Mình đang làm MLOps engineer.",
    )
    for message in fact_messages:
        baseline.reply("dungct", "thread-a", message)
        advanced.reply("dungct", "thread-a", message)

    question = "Mình tên gì và hiện làm nghề gì?"
    baseline_answer = baseline.reply("dungct", "thread-b", question)["answer"]
    advanced_answer = advanced.reply("dungct", "thread-b", question)["answer"]

    assert "DũngCT" not in baseline_answer
    assert "MLOps engineer" not in baseline_answer
    assert "DũngCT" in advanced_answer
    assert "MLOps engineer" in advanced_answer
    assert advanced.memory_file_size("dungct") > 0
    assert advanced.profile_store.facts("dungct") == {
        "name": "DũngCT",
        "profession": "MLOps engineer",
    }


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    """Verify compaction lowers cumulative context processing on long input."""

    config = make_config(tmp_path)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)
    long_turns = [
        f"Đây là đoạn hội thoại dài số {index}. " + ("token context " * 30)
        for index in range(14)
    ]

    for turn in long_turns:
        baseline.reply("dungct", "long-thread", turn)
        advanced.reply("dungct", "long-thread", turn)

    assert advanced.compaction_count("long-thread") > 0
    assert baseline.compaction_count("long-thread") == 0
    assert (
        advanced.prompt_token_usage("long-thread")
        < baseline.prompt_token_usage("long-thread")
    )


def test_confidence_threshold_rejects_uncertain_fact(tmp_path: Path) -> None:
    """An uncertain statement stays in thread memory instead of polluting User.md."""

    config = make_config(tmp_path)
    agent = AdvancedAgent(config, force_offline=True)

    result = agent.reply(
        "dungct",
        "thread-a",
        "Mình đang ở Huế, nhưng mình không chắc vì có thể sắp chuyển đi.",
    )

    candidates = extract_profile_candidates(
        "Mình đang ở Huế, nhưng mình không chắc vì có thể sắp chuyển đi."
    )
    assert candidates[0].confidence < config.memory_confidence_threshold
    assert result["profile_updates"] == {}
    assert agent.profile_store.facts("dungct") == {}


def test_confidence_threshold_accepts_explicit_correction(tmp_path: Path) -> None:
    """A confident correction replaces the old persistent value."""

    config = make_config(tmp_path)
    agent = AdvancedAgent(config, force_offline=True)
    agent.reply("dungct", "thread-a", "Mình đang ở Đà Nẵng.")
    result = agent.reply(
        "dungct",
        "thread-a",
        "Mình đính chính: giờ mình đang ở Huế chứ không còn ở Đà Nẵng.",
    )

    assert result["profile_updates"] == {"location": "Huế"}
    assert agent.profile_store.facts("dungct")["location"] == "Huế"
