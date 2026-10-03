from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import (
    CompactMemoryManager,
    UserProfileStore,
    estimate_tokens,
    extract_profile_candidates,
)
from model_provider import build_chat_model


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Agent with short-term, persistent, and compact memory."""

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}
        self.langchain_agent = None if force_offline else self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Reply through the live model when available, otherwise offline."""

        if self.langchain_agent is not None and not self.force_offline:
            try:
                return self._reply_live(user_id, thread_id, message)
            except Exception:
                # Live state is committed only after a successful invocation,
                # so falling back cannot duplicate the incoming turn.
                self.langchain_agent = None
        return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Run a deterministic turn using all three memory layers."""

        updates = self._profile_updates(user_id, message)
        self._persist_updates(user_id, updates)

        self.compact_memory.append(thread_id, "user", message)
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        self.thread_prompt_tokens[thread_id] = (
            self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens
        )

        response = self._offline_response(user_id, thread_id, message)
        response_tokens = estimate_tokens(response)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + response_tokens
        self.compact_memory.append(thread_id, "assistant", response)

        return self._result(
            user_id,
            thread_id,
            response,
            response_tokens,
            prompt_tokens,
            updates,
        )

    def _reply_live(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Invoke a provider model, committing memory only after success."""

        updates = self._profile_updates(user_id, message)
        profile = _render_profile(self.profile_store.facts(user_id), updates)
        context = self.compact_memory.context(thread_id)
        prompt = _build_live_prompt(profile, context, message)
        result = self.langchain_agent.invoke(prompt)
        response = getattr(result, "content", str(result))
        if not isinstance(response, str):
            response = str(response)

        # Commit after invoke so reply() can safely fall back on provider errors.
        self._persist_updates(user_id, updates)
        self.compact_memory.append(thread_id, "user", message)
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        self.thread_prompt_tokens[thread_id] = (
            self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens
        )
        response_tokens = estimate_tokens(response)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + response_tokens
        self.compact_memory.append(thread_id, "assistant", response)

        return self._result(
            user_id,
            thread_id,
            response,
            response_tokens,
            prompt_tokens,
            updates,
        )

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        """Estimate profile + compact summary + recent-message prompt load."""

        context = self.compact_memory.context(thread_id)
        messages = context.get("messages", [])
        recent_text = _messages_text(messages if isinstance(messages, list) else [])
        return sum(
            (
                estimate_tokens(self.profile_store.read_text(user_id)),
                estimate_tokens(str(context.get("summary", ""))),
                estimate_tokens(recent_text),
            )
        )

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        """Answer deterministically from persistent facts and compact context."""

        facts = self.profile_store.facts(user_id)
        question = _ascii_lower(message)
        requested = _requested_categories(question)

        if _asks_for_profile_recall(question) or len(requested) > 1:
            summary = _facts_summary(facts)
            if summary:
                return "Minh nho trong User.md: " + summary
            return "Minh chua co fact ben vung nao trong User.md cho ban."

        if "name" in requested:
            return _answer_one_fact(facts, "name", "ten")
        if "location" in requested:
            return _answer_one_fact(facts, "location", "noi o hien tai")
        if "profession" in requested:
            return _answer_one_fact(facts, "profession", "nghe hien tai")
        if "response_style" in requested:
            return _answer_one_fact(facts, "response_style", "style tra loi")
        if "interests" in requested:
            return _answer_one_fact(facts, "interests", "moi quan tam ky thuat")
        if "pet" in requested:
            return _answer_one_fact(facts, "pet", "thu cung")
        if "food_or_drink" in requested:
            parts: list[str] = []
            if facts.get("favorite_drink"):
                parts.append(f"do uong yeu thich la {facts['favorite_drink']}")
            if facts.get("favorite_food"):
                parts.append(f"mon an yeu thich la {facts['favorite_food']}")
            if parts:
                return "Minh nho trong User.md: " + "; ".join(parts) + "."
            return "Minh chua thay fact ve mon an hoac do uong yeu thich trong User.md."

        updates = self._profile_updates(user_id, message)
        if updates:
            labels = ", ".join(sorted(updates))
            return f"Da cap nhat User.md cac fact: {labels}."

        context = self.compact_memory.context(thread_id)
        if context.get("summary"):
            return "Minh da nhan thong tin va se giu phan gan nhat kem summary compact cho thread dai."
        return "Minh da nhan thong tin va se luu cac fact on dinh vao User.md khi phat hien duoc."

    def _profile_updates(self, user_id: str, message: str) -> dict[str, str]:
        """Accept only stable facts meeting the persistence confidence threshold."""

        if _skip_profile_write(message):
            return {}
        updates = {
            candidate.key: candidate.value
            for candidate in extract_profile_candidates(message)
            if candidate.confidence >= self.config.memory_confidence_threshold
        }
        if "interests" in updates:
            existing = self.profile_store.facts(user_id).get("interests")
            updates["interests"] = _merge_terms(existing, updates["interests"])
        return updates

    def _persist_updates(self, user_id: str, updates: dict[str, str]) -> None:
        for key, value in updates.items():
            self.profile_store.upsert_fact(user_id, key, value)

    def _result(
        self,
        user_id: str,
        thread_id: str,
        response: str,
        response_tokens: int,
        prompt_tokens: int,
        updates: dict[str, str],
    ) -> dict[str, Any]:
        return {
            "answer": response,
            "content": response,
            "agent_tokens": response_tokens,
            "prompt_tokens": prompt_tokens,
            "thread_id": thread_id,
            "memory": "short-term+persistent+compact",
            "memory_path": str(self.profile_store.path_for(user_id)),
            "profile_updates": updates,
            "compactions": self.compaction_count(thread_id),
        }

    def _maybe_build_langchain_agent(self):
        """Build the configured provider model when its dependency is available."""

        try:
            return build_chat_model(self.config.model)
        except Exception:
            return None


def _skip_profile_write(message: str) -> bool:
    """Questions should retrieve memory rather than become profile facts."""

    text = _ascii_lower(message)
    retrieval_markers = (
        "nhac lai",
        "tom tat",
        "ban biet",
        "minh la ai",
        "dau moi la",
    )
    return message.rstrip().endswith("?") or any(
        marker in text for marker in retrieval_markers
    )


def _merge_terms(existing: str | None, incoming: str) -> str:
    terms: list[str] = []
    for value in (existing or "", incoming):
        for item in value.split(","):
            cleaned = item.strip()
            if cleaned and cleaned.casefold() not in {term.casefold() for term in terms}:
                terms.append(cleaned)
    return ", ".join(terms)


def _facts_summary(facts: dict[str, str]) -> str:
    order = (
        ("name", "ten"),
        ("profession", "nghe hien tai"),
        ("location", "noi o hien tai"),
        ("favorite_drink", "do uong yeu thich"),
        ("favorite_food", "mon an yeu thich"),
        ("pet", "thu cung"),
        ("response_style", "style tra loi"),
        ("interests", "moi quan tam ky thuat"),
    )
    parts = [f"{label}: {facts[key]}" for key, label in order if facts.get(key)]
    return "; ".join(parts) + "." if parts else ""


def _answer_one_fact(facts: dict[str, str], key: str, label: str) -> str:
    if facts.get(key):
        return f"Minh nho trong User.md: {label} cua ban la {facts[key]}."
    return f"Minh chua thay fact '{label}' trong User.md."


def _requested_categories(text: str) -> set[str]:
    categories: set[str] = set()
    checks = (
        ("name", _asks_for_name),
        ("location", _asks_for_location),
        ("profession", _asks_for_profession),
        ("response_style", _asks_for_style),
        ("food_or_drink", _asks_for_food_or_drink),
        ("pet", _asks_for_pet),
        ("interests", _asks_for_interests),
    )
    for category, predicate in checks:
        if predicate(text):
            categories.add(category)
    return categories


def _messages_text(messages: list[dict[str, str]]) -> str:
    return "\n".join(
        f"{item.get('role', '')}: {item.get('content', '')}" for item in messages
    )


def _render_profile(existing: dict[str, str], updates: dict[str, str]) -> str:
    facts = {**existing, **updates}
    lines = ["# User Profile", ""]
    lines.extend(f"- {key}: {value}" for key, value in facts.items())
    return "\n".join(lines) + "\n"


def _build_live_prompt(
    profile: str, context: dict[str, object], latest_message: str
) -> str:
    messages = context.get("messages", [])
    recent_text = _messages_text(messages if isinstance(messages, list) else [])
    return (
        "You are the advanced agent in a memory-system lab. Use User.md as persistent memory, "
        "the compact summary as older thread context, and recent messages for immediate follow-up.\n\n"
        f"User.md:\n{profile}\n"
        f"Compact summary:\n{context.get('summary', '')}\n\n"
        f"Recent messages:\n{recent_text}\n\n"
        f"Latest user message:\n{latest_message}\n\n"
        "Answer concisely and prefer persisted corrected facts over stale details."
    )


def _ascii_lower(text: str) -> str:
    text = text.replace("\u0111", "d").replace("\u0110", "D")
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(char for char in normalized if not unicodedata.combining(char)).lower()


def _asks_for_profile_recall(text: str) -> bool:
    markers = ("nhac lai", "tom tat", "ban biet", "recall", "minh la ai", "dau moi la")
    return any(marker in text for marker in markers)


def _asks_for_name(text: str) -> bool:
    return any(marker in text for marker in ("ten", "name", "toi la ai", "minh la ai"))


def _asks_for_location(text: str) -> bool:
    return any(
        marker in text
        for marker in ("o dau", "dang o", "con o", "noi o", "location", "live")
    )


def _asks_for_profession(text: str) -> bool:
    return any(marker in text for marker in ("nghe", "cong viec", "job", "profession", "work"))


def _asks_for_style(text: str) -> bool:
    return any(marker in text for marker in ("style", "kieu tra loi", "tra loi", "bullet"))


def _asks_for_food_or_drink(text: str) -> bool:
    return any(
        marker in text
        for marker in (
            "do uong",
            "mon an",
            "favorite drink",
            "favorite food",
            "an yeu thich",
            "uong yeu thich",
        )
    )


def _asks_for_pet(text: str) -> bool:
    return any(marker in text for marker in ("nuoi con gi", "thu cung", "pet", "corgi"))


def _asks_for_interests(text: str) -> bool:
    return any(marker in text for marker in ("moi quan tam", "quan tam ky thuat", "so thich ky thuat"))
