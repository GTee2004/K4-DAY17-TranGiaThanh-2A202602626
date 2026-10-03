from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens, extract_profile_updates
from model_provider import build_chat_model


@dataclass
class SessionState:
    """In-memory state scoped to exactly one conversation thread."""

    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Baseline agent with short-term, within-thread memory only.

    The baseline deliberately has no persistent profile and no compaction. A
    new ``thread_id`` therefore starts empty, including for a known user.
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = None if force_offline else self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Reply using only the history associated with ``thread_id``.

        ``user_id`` is intentionally not used as a memory key. It remains in
        the public API so baseline and advanced agents can be benchmarked with
        the same call signature.
        """

        del user_id
        if self.langchain_agent is not None and not self.force_offline:
            try:
                return self._reply_live(thread_id, message)
            except Exception:
                # Keep the lab usable when credentials or network access fail.
                # _reply_live commits state only after a successful invocation.
                self.langchain_agent = None
        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        """Return cumulative response tokens generated in this thread."""

        return self._session(thread_id).token_usage

    def prompt_token_usage(self, thread_id: str) -> int:
        """Return cumulative prompt-context tokens processed in this thread."""

        return self._session(thread_id).prompt_tokens_processed

    def compaction_count(self, thread_id: str) -> int:
        """Baseline never compacts its conversation history."""

        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        """Provide deterministic behavior for tests and offline benchmarks."""

        session = self._session(thread_id)
        prompt_tokens = estimate_tokens(_messages_text(session.messages)) + estimate_tokens(message)
        session.prompt_tokens_processed += prompt_tokens

        session.messages.append({"role": "user", "content": message})
        response = self._offline_response(session, message)
        response_tokens = estimate_tokens(response)
        session.token_usage += response_tokens
        session.messages.append({"role": "assistant", "content": response})

        return {
            "answer": response,
            "content": response,
            "agent_tokens": response_tokens,
            "prompt_tokens": prompt_tokens,
            "thread_id": thread_id,
            "memory": "short-term-only",
        }

    def _reply_live(self, thread_id: str, message: str) -> dict[str, Any]:
        """Invoke the configured chat model with the current thread transcript."""

        session = self._session(thread_id)
        prompt_tokens = estimate_tokens(_messages_text(session.messages)) + estimate_tokens(message)
        pending_messages = [*session.messages, {"role": "user", "content": message}]
        result = self.langchain_agent.invoke(_build_prompt(pending_messages))
        response = getattr(result, "content", str(result))
        response_tokens = estimate_tokens(response)

        # Commit only after invocation succeeds, so the offline fallback does
        # not duplicate the user message or token accounting.
        session.prompt_tokens_processed += prompt_tokens
        session.messages = pending_messages
        session.token_usage += response_tokens
        session.messages.append({"role": "assistant", "content": response})

        return {
            "answer": response,
            "content": response,
            "agent_tokens": response_tokens,
            "prompt_tokens": prompt_tokens,
            "thread_id": thread_id,
            "memory": "short-term-only",
        }

    def _offline_response(self, session: SessionState, message: str) -> str:
        """Answer recall questions from this thread, never from another one."""

        facts = _session_facts(session.messages)
        question = _ascii_lower(message)

        if _asks_for_name(question):
            name = facts.get("name")
            if name:
                return f"Trong thread nay, minh thay ban ten la {name}."
            return "Minh khong co long-term memory nen khong biet ten ban trong thread moi."

        if _asks_for_location(question):
            location = facts.get("location")
            if location:
                return f"Trong thread nay, minh thay ban dang o {location}."
            return "Minh khong thay thong tin noi o trong thread hien tai."

        if _asks_for_profession(question):
            profession = facts.get("profession")
            if profession:
                return f"Trong thread nay, nghe hien tai cua ban la {profession}."
            return "Minh khong thay thong tin nghe nghiep trong thread hien tai."

        if _asks_for_style(question):
            style = facts.get("response_style")
            if style:
                return f"Trong thread nay, ban thich style tra loi {style}."
            return "Minh khong thay preference ve style tra loi trong thread hien tai."

        if _asks_for_food_or_drink(question):
            parts: list[str] = []
            if facts.get("favorite_drink"):
                parts.append(f"do uong yeu thich: {facts['favorite_drink']}")
            if facts.get("favorite_food"):
                parts.append(f"mon an yeu thich: {facts['favorite_food']}")
            if parts:
                return "Trong thread nay, minh thay " + "; ".join(parts) + "."
            return "Minh khong thay thong tin mon an/do uong yeu thich trong thread hien tai."

        updates = extract_profile_updates(message)
        if updates:
            labels = ", ".join(sorted(updates))
            return f"Da ghi nho tam thoi trong thread nay cac thong tin: {labels}."

        return "Minh da nhan thong tin trong thread hien tai, nhung se khong nho sang thread moi."

    def _maybe_build_langchain_agent(self):
        """Build the configured live model, falling back cleanly if unavailable."""

        try:
            return build_chat_model(self.config.model)
        except Exception:
            return None

    def _session(self, thread_id: str) -> SessionState:
        if not isinstance(thread_id, str) or not thread_id.strip():
            raise ValueError("thread_id must be a non-empty string")
        if thread_id not in self.sessions:
            self.sessions[thread_id] = SessionState()
        return self.sessions[thread_id]


def _session_facts(messages: list[dict[str, str]]) -> dict[str, str]:
    """Reconstruct the latest facts found only in user turns of one thread."""

    facts: dict[str, str] = {}
    for item in messages:
        if item.get("role") == "user":
            facts.update(extract_profile_updates(item.get("content", "")))
    return facts


def _messages_text(messages: list[dict[str, str]]) -> str:
    return "\n".join(
        f"{item.get('role', '')}: {item.get('content', '')}" for item in messages
    )


def _build_prompt(messages: list[dict[str, str]]) -> str:
    return (
        "You are the baseline agent in a memory-system lab. "
        "Use only the current thread transcript. Do not claim long-term memory.\n\n"
        f"Transcript:\n{_messages_text(messages)}\n\n"
        "Answer the latest user message concisely."
    )


def _ascii_lower(text: str) -> str:
    import unicodedata

    text = text.replace("\u0111", "d").replace("\u0110", "D")
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(char for char in normalized if not unicodedata.combining(char)).lower()


def _asks_for_name(text: str) -> bool:
    return any(marker in text for marker in ("ten", "name", "toi la ai", "minh la ai"))


def _asks_for_location(text: str) -> bool:
    return any(marker in text for marker in ("o dau", "dang o", "noi o", "location", "live"))


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
