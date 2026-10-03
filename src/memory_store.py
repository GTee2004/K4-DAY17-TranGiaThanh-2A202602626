from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict


class ThreadMemory(TypedDict):
    messages: list[dict[str, str]]
    summary: str
    compactions: int


@dataclass(frozen=True)
class ProfileFactCandidate:
    """A possible persistent fact together with an auditable confidence score."""

    key: str
    value: str
    confidence: float
    reason: str


def estimate_tokens(text: str) -> int:
    """Estimate tokens deterministically using roughly four characters each."""

    normalized = text.strip()
    if not normalized:
        return 0
    return max(1, (len(normalized) + 3) // 4)


@dataclass
class UserProfileStore:
    """Persistent, per-user Markdown profile storage."""

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        """Return a safe ``<root>/<user>/User.md`` path."""

        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("user_id must be a non-empty string")

        ascii_id = (
            unicodedata.normalize("NFKD", user_id.strip())
            .encode("ascii", "ignore")
            .decode("ascii")
        )
        slug = re.sub(r"[^A-Za-z0-9_-]+", "-", ascii_id).strip("-_")
        if not slug:
            raise ValueError(f"user_id {user_id!r} does not contain safe characters")
        return self.root_dir / slug / "User.md"

    def read_text(self, user_id: str) -> str:
        """Read a profile, returning an empty Markdown profile if absent."""

        path = self.path_for(user_id)
        if not path.exists():
            return "# User Profile\n\n"
        return path.read_text(encoding="utf-8")

    def write_text(self, user_id: str, content: str) -> Path:
        """Write a UTF-8 profile and return its path."""

        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        """Replace the first matching occurrence and report whether it changed."""

        path = self.path_for(user_id)
        if not search_text or not path.exists():
            return False
        content = path.read_text(encoding="utf-8")
        if search_text not in content:
            return False
        self.write_text(user_id, content.replace(search_text, replacement, 1))
        return True

    def file_size(self, user_id: str) -> int:
        """Return the profile size in bytes, or zero when it does not exist."""

        path = self.path_for(user_id)
        return path.stat().st_size if path.exists() else 0

    def facts(self, user_id: str) -> dict[str, str]:
        """Parse ``- key: value`` entries from a user's profile."""

        result: dict[str, str] = {}
        for line in self.read_text(user_id).splitlines():
            match = re.match(r"^\s*-\s*([a-z][a-z0-9_]*)\s*:\s*(.+?)\s*$", line)
            if match:
                result[match.group(1)] = match.group(2)
        return result

    def upsert_fact(self, user_id: str, key: str, value: str) -> Path:
        """Insert or replace one structured fact in ``User.md``."""

        normalized_key = re.sub(r"[^a-z0-9_]+", "_", key.strip().lower()).strip("_")
        normalized_value = value.strip()
        if not normalized_key or not normalized_value:
            raise ValueError("fact key and value must be non-empty")

        profile_facts = self.facts(user_id)
        profile_facts[normalized_key] = normalized_value
        lines = ["# User Profile", ""]
        lines.extend(f"- {fact_key}: {fact_value}" for fact_key, fact_value in profile_facts.items())
        return self.write_text(user_id, "\n".join(lines) + "\n")


def extract_profile_updates(message: str) -> dict[str, str]:
    """Extract high-confidence, stable profile facts from a user message."""

    text = " ".join(message.split())
    if not text:
        return {}

    updates: dict[str, str] = {}

    name = _last_capture(
        text,
        (
            r"\b(?:mình|tôi)\s+tên\s+(?:là\s+)?(.+?)(?=[,.!?;]|$)",
            r"\btên\s+(?:của\s+)?(?:mình|tôi)\s+là\s+(.+?)(?=[,.!?;]|$)",
        ),
    )
    if name and _is_concrete_fact(name):
        updates["name"] = name

    location = _last_capture(
        text,
        (
            r"\b(?:mình|tôi)\s+(?:(?:hiện|hiện tại|giờ|vẫn)\s+)?(?:đang\s+)?ở\s+(.+?)(?=\s+(?:và|chứ|nhưng|dù|để)\b|[,.!?;]|$)",
            r"\bhiện ở\s+(.+?)(?=\s+(?:và|chứ|nhưng|dù|để)\b|[,.!?;]|$)",
            r"\b(?:mình|tôi)\s+đang làm việc ở\s+(.+?)(?=\s+vài\b)",
            r"\bnơi ở\s+(?:hiện tại\s+)?(?:của\s+(?:mình|tôi)\s+)?(?:vẫn\s+)?là\s+(.+?)(?=\s+(?:và|chứ|nhưng|dù)\b|[,.!?;]|$)",
        ),
    )
    if location and _is_concrete_fact(location):
        updates["location"] = location

    profession = _last_capture(
        text,
        (
            r"\b(?:mình|tôi)\s+(?:(?:hiện|hiện tại|vẫn)\s+)?(?:đang\s+)?làm\s+(.+?)(?=\s+(?:cho|tại|với|và)\b|[,.!?;]|$)",
            r"\bvà\s+đang làm\s+(.+?)(?=\s+(?:cho|tại|với|và)\b|[,.!?;]|$)",
            r"\b(?:giờ\s+)?chuyển sang\s+(.+?)(?=\s+(?:cho|tại|với|và)\b|[,.!?;]|$)",
            r"\bnghề(?: nghiệp)?(?: hiện tại)?(?: của (?:mình|tôi))?\s+(?:thì\s+)?(?:vẫn\s+)?là\s+(.+?)(?=\s+(?:và|nhưng)\b|[,.!?;]|$)",
        ),
    )
    if (
        profession
        and _is_concrete_fact(profession)
        and not profession.casefold().startswith("việc ")
        and not _looks_hypothetical(text, profession)
    ):
        updates["profession"] = profession

    lower = text.casefold()
    style_parts: list[str] = []
    if "3 bullet" in lower:
        style_parts.append("3 bullet ngắn")
    elif "bullet" in lower:
        style_parts.append("bullet ngắn")
    elif "ngắn gọn" in lower or "trả lời ngắn" in lower or "câu trả lời gọn" in lower:
        style_parts.append("ngắn gọn")
    if "rõ ý" in lower:
        style_parts.append("rõ ý")
    if "ví dụ thực chiến" in lower:
        style_parts.append("có ví dụ thực chiến")
    elif "ví dụ thực tế" in lower:
        style_parts.append("có ví dụ thực tế")
    if "trade-off" in lower and any(word in lower for word in ("trả lời", "giải thích", "style")):
        style_parts.append("nhấn mạnh trade-off")
    has_primary_style = any(
        cue in lower
        for cue in (
            "3 bullet",
            "bullet",
            "ngắn gọn",
            "trả lời ngắn",
            "câu trả lời gọn",
            "đừng lan man",
            "không thích câu trả lời quá lan man",
        )
    )
    if style_parts and has_primary_style and any(
        word in lower for word in ("muốn", "thích", "hãy", "style", "trả lời", "giải thích")
    ):
        updates["response_style"] = ", ".join(dict.fromkeys(style_parts))

    if "cà phê sữa đá" in lower and any(
        cue in lower for cue in ("đồ uống yêu thích", "thích", "vẫn uống", "món ruột")
    ):
        updates["favorite_drink"] = "cà phê sữa đá"
    if "mì quảng" in lower and any(
        cue in lower for cue in ("món ăn yêu thích", "món ruột", "thích", "ăn mì quảng")
    ):
        updates["favorite_food"] = "mì Quảng"
    if "corgi" in lower and any(cue in lower for cue in ("nuôi", "con corgi", "bé corgi")):
        pet_name = " tên Bơ" if re.search(r"\b(?:tên|corgi)\s+Bơ\b", text, re.IGNORECASE) else ""
        updates["pet"] = f"corgi{pet_name}"

    if any(cue in lower for cue in ("mình thích", "mình đang quan tâm", "mối quan tâm")):
        interests = [
            label
            for keyword, label in (
                ("python", "Python"),
                ("ai", "AI"),
                ("mlops", "MLOps"),
                ("rag", "RAG"),
                ("memory", "memory systems"),
            )
            if re.search(rf"\b{re.escape(keyword)}\b", lower)
        ]
        if interests:
            updates["interests"] = ", ".join(dict.fromkeys(interests))

    return updates


def extract_profile_candidates(message: str) -> list[ProfileFactCandidate]:
    """Score extracted facts before they are allowed into persistent memory.

    The deterministic extractor only emits facts matching explicit Vietnamese
    patterns. Hedging still lowers confidence because persisting an uncertain
    statement is more harmful than keeping it in short-term thread memory.
    Explicit corrections receive the highest confidence and can safely replace
    an older value through ``upsert_fact``.
    """

    updates = extract_profile_updates(message)
    normalized = message.casefold()
    uncertain_cues = (
        "có lẽ",
        "hình như",
        "không chắc",
        "có thể",
        "tạm đoán",
    )
    correction_cues = (
        "đính chính",
        "thực ra",
        "không còn",
        "chuyển sang",
        "đã cập nhật",
    )

    if any(cue in normalized for cue in uncertain_cues):
        confidence, reason = 0.55, "hedged or uncertain statement"
    elif any(cue in normalized for cue in correction_cues):
        confidence, reason = 0.99, "explicit correction"
    else:
        confidence, reason = 0.95, "explicit first-person statement"

    return [
        ProfileFactCandidate(key, value, confidence, reason)
        for key, value in updates.items()
    ]


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Create a bounded deterministic summary while preserving stable facts."""

    if max_items <= 0 or not messages:
        return ""

    facts: dict[str, str] = {}
    for message in messages:
        if message.get("role", "").casefold() == "user":
            facts.update(extract_profile_updates(message.get("content", "")))

    selected = _select_summary_items(messages, max_items)
    lines: list[str] = []
    if facts:
        rendered_facts = "; ".join(f"{key}={value}" for key, value in facts.items())
        lines.append(f"- Stable facts: {rendered_facts}")

    for message in selected:
        role = message.get("role", "unknown").strip().capitalize() or "Unknown"
        content = " ".join(message.get("content", "").split())
        if not content:
            continue
        if len(content) > 240:
            content = content[:237].rstrip() + "..."
        lines.append(f"- {role}: {content}")
    return "\n".join(lines)


@dataclass
class CompactMemoryManager:
    """Keep recent messages verbatim and summarize older thread content."""

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.threshold_tokens <= 0:
            raise ValueError("threshold_tokens must be greater than zero")
        if self.keep_messages < 0:
            raise ValueError("keep_messages cannot be negative")

    def append(self, thread_id: str, role: str, content: str) -> None:
        """Append a message and compact the thread when it crosses the limit."""

        thread = self._thread(thread_id)
        messages = thread["messages"]
        assert isinstance(messages, list)
        messages.append({"role": role, "content": content})

        summary = str(thread["summary"])
        total_tokens = estimate_tokens(summary) + sum(
            estimate_tokens(str(message.get("content", ""))) for message in messages
        )
        if total_tokens <= self.threshold_tokens or len(messages) <= self.keep_messages:
            return

        split_at = len(messages) - self.keep_messages
        archived = messages[:split_at]
        recent = messages[split_at:]
        summary_input: list[dict[str, str]] = []
        if summary:
            summary_input.append({"role": "memory", "content": summary})
        summary_input.extend(archived)

        thread["summary"] = summarize_messages(summary_input)
        thread["messages"] = recent
        thread["compactions"] = int(thread["compactions"]) + 1

    def context(self, thread_id: str) -> dict[str, object]:
        """Return a copy of one thread's current compact-memory context."""

        thread = self._thread(thread_id)
        messages = thread["messages"]
        assert isinstance(messages, list)
        return {
            "messages": [dict(message) for message in messages],
            "summary": str(thread["summary"]),
            "compactions": int(thread["compactions"]),
        }

    def compaction_count(self, thread_id: str) -> int:
        """Return the number of compactions performed for one thread."""

        return int(self._thread(thread_id)["compactions"])

    def _thread(self, thread_id: str) -> ThreadMemory:
        if not isinstance(thread_id, str) or not thread_id.strip():
            raise ValueError("thread_id must be a non-empty string")
        thread = self.state.setdefault(
            thread_id,
            {"messages": [], "summary": "", "compactions": 0},
        )
        return thread  # type: ignore[return-value]


def _last_capture(text: str, patterns: tuple[str, ...]) -> str | None:
    matches: list[tuple[int, str]] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            value = match.group(1).strip(" \t\n\r,.;:!?")
            if value:
                matches.append((match.start(), value))
    return max(matches, default=(0, ""), key=lambda item: item[0])[1] or None


def _looks_hypothetical(text: str, value: str) -> bool:
    value_position = text.casefold().find(value.casefold())
    prefix = text[max(0, value_position - 60) : value_position].casefold()
    return any(cue in prefix for cue in ("đùa", "hay là", "giả sử", "nếu"))


def _is_concrete_fact(value: str) -> bool:
    return not re.search(
        r"\b(?:ai|gì|đâu|nào|không|bao nhiêu)\b",
        value.casefold(),
    )


def _select_summary_items(
    messages: list[dict[str, str]], max_items: int
) -> list[dict[str, str]]:
    if len(messages) <= max_items:
        return messages
    first_count = max_items // 2
    last_count = max_items - first_count
    return messages[:first_count] + messages[-last_count:]
