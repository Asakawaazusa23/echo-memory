from dataclasses import dataclass
from typing import Any


def _usage_int(data: dict[str, Any], key: str) -> int | None:
    try:
        return int(data[key]) if key in data else None
    except (TypeError, ValueError):
        return None


def _first_usage_int(data: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = _usage_int(data, key)
        if value is not None:
            return value
    return None


def _nested_usage_int(data: dict[str, Any], outer_key: str, inner_key: str) -> int | None:
    nested = data.get(outer_key)
    return _usage_int(nested, inner_key) if isinstance(nested, dict) else None


@dataclass(slots=True)
class LLMUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    prompt_cache_hit_tokens: int = 0
    prompt_cache_miss_tokens: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "LLMUsage":
        if not data:
            return cls()
        prompt = _first_usage_int(data, "prompt_tokens", "input_tokens") or 0
        completion = _first_usage_int(data, "completion_tokens", "output_tokens") or 0
        total = _usage_int(data, "total_tokens")
        hit = _usage_int(data, "prompt_cache_hit_tokens")
        if hit is None:
            hit = (
                _nested_usage_int(data, "prompt_tokens_details", "cached_tokens")
                or _nested_usage_int(data, "input_tokens_details", "cached_tokens") or 0
            )
        miss = _usage_int(data, "prompt_cache_miss_tokens")
        return cls(prompt, completion, prompt + completion if total is None else total,
                   hit, max(prompt - hit, 0) if miss is None else miss)

    def to_dict(self) -> dict[str, int | float | None]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "prompt_cache_hit_tokens": self.prompt_cache_hit_tokens,
            "prompt_cache_miss_tokens": self.prompt_cache_miss_tokens,
            "prompt_cache_hit_rate_percent": self.prompt_cache_hit_rate_percent(),
        }

    def prompt_cache_hit_rate_percent(self) -> float | None:
        if self.prompt_tokens <= 0:
            return None
        return round(self.prompt_cache_hit_tokens / self.prompt_tokens * 100, 2)
