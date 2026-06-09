from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
import os
import pathlib
import random
import re
import time
import warnings
from collections import deque
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type
from urllib.parse import urlparse

import httpx
import nest_asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel
from tqdm import tqdm

from model.model import LLM

nest_asyncio.apply()


_VISION_MAX_IMAGE_BYTES_WARNING = 10 * 1024 * 1024
_VISION_MAX_IMAGE_BYTES_ERROR = 50 * 1024 * 1024
_VISION_MAX_IMAGES_PER_CALL = 10
_VISION_MIME_WHITELIST = {"image/png", "image/jpeg", "image/webp", "image/gif"}
_VISION_EXT_WHITELIST = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
_QWEN_REASONING_MAX_TOKENS = 16_384
_QWEN_NON_REASONING_MAX_TOKENS = 8_192
_TRUE_STR = {"1", "true", "t", "yes", "y", "on", "enabled"}
_FALSE_STR = {"0", "false", "f", "no", "n", "off", "disabled", "disable"}
_QWEN_MODE_ALIASES = {
    "thinking": "thinking",
    "thinking_general": "thinking",
    "thinking_coding": "thinking_precise_coding",
    "thinking_precise_coding": "thinking_precise_coding",
    "reasoning": "non_thinking_reasoning",
    "instruct": "non_thinking_reasoning",
    "instruct_general": "non_thinking",
    "instruct_reasoning": "non_thinking_reasoning",
    "nothinking": "non_thinking",
    "nonthinking": "non_thinking",
    "non_thinking": "non_thinking",
    "off": "non_thinking",
    "false": "non_thinking",
    "0": "non_thinking",
    "on": "thinking",
    "true": "thinking",
    "1": "thinking",
}
_QWEN_HYBRID_MODEL_IDS = {
    "Qwen/Qwen3.5-397B-A17B",
    "Qwen/Qwen3.5-9B",
    "Qwen/Qwen3.6-Plus",
}
_QWEN_NON_THINKING_ONLY_MODEL_IDS = {
    "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
    "Qwen/Qwen3-235B-A22B-Instruct-2507-tput",
    "Qwen/Qwen2.5-7B-Instruct-Turbo",
}
# Sampling defaults below are taken from primary model docs where available:
# Hugging Face model cards / generation_config for open-weight Qwen models and
# Together or provider docs for hosted/closed models. Qwen3.6-Plus reuses the
# published Qwen3.6-35B-A3B recommendations as the closest public sibling.
_SAMPLING_PROFILES = {
    "qwen35_397_thinking": {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "repetition_penalty": 1.0,
    },
    "qwen35_397_non_thinking": {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "repetition_penalty": 1.0,
    },
    "qwen35_9b_thinking": {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "repetition_penalty": 1.0,
    },
    "qwen35_9b_thinking_precise_coding": {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "repetition_penalty": 1.0,
    },
    "qwen35_9b_non_thinking": {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "repetition_penalty": 1.0,
    },
    "qwen35_9b_non_thinking_reasoning": {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 40,
        "min_p": 0.0,
        "presence_penalty": 2.0,
        "repetition_penalty": 1.0,
    },
    "qwen36_thinking": {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "repetition_penalty": 1.0,
    },
    "qwen36_thinking_precise_coding": {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "repetition_penalty": 1.0,
    },
    "qwen36_non_thinking": {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "repetition_penalty": 1.0,
    },
    "qwen3_coder_non_thinking": {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "repetition_penalty": 1.05,
    },
    "qwen3_235b_instruct_non_thinking": {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
    },
    "qwen25_7b_instruct_non_thinking": {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "repetition_penalty": 1.05,
    },
    "kimi_k25_thinking": {
        "temperature": 1.0,
        "top_p": 0.95,
    },
    "kimi_k25_non_thinking": {
        "temperature": 0.6,
        "top_p": 0.95,
    },
    "glm5_default": {
        "temperature": 1.0,
        "top_p": 0.95,
    },
    "deepseek_v4_pro_default": {
        "temperature": 1.0,
        "top_p": 1.0,
    },
    "minimax_m27_default": {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 40,
    },
}
_QWEN_MODEL_PROFILE_KEYS = {
    "Qwen/Qwen3.5-397B-A17B": {
        "thinking": "qwen35_397_thinking",
        "thinking_precise_coding": "qwen35_397_thinking",
        "non_thinking": "qwen35_397_non_thinking",
        "non_thinking_reasoning": "qwen35_397_non_thinking",
    },
    "Qwen/Qwen3.5-9B": {
        "thinking": "qwen35_9b_thinking",
        "thinking_precise_coding": "qwen35_9b_thinking_precise_coding",
        "non_thinking": "qwen35_9b_non_thinking",
        "non_thinking_reasoning": "qwen35_9b_non_thinking_reasoning",
    },
    "Qwen/Qwen3.6-Plus": {
        "thinking": "qwen36_thinking",
        "thinking_precise_coding": "qwen36_thinking_precise_coding",
        "non_thinking": "qwen36_non_thinking",
        "non_thinking_reasoning": "qwen36_non_thinking",
    },
    "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8": {
        "non_thinking": "qwen3_coder_non_thinking",
        "non_thinking_reasoning": "qwen3_coder_non_thinking",
    },
    "Qwen/Qwen3-235B-A22B-Instruct-2507-tput": {
        "non_thinking": "qwen3_235b_instruct_non_thinking",
        "non_thinking_reasoning": "qwen3_235b_instruct_non_thinking",
    },
    "Qwen/Qwen2.5-7B-Instruct-Turbo": {
        "non_thinking": "qwen25_7b_instruct_non_thinking",
        "non_thinking_reasoning": "qwen25_7b_instruct_non_thinking",
    },
}
_MODEL_DEFAULT_PROFILE_KEYS = {
    "moonshotai/Kimi-K2.5": {
        "thinking": "kimi_k25_thinking",
        "non_thinking": "kimi_k25_non_thinking",
    },
    "moonshotai/Kimi-K2.6": {
        "thinking": "kimi_k25_thinking",
        "non_thinking": "kimi_k25_non_thinking",
    },
    "zai-org/GLM-5": {"default": "glm5_default"},
    "zai-org/GLM-5.1": {"default": "glm5_default"},
    "deepseek-ai/DeepSeek-V4-Pro": {"default": "deepseek_v4_pro_default"},
    "MiniMaxAI/MiniMax-M2.7": {"default": "minimax_m27_default"},
}


def _warn_once(message: str) -> None:
    warnings.warn(f"[TogetherVision] {message}")


def _warn_model_once(message: str) -> None:
    warnings.warn(f"[TogetherModel] {message}")


def _obj_get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _model_dump(obj: Any) -> Any:
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, (dict, list, str, int, float, bool)):
        return obj
    return str(obj)


def _b64_image_url(src: str) -> str:
    src = str(src or "").strip()
    if not src:
        raise ValueError("Empty image source.")

    if src.startswith("data:"):
        if ";base64," not in src.lower():
            _warn_once(
                "Data URI is not explicitly base64-encoded; expected pattern data:image/<mime>;base64,<payload>. "
                "Proceeding but the request may fail if payload is not base64."
            )
        try:
            header, payload = src.split(",", 1)
        except ValueError as exc:
            raise ValueError("Invalid image data URI: missing comma separator.") from exc
        if not payload:
            raise ValueError("Invalid image data URI: empty payload.")

        mime = (header[5:].split(";", 1)[0] or "").strip().lower()
        if mime and mime != "application/octet-stream" and mime not in _VISION_MIME_WHITELIST:
            _warn_once(
                f"Image MIME '{mime}' may be unsupported; prefer png/jpeg/webp/gif in data URI."
            )

        try:
            raw = base64.b64decode(payload, validate=True)
        except Exception as exc:
            raw = base64.b64decode(payload, validate=False)
            if not raw:
                raise ValueError(f"Invalid image data URI base64 payload: {exc}") from exc

        if not raw:
            raise ValueError("Invalid image data URI base64 payload is empty.")
        if len(raw) > _VISION_MAX_IMAGE_BYTES_ERROR:
            raise ValueError(
                f"Image data URL payload is too large ({len(raw)} bytes). "
                f"Maximum is {_VISION_MAX_IMAGE_BYTES_ERROR} bytes (~50MB)."
            )
        if len(raw) > _VISION_MAX_IMAGE_BYTES_WARNING:
            _warn_once(
                "Image data URL payload >10MB. Together fine-tuning docs use 10MB/image; large inputs may be rejected."
            )
        return src

    if src.startswith(("http://", "https://")):
        parsed = urlparse(src)
        if not parsed.netloc:
            raise ValueError(f"Invalid image URL (missing host): {src}")
        ext = pathlib.Path(parsed.path).suffix.lower()
        if ext and ext not in _VISION_EXT_WHITELIST:
            _warn_once(
                f"Image URL extension '{ext}' is not in the typical png/jpeg/webp/gif set; "
                "confirm the URL serves a valid image format supported by Together."
            )
        if parsed.scheme.lower() not in {"http", "https"}:
            raise ValueError(f"Unsupported image URL scheme '{parsed.scheme}'. Use http/https/data:")
        return src

    path = pathlib.Path(src).expanduser()
    if not path.is_absolute():
        path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Image path is not a file: {path}")

    file_size = path.stat().st_size
    if file_size <= 0:
        raise ValueError(f"Image file is empty: {path}")
    if file_size > _VISION_MAX_IMAGE_BYTES_ERROR:
        raise ValueError(
            f"Image file is too large ({file_size} bytes): {path}. "
            f"Maximum is {_VISION_MAX_IMAGE_BYTES_ERROR} bytes (~50MB)."
        )
    if file_size > _VISION_MAX_IMAGE_BYTES_WARNING:
        _warn_once(
            f"Image file {path} is >10MB ({file_size} bytes). Consider downscaling or compressing to reduce parsing risk."
        )

    mime = mimetypes.guess_type(path)[0] or "application/octet-stream"
    if mime not in _VISION_MIME_WHITELIST:
        _warn_once(
            f"Image MIME '{mime}' from {path} is not a common supported vision format (png/jpeg/webp/gif)."
        )
    with path.open("rb") as handle:
        b64 = base64.b64encode(handle.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _json_schema_name(schema: Type[BaseModel]) -> str:
    raw = getattr(schema, "__name__", "response") or "response"
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", raw).strip("_").lower()
    return name or "response"


def _strip_json_noise(text: str) -> str:
    s = str(text or "").strip()
    if not s:
        return s
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.IGNORECASE | re.DOTALL).strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE).strip()
        s = re.sub(r"\s*```$", "", s).strip()
    return s


def _loads_json_lenient(text: str) -> Any:
    s = _strip_json_noise(text)
    if not s:
        return {}
    try:
        return json.loads(s)
    except Exception:
        pass
    start = s.find("{")
    end = s.rfind("}")
    if 0 <= start < end:
        try:
            return json.loads(s[start : end + 1])
        except Exception:
            pass
    return text or {}


class TogetherModel(LLM):
    """Together AI chat/vision backend via the official OpenAI-compatible API."""

    DEFAULT_BASE_URL = "https://api.together.ai/v1"
    DEFAULT_MAX_TOKENS = 2048
    DEFAULT_MAX_CONCURRENCY = 16
    DEFAULT_MAX_RETRIES = 5
    DEFAULT_INVALID_RESPONSE_RETRIES = 1
    DEFAULT_RPM_LIMIT = 30_000
    DEFAULT_TPM_LIMIT = 150_000_000

    # Official serverless model table fallback, checked 2026-05-03:
    # https://docs.together.ai/docs/serverless-models
    # Dynamic pricing from GET /models is preferred whenever available.
    MODEL_PRICING_USD_PER_1M = {
        "MiniMaxAI/MiniMax-M2.7": {"input": 0.30, "cached_input": 0.06, "output": 1.20},
        "Qwen/Qwen3.5-397B-A17B": {"input": 0.60, "cached_input": None, "output": 3.60},
        "Qwen/Qwen3.6-Plus": {"input": 0.50, "cached_input": None, "output": 3.00},
        "Qwen/Qwen3.5-9B": {"input": 0.10, "cached_input": None, "output": 0.15},
        "moonshotai/Kimi-K2.6": {"input": 1.20, "cached_input": 0.20, "output": 4.50},
        "moonshotai/Kimi-K2.5": {"input": 0.50, "cached_input": None, "output": 2.80},
        "zai-org/GLM-5.1": {"input": 1.40, "cached_input": None, "output": 4.40},
        "zai-org/GLM-5": {"input": 1.00, "cached_input": None, "output": 3.20},
        "openai/gpt-oss-120b": {"input": 0.15, "cached_input": None, "output": 0.60},
        "openai/gpt-oss-20b": {"input": 0.05, "cached_input": None, "output": 0.20},
        "deepseek-ai/DeepSeek-V4-Pro": {"input": 2.10, "cached_input": 0.20, "output": 4.40},
        "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8": {
            "input": 2.00,
            "cached_input": None,
            "output": 2.00,
        },
        "Qwen/Qwen3-235B-A22B-Instruct-2507-tput": {
            "input": 0.20,
            "cached_input": None,
            "output": 0.60,
        },
        "meta-llama/Llama-3.3-70B-Instruct-Turbo": {
            "input": 0.88,
            "cached_input": None,
            "output": 0.88,
        },
        "essentialai/rnj-1-instruct": {"input": 0.15, "cached_input": None, "output": 0.15},
        "Qwen/Qwen2.5-7B-Instruct-Turbo": {
            "input": 0.30,
            "cached_input": None,
            "output": 0.30,
        },
        "google/gemma-4-31B-it": {"input": 0.20, "cached_input": None, "output": 0.50},
        "google/gemma-3n-E4B-it": {"input": 0.06, "cached_input": None, "output": 0.12},
        "LiquidAI/LFM2-24B-A2B": {"input": 0.03, "cached_input": None, "output": 0.12},
        "meta-llama/Meta-Llama-3-8B-Instruct-Lite": {
            "input": 0.10,
            "cached_input": None,
            "output": 0.10,
        },
        "deepcogito/cogito-v2-1-671b": {"input": 1.25, "cached_input": None, "output": 1.25},
    }

    _OPENAI_PARAM_KEYS = {
        "frequency_penalty",
        "function_call",
        "functions",
        "logit_bias",
        "logprobs",
        "max_completion_tokens",
        "max_tokens",
        "n",
        "parallel_tool_calls",
        "presence_penalty",
        "reasoning_effort",
        "response_format",
        "seed",
        "stop",
        "temperature",
        "tool_choice",
        "tools",
        "top_logprobs",
        "top_p",
        "user",
    }
    _TOGETHER_EXTRA_KEYS = {
        "chat_template_kwargs",
        "min_p",
        "reasoning",
        "repetition_penalty",
        "safety_model",
        "top_k",
    }
    _ADVANCED_FALLBACK_KEYS = {
        "chat_template_kwargs",
        "min_p",
        "reasoning",
        "reasoning_effort",
        "repetition_penalty",
        "response_format",
        "safety_model",
        "tool_choice",
        "tools",
        "top_k",
        "top_logprobs",
    }

    def __init__(
        self,
        model_name: str,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_concurrency: Optional[int] = None,
        rpm_limit: Optional[int] = None,
        tpm_limit: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        min_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        stop: Optional[str | Sequence[str]] = None,
        seed: Optional[int] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
        reasoning: Optional[dict | bool] = None,
        reasoning_effort: Optional[str] = None,
        qwen_mode: Optional[str] = None,
        chat_template_kwargs: Optional[dict] = None,
        tools: Optional[list[dict[str, Any]]] = None,
        tool_choice: Optional[str | dict[str, Any]] = None,
        response_format: Optional[dict[str, Any]] = None,
        safety_model: Optional[str] = None,
        extra_body: Optional[dict[str, Any]] = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        max_invalid_response_retries: int = DEFAULT_INVALID_RESPONSE_RETRIES,
        request_timeout_s: Optional[float] = 300.0,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        fetch_pricing: bool = True,
        print_cost: bool = True,
    ) -> None:
        api_model_name = self._normalize_model_name(model_name)
        init_temperature = 0.0 if temperature is None else float(temperature)
        super().__init__(model_name=f"Together/{api_model_name}", temperature=init_temperature)
        self.model_name = api_model_name
        self.model_type = "Together"
        self.model_name_full = f"Together/{api_model_name}"
        load_dotenv()

        self.api_key = api_key or os.getenv("TOGETHER_API_KEY")
        if not self.api_key:
            raise RuntimeError("TOGETHER_API_KEY is not set.")

        self.base_url = (base_url or os.getenv("TOGETHER_BASE_URL") or self.DEFAULT_BASE_URL).rstrip("/")
        self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

        self.temperature = init_temperature
        self.qwen_mode_requested = qwen_mode
        self.qwen_mode = self._normalize_qwen_mode(qwen_mode)
        self._user_option_keys = {
            key
            for key, value in {
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
                "top_k": top_k,
                "min_p": min_p,
                "repetition_penalty": repetition_penalty,
                "presence_penalty": presence_penalty,
                "frequency_penalty": frequency_penalty,
                "reasoning": reasoning,
                "reasoning_effort": reasoning_effort,
                "chat_template_kwargs": chat_template_kwargs,
            }.items()
            if value is not None
        }
        self._user_max_tokens = max_tokens is not None
        normalized_reasoning = self._normalize_reasoning(reasoning)
        normalized_chat_template_kwargs = (
            dict(chat_template_kwargs) if isinstance(chat_template_kwargs, dict) else chat_template_kwargs
        )
        normalized_reasoning, normalized_chat_template_kwargs = self._apply_qwen_mode_overrides(
            normalized_reasoning,
            normalized_chat_template_kwargs,
        )
        if self._user_max_tokens:
            self.max_tokens = int(max_tokens)
        elif self._is_qwen_model(self.model_name):
            self.max_tokens = self._resolve_qwen_max_tokens_from_config(
                normalized_reasoning, normalized_chat_template_kwargs
            )
        else:
            self.max_tokens = int(self.DEFAULT_MAX_TOKENS)
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer.")

        self.max_concurrency = (
            int(max_concurrency) if max_concurrency is not None else self.DEFAULT_MAX_CONCURRENCY
        )
        if self.max_concurrency <= 0:
            raise ValueError("max_concurrency must be a positive integer.")
        self.rpm_limit = int(rpm_limit) if rpm_limit is not None else self.DEFAULT_RPM_LIMIT
        self.tpm_limit = int(tpm_limit) if tpm_limit is not None else self.DEFAULT_TPM_LIMIT
        self.max_retries = max(1, int(max_retries))
        self.max_invalid_response_retries = max(0, int(max_invalid_response_retries))
        self.request_timeout_s = (
            float(request_timeout_s) if request_timeout_s is not None else None
        )
        if self.request_timeout_s is not None and self.request_timeout_s <= 0:
            raise ValueError("request_timeout_s must be positive when provided.")

        self.default_options: Dict[str, Any] = {}
        for key, value in {
            "temperature": temperature,
            "max_tokens": self.max_tokens,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "repetition_penalty": repetition_penalty,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "stop": stop,
            "seed": seed,
            "logprobs": logprobs,
            "top_logprobs": top_logprobs,
            "reasoning": normalized_reasoning,
            "reasoning_effort": reasoning_effort,
            "chat_template_kwargs": normalized_chat_template_kwargs,
            "tools": tools,
            "tool_choice": tool_choice,
            "response_format": response_format,
            "safety_model": safety_model,
        }.items():
            if value is not None:
                self.default_options[key] = value
        self.default_extra_body = dict(extra_body or {})
        self.print_cost = bool(print_cost)
        self.fetch_pricing = bool(fetch_pricing)

        self._req_times, self._token_times = deque(), deque()
        self._stats_lock = asyncio.Lock()
        self._pricing_lock = asyncio.Lock()
        self._pricing_remote_loaded = False
        self._pricing_warned = False
        self._pricing_by_model: Dict[str, dict[str, Any]] = {
            key: {**value, "source": "https://docs.together.ai/docs/serverless-models"}
            for key, value in self.MODEL_PRICING_USD_PER_1M.items()
        }

        self.tokens_used = 0
        self.cached_input_tokens_used = 0
        self.reasoning_output_tokens_used = 0
        self.requests_made = 0
        self.total_cost_usd = 0.0
        self.costed_requests = 0
        self.uncosted_requests = 0
        self.last_cost_info: Optional[dict[str, Any]] = None

        try:
            import tiktoken

            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self.tokenizer = None

        print(
            f"{self.model_name} | max_out={self.max_tokens} max_conc={self.max_concurrency} "
            f"rpm={self.rpm_limit} tpm={self.tpm_limit} base_url={self.base_url} "
            f"fetch_pricing={self.fetch_pricing}"
        )

    @staticmethod
    def _normalize_model_name(model_name: str) -> str:
        raw = str(model_name or "").strip()
        if raw.lower().startswith("together/"):
            return raw.split("/", 1)[1]
        if raw.lower().startswith("together:"):
            return raw.split(":", 1)[1]
        return raw

    @staticmethod
    def _coerce_bool_like(value: Any) -> Optional[bool]:
        if isinstance(value, bool):
            return bool(value)
        if value is None:
            return None
        if isinstance(value, (int, float)):
            if value == 1:
                return True
            if value == 0:
                return False
            return None
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in _TRUE_STR:
                return True
            if normalized in _FALSE_STR:
                return False
            return None
        return None

    @staticmethod
    def _normalize_reasoning(reasoning: Optional[dict | bool]) -> Optional[dict]:
        if reasoning is None:
            return None
        if isinstance(reasoning, bool):
            return {"enabled": bool(reasoning)}
        if isinstance(reasoning, dict):
            output = dict(reasoning)
            if "enabled" in output:
                parsed_enabled = TogetherModel._coerce_bool_like(output.get("enabled"))
                if parsed_enabled is not None:
                    output["enabled"] = parsed_enabled
            return output
        raise ValueError("reasoning must be a dict, bool, or None.")

    @staticmethod
    def _normalize_qwen_mode(mode: Optional[str]) -> Optional[str]:
        if mode is None:
            return None
        normalized = str(mode).strip().lower()
        if not normalized:
            return None
        if normalized in _QWEN_MODE_ALIASES:
            return _QWEN_MODE_ALIASES[normalized]
        if normalized in {"thinking", "thinking_precise_coding", "non_thinking", "non_thinking_reasoning"}:
            return normalized
        _warn_model_once(
            f"Unsupported qwen_mode={mode!r}; ignoring it for Together defaults."
        )
        return None

    @staticmethod
    def _as_int(value: Any) -> int:
        try:
            return int(value or 0)
        except Exception:
            return 0

    @staticmethod
    def _as_float_or_none(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except Exception:
            return None

    @staticmethod
    def _cost_component(tokens: int, rate_per_1m: Optional[float]) -> Optional[float]:
        if tokens <= 0:
            return 0.0
        if rate_per_1m is None:
            return None
        return (tokens / 1_000_000.0) * float(rate_per_1m)

    def _schema_instruction(self, schema: Type[BaseModel]) -> str:
        schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False, sort_keys=True)
        return (
            "Return only valid JSON. The JSON must conform to this JSON Schema: "
            f"{schema_json}"
        )

    @staticmethod
    def _is_qwen_model(model_name: str) -> bool:
        return str(model_name or "").lower().startswith("qwen/")

    @staticmethod
    def _is_qwen_hybrid_model(model_name: str) -> bool:
        return str(model_name or "") in _QWEN_HYBRID_MODEL_IDS

    @staticmethod
    def _is_qwen_non_thinking_only_model(model_name: str) -> bool:
        return str(model_name or "") in _QWEN_NON_THINKING_ONLY_MODEL_IDS

    def _apply_qwen_mode_overrides(
        self,
        reasoning: Optional[dict],
        chat_template_kwargs: Optional[dict],
    ) -> Tuple[Optional[dict], Optional[dict]]:
        if not self._is_qwen_model(self.model_name):
            return reasoning, chat_template_kwargs

        out_reasoning = dict(reasoning) if isinstance(reasoning, dict) else reasoning
        out_chat_kwargs = (
            dict(chat_template_kwargs) if isinstance(chat_template_kwargs, dict) else None
        )
        if self.qwen_mode is None:
            return out_reasoning, out_chat_kwargs

        if self._is_qwen_non_thinking_only_model(self.model_name):
            if self.qwen_mode.startswith("thinking"):
                _warn_model_once(
                    f"{self.model_name} is a non-thinking-only Qwen model; "
                    f"qwen_mode={self.qwen_mode_requested!r} will fall back to non-thinking defaults."
                )
            return out_reasoning, out_chat_kwargs

        enabled = self.qwen_mode.startswith("thinking")
        if out_reasoning is None:
            out_reasoning = {"enabled": enabled}
        else:
            out_reasoning["enabled"] = enabled
        if out_chat_kwargs is None:
            out_chat_kwargs = {}
        out_chat_kwargs["enable_thinking"] = enabled
        return out_reasoning, out_chat_kwargs

    def _is_qwen_reasoning_enabled(self, options: Dict[str, Any]) -> bool:
        if self._is_qwen_non_thinking_only_model(self.model_name):
            reasoning = options.get("reasoning")
            chat_template_kwargs = options.get("chat_template_kwargs")
            explicit_thinking = None
            if isinstance(reasoning, dict):
                explicit_thinking = self._coerce_bool_like(reasoning.get("enabled"))
            elif reasoning is not None:
                explicit_thinking = self._coerce_bool_like(reasoning)
            if explicit_thinking is None and isinstance(chat_template_kwargs, dict):
                for key in ("thinking", "enable_thinking"):
                    explicit_thinking = self._coerce_bool_like(chat_template_kwargs.get(key))
                    if explicit_thinking is not None:
                        break
            if explicit_thinking:
                _warn_model_once(
                    f"{self.model_name} does not support thinking mode; "
                    "reasoning-related flags may be ignored or rejected by the backend."
                )
            return False

        reasoning = options.get("reasoning")
        parsed_reasoning = self._coerce_bool_like(reasoning)
        if parsed_reasoning is not None:
            return parsed_reasoning
        if isinstance(reasoning, dict) and isinstance(reasoning.get("enabled"), bool):
            return bool(reasoning.get("enabled"))
        if isinstance(reasoning, dict):
            parsed_enabled = self._coerce_bool_like(reasoning.get("enabled"))
            if parsed_enabled is not None:
                return parsed_enabled

        if self.qwen_mode is not None:
            return self.qwen_mode.startswith("thinking")

        chat_template_kwargs = options.get("chat_template_kwargs")
        if isinstance(chat_template_kwargs, dict):
            for key in ("thinking", "enable_thinking"):
                parsed = self._coerce_bool_like(chat_template_kwargs.get(key))
                if parsed is not None:
                    return parsed
        return True

    def _select_qwen_profile_name(self, options: Dict[str, Any]) -> Optional[str]:
        profile_map = _QWEN_MODEL_PROFILE_KEYS.get(self.model_name)
        if not profile_map:
            return None
        if self._is_qwen_non_thinking_only_model(self.model_name):
            return profile_map.get("non_thinking")

        canonical_mode = self.qwen_mode
        if canonical_mode is None:
            canonical_mode = "thinking" if self._is_qwen_reasoning_enabled(options) else "non_thinking"
        if canonical_mode not in profile_map:
            canonical_mode = "non_thinking" if canonical_mode.startswith("non_") else "thinking"
        return profile_map.get(canonical_mode)

    def _select_default_profile_name(self, options: Dict[str, Any]) -> Optional[str]:
        if self._is_qwen_model(self.model_name):
            return self._select_qwen_profile_name(options)

        profile_map = _MODEL_DEFAULT_PROFILE_KEYS.get(self.model_name)
        if not profile_map:
            return None
        if self.model_name in {"moonshotai/Kimi-K2.5", "moonshotai/Kimi-K2.6"}:
            mode = "thinking" if bool(_obj_get(options.get("reasoning"), "enabled", True)) else "non_thinking"
            parsed = self._coerce_bool_like(options.get("reasoning"))
            if parsed is not None:
                mode = "thinking" if parsed else "non_thinking"
            return profile_map.get(mode)
        return profile_map.get("default")

    def _recommended_sampling_defaults(self, options: Dict[str, Any]) -> Dict[str, Any]:
        profile_name = self._select_default_profile_name(options)
        if not profile_name:
            return {}
        return dict(_SAMPLING_PROFILES.get(profile_name) or {})

    def _apply_recommended_defaults(
        self,
        merged: Dict[str, Any],
        call_options: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        defaults = self._recommended_sampling_defaults(merged)
        if not defaults:
            return merged

        blocked_keys = set(self._user_option_keys)
        if call_options:
            blocked_keys.update(call_options.keys())
        for key, value in defaults.items():
            if key in blocked_keys:
                continue
            merged.setdefault(key, value)
        return merged

    def _resolve_qwen_max_tokens_from_config(
        self,
        reasoning: Optional[dict | bool],
        chat_template_kwargs: Optional[dict],
    ) -> int:
        if self._is_qwen_reasoning_enabled(
            {"reasoning": reasoning, "chat_template_kwargs": chat_template_kwargs}
        ):
            return _QWEN_REASONING_MAX_TOKENS
        return _QWEN_NON_REASONING_MAX_TOKENS

    def _resolve_qwen_request_max_tokens(self, options: Dict[str, Any]) -> int:
        if self._is_qwen_reasoning_enabled(options):
            return _QWEN_REASONING_MAX_TOKENS
        return _QWEN_NON_REASONING_MAX_TOKENS

    def _build_messages(
        self,
        system_msg: str,
        user_msg: str,
        images: Optional[List[str]],
        schema: Optional[Type[BaseModel]],
    ) -> List[Dict[str, Any]]:
        system_text = str(system_msg or "")
        if schema is not None:
            schema_instruction = self._schema_instruction(schema)
            system_text = (
                f"{system_text}\n\n{schema_instruction}" if system_text else schema_instruction
            )

        messages: List[Dict[str, Any]] = []
        if system_text:
            messages.append({"role": "system", "content": system_text})

        user_text = str(user_msg or "")
        if images:
            if len(images) > _VISION_MAX_IMAGES_PER_CALL:
                _warn_once(
                    f"Request has {len(images)} images. For robustness, split into smaller batches if needed. "
                    "Models may have implementation-specific image count/size constraints."
                )
            content: List[Dict[str, Any]] = []
            if user_text:
                content.append({"type": "text", "text": user_text})
            for image in images:
                if image is None or str(image).strip() == "":
                    raise ValueError("Vision request contains an empty image source.")
                content.append({"type": "image_url", "image_url": {"url": _b64_image_url(image)}})
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": user_text})
        return messages

    def _schema_response_format(self, schema: Type[BaseModel]) -> Dict[str, Any]:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": _json_schema_name(schema),
                "schema": schema.model_json_schema(),
            },
        }

    def _split_options(self, options: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        openai_params: Dict[str, Any] = {}
        extra_body = dict(self.default_extra_body)
        for key, value in options.items():
            if value is None:
                continue
            if key == "extra_body":
                if not isinstance(value, dict):
                    raise ValueError("extra_body must be a dict when provided.")
                extra_body.update(value)
            elif key in self._OPENAI_PARAM_KEYS:
                openai_params[key] = value
            elif key in self._TOGETHER_EXTRA_KEYS:
                extra_body[key] = value
            else:
                extra_body[key] = value
        if extra_body:
            openai_params["extra_body"] = extra_body
        return openai_params, extra_body

    def _build_request(
        self,
        messages: List[Dict[str, Any]],
        schema: Optional[Type[BaseModel]],
        call_options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        merged = dict(self.default_options)
        if call_options:
            merged.update({k: v for k, v in call_options.items() if v is not None})
        merged = self._apply_recommended_defaults(merged, call_options)
        if schema is not None and "response_format" not in merged:
            merged["response_format"] = self._schema_response_format(schema)
        if (
            self._is_qwen_model(self.model_name)
            and (not self._user_max_tokens)
            and ("max_tokens" not in call_options if call_options else True)
        ):
            merged["max_tokens"] = self._resolve_qwen_request_max_tokens(merged)

        openai_params, _extra = self._split_options(merged)
        req = {"model": self.model_name, "messages": messages}
        req.update(openai_params)
        return req

    async def _await_with_timeout(self, coro: Any) -> Any:
        if self.request_timeout_s is None:
            return await coro
        return await asyncio.wait_for(coro, timeout=float(self.request_timeout_s))

    @staticmethod
    def _status_code(exc: BaseException) -> Optional[int]:
        return getattr(exc, "status_code", None) or getattr(
            getattr(exc, "response", None), "status_code", None
        )

    @classmethod
    def _is_param_error(cls, exc: BaseException) -> bool:
        status = cls._status_code(exc)
        if status not in {400, 422}:
            return False
        text = str(exc).lower()
        needles = (
            "additional properties",
            "extra inputs",
            "invalid parameter",
            "not supported",
            "repetition_penalty",
            "response_format",
            "reasoning",
            "safety_model",
            "tool_choice",
            "tools",
            "top_k",
            "unexpected",
            "unknown parameter",
            "unsupported",
            "unrecognized",
        )
        return any(needle in text for needle in needles)

    @classmethod
    def _is_image_api_error(cls, exc: BaseException) -> bool:
        status = cls._status_code(exc)
        if status not in {400, 413, 415, 422}:
            return False
        text = str(exc).lower()
        needles = (
            "image",
            "vision",
            "image_url",
            "image input",
            "could not be processed",
            "does not support",
            "media type",
            "invalid image",
            "invalid media",
            "payload",
            "max image",
            "too large",
            "base64",
            "unsupported",
            "unrecognized image",
        )
        return any(needle in text for needle in needles)

    def _downgrade_request(self, req: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        out = dict(req)
        removed: List[str] = []
        extra_body = dict(out.get("extra_body") or {})

        for key in list(self._ADVANCED_FALLBACK_KEYS):
            if key in out:
                out.pop(key, None)
                removed.append(key)
            if key in extra_body:
                extra_body.pop(key, None)
                removed.append(key)

        if extra_body:
            out["extra_body"] = extra_body
        else:
            out.pop("extra_body", None)
        return out, sorted(set(removed))

    async def _chat_call_with_retries(self, req: Dict[str, Any]) -> Any:
        last_err: Optional[BaseException] = None
        call_req = dict(req)
        downgraded = False

        for attempt in range(self.max_retries):
            try:
                return await self._await_with_timeout(
                    self.client.chat.completions.create(**call_req)
                )
            except Exception as exc:
                last_err = exc
                if self._is_image_api_error(exc):
                    _warn_once(
                        f"Together rejected vision input for model={self.model_name}; "
                        "please validate image source/type/size before retrying or switch to a vision-capable model."
                    )
                    raise
                if (not downgraded) and self._is_param_error(exc):
                    fallback_req, removed = self._downgrade_request(call_req)
                    if removed:
                        call_req = fallback_req
                        downgraded = True
                        print(
                            "WARNING: Together request failed with advanced parameters; "
                            f"retrying without {', '.join(removed)}. "
                            f"error={type(exc).__name__}: {exc}"
                        )
                        await asyncio.sleep(0.2)
                        continue
                if attempt < self.max_retries - 1:
                    backoff = min(2**attempt, 10) + random.random()
                    await asyncio.sleep(backoff)
                    continue
                raise

        raise last_err if last_err is not None else RuntimeError("Together request failed.")

    def _extract_usage_tokens(self, usage: Any) -> Tuple[int, int, int, int]:
        in_tok = _obj_get(usage, "prompt_tokens", None)
        if in_tok is None:
            in_tok = _obj_get(usage, "input_tokens", 0)

        out_tok = _obj_get(usage, "completion_tokens", None)
        if out_tok is None:
            out_tok = _obj_get(usage, "output_tokens", 0)

        prompt_details = _obj_get(usage, "prompt_tokens_details", None)
        input_details = _obj_get(usage, "input_tokens_details", None)
        cached_tok = _obj_get(prompt_details, "cached_tokens", None)
        if cached_tok is None:
            cached_tok = _obj_get(input_details, "cached_tokens", None)
        if cached_tok is None:
            cached_tok = _obj_get(usage, "cached_tokens", 0)

        completion_details = _obj_get(usage, "completion_tokens_details", None)
        output_details = _obj_get(usage, "output_tokens_details", None)
        reasoning_tok = _obj_get(completion_details, "reasoning_tokens", None)
        if reasoning_tok is None:
            reasoning_tok = _obj_get(output_details, "reasoning_tokens", None)
        if reasoning_tok is None:
            reasoning_tok = _obj_get(usage, "reasoning_tokens", 0)

        in_i = max(0, self._as_int(in_tok))
        out_i = max(0, self._as_int(out_tok))
        cached_i = min(max(0, self._as_int(cached_tok)), in_i)
        reasoning_i = min(max(0, self._as_int(reasoning_tok)), out_i)
        return in_i, out_i, cached_i, reasoning_i

    async def _throttle(self, tokens_est: int) -> List[Any]:
        async with self._stats_lock:
            now = time.monotonic()
            while self._req_times and now - self._req_times[0] > 60:
                self._req_times.popleft()
            while self._token_times and now - self._token_times[0][0] > 60:
                self._token_times.popleft()

            rpm_wait = 0.0
            if self.rpm_limit and len(self._req_times) >= self.rpm_limit:
                rpm_wait = (self._req_times[0] + 60) - now

            tpm_wait = 0.0
            if self.tpm_limit:
                used = sum(int(t) for _, t in self._token_times)
                need = used + int(tokens_est) - int(self.tpm_limit)
                if need > 0 and self._token_times:
                    freed = 0
                    for ts, tk in self._token_times:
                        freed += int(tk)
                        if freed >= need:
                            tpm_wait = (ts + 60) - now
                            break

        wait_s = max(rpm_wait, tpm_wait, 0.0)
        if wait_s > 0:
            await asyncio.sleep(wait_s)
            return await self._throttle(tokens_est)

        async with self._stats_lock:
            now = time.monotonic()
            self._req_times.append(now)
            entry = [now, int(tokens_est)]
            self._token_times.append(entry)
            return entry

    async def _ensure_remote_pricing_loaded(self) -> None:
        if not self.fetch_pricing or self._pricing_remote_loaded:
            return
        async with self._pricing_lock:
            if self._pricing_remote_loaded:
                return
            try:
                async with httpx.AsyncClient(timeout=20.0) as client:
                    resp = await client.get(
                        f"{self.base_url}/models",
                        headers={"Authorization": f"Bearer {self.api_key}"},
                    )
                    resp.raise_for_status()
                    payload = resp.json()
                rows = payload.get("data") if isinstance(payload, dict) else payload
                if not isinstance(rows, list):
                    raise ValueError("Unexpected /models response shape.")
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    model_id = str(row.get("id") or "").strip()
                    pricing = row.get("pricing")
                    if not model_id or not isinstance(pricing, dict):
                        continue
                    input_rate = self._as_float_or_none(pricing.get("input"))
                    output_rate = self._as_float_or_none(pricing.get("output"))
                    cached_rate = self._as_float_or_none(pricing.get("cached_input"))
                    if input_rate is None or output_rate is None:
                        continue
                    self._pricing_by_model[model_id] = {
                        "input": input_rate,
                        "cached_input": cached_rate,
                        "output": output_rate,
                        "source": "https://docs.together.ai/reference/models",
                    }
                self._pricing_remote_loaded = True
            except Exception as exc:
                self._pricing_remote_loaded = True
                if not self._pricing_warned:
                    self._pricing_warned = True
                    print(
                        "WARNING: Could not refresh Together pricing from /models; "
                        "using built-in official serverless table fallback when available. "
                        f"error={type(exc).__name__}: {exc}"
                    )

    def _pricing_for_model(self) -> Optional[dict[str, Any]]:
        if self.model_name in self._pricing_by_model:
            return self._pricing_by_model[self.model_name]
        lower = self.model_name.lower()
        for key, pricing in self._pricing_by_model.items():
            if key.lower() == lower:
                return pricing
        return None

    def _estimate_cost(
        self, in_tok: int, out_tok: int, cached_in_tok: int, reasoning_out_tok: int = 0
    ) -> Optional[dict[str, Any]]:
        pricing = self._pricing_for_model()
        if pricing is None:
            return None

        in_tok_i = max(0, self._as_int(in_tok))
        out_tok_i = max(0, self._as_int(out_tok))
        cached_i = min(max(0, self._as_int(cached_in_tok)), in_tok_i)
        uncached_i = max(0, in_tok_i - cached_i)
        reasoning_i = min(max(0, self._as_int(reasoning_out_tok)), out_tok_i)
        visible_i = max(0, out_tok_i - reasoning_i)

        input_rate = self._as_float_or_none(pricing.get("input"))
        cached_rate = self._as_float_or_none(pricing.get("cached_input"))
        output_rate = self._as_float_or_none(pricing.get("output"))

        input_cost = self._cost_component(uncached_i, input_rate)
        if cached_i <= 0:
            cached_cost = 0.0
            cached_rate_effective = cached_rate
        elif cached_rate is None:
            # Some Together models report cached_input_tokens in usage even when
            # official pricing lists no cached-input discount. In that case,
            # bill cached tokens conservatively at the normal input rate instead
            # of treating the whole request as unpriced.
            cached_rate_effective = input_rate
            cached_cost = self._cost_component(cached_i, cached_rate_effective)
        else:
            cached_rate_effective = cached_rate
            cached_cost = self._cost_component(cached_i, cached_rate_effective)
        output_cost = self._cost_component(out_tok_i, output_rate)

        if input_cost is None or cached_cost is None or output_cost is None:
            total_cost = None
        else:
            total_cost = input_cost + cached_cost + output_cost

        return {
            "model_key": self.model_name,
            "pricing_source": pricing.get("source"),
            "pricing_input_rate_per_1m": input_rate,
            "pricing_cached_input_rate_per_1m": cached_rate,
            "pricing_cached_input_effective_rate_per_1m": cached_rate_effective,
            "pricing_output_rate_per_1m": output_rate,
            "input_tokens": in_tok_i,
            "cached_input_tokens": cached_i,
            "uncached_input_tokens": uncached_i,
            "output_tokens": out_tok_i,
            "reasoning_output_tokens": reasoning_i,
            "visible_output_tokens": visible_i,
            "input_cost_usd": input_cost,
            "cached_input_cost_usd": cached_cost,
            "output_cost_usd": output_cost,
            "total_cost_usd": total_cost,
            "cached_input_pricing_missing": (cached_i > 0 and cached_rate is None),
            "cached_input_pricing_fallback": (
                "input_rate" if cached_i > 0 and cached_rate is None else None
            ),
        }

    def _extract_content_and_tool_calls(self, resp: Any) -> Tuple[str, list[Any], Any]:
        choices = getattr(resp, "choices", None) or []
        if not choices:
            return "", [], None
        msg = getattr(choices[0], "message", None)
        if msg is None:
            return "", [], None
        content = getattr(msg, "content", "") or ""
        tool_calls = getattr(msg, "tool_calls", None) or []
        reasoning = (
            getattr(msg, "reasoning", None)
            or getattr(msg, "reasoning_content", None)
            or getattr(msg, "reasoning_details", None)
        )
        return str(content or ""), [_model_dump(item) for item in tool_calls], reasoning

    @staticmethod
    def _is_empty_payload(obj: Any, text: str, tool_calls: list[Any]) -> bool:
        if tool_calls:
            return False
        if isinstance(text, str) and text.strip():
            return False
        if obj is None:
            return True
        if isinstance(obj, str):
            return not obj.strip()
        if isinstance(obj, (dict, list, tuple, set)):
            return len(obj) == 0
        return False

    async def _generate_single(
        self,
        system_msg: str,
        user_msg: str,
        images: Optional[List[str]] = None,
        schema: Optional[Type[BaseModel]] = None,
        call_options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, int, int]:
        est_tokens = (self.tokens_used / self.requests_made) * 1.5 if self.requests_made else self.max_tokens
        reserve = await self._throttle(int(est_tokens))

        messages = self._build_messages(system_msg, user_msg, images, schema)
        req = self._build_request(messages, schema, call_options=call_options)
        await self._ensure_remote_pricing_loaded()

        total_in_tok = 0
        total_out_tok = 0
        total_cached_in_tok = 0
        total_reasoning_out_tok = 0
        total_cost_usd = 0.0
        has_priced_attempt = False
        has_unpriced_attempt = False
        attempts_made = 0
        obj: Any = {"__error__": "EmptyResponseError: uninitialized Together result."}
        last_schema_err: Optional[BaseException] = None

        attempts = max(1, int(self.max_invalid_response_retries) + 1)
        for attempt_idx in range(attempts):
            resp = await self._chat_call_with_retries(req)
            attempts_made += 1
            usage = getattr(resp, "usage", None)
            in_tok, out_tok, cached_in_tok, reasoning_out_tok = self._extract_usage_tokens(usage)
            in_tok = int(in_tok or 0)
            out_tok = int(out_tok or 0)
            cached_in_tok = int(cached_in_tok or 0)
            reasoning_out_tok = int(reasoning_out_tok or 0)

            total_in_tok += in_tok
            total_out_tok += out_tok
            total_cached_in_tok += cached_in_tok
            total_reasoning_out_tok += reasoning_out_tok

            attempt_cost = self._estimate_cost(in_tok, out_tok, cached_in_tok, reasoning_out_tok)
            if attempt_cost is not None and attempt_cost.get("total_cost_usd") is not None:
                total_cost_usd += float(attempt_cost["total_cost_usd"])
                has_priced_attempt = True
                self.last_cost_info = attempt_cost
            else:
                has_unpriced_attempt = True
                self.last_cost_info = attempt_cost

            text, tool_calls, reasoning = self._extract_content_and_tool_calls(resp)
            if tool_calls and not text.strip() and schema is None:
                obj = {"tool_calls": tool_calls}
                if reasoning is not None:
                    obj["reasoning"] = reasoning
            else:
                obj = _loads_json_lenient(text)
                if tool_calls and isinstance(obj, dict):
                    obj.setdefault("tool_calls", tool_calls)
                elif tool_calls:
                    obj = {"content": obj, "tool_calls": tool_calls}

            payload_empty = self._is_empty_payload(obj, text, tool_calls)
            schema_ok = True
            if schema is not None and not (isinstance(obj, dict) and "__error__" in obj):
                try:
                    validated = schema.model_validate(obj)
                    obj = validated.model_dump()
                except Exception as exc:
                    schema_ok = False
                    last_schema_err = exc

            if not payload_empty and schema_ok:
                break

            if attempt_idx < attempts - 1:
                reason_bits = []
                if payload_empty:
                    reason_bits.append("empty_payload")
                if not schema_ok:
                    reason_bits.append("schema_invalid")
                reason = ",".join(reason_bits) or "invalid_output"
                print(
                    f"WARNING: Together returned {reason} for model={self.model_name}; "
                    f"retrying ({attempt_idx + 1}/{attempts - 1})."
                )
                await asyncio.sleep(min(2**attempt_idx, 4) + random.random())
                continue

            if payload_empty:
                obj = {
                    "__error__": (
                        f"EmptyResponseError: Together returned no parseable content "
                        f"after {attempts} attempt(s)."
                    )
                }
            elif schema is not None and not schema_ok:
                if isinstance(obj, str) and obj.strip():
                    pass
                else:
                    err_msg = (
                        f"{type(last_schema_err).__name__}: {last_schema_err}"
                        if last_schema_err
                        else "unknown"
                    )
                    obj = {"__error__": f"SchemaValidationError: {err_msg}"}
            break

        async with self._stats_lock:
            total = int(total_in_tok or 0) + int(total_out_tok or 0)
            self.tokens_used += total
            self.cached_input_tokens_used += int(total_cached_in_tok or 0)
            self.reasoning_output_tokens_used += int(total_reasoning_out_tok or 0)
            self.requests_made += max(1, int(attempts_made))
            if has_priced_attempt:
                self.total_cost_usd += float(total_cost_usd)
                self.costed_requests += 1
            if has_unpriced_attempt or not has_priced_attempt:
                self.uncosted_requests += 1
            reserve[1] = total

        return obj, int(total_in_tok or 0), int(total_out_tok or 0)

    async def generate_async(
        self,
        prompts: List[Tuple[str, str, Optional[List[str]]]],
        schema: Optional[Type[BaseModel]] = None,
        show_progress: bool = True,
        **kwargs: Any,
    ) -> List[Tuple[Any, int, int]]:
        print_cost: bool = bool(kwargs.pop("print_cost", self.print_cost))
        kwargs.pop("batch_size", None)
        max_concurrency = int(kwargs.pop("max_concurrency", self.max_concurrency))
        if max_concurrency <= 0:
            raise ValueError("max_concurrency must be a positive integer.")
        call_options = {k: v for k, v in kwargs.items() if v is not None}
        sem = asyncio.Semaphore(max_concurrency)

        async with self._stats_lock:
            base_cached = int(self.cached_input_tokens_used)
            base_reasoning = int(self.reasoning_output_tokens_used)
            base_cost = float(self.total_cost_usd)
            base_costed = int(self.costed_requests)
            base_uncosted = int(self.uncosted_requests)

        async def _wrap(idx: int, sys_msg: str, user_msg: str, img_list: Optional[List[str]]):
            async with sem:
                try:
                    out = await self._generate_single(
                        sys_msg, user_msg, img_list, schema, call_options=call_options
                    )
                    return idx, out
                except Exception as exc:
                    placeholder = ({"__error__": f"{type(exc).__name__}: {exc}"}, 0, 0)
                    return idx, placeholder

        tasks = []
        for i, tup in enumerate(prompts):
            sys_msg, user_msg = tup[0], tup[1]
            img_list = tup[2] if len(tup) > 2 else None
            tasks.append(asyncio.create_task(_wrap(i, sys_msg, user_msg, img_list)))

        final_results: List[Tuple[Any, int, int]]
        if not show_progress:
            done = await asyncio.gather(*tasks)
            done.sort(key=lambda x: x[0])
            final_results = [r for _, r in done]
        else:
            results: List[Optional[Tuple[Any, int, int]]] = [None] * len(prompts)
            for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"{self.model_name} generating..."):
                idx, res = await fut
                results[idx] = res
            final_results = [
                r if r is not None else ({"__error__": "RuntimeError: missing async result"}, 0, 0)
                for r in results
            ]

        if print_cost:
            call_input_tok = sum(int((r[1] if len(r) > 1 else 0) or 0) for r in final_results)
            call_output_tok = sum(int((r[2] if len(r) > 2 else 0) or 0) for r in final_results)
            call_failed = sum(
                1
                for r in final_results
                if isinstance(r, (list, tuple))
                and len(r) > 0
                and isinstance(r[0], dict)
                and "__error__" in r[0]
            )

            async with self._stats_lock:
                call_cached_tok = int(self.cached_input_tokens_used) - base_cached
                call_reasoning_tok = int(self.reasoning_output_tokens_used) - base_reasoning
                call_cost_usd = float(self.total_cost_usd) - base_cost
                call_costed = int(self.costed_requests) - base_costed
                call_uncosted = int(self.uncosted_requests) - base_uncosted

            call_cached_tok = max(0, call_cached_tok)
            call_uncached_tok = max(0, call_input_tok - call_cached_tok)
            call_reasoning_tok = max(0, min(call_reasoning_tok, call_output_tok))
            call_visible_out_tok = max(0, call_output_tok - call_reasoning_tok)

            summary = (
                f"[COST] model={self.model_name} n={len(final_results)} "
                f"input={call_input_tok} (cached={call_cached_tok}, uncached={call_uncached_tok}) "
                f"output={call_output_tok} (reasoning={call_reasoning_tok}, visible={call_visible_out_tok}) "
                f"est_cost=${call_cost_usd:.6f}"
            )
            if call_costed < len(final_results):
                unpriced_count = max(0, len(final_results) - call_costed)
                if call_uncosted > 0:
                    unpriced_count = max(unpriced_count, call_uncosted)
                summary += f" | priced={call_costed} unpriced={unpriced_count}"
            if call_failed > 0:
                summary += f" | failed={call_failed}"
            print(summary)

        return final_results

    def generate(
        self,
        prompts: List[Tuple[str, str, Optional[List[str]]]],
        schema: Optional[Type[BaseModel]] = None,
        show_progress: bool = True,
        **kwargs: Any,
    ) -> List[Tuple[Any, int, int]]:
        coro = self.generate_async(prompts, schema, show_progress, **kwargs)
        try:
            loop = asyncio.get_running_loop()
            return loop.run_until_complete(coro)  # type: ignore[no-any-return]
        except RuntimeError:
            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                return loop.run_until_complete(coro)  # type: ignore[no-any-return]
            finally:
                try:
                    loop.close()
                except Exception:
                    pass
