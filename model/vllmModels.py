# vllm_models_fixed.py
import os, time, json, logging, warnings, re, math, shutil
from typing import List, Tuple, Optional, Type, Union, Dict, Any
import html
from PIL import Image
from pydantic import BaseModel
from dotenv import load_dotenv

import torch
import torch.nn.functional as F

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoProcessor,
    AutoModelForImageTextToText,
)
try:
    from transformers import AutoModelForVision2Seq
except ImportError:
    AutoModelForVision2Seq = None

from model.model import LLM

logger = logging.getLogger(__name__)


def _memory_debug_enabled() -> bool:
    return str(os.environ.get("VLLM_MODELS_MEM_DEBUG", "0")).strip().lower() in {
        "1", "true", "yes", "on"
    }


def _read_rss_mb(pid: int) -> Optional[float]:
    try:
        with open(f"/proc/{pid}/status", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except Exception:
        return None
    return None


def _child_pids(pid: int) -> List[int]:
    try:
        with open(f"/proc/{pid}/task/{pid}/children", "r", encoding="utf-8") as handle:
            return [int(item) for item in handle.read().split() if item.strip()]
    except Exception:
        return []


def _process_cmd(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as handle:
            raw = handle.read().replace(b"\x00", b" ").decode("utf-8", errors="replace")
            return raw.strip()[:160]
    except Exception:
        return ""


def _log_memory_tree(label: str) -> None:
    if not _memory_debug_enabled():
        return
    root = os.getpid()
    seen: set[int] = set()
    stack = [root]
    rows = []
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        rss = _read_rss_mb(pid)
        if rss is not None:
            rows.append((rss, pid, _process_cmd(pid)))
        stack.extend(_child_pids(pid))
    rows.sort(reverse=True)
    total = sum(row[0] for row in rows)
    detail = "; ".join(f"pid={pid} rss={rss:.1f}MB cmd={cmd}" for rss, pid, cmd in rows[:8])
    logger.info("[mem] %s total_tree_rss=%.1fMB %s", label, total, detail)

class vllmModels(LLM):
    """
    Multimodal wrapper with vLLM backend and (NEW) Transformers fallback for InternVL:
      • generate() → vLLM path (supports single/multi-image)
      • (NEW) InternVL family uses Transformers branch by default (per official docs)
      • Optional structured outputs via Pydantic schema (vLLM guided decoding)
      • Robust fallback for non-instruct (no chat_template) models
      • (NEW) Optional per-token logprobs/top-k for outputs (+ prompt logprobs via vLLM; HF branch outputs-only)
    """

    # Known non-instruct models (no chat template); lowercase match
    NON_INSTRUCT_MODELS = {
        "meta-llama/llama-3.1-70b",
        # Add more as needed
        # "meta-llama/llama-3.1-8b",
        # "meta-llama/llama-3.1-405b",
    }
    QWEN35_SAMPLING_PROFILES = {
        "thinking": {
            "enable_thinking": True,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 1.5,
            "repetition_penalty": 1.0,
        },
        "thinking_precise_coding": {
            "enable_thinking": True,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 0.0,
            "repetition_penalty": 1.0,
        },
        "instruct_general": {
            "enable_thinking": False,
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 1.5,
            "repetition_penalty": 1.0,
        },
        "instruct_reasoning": {
            "enable_thinking": False,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 40,
            "min_p": 0.0,
            "presence_penalty": 2.0,
            "repetition_penalty": 1.0,
        },
        "instruct_reasoning_temp0": {
            "enable_thinking": False,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "min_p": 0.0,
            "presence_penalty": 2.0,
            "repetition_penalty": 1.0,
        },
    }
    QWEN36_SAMPLING_PROFILES = {
        "thinking": {
            "enable_thinking": True,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 1.5,
            "repetition_penalty": 1.0,
        },
        "thinking_precise_coding": {
            "enable_thinking": True,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 0.0,
            "repetition_penalty": 1.0,
        },
        "instruct_general": {
            "enable_thinking": False,
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 1.5,
            "repetition_penalty": 1.0,
        },
        "instruct_reasoning": {
            "enable_thinking": False,
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 1.5,
            "repetition_penalty": 1.0,
        },
        "instruct_reasoning_temp0": {
            "enable_thinking": False,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "min_p": 0.0,
            "presence_penalty": 1.5,
            "repetition_penalty": 1.0,
        },
    }
    QWEN35_MODE_ALIASES = {
        "thinking_general": "thinking",
        "thinking_coding": "thinking_precise_coding",
        "reasoning": "instruct_reasoning",
        "instruct": "instruct_reasoning",
        "nothinking": "instruct_reasoning",
        "nonthinking": "instruct_reasoning",
        "non_thinking": "instruct_reasoning",
        "nonthinking_temp0": "instruct_reasoning_temp0",
        "non_thinking_temp0": "instruct_reasoning_temp0",
        "nothinking_temp0": "instruct_reasoning_temp0",
        "off": "instruct_reasoning",
        "false": "instruct_reasoning",
        "0": "instruct_reasoning",
        "on": "thinking",
        "true": "thinking",
        "1": "thinking",
    }
    VLLM_COMPILATION_MODE_ALIASES = {
        "0": "NONE",
        "1": "STOCK_TORCH_COMPILE",
        "2": "DYNAMO_TRACE_ONCE",
        "3": "VLLM_COMPILE",
        "none": "NONE",
        "off": "NONE",
        "false": "NONE",
        "stock_torch_compile": "STOCK_TORCH_COMPILE",
        "stock": "STOCK_TORCH_COMPILE",
        "torch_compile": "STOCK_TORCH_COMPILE",
        "dynamo_trace_once": "DYNAMO_TRACE_ONCE",
        "dynamo": "DYNAMO_TRACE_ONCE",
        "trace_once": "DYNAMO_TRACE_ONCE",
        "vllm_compile": "VLLM_COMPILE",
        "vllm": "VLLM_COMPILE",
        "default": "VLLM_COMPILE",
        "true": "VLLM_COMPILE",
    }
    VLLM_CUDAGRAPH_MODE_ALIASES = {
        "0": "NONE",
        "1": "PIECEWISE",
        "2": "FULL",
        "none": "NONE",
        "off": "NONE",
        "false": "NONE",
        "piecewise": "PIECEWISE",
        "full": "FULL",
        "full_decode_only": "FULL_DECODE_ONLY",
        "decode_only": "FULL_DECODE_ONLY",
        "full_and_piecewise": "FULL_AND_PIECEWISE",
        "full_piecewise": "FULL_AND_PIECEWISE",
        "hybrid": "FULL_AND_PIECEWISE",
        "default": "FULL_AND_PIECEWISE",
        "true": "FULL_AND_PIECEWISE",
    }

    def __init__(
        self,
        model_name: str,
        max_tokens: int = 8192,
        gpu_id: Optional[int] = None,
        temperature: float = 0.6,
        seed: Optional[int] = None,
        top_p: float = 0.95,
        top_k: int = -1,
        min_p: float = 0.0,
        presence_penalty: float = 0.0,
        repetition_penalty: float = 1.0,
        max_images: int = 47,
        trust_remote_code: bool = True,
        max_model_len: Optional[int] = None,   # if None, let vLLM infer from the model
        # vLLM upstream default is 0.90; this project defaults to 0.95.
        # Setting this too high can OOM during warmup (e.g., CUDA graph capture).
        gpu_memory_utilization: float = 0.95,
        # Scheduler token budget per iteration. Lowering this can avoid
        # initialization/warmup OOM for very large models.
        max_num_batched_tokens: int = 16384,
        quantization: Optional[str] = None,    # e.g., "awq", "gptq", "fp8"
        tensor_parallel_size: Optional[int] = None,  # None => auto detect from visible GPUs
        pipeline_parallel_size: Optional[int] = None,
        decode_context_parallel_size: Optional[int] = None,
        distributed_executor_backend: Optional[str] = None,
        all2all_backend: Optional[str] = None,
        enable_expert_parallel: bool = False,
        attention_backend: Optional[str] = None,  # e.g., "TRITON_ATTN", "FLASH_ATTN"
        qwen35_mode: Optional[str] = None,
        enforce_eager: Optional[bool] = None,
        compilation_mode: Optional[str] = None,
        cudagraph_mode: Optional[str] = None,
        enable_chunked_prefill: Optional[bool] = None,
        async_scheduling: Optional[bool] = None,
        enable_prefix_caching: Optional[bool] = None,
        mm_processor_cache_gb: Optional[float] = None,
        mm_processor_cache_type: Optional[str] = None,
        mm_processor_kwargs: Optional[Dict[str, Any]] = None,
        gemma4_thinking: bool = False,
    ):
        super().__init__(model_name=model_name)
        load_dotenv()
        self.model_name_full = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.presence_penalty = presence_penalty
        self.repetition_penalty = repetition_penalty
        self.seed = seed or int(time.time() * 1000) % (2**32 - 1)
        self.max_images = max_images
        self._qwen3_enable_thinking = str(os.environ.get("QWEN3_THINKING", "0")).strip().lower() in {
            "1", "true", "yes", "on"
        }
        self._qwen_soft_switch_enabled = str(
            os.environ.get("QWEN_DISABLE_SOFT_SWITCH", "0")
        ).strip().lower() not in {"1", "true", "yes", "on"}
        self._qwen35_mode_requested = qwen35_mode
        self._qwen35_mode: Optional[str] = None
        self._enforce_eager_override = enforce_eager
        self._compilation_mode_override = compilation_mode
        self._cudagraph_mode_override = cudagraph_mode
        self._enable_chunked_prefill_override = enable_chunked_prefill
        self._async_scheduling_override = async_scheduling
        self._enable_prefix_caching_override = enable_prefix_caching
        self._mm_processor_cache_gb_override = mm_processor_cache_gb
        self._mm_processor_cache_type_override = mm_processor_cache_type
        self._mm_processor_kwargs_override = mm_processor_kwargs
        self._gemma4_enable_thinking = bool(gemma4_thinking)
        if distributed_executor_backend is None:
            self.distributed_executor_backend = None
        else:
            backend_value = str(distributed_executor_backend).strip().lower()
            self.distributed_executor_backend = (
                None
                if backend_value in {"", "none", "auto", "default"}
                else backend_value
            )

        # Bind to a specific GPU (must set before initializing LLM)
        if gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        self._runtime_model_name = self._resolve_runtime_model_name(model_name)

        self.tensor_parallel_size = self._resolve_tensor_parallel_size(
            requested=tensor_parallel_size,
            gpu_id=gpu_id,
            distributed_executor_backend=self.distributed_executor_backend,
        )

        # Tokenizer (used for chat_template and token counting)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._runtime_model_name, trust_remote_code=trust_remote_code
        )
        self._model_metadata = self._load_model_metadata(
            model_name=self._runtime_model_name,
            trust_remote_code=trust_remote_code,
        )
        self._model_type = str(self._model_metadata.get("model_type") or "").strip().lower()
        self._processor_class = str(self._model_metadata.get("processor_class") or "").strip().lower()
        self._architectures = tuple(
            str(arch).strip().lower()
            for arch in (self._model_metadata.get("architectures") or [])
            if str(arch).strip()
        )
        if self._is_qwen35_family() or self._is_qwen36_family():
            self._qwen35_mode = self._resolve_qwen35_mode(qwen35_mode)
            self._apply_qwen35_mode(self._qwen35_mode)
            # Current vLLM offline multimodal inference is materially more stable
            # for Qwen3.5/3.6 when torch.compile remains enabled but CUDA graphs are
            # disabled. Keep this as the family default unless the caller passes
            # an explicit override.
            if self._cudagraph_mode_override is None:
                self._cudagraph_mode_override = "none"
                logger.info(
                    "Defaulting Qwen3.5 family to cudagraph_mode=none for more stable offline multimodal inference."
                )

        # Heuristic multimodal detection
        def _guess_is_multimodal(name: str, tokenizer) -> bool:
            name = (name or "").lower()
            tpl = getattr(tokenizer, "chat_template", "") or ""
            # NEW: explicitly handle medgemma (text-only variants)
            if "medgemma" in name:
                return ("text" not in name)  # medgemma-27b-it is multimodal; -text-it is text-only
            if any(k in name for k in [
                "gemma-3", "gemma-4", "gemma3", "gemma4", "-vl", "vision",
                "multimodal", "minicpm-v", "llava", "qwen3.5", "qwen3_5",
                "qwen3.6", "qwen3_6", "qwen2.5-vl", "qwen2-vl",
                "phi-3-vision", "internvl3", "internvl3_5", "internvl"
            ]):
                return True
            if any(tok in tpl for tok in ["<|vision_start|>", "<|image_pad|>", "<start_of_image>", "<image>"]):
                return True
            return False

        self.is_multimodal = _guess_is_multimodal(model_name, self.tokenizer)
        if self.is_multimodal:
            previous_v1_multiproc = os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING")
            if previous_v1_multiproc != "0":
                os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
                logger.info(
                    "Multimodal model detected; forcing VLLM_ENABLE_V1_MULTIPROCESSING=0 "
                    "for more stable offline inference."
                )

        # Whether model is from InternVL family.
        self._is_internvl = self._detect_is_internvl()
        # Official OpenGVLab "-HF" checkpoints are Transformers-standard and
        # should run on vLLM directly. Legacy "Github format" checkpoints are
        # kept on the Transformers fallback branch.
        self._is_internvl_hf_variant = self._detect_is_internvl_hf_variant()
        self._use_internvl_transformers = self._is_internvl and (not self._is_internvl_hf_variant)
        self._is_lingshu = self._detect_is_lingshu()

        # At init, decide whether to use chat template
        self._force_plain_text = self._is_non_instruct_model()

        # Lazy-load for the Transformers branch
        self._hf_processor: Optional[AutoProcessor] = None
        # Potential HF model families: ImageText2Text (new) or Vision2Seq (legacy)
        self._hf_model: Optional[torch.nn.Module] = None
        # Lazy multimodal chat processor for model-card-compliant prompt formatting.
        self._mm_chat_processor: Optional[Any] = None

        # Force the Transformers path (e.g., INTERNVL_FORCE_TRANSFORMERS=1)
        self._force_hf_env = os.environ.get("INTERNVL_FORCE_TRANSFORMERS", "0") == "1"

        # For legacy InternVL family: use HF directly; skip vLLM.
        # For InternVL "-HF" checkpoints, use the vLLM path.
        self.llm = None
        if not (self._use_internvl_transformers or self._force_hf_env):
            # Initialize vLLM for all models except legacy InternVL forced to HF.
            vllm_cls = None
            llm_kwargs: Dict[str, Any] = {}
            requested_dcp: Optional[int] = None
            try:
                from vllm import LLM as vLLM
                vllm_cls = vLLM
                llm_kwargs = dict(
                    model=self._runtime_model_name,
                    trust_remote_code=trust_remote_code,
                    seed=self.seed,
                    gpu_memory_utilization=gpu_memory_utilization,
                    max_num_batched_tokens=max_num_batched_tokens,
                    tensor_parallel_size=self.tensor_parallel_size,
                )
                if self.distributed_executor_backend is not None:
                    llm_kwargs["distributed_executor_backend"] = self.distributed_executor_backend
                if pipeline_parallel_size is not None:
                    llm_kwargs["pipeline_parallel_size"] = int(pipeline_parallel_size)
                if decode_context_parallel_size is not None:
                    requested_dcp = int(decode_context_parallel_size)
                    llm_kwargs["decode_context_parallel_size"] = requested_dcp
                if all2all_backend:
                    llm_kwargs["all2all_backend"] = str(all2all_backend)
                if enable_expert_parallel:
                    llm_kwargs["enable_expert_parallel"] = True
                compilation_config: Dict[str, Any] = {}
                # Optional escape hatch for CUDA graph / torch.compile issues.
                requested_enforce_eager = self._enforce_eager_override
                if requested_enforce_eager is None:
                    requested_enforce_eager = str(
                        os.environ.get("VLLM_ENFORCE_EAGER", "0")
                    ).strip().lower() in {"1", "true", "yes", "on"}
                if requested_enforce_eager:
                    llm_kwargs["enforce_eager"] = True
                    logger.info("vLLM eager mode enabled -> enforce_eager=True")
                if self._compilation_mode_override is not None:
                    from vllm.config.compilation import CompilationMode

                    compilation_config["mode"] = self._resolve_vllm_enum(
                        self._compilation_mode_override,
                        aliases=self.VLLM_COMPILATION_MODE_ALIASES,
                        enum_cls=CompilationMode,
                        kind="compilation_mode",
                    )
                    logger.info(
                        "Overriding vLLM compilation mode -> %s",
                        compilation_config["mode"],
                    )
                if self._cudagraph_mode_override is not None:
                    from vllm.config.compilation import CUDAGraphMode

                    compilation_config["cudagraph_mode"] = self._resolve_vllm_enum(
                        self._cudagraph_mode_override,
                        aliases=self.VLLM_CUDAGRAPH_MODE_ALIASES,
                        enum_cls=CUDAGraphMode,
                        kind="cudagraph_mode",
                    )
                    logger.info(
                        "Overriding vLLM cudagraph mode -> %s",
                        compilation_config["cudagraph_mode"],
                    )
                # On Hopper/Blackwell with TP>1, vLLM may enable flashinfer
                # allreduce-rms fusion by default. On some clusters this path
                # can fail during workspace creation (cudaErrorInsufficientDriver).
                disable_allreduce_rms_fusion = str(
                    os.environ.get("VLLM_DISABLE_ALLREDUCE_RMS_FUSION", "1")
                ).strip().lower() in {"1", "true", "yes", "on"}
                if disable_allreduce_rms_fusion and self.tensor_parallel_size > 1:
                    pass_config = dict(compilation_config.get("pass_config") or {})
                    pass_config["fuse_allreduce_rms"] = False
                    compilation_config["pass_config"] = pass_config
                    logger.info(
                        "VLLM_DISABLE_ALLREDUCE_RMS_FUSION enabled -> "
                        "compilation_config.pass_config.fuse_allreduce_rms=False"
                    )
                if compilation_config:
                    llm_kwargs["compilation_config"] = compilation_config
                # For image-text workloads, default to video=0 to reduce memory.
                # Override with env var, e.g. VLLM_MAX_VIDEOS_PER_PROMPT=1.
                if self.is_multimodal:
                    video_limit = 0
                    video_limit_raw = os.environ.get("VLLM_MAX_VIDEOS_PER_PROMPT")
                    if video_limit_raw is not None:
                        try:
                            video_limit = max(0, int(video_limit_raw))
                        except Exception:
                            warnings.warn(
                                "Invalid VLLM_MAX_VIDEOS_PER_PROMPT="
                                f"{video_limit_raw!r}; fallback to 0."
                            )
                            video_limit = 0
                    llm_kwargs["limit_mm_per_prompt"] = {
                        "image": max_images,
                        "video": video_limit,
                    }
                    logger.info(
                        "limit_mm_per_prompt configured as image=%s, video=%s",
                        max_images,
                        video_limit,
                    )
                if self._enable_chunked_prefill_override is not None:
                    llm_kwargs["enable_chunked_prefill"] = bool(
                        self._enable_chunked_prefill_override
                    )
                    logger.info(
                        "Overriding vLLM chunked prefill -> %s",
                        llm_kwargs["enable_chunked_prefill"],
                    )
                if self._async_scheduling_override is not None:
                    llm_kwargs["async_scheduling"] = bool(
                        self._async_scheduling_override
                    )
                    logger.info(
                        "Overriding vLLM async scheduling -> %s",
                        llm_kwargs["async_scheduling"],
                    )
                if self._enable_prefix_caching_override is not None:
                    llm_kwargs["enable_prefix_caching"] = bool(
                        self._enable_prefix_caching_override
                    )
                    logger.info(
                        "Overriding vLLM prefix caching -> %s",
                        llm_kwargs["enable_prefix_caching"],
                    )
                if self._mm_processor_cache_gb_override is not None:
                    llm_kwargs["mm_processor_cache_gb"] = float(
                        self._mm_processor_cache_gb_override
                    )
                    logger.info(
                        "Overriding vLLM mm_processor_cache_gb -> %s",
                        llm_kwargs["mm_processor_cache_gb"],
                    )
                if self._mm_processor_cache_type_override is not None:
                    llm_kwargs["mm_processor_cache_type"] = str(
                        self._mm_processor_cache_type_override
                    ).strip().lower()
                    logger.info(
                        "Overriding vLLM mm_processor_cache_type -> %s",
                        llm_kwargs["mm_processor_cache_type"],
                    )
                if self._mm_processor_kwargs_override:
                    llm_kwargs["mm_processor_kwargs"] = dict(
                        self._mm_processor_kwargs_override
                    )
                    logger.info(
                        "Overriding vLLM mm_processor_kwargs -> %s",
                        llm_kwargs["mm_processor_kwargs"],
                    )
                if max_model_len is not None:
                    llm_kwargs["max_model_len"] = max_model_len
                if quantization:
                    llm_kwargs["quantization"] = quantization
                if attention_backend:
                    llm_kwargs["attention_config"] = {"backend": attention_backend}
                if self._is_qwen3_family():
                    llm_kwargs["reasoning_parser"] = "qwen3"

                self.llm = vllm_cls(**llm_kwargs)
                logger.info("Loaded model via vLLM ✔")
                _log_memory_tree("after vLLM init")
            except Exception as e:
                should_retry_without_dcp = (
                    vllm_cls is not None
                    and requested_dcp is not None
                    and requested_dcp > 1
                    and self._is_dcp_tp_kv_heads_validation_error(e)
                )
                if should_retry_without_dcp:
                    retry_kwargs = dict(llm_kwargs)
                    retry_kwargs.pop("decode_context_parallel_size", None)
                    warnings.warn(
                        "vLLM init failed with decode_context_parallel_size="
                        f"{requested_dcp} due to TP/KV-head constraint; retrying with DCP disabled."
                    )
                    try:
                        self.llm = vllm_cls(**retry_kwargs)
                        logger.info(
                            "Loaded model via vLLM ✔ (retry succeeded with decode_context_parallel_size=1)"
                        )
                    except Exception as retry_e:
                        warnings.warn(
                            "vLLM retry without decode_context_parallel_size failed: "
                            f"{retry_e}"
                        )
                        warnings.warn(
                            "vLLM failed to initialize (will use Transformers if possible): "
                            f"{e}"
                        )
                        self.llm = None
                else:
                    warnings.warn(f"vLLM failed to initialize (will use Transformers if possible): {e}")
                    self.llm = None

    def _resolve_tensor_parallel_size(
        self,
        requested: Optional[int],
        gpu_id: Optional[int],
        distributed_executor_backend: str,
    ) -> int:
        """
        Decide tensor parallel size from runtime environment.
        Priority:
          1) explicit user request (if provided)
          2) auto-detect from visible CUDA devices (gpu_id / CUDA_VISIBLE_DEVICES / torch)
        """
        visible = self._detect_visible_cuda_count(gpu_id=gpu_id)

        if requested is None:
            chosen = visible if visible > 0 else 1
            if visible > 0:
                logger.info(f"Auto tensor_parallel_size={chosen} (visible_gpus={visible})")
            else:
                logger.info("Auto tensor_parallel_size=1 (no visible CUDA device detected)")
            return max(1, int(chosen))

        try:
            chosen = max(1, int(requested))
        except Exception:
            warnings.warn(f"Invalid tensor_parallel_size={requested}; fallback to auto.")
            return self._resolve_tensor_parallel_size(
                requested=None,
                gpu_id=gpu_id,
                distributed_executor_backend=distributed_executor_backend,
            )

        backend = (distributed_executor_backend or "").strip().lower()
        allow_cross_node_tp = backend in {"ray", "external_launcher"}

        if visible > 0 and chosen > visible:
            if allow_cross_node_tp:
                logger.info(
                    "tensor_parallel_size=%s > visible_gpus=%s with backend=%s; "
                    "keeping requested value for multi-node sharding.",
                    chosen,
                    visible,
                    backend,
                )
            else:
                warnings.warn(
                    f"tensor_parallel_size={chosen} exceeds visible_gpus={visible}; "
                    f"clamping to {visible}."
                )
                chosen = visible
        elif visible <= 0 and chosen > 1:
            warnings.warn(
                "No visible CUDA device detected but tensor_parallel_size>1 was requested; "
                "clamping to 1."
            )
            chosen = 1

        return max(1, int(chosen))

    def _detect_visible_cuda_count(self, gpu_id: Optional[int]) -> int:
        # If caller bound a single GPU via gpu_id, this process should only use one GPU.
        if gpu_id is not None:
            return 1

        env_count = self._parse_cuda_visible_devices_count(os.environ.get("CUDA_VISIBLE_DEVICES"))
        if env_count is not None:
            return env_count

        try:
            if torch.cuda.is_available():
                return int(torch.cuda.device_count())
        except Exception:
            pass
        return 0

    @staticmethod
    def _parse_cuda_visible_devices_count(raw: Optional[str]) -> Optional[int]:
        """
        Parse CUDA_VISIBLE_DEVICES into a count.
        Returns None when variable is unset/empty (unknown).
        """
        if raw is None:
            return None
        s = str(raw).strip()
        if s == "":
            return None
        if s == "-1":
            return 0
        parts = [p.strip() for p in s.split(",") if p.strip()]
        if not parts:
            return 0
        return len(parts)

    @staticmethod
    def _is_dcp_tp_kv_heads_validation_error(exc: Exception) -> bool:
        """
        Detect the vLLM validation error:
        tensor parallel size X must be greater than total num kv heads Y
        when decode context parallel is enabled.
        """
        msg = str(exc).lower()
        return (
            "tensor parallel size" in msg
            and "kv heads" in msg
            and (
                "decode context parallel" in msg
                or "decode_context_parallel" in msg
            )
        )

    # ---------- public API ----------
    def generate(
        self,
        prompts: List[Union[Tuple[str, str], Tuple[str, str, List[str]]]],
        batch_size: Optional[int] = None,
        schema: Optional[Type[BaseModel]] = None,
        return_logprobs: bool = False,
        top_logprobs: Optional[int] = None,
        include_prompt_logprobs: bool = False,
        show_progress: bool = False,
    ):
        """
        Returns:
            - When return_logprobs=False (default):
                List[(text: str, in_tok: int, out_tok: int)]
            - When return_logprobs=True:
                List[(text, in_tok, out_tok, extras)], where:
                extras = {
                  "output_logprobs": List[Dict],  # step-wise info
                  "prompt_logprobs": Optional[List[Dict]]  # available in vLLM; None in HF branch
                }
        """
        # Normalize to (sys, user, [images]) structure
        triples = [(p + ([],)) if len(p) == 2 else p for p in prompts]
        requested_batch_size = None if batch_size is None else int(batch_size)
        effective_batch_size = len(triples) if triples else 1

        # InternVL family or explicitly forced: use Transformers branch.
        # For non-InternVL models, a failed vLLM init should surface directly
        # instead of attempting an incompatible vision HF fallback.
        use_hf = self._use_internvl_transformers or self._force_hf_env
        if use_hf:
            hf_batch_size = len(triples) if (requested_batch_size is None or requested_batch_size <= 0) else requested_batch_size
            hf_batch_size = max(1, hf_batch_size)
            return self._generate_with_transformers_internvl(
                triples=triples,
                batch_size=hf_batch_size,
                return_logprobs=return_logprobs,
                top_logprobs=top_logprobs,
                include_prompt_logprobs=include_prompt_logprobs,
                schema=schema,  # HF does not support guided decoding; warn only
            )
        if self.llm is None:
            raise RuntimeError(
                "vLLM initialization failed for a non-InternVL model. "
                "Check earlier vLLM startup errors (typically CUDA OOM / config)."
            )
        if requested_batch_size not in (None, 0, len(triples)):
            logger.info(
                "Ignoring batch_size=%s for vLLM inference; sending all %s requests to the backend in one generate() call.",
                requested_batch_size,
                len(triples),
            )

        # ---------- vLLM path (non-InternVL) ----------
        from vllm import SamplingParams
        sp_kwargs = dict(
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            min_p=self.min_p,
            presence_penalty=self.presence_penalty,
            repetition_penalty=self.repetition_penalty,
        )
        # Structured outputs (if needed)
        if schema is not None:
            try:
                from vllm.sampling_params import GuidedDecodingParams
                sp_kwargs["guided_decoding"] = GuidedDecodingParams(
                    json=schema.model_json_schema()
                )
            except Exception as e:
                warnings.warn(f"Structured outputs unavailable ({e}); continuing without.")

        # Log probabilities (logprobs/top-K)
        want_logprobs = bool(return_logprobs)
        if want_logprobs:
            k = 5 if (top_logprobs is None) else max(0, int(top_logprobs))
            sp_kwargs["logprobs"] = k
            if include_prompt_logprobs:
                try:
                    sp_kwargs["prompt_logprobs"] = k
                except Exception as e:
                    warnings.warn(f"prompt_logprobs not supported in this vLLM build: {e}")
        sp = SamplingParams(**sp_kwargs)

        outs, in_tok, out_tok = [], [], []
        extras_all: List[Optional[Dict[str, Any]]] = []

        batch_starts = range(0, len(triples), effective_batch_size)
        if show_progress and len(triples) > effective_batch_size:
            try:
                from tqdm.auto import tqdm

                batch_starts = tqdm(
                    batch_starts,
                    total=(len(triples) + effective_batch_size - 1) // effective_batch_size,
                    desc=f"{self.model_name} generating...",
                )
            except Exception:
                pass

        for i in batch_starts:
            batch = triples[i: i + effective_batch_size]
            request_list = []
            pil_handles: List[List[Image.Image]] = []

            for sys_msg, user_msg, img_paths in batch:
                user_core = user_msg
                chat_template_source = self.tokenizer
                chat: List[Dict[str, Any]]

                # Some model families require processor-level multimodal chat
                # formatting to produce exact placeholder tokens.
                use_structured_mm_chat = bool(
                    self.is_multimodal
                    and img_paths
                    and (
                        self._is_lingshu
                        or self._is_internvl_hf_variant
                        or self._is_gemma_family()
                        or self._is_medgemma_family()
                        or (self._model_hint_contains("qwen") and self._has_qwen_vision_chat_template())
                    )
                )
                if use_structured_mm_chat:
                    try:
                        if (
                            self._is_lingshu
                            or self._is_internvl_hf_variant
                            or self._is_gemma_family()
                            or self._is_medgemma_family()
                        ):
                            self._ensure_mm_chat_processor_ready()
                            if not hasattr(self._mm_chat_processor, "apply_chat_template"):
                                raise AttributeError("processor has no apply_chat_template")
                            chat_template_source = self._mm_chat_processor
                        else:
                            chat_template_source = self.tokenizer
                        user_content = [{"type": "image", "image": "<image>"} for _ in img_paths]
                        user_content.append({"type": "text", "text": user_msg})
                        chat = []
                        if (sys_msg or "").strip():
                            if chat_template_source is self.tokenizer:
                                chat.append({"role": "system", "content": sys_msg})
                            else:
                                chat.append({
                                    "role": "system",
                                    "content": [{"type": "text", "text": sys_msg}],
                                })
                        chat.append({"role": "user", "content": user_content})
                    except Exception as e:
                        warnings.warn(
                            "Structured multimodal chat-template formatting failed "
                            f"({e}); fallback to explicit placeholder injection."
                        )
                        user_core = self._inject_image_placeholders(user_msg, img_paths)
                        chat = [
                            {"role": "system", "content": sys_msg},
                            {"role": "user", "content": user_core},
                        ]
                else:
                    user_core = (
                        self._inject_image_placeholders(user_msg, img_paths)
                        if self.is_multimodal else user_msg
                    )
                    chat = [
                        {"role": "system", "content": sys_msg},
                        {"role": "user", "content": user_core},
                    ]

                # Qwen3 safeguard: default no-think unless explicitly enabled.
                if self._supports_qwen_soft_switch() and (not self._qwen3_enable_thinking):
                    no_think_applied = False
                    if use_structured_mm_chat:
                        no_think_applied = self._prepend_no_think_to_structured_chat(chat)
                    if (not no_think_applied) and (not user_core.lstrip().startswith("/no_think")):
                        user_core = "/no_think\n" + user_core
                        chat = [
                            {"role": "system", "content": sys_msg},
                            {"role": "user", "content": user_core},
                        ]
                if self._is_gemma4_family() and self._gemma4_enable_thinking:
                    if use_structured_mm_chat:
                        self._prepend_think_to_structured_system(chat)
                    else:
                        sys_msg = self._prepend_gemma4_think_token(sys_msg)
                        chat = [
                            {"role": "system", "content": sys_msg},
                            {"role": "user", "content": user_core},
                        ]

                # Compose final prompt
                if self._should_use_chat_template():
                    tpl_kwargs = {"tokenize": False, "add_generation_prompt": True}
                    if self._is_qwen3_family():
                        tpl_kwargs["enable_thinking"] = self._qwen3_enable_thinking
                    try:
                        prompt_txt = chat_template_source.apply_chat_template(chat, **tpl_kwargs)
                    except TypeError:
                        tpl_kwargs.pop("enable_thinking", None)
                        try:
                            prompt_txt = chat_template_source.apply_chat_template(chat, **tpl_kwargs)
                        except Exception as e2:
                            warnings.warn(f"apply_chat_template failed ({e2}); fallback to plain text.")
                            prompt_txt = self._compose_plain_prompt(sys_msg, user_core)
                    except AttributeError as e:
                        warnings.warn(f"No chat_template on tokenizer ({e}); fallback to plain text.")
                        prompt_txt = self._compose_plain_prompt(sys_msg, user_core)
                    except Exception as e:
                        warnings.warn(f"apply_chat_template error ({e}); fallback to plain text.")
                        prompt_txt = self._compose_plain_prompt(sys_msg, user_core)
                else:
                    prompt_txt = self._compose_plain_prompt(sys_msg, user_core)

                # Load images (vLLM multimodal path)
                imgs = []
                if self.is_multimodal and img_paths:
                    for pth in img_paths:
                        im = Image.open(pth).convert("RGB")
                        imgs.append(im)
                pil_handles.append(imgs)

                req: Dict[str, Any] = {"prompt": prompt_txt}
                if self.is_multimodal and len(imgs) == 1:
                    req["multi_modal_data"] = {"image": imgs[0]}
                elif self.is_multimodal and len(imgs) > 1:
                    req["multi_modal_data"] = {"image": imgs}
                request_list.append(req)

            _log_memory_tree(f"before llm.generate batch_size={len(request_list)}")
            results = self.llm.generate(request_list, sp)
            _log_memory_tree(f"after llm.generate batch_size={len(request_list)}")

            # Extract text, token counts, and optional logprobs
            for j, r in enumerate(results):
                out_item = r.outputs[0]
                recovered_text, recovery_out_tok = self._recover_qwen_thinking_answer(
                    request=request_list[j],
                    generated_text=out_item.text,
                )
                text = self._postprocess(recovered_text)
                outs.append(text)

                this_in_tok = self._count_tokens(request_list[j]["prompt"])
                this_out_tok = self._count_tokens(recovered_text)
                in_tok.append(this_in_tok)
                out_tok.append(this_out_tok)

                if want_logprobs:
                    extras = {
                        "output_logprobs": self._format_output_logprobs(out_item),
                        "prompt_logprobs": self._format_prompt_logprobs(r) if include_prompt_logprobs else None,
                    }
                    extras_all.append(extras)
                else:
                    extras_all.append(None)

            # Close PIL handles
            for imgs in pil_handles:
                for im in imgs:
                    try:
                        im.close()
                    except Exception:
                        pass
            _log_memory_tree(f"after closing PIL batch_size={len(request_list)}")

        if want_logprobs:
            return [(t, it, ot, ex) for t, it, ot, ex in zip(outs, in_tok, out_tok, extras_all)]
        else:
            return list(zip(outs, in_tok, out_tok))

    # ---------- Transformers branch (InternVL-specific, official usage) ----------
    def _generate_with_transformers_internvl(
        self,
        triples: List[Tuple[str, str, List[str]]],
        batch_size: int,
        return_logprobs: bool,
        top_logprobs: Optional[int],
        include_prompt_logprobs: bool,
        schema: Optional[Type[BaseModel]] = None,
    ):
        # HF does not support vLLM guided decoding; warn once
        if schema is not None:
            warnings.warn("Transformers branch does not support guided decoding; ignoring `schema`.")

        self._ensure_hf_internvl_ready()

        outs, in_tok, out_tok = [], [], []
        extras_all: List[Optional[Dict[str, Any]]] = []
        want_logprobs = bool(return_logprobs)
        k = 5 if (top_logprobs is None) else max(0, int(top_logprobs))

        for i in range(0, len(triples), batch_size):
            batch = triples[i: i + batch_size]

            messages_batch: List[List[Dict[str, Any]]] = []
            pil_handles: List[List[Image.Image]] = []

            for sys_msg, user_msg, img_paths in batch:
                content_blocks: List[Dict[str, Any]] = []

                # Load images (PIL objects; official chat template supports {"type":"image","image": PIL.Image})
                imgs = []
                if img_paths:
                    for pth in img_paths:
                        im = Image.open(pth).convert("RGB")
                        imgs.append(im)
                        content_blocks.append({"type": "image", "image": im})
                # Text content
                content_blocks.append({"type": "text", "text": user_msg})

                one_dialog: List[Dict[str, Any]] = []
                if (sys_msg or "").strip():
                    one_dialog.append({"role": "system", "content": [{"type": "text", "text": sys_msg}]})
                one_dialog.append({"role": "user", "content": content_blocks})

                messages_batch.append(one_dialog)
                pil_handles.append(imgs)

            # Official: tokenize=True, return_dict=True (returns tensors)
            # Prepare tensors (must set tokenize=True, return_dict=True)
            inputs = self._hf_processor.apply_chat_template(
                messages_batch,
                padding=True,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )

            # Key fix: align dtype only for pixel_values to model dtype (usually bf16); other tensors move to device only
            inputs = self._align_mm_inputs_dtype_device(inputs)

            # Move tensors to device without forcing dtype changes (pixel tensors often remain float32 for stability)
            device = next(self._hf_model.parameters()).device
            inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}

            gen_kwargs = dict(
                max_new_tokens=self.max_tokens,
                do_sample=(self.temperature is not None and self.temperature > 0),
                temperature=float(self.temperature),
                top_p=float(self.top_p),
                repetition_penalty=float(self.repetition_penalty),
                use_cache=True,
            )
            if want_logprobs:
                gen_kwargs.update(dict(output_scores=True, return_dict_in_generate=True))

            with torch.no_grad():
                output = self._hf_model.generate(**inputs, **gen_kwargs)

            # Retrieve generated sequences
            if want_logprobs:
                sequences = output.sequences  # (B, prompt+gen)
                scores = list(output.scores)  # list(len=gen_len) of (B, vocab)
            else:
                sequences = output          # (B, prompt+gen)
                scores = None

            # Actual prompt length (from attention_mask)
            attn = inputs["attention_mask"]  # (B, L)
            input_lens = attn.sum(dim=1).tolist()

            # Decode newly generated tokens per sample and postprocess consistent with vLLM
            B = sequences.size(0)
            for b in range(B):
                in_len = int(input_lens[b])
                full_ids = sequences[b]
                gen_ids = full_ids[in_len:]
                text = self._safe_decode_ids_hf(gen_ids)
                text = self._postprocess(text)
                outs.append(text)

                in_tok.append(in_len)
                out_tok.append(int(gen_ids.numel()))

                if want_logprobs:
                    step_max = min(len(scores), gen_ids.numel())
                    per_steps: List[Dict[str, Any]] = []
                    for s in range(step_max):
                        logits_s = scores[s][b]  # (vocab,)
                        logp_s = F.log_softmax(logits_s, dim=-1)

                        chosen_id = int(gen_ids[s].item())

                        # top-K
                        topv, topi = torch.topk(logp_s, k=max(k, 1), dim=-1)
                        top_list = []
                        for idx in range(topi.size(0)):
                            tok_id = int(topi[idx].item())
                            lp = float(topv[idx].item())
                            tok_str = self._safe_decode([tok_id])
                            top_list.append({
                                "token_id": tok_id,
                                "token": tok_str,
                                "logprob": lp,
                                "prob": math.exp(lp) if math.isfinite(lp) else 0.0,
                            })
                        chosen_lp = float(logp_s[chosen_id].item()) if chosen_id < logp_s.size(0) else None
                        per_steps.append({
                            "index": s,
                            "token_id": chosen_id,
                            "token": self._safe_decode([chosen_id]),
                            "top_logprobs": sorted(top_list, key=lambda d: d["logprob"], reverse=True),
                            "chosen_logprob": chosen_lp,
                            "chosen_prob": (math.exp(chosen_lp) if (chosen_lp is not None and math.isfinite(chosen_lp)) else None),
                        })
                    extras_all.append({
                        "output_logprobs": per_steps,
                        "prompt_logprobs": None if not include_prompt_logprobs else None,
                    })
                else:
                    extras_all.append(None)

            # Close PIL handles
            for imgs in pil_handles:
                for im in imgs:
                    try:
                        im.close()
                    except Exception:
                        pass

        if return_logprobs:
            return [(t, it, ot, ex) for t, it, ot, ex in zip(outs, in_tok, out_tok, extras_all)]
        else:
            return list(zip(outs, in_tok, out_tok))

    # ---------- helpers ----------
    def _resolve_runtime_model_name(self, model_name: str) -> str:
        if not self._looks_like_gemma4_name(model_name):
            return model_name
        patched = self._create_gemma4_tokenizer_overlay(model_name)
        return patched or model_name

    @staticmethod
    def _looks_like_gemma4_name(model_name: str) -> bool:
        lowered = str(model_name or "").lower()
        return "gemma-4" in lowered or "gemma4" in lowered

    def _create_gemma4_tokenizer_overlay(self, model_name: str) -> Optional[str]:
        if os.path.isdir(str(model_name)):
            return None
        try:
            from huggingface_hub import snapshot_download

            snapshot_dir = snapshot_download(
                repo_id=model_name,
                local_files_only=False,
                resume_download=True,
            )
        except Exception as e:
            warnings.warn(f"Could not prepare Gemma4 tokenizer overlay; using original model id: {e}")
            return None

        tokenizer_config = os.path.join(snapshot_dir, "tokenizer_config.json")
        if not os.path.exists(tokenizer_config):
            return None
        try:
            with open(tokenizer_config, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            warnings.warn(f"Could not read Gemma4 tokenizer_config.json; using original model id: {e}")
            return None

        extra_special_tokens = data.get("extra_special_tokens")
        if not isinstance(extra_special_tokens, list):
            return snapshot_dir

        patched = dict(data)
        patched["extra_special_tokens"] = self._gemma4_extra_special_tokens_dict(
            extra_special_tokens
        )

        hf_home = os.environ.get("HF_HOME") or os.path.join(
            os.path.expanduser("~"), ".cache", "huggingface"
        )
        job_id = os.environ.get("SLURM_JOB_ID") or str(os.getpid())
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "--", str(model_name).strip("/"))
        snapshot_id = os.path.basename(os.path.realpath(snapshot_dir.rstrip(os.sep)))
        overlay_dir = os.path.join(
            hf_home,
            "patched_models",
            "gemma4-tokenizer-config",
            f"{safe_name}-{snapshot_id}-{job_id}",
        )
        tmp_dir = f"{overlay_dir}.tmp.{os.getpid()}"
        try:
            if os.path.exists(overlay_dir):
                return overlay_dir
            shutil.rmtree(tmp_dir, ignore_errors=True)
            os.makedirs(tmp_dir, exist_ok=False)
            for entry in os.listdir(snapshot_dir):
                src = os.path.join(snapshot_dir, entry)
                dst = os.path.join(tmp_dir, entry)
                if entry == "tokenizer_config.json":
                    with open(dst, "w", encoding="utf-8") as f:
                        json.dump(patched, f, ensure_ascii=False, indent=2)
                        f.write("\n")
                else:
                    os.symlink(os.path.realpath(src), dst)
            os.replace(tmp_dir, overlay_dir)
            logger.info("Using Gemma4 tokenizer compatibility overlay: %s", overlay_dir)
            return overlay_dir
        except Exception as e:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            warnings.warn(f"Could not create Gemma4 tokenizer overlay; using original model id: {e}")
            return None

    @staticmethod
    def _gemma4_extra_special_tokens_dict(tokens: List[str]) -> Dict[str, str]:
        result: Dict[str, str] = {}
        for idx, token in enumerate(tokens):
            token_str = str(token)
            lowered = token_str.lower()
            if "video" in lowered and "video_token" not in result:
                key = "video_token"
            elif "image" in lowered and "image_token" not in result:
                key = "image_token"
            elif "audio" in lowered and "audio_token" not in result:
                key = "audio_token"
            else:
                stem = re.sub(r"[^a-z0-9]+", "_", lowered).strip("_") or f"extra_{idx}"
                key = f"{stem}_token"
            base = key
            suffix = 2
            while key in result:
                key = f"{base}_{suffix}"
                suffix += 1
            result[key] = token_str
        return result

    def _load_model_metadata(self, model_name: str, trust_remote_code: bool) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {
            "model_type": "",
            "processor_class": "",
            "architectures": [],
        }
        try:
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
            metadata["model_type"] = str(getattr(config, "model_type", "") or "").strip().lower()
            metadata["architectures"] = list(getattr(config, "architectures", []) or [])
        except Exception:
            pass

        model_dir = str(model_name or "")
        if os.path.isdir(model_dir):
            for filename in ("preprocessor_config.json", "processor_config.json"):
                path = os.path.join(model_dir, filename)
                if not os.path.exists(path):
                    continue
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        raw = json.load(f)
                except Exception:
                    continue
                processor_class = raw.get("processor_class") or raw.get("image_processor_type")
                if processor_class:
                    metadata["processor_class"] = str(processor_class).strip().lower()
                    break
        return metadata

    def _model_hint_strings(self) -> Tuple[str, ...]:
        hints = [self.model_name_full, self._model_type, self._processor_class, *self._architectures]
        return tuple(str(hint).strip().lower() for hint in hints if str(hint).strip())

    def _model_hint_contains(self, *needles: str) -> bool:
        lowered_needles = [str(needle).strip().lower() for needle in needles if str(needle).strip()]
        if not lowered_needles:
            return False
        hints = self._model_hint_strings()
        return any(needle in hint for hint in hints for needle in lowered_needles)

    def _resolve_qwen35_mode(self, requested: Optional[str]) -> str:
        raw = requested
        if raw is None:
            raw = (
                os.environ.get("QWEN35_MODE")
                or os.environ.get("QWEN3_5_MODE")
                or os.environ.get("QWEN3_MODE")
            )

        if raw is None:
            legacy = os.environ.get("QWEN3_THINKING")
            if legacy is not None:
                return "thinking" if str(legacy).strip().lower() in {"1", "true", "yes", "on"} else "instruct_reasoning"
            return "thinking"

        mode = str(raw).strip().lower().replace("-", "_")
        mode = self.QWEN35_MODE_ALIASES.get(mode, mode)
        if mode not in self._qwen_sampling_profiles():
            raise ValueError(
                "Unsupported qwen35_mode="
                f"{raw!r}. Expected one of: {', '.join(sorted(self._qwen_sampling_profiles()))}."
            )
        return mode

    def _apply_qwen35_mode(self, mode: str) -> None:
        config = dict(self._qwen_sampling_profiles()[mode])
        self._qwen3_enable_thinking = bool(config["enable_thinking"])
        self.temperature = float(config["temperature"])
        self.top_p = float(config["top_p"])
        self.top_k = int(config["top_k"])
        self.min_p = float(config["min_p"])
        self.presence_penalty = float(config["presence_penalty"])
        self.repetition_penalty = float(config["repetition_penalty"])
        logger.info(
            "Applied Qwen3.5/3.6 profile mode=%s enable_thinking=%s temperature=%s top_p=%s top_k=%s min_p=%s presence_penalty=%s repetition_penalty=%s",
            mode,
            self._qwen3_enable_thinking,
            self.temperature,
            self.top_p,
            self.top_k,
            self.min_p,
            self.presence_penalty,
            self.repetition_penalty,
        )

    def _qwen_sampling_profiles(self) -> Dict[str, Dict[str, Any]]:
        if self._is_qwen36_family():
            profiles = {
                key: dict(value)
                for key, value in self.QWEN36_SAMPLING_PROFILES.items()
            }
            name = (self.model_name_full or "").lower()
            if "qwen3.6-27b" in name or "qwen3_6-27b" in name:
                profiles["thinking"]["presence_penalty"] = 0.0
            return profiles
        return {
            key: dict(value)
            for key, value in self.QWEN35_SAMPLING_PROFILES.items()
        }

    @classmethod
    def _resolve_vllm_enum(
        cls,
        value: Any,
        *,
        aliases: Dict[str, str],
        enum_cls: Any,
        kind: str,
    ) -> Any:
        raw = str(value).strip()
        if not raw:
            raise ValueError(f"{kind} must not be empty")
        key = raw.lower()
        enum_name = aliases.get(key)
        if enum_name is None and raw.upper() in getattr(enum_cls, "__members__", {}):
            enum_name = raw.upper()
        if enum_name is None:
            valid = ", ".join(sorted(set(aliases.keys())))
            raise ValueError(
                f"Unsupported {kind}={value!r}. Valid aliases include: {valid}"
            )
        return getattr(enum_cls, enum_name)

    def _is_qwen_vl_family(self) -> bool:
        return self._model_hint_contains(
            "qwen2_5_vl",
            "qwen2.5-vl",
            "qwen2_vl",
            "qwen2-vl",
            "qwen3_vl",
            "qwen3-vl",
            "qwen3_5",
            "qwen3.5",
            "qvq",
        )

    def _is_qwen35_family(self) -> bool:
        return self._model_hint_contains("qwen3_5", "qwen3.5")

    def _is_qwen36_family(self) -> bool:
        return self._model_hint_contains("qwen3_6", "qwen3.6")

    def _is_qwen3_family(self) -> bool:
        tpl = getattr(self.tokenizer, "chat_template", "") or ""
        return self._model_hint_contains("qwen3", "qwen3_vl") or ("qwen3" in tpl)

    def _supports_qwen_soft_switch(self) -> bool:
        return (
            self._qwen_soft_switch_enabled
            and self._is_qwen3_family()
            and (not self._is_qwen35_family())
            and (not self._is_qwen36_family())
        )

    def _has_qwen_vision_chat_template(self) -> bool:
        tpl = getattr(self.tokenizer, "chat_template", "") or ""
        return all(tok in tpl for tok in ("<|vision_start|>", "<|image_pad|>", "<|vision_end|>"))

    def _prepend_no_think_to_structured_chat(self, chat: List[Dict[str, Any]]) -> bool:
        if not chat:
            return False
        user_turn = chat[-1]
        content = user_turn.get("content")
        if not isinstance(content, list):
            return False

        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "text":
                continue
            text = str(item.get("text", ""))
            if text.lstrip().startswith("/no_think"):
                return True
            item["text"] = "/no_think\n" + text
            return True

        content.append({"type": "text", "text": "/no_think"})
        return True

    def _prepend_gemma4_think_token(self, sys_msg: str) -> str:
        sys_msg = (sys_msg or "").strip()
        if sys_msg.startswith("<|think|>"):
            return sys_msg
        return "<|think|>" if not sys_msg else f"<|think|>\n{sys_msg}"

    def _prepend_think_to_structured_system(self, chat: List[Dict[str, Any]]) -> bool:
        if chat and chat[0].get("role") == "system":
            content = chat[0].get("content")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        item["text"] = self._prepend_gemma4_think_token(str(item.get("text", "")))
                        return True
                content.insert(0, {"type": "text", "text": "<|think|>"})
                return True
            if isinstance(content, str):
                chat[0]["content"] = self._prepend_gemma4_think_token(content)
                return True

        chat.insert(0, {"role": "system", "content": [{"type": "text", "text": "<|think|>"}]})
        return True

    def _is_gemma_family(self) -> bool:
        return self._model_hint_contains("gemma-3", "gemma3", "gemma-4", "gemma4")

    def _is_gemma4_family(self) -> bool:
        return self._model_hint_contains("gemma-4", "gemma4")

    def _is_medgemma_family(self) -> bool:
        return self._model_hint_contains("medgemma")

    def _detect_is_internvl(self) -> bool:
        return self._model_hint_contains("internvl3_5", "internvl3", "internvl")
    # or ("medgemma" in name)

    def _detect_is_internvl_hf_variant(self) -> bool:
        name = (self.model_name_full or "").lower()
        if not self._detect_is_internvl():
            return False
        return ("-hf" in name) or name.endswith("/hf")

    def _detect_is_lingshu(self) -> bool:
        return self._model_hint_contains("lingshu-medical-mllm/lingshu", "lingshu")

    def _ensure_hf_internvl_ready(self):
        if (self._hf_processor is not None) and (self._hf_model is not None):
            return
        # Official: AutoProcessor + AutoModelForImageTextToText (preferred)
        try:
            self._hf_processor = AutoProcessor.from_pretrained(
                self.model_name_full,
                trust_remote_code=True,
            )
        except Exception as e:
            msg = str(e)
            if ("start_image_token" in msg) or ("Qwen2TokenizerFast has no attribute" in msg):
                raise RuntimeError(
                    "InternVL legacy checkpoint is not fully compatible with the "
                    "Transformers AutoProcessor path in this environment. "
                    "Use an official '-HF' checkpoint (e.g., OpenGVLab/InternVL3_5-1B-HF) "
                    "or run through vLLM-compatible path."
                ) from e
            raise

        # Try ImageText2Text → fallback to Vision2Seq if it fails
        last_err = None
        for ctor in tuple(c for c in (AutoModelForImageTextToText, AutoModelForVision2Seq) if c is not None):
            try:
                self._hf_model = ctor.from_pretrained(
                    self.model_name_full,
                    device_map="auto" if torch.cuda.is_available() else None,
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                    trust_remote_code=True,
                )
                self._hf_model.eval()
                return
            except TypeError as e:
                # Compatibility for older transformers: no device_map parameter
                try:
                    self._hf_model = ctor.from_pretrained(
                        self.model_name_full,
                        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                        trust_remote_code=True,
                    )
                    if torch.cuda.is_available():
                        self._hf_model = self._hf_model.cuda()
                    self._hf_model.eval()
                    return
                except Exception as e2:
                    last_err = e2
            except Exception as e:
                last_err = e
        raise RuntimeError(f"Failed to load InternVL HF model with both heads: {last_err}")

    def _ensure_mm_chat_processor_ready(self):
        if self._mm_chat_processor is not None:
            return
        self._mm_chat_processor = AutoProcessor.from_pretrained(
            self.model_name_full,
            trust_remote_code=True,
        )

    def _is_non_instruct_model(self) -> bool:
        name = (self.model_name_full or "").lower()
        if name in self.NON_INSTRUCT_MODELS:
            return True
        tpl = getattr(self.tokenizer, "chat_template", None)
        return not bool(tpl)

    def _should_use_chat_template(self) -> bool:
        if self._force_plain_text:
            return False
        if not hasattr(self.tokenizer, "apply_chat_template"):
            return False
        tpl = getattr(self.tokenizer, "chat_template", None)
        return bool(tpl)

    def _compose_plain_prompt(self, sys_msg: str, user_msg: str) -> str:
        sys_msg = (sys_msg or "").strip()
        if sys_msg:
            return f"{sys_msg}\n\n{user_msg}"
        return user_msg

    def _count_tokens(self, text) -> int:
        if not isinstance(text, str):
            try:
                text = json.dumps(text, ensure_ascii=False)
            except Exception:
                text = str(text)
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _looks_like_visible_reasoning(self, txt: str) -> bool:
        s = str(txt or "").strip()
        if not s:
            return False
        lowered = s.lower()
        reasoning_markers = (
            "got it",
            "let's",
            "let me",
            "i need to",
            "i should",
            "i'll",
            "the user",
            "the user wants",
            "the user wants me",
            "wait,",
            "wait ",
            "we need to",
            "task:",
            "case summary:",
            "image analysis:",
            "drafting the",
            "refining the",
        )
        if any(marker in lowered for marker in reasoning_markers):
            return True
        # Final caption/findings answers should usually be short. If a Qwen
        # thinking run emits an unusually long completion without </think>, it
        # is very likely visible chain-of-thought rather than the final answer.
        return self._count_tokens(s) >= 256

    def _recover_qwen_thinking_answer(
        self,
        request: Dict[str, Any],
        generated_text: str,
    ) -> Tuple[str, int]:
        text = str(generated_text or "")
        if (not self._is_qwen3_family()) or (not self._qwen3_enable_thinking):
            return text, 0
        if self._is_qwen35_family():
            return text, 0
        if "</think>" in text or (not self._looks_like_visible_reasoning(text)):
            return text, 0

        try:
            from vllm import SamplingParams

            recovery_prompt = str(request.get("prompt") or "")
            recovery_prompt += text
            if not recovery_prompt.endswith(("\n", " ", "\t")):
                recovery_prompt += "\n"
            recovery_prompt += (
                "</think>\n"
                "<|im_end|>\n"
                "<|im_start|>user\n"
                "/no_think\n"
                "Provide only the final answer in plain text, without any reasoning.\n"
                "<|im_end|>\n"
                "<|im_start|>assistant\n"
            )

            recovery_req: Dict[str, Any] = {"prompt": recovery_prompt}
            if "multi_modal_data" in request:
                recovery_req["multi_modal_data"] = request["multi_modal_data"]

            logger.info("Recovering visible Qwen thinking output with a /no_think follow-up turn.")
            recovery_sp = SamplingParams(
                max_tokens=min(512, max(64, self.max_tokens)),
                temperature=0.0,
                top_p=1.0,
                top_k=0,
                min_p=0.0,
                presence_penalty=0.0,
                repetition_penalty=1.0,
                stop=["<|im_end|>"],
            )
            recovery_result = self.llm.generate([recovery_req], recovery_sp)[0]
            recovery_text = str(recovery_result.outputs[0].text or "")
            recovery_text = recovery_text.strip()
            if not recovery_text:
                return text, 0

            combined = f"<think>{text}</think>{recovery_text}"
            return combined, self._count_tokens(recovery_text)
        except Exception as exc:
            warnings.warn(
                "Qwen thinking answer recovery failed; keeping original visible reasoning. "
                f"Details: {exc}"
            )
            return text, 0

    def _extract_final_visible_answer(self, txt: str) -> str:
        s = str(txt or "").strip()
        if not s:
            return s

        paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", s) if p and p.strip()]
        if not paragraphs:
            return s

        bad_prefixes = (
            "got it",
            "let's",
            "let me",
            "wait",
            "modality:",
            "view:",
            "key findings:",
            "constraint checklist",
            "drafting the",
            "refining the",
            "final check",
            "case summary:",
            "image analysis:",
            "synthesizing findings:",
        )

        for para in reversed(paragraphs):
            cleaned = para.strip().strip('"').strip("'").strip()
            if not cleaned:
                continue
            lowered = cleaned.lower()
            if lowered.startswith(bad_prefixes):
                continue
            if re.match(r"^[-*]\s", cleaned):
                continue
            if re.match(r"^\d+\.\s", cleaned):
                continue
            if self._count_tokens(cleaned) < 6:
                continue
            return cleaned

        last = paragraphs[-1].strip().strip('"').strip("'").strip()
        return last or s

    def _postprocess(self, txt: str):
        s = html.unescape(txt or "")
        m = re.search(r"(?is)<\s*answer\b[^>]*>\s*(.*?)\s*(?:</\s*answer\s*>|$)", s)
        if m:
            return m.group(1).strip()
        s = re.sub(
            r"(?is)<\s*think\b[^>]*>.*?(?=</\s*think\s*>|<\s*answer\b|$)",
            "", s,
        )
        s = re.sub(r"(?is)</\s*think\s*>", "", s)
        s = re.sub(
            r"(?is)&lt;\s*think\b[^&]*&gt;.*?(?=&lt;/\s*think\s*&gt;|&lt;\s*answer\b|$)",
            "", s,
        )
        s = re.sub(r"(?is)&lt;/\s*think\s*&gt;", "", s)
        s = re.sub(r"(?is)<\|channel\>thought\s*.*?<channel\|>", "", s)
        s = re.sub(r"(?is)<\|im_end\|>", "", s)
        s = re.sub(r"(?is)<\|endoftext\|>", "", s)
        s = s.strip()
        if self._is_qwen3_family() and self._qwen3_enable_thinking and self._looks_like_visible_reasoning(s):
            extracted = self._extract_final_visible_answer(s)
            extracted = extracted.strip()
            if extracted and extracted != s:
                logger.info("Extracted final answer from visible Qwen thinking output.")
                return extracted
        return s

    def _detect_image_placeholder(self):
        tpl = getattr(self.tokenizer, "chat_template", "") or ""
        name = (self.model_name_full or "").lower()

        qwen_has_vision_tokens = (
            "<|vision_start|>" in tpl and "<|image_pad|>" in tpl and "<|vision_end|>" in tpl
        )

        if self._is_internvl_hf_variant:
            # vLLM maps InternVLForConditionalGeneration to InternS1 backend.
            # That processor expects "<IMG_CONTEXT>" placeholders.
            return {
                "family": "internvl-hf",
                "gen_block": lambda n: "".join("<IMG_CONTEXT>\n" for _ in range(n)),
                "append_newline": False,
            }

        if self._is_lingshu and qwen_has_vision_tokens:
            # Lingshu model card uses Qwen-VL vision token triplets without an
            # extra newline between placeholder block and text.
            return {
                "family": "lingshu",
                "gen_block": lambda n: "<|vision_start|><|image_pad|><|vision_end|>" * n,
                "append_newline": False,
            }

        if ("internvl3" in name) or ("internvl3_5" in name) or ("internvl" in name) or ("<IMG_CONTEXT>" in tpl):
            return {"family": "internvl", "gen_block": lambda n: "\n".join("<image>" for _ in range(n))}

        # MedGemma 1.5 expects explicit image wrapper + soft image token.
        if "medgemma" in name:
            return {
                "family": "medgemma",
                "gen_block": lambda n: "<start_of_image><image_soft_token><end_of_image>" * n,
            }

        if (
            "gemma-3" in name
            or "gemma-4" in name
            or "gemma3" in tpl
            or "gemma4" in tpl
            or "<start_of_image>" in tpl
        ):
            return {"family": "gemma", "gen_block": lambda n: "<start_of_image>" * n}

        if qwen_has_vision_tokens and (self._is_qwen_vl_family() or self._model_hint_contains("qwen")):
            # Qwen-VL expects one full vision token triplet per image.
            return {
                "family": "qwen-vl",
                "gen_block": lambda n: "<|vision_start|><|image_pad|><|vision_end|>" * n,
            }

        if "minicpm" in name:
            return {"family": "minicpm", "gen_block": lambda n: "(<image>./</image>)" * n}

        if "phi" in name:
            return {"family": "phi3", "gen_block": lambda n: "".join(f"<|image_{i+1}|>" for i in range(n))}

        return {"family": "default", "gen_block": lambda n: "<image>" * n}

    def _inject_image_placeholders(self, user_msg: str, img_paths: List[str]) -> str:
        if not img_paths:
            return user_msg
        ph = self._detect_image_placeholder()
        block = ph["gen_block"](len(img_paths))
        append_newline = bool(ph.get("append_newline", True))
        if not append_newline:
            return f"{block}{user_msg}"
        if user_msg.startswith("\n"):
            return f"{block}{user_msg}"
        return f"{block}\n{user_msg}"

    def _is_qwen3(self) -> bool:
        return self._is_qwen3_family()

    # --------- vLLM: format logprobs (compatible with different vLLM versions) ---------
    def _format_output_logprobs(self, out_item) -> List[Dict[str, Any]]:
        lp = getattr(out_item, "logprobs", None)
        token_ids = list(getattr(out_item, "token_ids", []) or [])
        if lp is None:
            return []

        def _val_to_float(v) -> Optional[float]:
            if isinstance(v, (int, float)):
                return float(v)
            for attr in ("logprob", "log_prob", "value"):
                if hasattr(v, attr):
                    try:
                        return float(getattr(v, attr))
                    except Exception:
                        pass
            try:
                return float(v)  # fallback
            except Exception:
                return None

        formatted = []
        for i, step in enumerate(lp):
            entry: Dict[str, Any] = {"index": i}
            chosen_id = token_ids[i] if i < len(token_ids) else None
            if chosen_id is not None:
                entry["token_id"] = int(chosen_id)
                entry["token"] = self._safe_decode([chosen_id])

            top_list: List[Dict[str, Any]] = []

            if isinstance(step, dict):
                for k, v in step.items():
                    logp = _val_to_float(v)
                    if logp is None:
                        continue
                    if isinstance(k, int):
                        tok_id = k
                        tok_str = self._safe_decode([tok_id])
                    else:
                        tok_id = None
                        tok_str = str(k)
                    top_list.append({
                        "token_id": tok_id,
                        "token": tok_str,
                        "logprob": logp,
                        "prob": math.exp(logp) if math.isfinite(logp) else 0.0,
                    })

            else:
                ids = getattr(step, "logprob_token_ids", None) or getattr(step, "top_logprob_token_ids", None)
                vals = getattr(step, "logprobs", None)
                decs = getattr(step, "decoded_tokens", None)
                if ids is not None and vals is not None:
                    for idx, logp in enumerate(list(vals)):
                        tok_id = int(ids[idx]) if idx < len(ids) else None
                        tok_str = None
                        if decs is not None and idx < len(decs) and decs[idx] is not None:
                            tok_str = str(decs[idx])
                        elif tok_id is not None:
                            tok_str = self._safe_decode([tok_id])
                        if tok_str is None:
                            tok_str = ""
                        flp = float(logp)
                        top_list.append({
                            "token_id": tok_id,
                            "token": tok_str,
                            "logprob": flp,
                            "prob": math.exp(flp) if math.isfinite(flp) else 0.0,
                        })

            top_list.sort(key=lambda d: d["logprob"], reverse=True)
            entry["top_logprobs"] = top_list

            if chosen_id is not None:
                chosen = next((d for d in top_list if d.get("token_id") == chosen_id), None)
                if chosen is not None:
                    entry["chosen_logprob"] = chosen["logprob"]
                    entry["chosen_prob"] = chosen["prob"]
                else:
                    entry["chosen_logprob"] = None
                    entry["chosen_prob"] = None

            formatted.append(entry)

        return formatted

    def _format_prompt_logprobs(self, req_output) -> List[Dict[str, Any]]:
        lp = getattr(req_output, "prompt_logprobs", None)
        prompt_ids = list(getattr(req_output, "prompt_token_ids", []) or [])
        if lp is None:
            return []

        formatted = []
        for i, step in enumerate(list(lp)):
            entry: Dict[str, Any] = {"index": i}
            tok_id = prompt_ids[i] if i < len(prompt_ids) else None
            if tok_id is not None:
                entry["token_id"] = int(tok_id)
                entry["token"] = self._safe_decode([tok_id])

            top_list: List[Dict[str, Any]] = []

            if isinstance(step, dict):
                for k, v in step.items():
                    logp = None
                    if isinstance(v, (int, float)):
                        logp = float(v)
                    else:
                        for attr in ("logprob", "log_prob", "value"):
                            if hasattr(v, attr):
                                try:
                                    logp = float(getattr(v, attr))
                                    break
                                except Exception:
                                    pass
                    if logp is None:
                        continue
                    if isinstance(k, int):
                        kid = k
                        kstr = self._safe_decode([kid])
                    else:
                        kid = None
                        kstr = str(k)
                    top_list.append({
                        "token_id": kid,
                        "token": kstr,
                        "logprob": logp,
                        "prob": math.exp(logp) if math.isfinite(logp) else 0.0,
                    })
            else:
                ids = getattr(step, "logprob_token_ids", None) or getattr(step, "top_logprob_token_ids", None)
                vals = getattr(step, "logprobs", None)
                decs = getattr(step, "decoded_tokens", None)
                if ids is not None and vals is not None:
                    for idx, logp in enumerate(list(vals)):
                        kid = int(ids[idx]) if idx < len(ids) else None
                        kstr = None
                        if decs is not None and idx < len(decs) and decs[idx] is not None:
                            kstr = str(decs[idx])
                        elif kid is not None:
                            kstr = self._safe_decode([kid])
                        if kstr is None:
                            kstr = ""
                        flp = float(logp)
                        top_list.append({
                            "token_id": kid,
                            "token": kstr,
                            "logprob": flp,
                            "prob": math.exp(flp) if math.isfinite(flp) else 0.0,
                        })

            top_list.sort(key=lambda d: d["logprob"], reverse=True)
            entry["top_logprobs"] = top_list

            if tok_id is not None:
                chosen = next((d for d in top_list if d.get("token_id") == tok_id), None)
                entry["chosen_logprob"] = (chosen["logprob"] if chosen else None)
                entry["chosen_prob"] = (chosen["prob"] if chosen else None)

            formatted.append(entry)

        return formatted

    def _safe_decode(self, ids: List[int]) -> str:
        try:
            return self.tokenizer.decode(ids, skip_special_tokens=False)
        except Exception:
            try:
                return self.tokenizer.decode(ids)
            except Exception:
                return ""

    def _safe_decode_ids_hf(self, ids_tensor: torch.Tensor) -> str:
        try:
            return self.tokenizer.decode(ids_tensor.tolist(), skip_special_tokens=True)
        except Exception:
            return ""


    def _align_mm_inputs_dtype_device(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Move tensors to the model's device.
        Only cast floating-point image tensors (pixel_values) to the model parameter dtype (usually bf16).
        Do not change dtype for other tensors (e.g., input_ids/attention_mask).
        """
        device = next(self._hf_model.parameters()).device
        model_dtype = next(self._hf_model.parameters()).dtype

        for k, v in list(inputs.items()):
            if torch.is_tensor(v):
                # Align dtype only for pixel_values (or keys containing 'pixel_values')
                if v.dtype.is_floating_point and ("pixel_values" in k):
                    inputs[k] = v.to(device=device, dtype=model_dtype, non_blocking=True)
                else:
                    inputs[k] = v.to(device=device, non_blocking=True)
        return inputs
