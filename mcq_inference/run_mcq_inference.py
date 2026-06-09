#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcq_inference.runner import run_inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MCQ inference on the standardized smoke-test manifest.")
    parser.add_argument(
        "--manifest",
        default="artifacts/mcq_smoke_samples/seed_7_n10/examples.jsonl",
        help="Path to the standardized MCQ manifest JSONL.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write predictions.jsonl and summary.json.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name. Examples: OpenAI/gpt-5.4-nano or Qwen/Qwen3.5-9B.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size passed to the model wrapper.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional cap on the number of manifest rows to run.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional dataset filter. Uses manifest dataset keys.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Optional maximum completion tokens. If omitted, the backend default is used.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Optional sampling temperature override. If omitted, backend defaults are used.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Optional nucleus sampling top_p override for the vLLM backend.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Optional top_k override for the vLLM backend.",
    )
    parser.add_argument(
        "--min-p",
        type=float,
        default=None,
        help="Optional min_p override for the vLLM backend.",
    )
    parser.add_argument(
        "--presence-penalty",
        type=float,
        default=None,
        help="Optional presence penalty override for the vLLM backend.",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=None,
        help="Optional repetition penalty override for the vLLM backend.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=8,
        help="Concurrency for the OpenAI backend.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional max image count passed to the vLLM wrapper as limit_mm_per_prompt.image.",
    )
    parser.add_argument(
        "--vllm-max-num-batched-tokens",
        type=int,
        default=None,
        help="Optional override for vLLM max_num_batched_tokens.",
    )
    parser.add_argument(
        "--vllm-max-model-len",
        type=int,
        default=None,
        help="Optional override for vLLM max_model_len.",
    )
    parser.add_argument(
        "--vllm-gpu-memory-utilization",
        type=float,
        default=None,
        help="Optional override for vLLM gpu_memory_utilization.",
    )
    parser.add_argument(
        "--prompt-variant",
        default="normal",
        choices=["normal", "text_only", "options_only", "image_options_only"],
        help="Keep the same prompt template but optionally remove images or question text.",
    )
    parser.add_argument(
        "--duplicate-options-in-question",
        action="store_true",
        help="Append an inline (A) ... (B) ... options block to the question text before the explicit Options section.",
    )
    parser.add_argument(
        "--no-structured-output",
        action="store_true",
        help="Do not pass a Pydantic schema or guided decoding config to the model backend; rely on the prompt only.",
    )
    parser.add_argument(
        "--request-chunk-size",
        type=int,
        default=None,
        help="Optional vLLM request chunk size. When set below the dataset size, submits multiple generate() calls and checkpoints after each chunk.",
    )
    parser.add_argument(
        "--qwen35-mode",
        default=None,
        choices=[
            "thinking",
            "thinking_precise_coding",
            "instruct_general",
            "instruct_reasoning",
            "instruct_reasoning_temp0",
        ],
        help="Qwen3.5 family sampling profile. Aligns with the official Hugging Face recommendations.",
    )
    parser.add_argument(
        "--gemma4-thinking",
        action="store_true",
        help="Enable Gemma 4 thinking by prefixing the system prompt with the official <|think|> control token.",
    )
    parser.add_argument(
        "--vllm-timeout-seconds",
        type=int,
        default=None,
        help="Optional override for VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS.",
    )
    parser.add_argument(
        "--vllm-enforce-eager",
        action="store_true",
        help="Run vLLM in eager mode, disabling torch.compile and CUDAGraphs.",
    )
    parser.add_argument(
        "--vllm-compilation-mode",
        default=None,
        choices=[
            "none",
            "stock_torch_compile",
            "dynamo_trace_once",
            "vllm_compile",
        ],
        help="Optional vLLM compilation mode override.",
    )
    parser.add_argument(
        "--vllm-cudagraph-mode",
        default=None,
        choices=[
            "none",
            "piecewise",
            "full",
            "full_decode_only",
            "full_and_piecewise",
        ],
        help="Optional vLLM CUDAGraph mode override.",
    )
    parser.add_argument(
        "--vllm-enable-chunked-prefill",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Optional vLLM chunked prefill override.",
    )
    parser.add_argument(
        "--vllm-async-scheduling",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Optional vLLM async scheduling override.",
    )
    parser.add_argument(
        "--vllm-enable-prefix-caching",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Optional vLLM prefix caching override.",
    )
    parser.add_argument(
        "--vllm-mm-processor-cache-gb",
        type=float,
        default=None,
        help="Optional vLLM multimodal processor cache size override.",
    )
    parser.add_argument(
        "--vllm-mm-processor-cache-type",
        default=None,
        choices=["lru", "shm"],
        help="Optional vLLM multimodal processor cache type override.",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    if args.vllm_timeout_seconds is not None:
        os.environ["VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS"] = str(
            int(args.vllm_timeout_seconds)
        )
    results_path, summary_path = run_inference(
        manifest_path=Path(args.manifest).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        model_name=args.model,
        batch_size=int(args.batch_size),
        max_examples=args.max_examples,
        dataset_filter=set(args.datasets) if args.datasets else None,
        max_tokens=(int(args.max_tokens) if args.max_tokens is not None else None),
        temperature=(float(args.temperature) if args.temperature is not None else None),
        top_p=(float(args.top_p) if args.top_p is not None else None),
        top_k=(int(args.top_k) if args.top_k is not None else None),
        min_p=(float(args.min_p) if args.min_p is not None else None),
        presence_penalty=(
            float(args.presence_penalty)
            if args.presence_penalty is not None
            else None
        ),
        repetition_penalty=(
            float(args.repetition_penalty)
            if args.repetition_penalty is not None
            else None
        ),
        max_concurrency=int(args.max_concurrency),
        max_images=(int(args.max_images) if args.max_images is not None else None),
        vllm_max_model_len=(
            int(args.vllm_max_model_len)
            if args.vllm_max_model_len is not None
            else None
        ),
        vllm_max_num_batched_tokens=(
            int(args.vllm_max_num_batched_tokens)
            if args.vllm_max_num_batched_tokens is not None
            else None
        ),
        vllm_gpu_memory_utilization=(
            float(args.vllm_gpu_memory_utilization)
            if args.vllm_gpu_memory_utilization is not None
            else None
        ),
        qwen35_mode=args.qwen35_mode,
        gemma4_thinking=bool(args.gemma4_thinking),
        vllm_enforce_eager=bool(args.vllm_enforce_eager),
        vllm_compilation_mode=args.vllm_compilation_mode,
        vllm_cudagraph_mode=args.vllm_cudagraph_mode,
        vllm_enable_chunked_prefill=args.vllm_enable_chunked_prefill,
        vllm_async_scheduling=args.vllm_async_scheduling,
        vllm_enable_prefix_caching=args.vllm_enable_prefix_caching,
        vllm_mm_processor_cache_gb=args.vllm_mm_processor_cache_gb,
        vllm_mm_processor_cache_type=args.vllm_mm_processor_cache_type,
        prompt_variant=args.prompt_variant,
        duplicate_options_in_question=bool(args.duplicate_options_in_question),
        request_chunk_size=args.request_chunk_size,
        use_structured_output=(not bool(args.no_structured_output)),
    )
    print(
        json.dumps(
            {
                "results_path": str(results_path),
                "summary_path": str(summary_path),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
