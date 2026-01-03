#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run spr1_generation spr1_loop using an existing sglang OpenAI-compatible server.
Only supports --no-launch mode.
"""

import argparse
import asyncio
import importlib
import inspect
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

import ray
import requests
from omegaconf import OmegaConf, DictConfig


# ----------------------------
# Bootstrap sys.path so that `import verl...` works
# ----------------------------
_THIS_FILE = Path(__file__).resolve()
# .../verl/verl/spr1_generation/test.py -> parents:
# 0 spr1_generation, 1 verl (python package root), 2 repo root (/root/zzx/verl)
_REPO_ROOT = _THIS_FILE.parents[2]
_PKG_ROOT = _THIS_FILE.parents[1]

# Ensure repo root is on sys.path so `import verl` succeeds.
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
# Also ensure package root is on sys.path so `import spr1_generation` works robustly.
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))


# ----------------------------
# Test config
# ----------------------------
MODEL_PATH = "/data2/gjr/models/Qwen2.5-3B-Instruct"

PROMPT_TEXT = """You are a helpful and harmless assistant.
<system instruction>
When the user asks a question, the assistant should actively solve it. The assistant may think, tool_call, reflect, and then produce a final answer. Use the following structured tags to organize reasoning and tool_call steps. Precise formatting matters — follow the rules below.

Tags (semantic roles)
1. <think> ... </think>
2. <tool_call> ... </tool_call>
   - Use when the assistant must perform external or uncertain information retrieval.
   - The content must follow this exact format:
     "<tool_call> keyword_1 keyword_2 ... </tool_call>"
   - After sending the tool_call tag, the system/tool will return results wrapped in an "<tool_response> ... </tool_response>" block.
3. <reflect> ... </reflect>
4. <answer> ... </answer>
   - This tag must appear exactly once and must be placed at the very end of the response.
</system instruction>

<query>
Where did Khanzada Begum's father die?
</query>
""".strip()


# ----------------------------
# Minimal TokenOutput-like object (duck typing)
# ----------------------------
@dataclass
class TokenOutputLike:
    output_ids: list[int]
    token_ids: list[int]
    text: str
    logprobs: Optional[list[float]] = None
    routed_experts: Optional[Any] = None
    finish_reason: Optional[str] = None


def wait_http_ready(url: str, timeout_s: int = 60) -> None:
    t0 = time.time()
    last_err = None
    while time.time() - t0 < timeout_s:
        try:
            r = requests.get(url, timeout=2)
            if r.status_code < 500:
                return
        except Exception as e:
            last_err = e
        time.sleep(0.5)
    raise RuntimeError(f"Server not ready: {url}, last_err={last_err}")


def get_openai_model_id(base_url: str) -> str:
    r = requests.get(f"{base_url}/models", timeout=10)
    r.raise_for_status()
    data = r.json()
    return data["data"][0]["id"]


@ray.remote
class SGLangOpenAIActor:
    """
    Ray Actor that adapts OpenAI-compatible /v1/completions into the interface
    expected by AsyncLLMServerManager.generate(request_id, prompt_ids, sampling_params, image_data).
    """
    def __init__(self, base_url: str, model_id: str, tokenizer_path: str):
        from transformers import AutoTokenizer
        self.base_url = base_url.rstrip("/")
        self.model_id = model_id
        self.tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    async def generate(
        self,
        *,
        request_id: str,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
    ) -> TokenOutputLike:
        prompt_text = self.tok.decode(prompt_ids, skip_special_tokens=False)

        max_tokens = (
            sampling_params.get("max_tokens")
            or sampling_params.get("max_new_tokens")
            or sampling_params.get("max_new_token")
            or 512
        )
        temperature = float(sampling_params.get("temperature", 0.7))
        top_p = float(sampling_params.get("top_p", 0.9))
        stop = sampling_params.get("stop")

        payload = {
            "model": self.model_id,
            "prompt": prompt_text,
            "max_tokens": int(max_tokens),
            "temperature": temperature,
            "top_p": top_p,
            "stream": False,
        }
        if stop is not None:
            payload["stop"] = stop

        r = requests.post(f"{self.base_url}/completions", json=payload, timeout=300)
        r.raise_for_status()
        data = r.json()

        text = data["choices"][0]["text"]
        finish_reason = data["choices"][0].get("finish_reason")

        out_ids = self.tok.encode(text, add_special_tokens=False)
        return TokenOutputLike(
            output_ids=out_ids,
            token_ids=out_ids,
            text=text,
            logprobs=None,
            routed_experts=None,
            finish_reason=finish_reason,
        )


def build_minimal_config(model_path: str) -> DictConfig:
    """
    Minimal config to satisfy:
      - spr1_loop.init_class needs rollout.prompt_length
      - rollout postprocess/padding needs rollout.response_length
      - some shared utilities may read rollout.temperature/top_p/calculate_log_probs
      - tracing may read trainer.project_name / trainer.experiment_name
    """
    cfg = {
        "trainer": {
            "project_name": "spr1_debug",
            "experiment_name": "spr1_debug_local",
        },
        "data": {
            "apply_chat_template_kwargs": {
                # "add_generation_prompt": True
            }
        },

        "actor_rollout_ref": {
            "model": {
                "path": model_path,
            },
            "rollout": {
                # --- REQUIRED by your spr1_loop.init_class ---
                "prompt_length": 1024,     # 你可按需要调整
                # --- REQUIRED by padding/postprocess ---
                "response_length": 768,    # 你现在默认 max_new_tokens=768，保持一致即可
                # --- common rollout knobs (safe defaults) ---
                "temperature": 0.7,
                "top_p": 0.9,
                "calculate_log_probs": False,
                "val_kwargs": {"temperature": 0.7, "top_p": 0.9},
                "agent": {
                    "agent_loop_config_path": None,
                },
                # trace 字段可选；不给也行
                "trace": {
                    "backend": None,
                    "token2text": False,
                    "max_samples_per_step_per_worker": None,
                },
            },
        },
        "reward_model": {
            "use_reward_loop": False,
            # 有的路径会读 enable/enable_resource_pool，给默认更稳
            "enable": False,
            "enable_resource_pool": False,
        },
    }
    return OmegaConf.create(cfg)



def _try_import(modname: str):
    try:
        return importlib.import_module(modname)
    except Exception as e:
        print(f"[IMPORT-FAIL] {modname}: {type(e).__name__}: {e}", flush=True)
        return None


def _pick_spr1_entry(mod) -> Tuple[str, Any]:
    """
    Find a usable spr1 loop entry in module:
      - Prefer a class whose name contains 'spr1' or ends with 'loop' and has method 'run'
      - Otherwise accept a callable function entrypoint
    """
    # 1) classes
    for name, obj in inspect.getmembers(mod, inspect.isclass):
        lname = name.lower()
        if ("spr1" in lname) or (lname.endswith("loop")):
            if hasattr(obj, "run"):
                return ("class", obj)

    # 2) common function entrypoints
    for fname in ["spr1_loop", "run", "amain", "main", "rollout", "generate"]:
        if hasattr(mod, fname):
            f = getattr(mod, fname)
            if callable(f):
                return ("func", f)

    # 3) fallback: any class with run
    for name, obj in inspect.getmembers(mod, inspect.isclass):
        if hasattr(obj, "run"):
            return ("class", obj)

    raise RuntimeError(f"Cannot find spr1 entry in module {mod.__name__}")


def autodiscover_spr1() -> Tuple[str, Any, str]:
    """
    Deterministic discovery for your known layout:
      spr1_generation/spr1_loop.py
      spr1_generation/agent_loop.py
    """
    candidates = [
        "spr1_generation.spr1_loop",
        "spr1_generation.agent_loop",
    ]
    for m in candidates:
        mod = _try_import(m)
        if mod is None:
            continue
        kind, entry = _pick_spr1_entry(mod)
        return kind, entry, m

    raise RuntimeError(f"Failed to import spr1 modules. Tried: {candidates}")


async def run_spr1_once(
    spr1_kind: str,
    spr1_entry: Any,
    server_handles,
    tokenizer_path: str,
    prompt_text: str,
    sampling_params: dict[str, Any],
):
    # These imports require repo root on sys.path, which we ensured at the top.
    from verl.experimental.agent_loop.agent_loop import AsyncLLMServerManager, _DummyConfig
    from verl.utils import hf_tokenizer, hf_processor

    cfg = build_minimal_config(tokenizer_path)
    tokenizer = hf_tokenizer(tokenizer_path, trust_remote_code=True)
    processor = hf_processor(tokenizer_path, trust_remote_code=True)

    server_manager = AsyncLLMServerManager(cfg, server_handles)
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

    if spr1_kind == "class":
        loop = spr1_entry(
            trainer_config=_DummyConfig(cfg),
            server_manager=server_manager,
            tokenizer=tokenizer,
            processor=processor,
        )
        out = loop.run(
            sampling_params=sampling_params,
            prompt_ids=prompt_ids,
            raw_prompt=
            [{
                "role": "user",
                "content": prompt_text,
            }]
            # prompt_text=prompt_text,
        )
        if inspect.isawaitable(out):
            out = await out
        return out

    if spr1_kind == "func":
        kwargs = dict(
            sampling_params=sampling_params,
            prompt_ids=prompt_ids,
            # prompt_text=prompt_text,
            raw_prompt=[{
                "role": "user",
                "content": prompt_text,
            }],
            server_manager=server_manager,
            tokenizer=tokenizer,
            processor=processor,
            trainer_config=_DummyConfig(cfg),
        )
        try:
            sig = inspect.signature(spr1_entry)
            kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        except Exception:
            pass

        out = spr1_entry(**kwargs)
        if inspect.isawaitable(out):
            out = await out
        return out

    raise RuntimeError(f"Unknown spr1_kind={spr1_kind}")


async def amain():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-launch", action="store_true", default=True,
                    help="Only --no-launch is supported. This script will NOT start sglang.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--ray-address", default=None, help="Ray cluster address; default starts local Ray.")
    ap.add_argument("--model-path", default=MODEL_PATH)
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    args = ap.parse_args()

    if not args.no_launch:
        raise RuntimeError("This test script supports only --no-launch mode.")

    # Ray
    if args.ray_address:
        ray.init(address=args.ray_address)
    else:
        ray.init(ignore_reinit_error=True)

    base_url = f"http://{args.host}:{args.port}/v1"
    try:
        # Ensure sglang already up
        wait_http_ready(f"{base_url}/models", timeout_s=60)
        print(f"[BOOT] using existing sglang: {base_url}", flush=True)

        model_id = get_openai_model_id(base_url)
        print(f"[BOOT] sglang model_id={model_id}", flush=True)

        # Backend handle(s)
        backend = SGLangOpenAIActor.remote(
            base_url=base_url,
            model_id=model_id,
            tokenizer_path=args.model_path,
        )
        server_handles = [backend]

        # Discover spr1 loop
        spr1_kind, spr1_entry, src = autodiscover_spr1()
        print(f"[BOOT] spr1 entry from {src}: kind={spr1_kind}, entry={spr1_entry}", flush=True)

        sampling_params = {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
        }

        out = await run_spr1_once(
            spr1_kind=spr1_kind,
            spr1_entry=spr1_entry,
            server_handles=server_handles,
            tokenizer_path=args.model_path,
            prompt_text=PROMPT_TEXT,
            sampling_params=sampling_params,
        )

        print("\n========== SPR1 LOOP OUTPUT ==========")
        try:
            print(out.model_dump() if hasattr(out, "model_dump") else out)
        except Exception:
            print(out)
        print("========== END ==========\n")

    finally:
        ray.shutdown()


if __name__ == "__main__":
    asyncio.run(amain())
