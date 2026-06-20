"""llama.cpp (GGUF) wrapper: streaming generation with TTFT / TPS metrics.

The model is loaded lazily so that data-prep steps (scrape, build index) don't
pay to load the LLM. Streaming yields token text as it arrives and records the
two quantitative metrics the assignment asks for:

  * TTFT -- time to first token (prompt prefill latency).
  * TPS  -- decode throughput = (tokens - 1) / (total - TTFT).
"""

from __future__ import annotations

import time

from .config import cfg
from .prompt import STOP


class LlamaLLM:
    def __init__(
        self,
        model_path: str | None = None,
        n_ctx: int | None = None,
        n_gpu_layers: int | None = None,
        main_gpu: int | None = None,
        n_threads: int | None = None,
        flash_attn: bool | None = None,
    ):
        self.model_path = str(model_path or cfg.model_path)
        self.n_ctx = n_ctx if n_ctx is not None else cfg.n_ctx
        self.n_gpu_layers = n_gpu_layers if n_gpu_layers is not None else cfg.n_gpu_layers
        self.main_gpu = main_gpu if main_gpu is not None else cfg.main_gpu
        self.n_threads = n_threads if n_threads is not None else cfg.n_threads
        self.flash_attn = flash_attn if flash_attn is not None else cfg.flash_attn
        self._llm = None
        self.last_metrics: dict = {}

    def _ensure_loaded(self):
        if self._llm is not None:
            return
        from pathlib import Path

        from llama_cpp import Llama

        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"GGUF model not found at {self.model_path}. "
                f"Run: uv run python scripts/download_model.py"
            )
        self._llm = Llama(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_gpu_layers=self.n_gpu_layers,
            main_gpu=self.main_gpu,
            n_threads=self.n_threads,
            flash_attn=self.flash_attn,
            verbose=False,
        )

    def count_tokens(self, text: str) -> int:
        """Token count for `text` using THIS model's tokenizer (loads it lazily).

        Used to budget conversation history against n_ctx so the prompt plus the
        reserved output (max_tokens) never exceeds the context window.
        """
        self._ensure_loaded()
        return len(self._llm.tokenize(text.encode("utf-8"), add_bos=False))

    def stream(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        repeat_penalty: float | None = None,
        stop: list[str] | None = None,
    ):
        """Yield generated token text; populate self.last_metrics when done."""
        self._ensure_loaded()
        t0 = time.perf_counter()
        ttft = None
        n_tokens = 0

        completion = self._llm.create_completion(
            prompt=prompt,
            max_tokens=max_tokens if max_tokens is not None else cfg.max_tokens,
            temperature=temperature if temperature is not None else cfg.temperature,
            top_p=top_p if top_p is not None else cfg.top_p,
            repeat_penalty=repeat_penalty if repeat_penalty is not None else cfg.repeat_penalty,
            stop=stop if stop is not None else STOP,
            stream=True,
        )
        for part in completion:
            text = part["choices"][0]["text"]
            if not text:
                continue
            if ttft is None:
                ttft = time.perf_counter() - t0
            n_tokens += 1
            yield text

        total = time.perf_counter() - t0
        ttft = ttft if ttft is not None else total
        decode_time = max(total - ttft, 1e-9)
        decode_tps = (n_tokens - 1) / decode_time if n_tokens > 1 else 0.0
        self.last_metrics = {
            "ttft_s": round(ttft, 4),
            "total_s": round(total, 4),
            "completion_tokens": n_tokens,
            "tps": round(decode_tps, 2),                       # decode throughput
            "tps_overall": round(n_tokens / total, 2) if total > 0 else 0.0,
        }

    def complete(self, prompt: str, **kwargs) -> str:
        """Non-streaming convenience: collect the full string."""
        return "".join(self.stream(prompt, **kwargs))
