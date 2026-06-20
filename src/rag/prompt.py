"""Prompt construction for the active LLM (selected by cfg.prompt_format).

Two model families are supported, sharing the same system prompt and retrieved
context but rendered with each family's native chat format:

  * ``qwen``  -- Qwen3-1.7B ChatML. Qwen3 is a *hybrid* reasoning model; the
    official template disables reasoning (``enable_thinking=False``) by
    pre-filling an empty, already-closed ``<think>\\n\\n</think>`` block right
    after the assistant turn begins, so the model emits the final answer
    directly (low TTFT, no <think> leakage).
  * ``llama3`` -- Llama-3.2 header template (``<|start_header_id|>role
    <|end_header_id|>`` … ``<|eot_id|>``). No thinking block.

The format is chosen by ``MODEL`` / ``PROMPT_FORMAT`` (see rag.config).
"""

from __future__ import annotations

from .config import cfg

# --- Qwen3 ChatML tokens ---
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
# Pre-filled empty reasoning block == enable_thinking=False in Qwen3's template.
THINK_OFF = "<think>\n\n</think>\n\n"
QWEN_STOP = [IM_END, "<|endoftext|>"]

# --- Llama-3 header-template tokens ---
LLAMA_STOP = ["<|eot_id|>", "<|end_of_text|>", "<|start_header_id|>"]


def stop_tokens(fmt: str | None = None) -> list[str]:
    """Stop strings for the active prompt format."""
    fmt = fmt or cfg.prompt_format
    return LLAMA_STOP if fmt == "llama3" else QWEN_STOP


# Computed once per process (the active model is fixed by env at startup).
STOP = stop_tokens()

SYSTEM_PROMPT = (
    "你是 GIGABYTE 技嘉 AORUS MASTER 16 AM6H 筆記型電腦的產品規格助理。"
    "You are a product-spec assistant for the GIGABYTE AORUS MASTER 16 AM6H laptop.\n"
    "規則 / Rules:\n"
    "1. 只能根據 規格內容 回答；不得編造或臆測規格。\n"
    "   Answer ONLY from the spec context . Never invent specifications.\n"
    "2. 若內容中找不到答案，明確說明官方規格文件未提供相關內容，不要猜測。\n"
    "   If the answer is not in the context, say the official spec document does "
    "not provide that information.\n"
    "3. 使用與提問相同的語言回答（中文問題用繁體中文，英文問題用英文）。\n"
    "   Reply in the SAME language as the question.\n"
    "4. 回答簡潔、精確，保留型號與數值單位（如 240Hz、99Wh、DDR5 5600MHz）。\n"
    "   Be concise and keep exact model names, numbers and units.\n"
    "5. 規格會依型號（BZH／BYH／BXH）而不同。若提問的型號或數值未明確出現在規格內容中，"
    "請誠實說明官方規格文件未提供相關內容，切勿把其他型號的數值套用上去。\n"
    "   Specs differ by model (BZH/BYH/BXH). If the asked-for model or value is "
    "not explicitly in the context, say so honestly — never apply another "
    "model's value to fill the gap.\n"
    "6. 若問題要求列出多個項目（作業系統、連接埠、認證、插槽、無線功能等），"
    "請依規格內容完整列出所有項目，不可只列其中一項。\n"
    "   When asked to list multiple items, list ALL of them from the context, "
    "not just one.\n"
    "7. 顯示記憶體（VRAM，如 24GB GDDR7）與系統記憶體（RAM，如 64GB DDR5）是不同的東西，"
    "回答記憶體容量時切勿把顯示卡的 GDDR7 容量當成系統記憶體。\n"
    "   Video memory (VRAM, e.g. 24GB GDDR7) is NOT system memory (RAM, e.g. "
    "64GB DDR5); never report a GPU's GDDR7 size as system memory."
)


def detect_lang(text: str) -> str:
    """Pick the answer language for a (possibly mixed) question.

    Counts CJK vs ASCII letters. No CJK -> English. Otherwise lean Chinese
    unless the text is overwhelmingly Latin (e.g. an English sentence with one
    stray CJK char). This handles spec questions that mix a Chinese sentence
    with English model names, e.g. "有沒有 Thunderbolt 5？".
    """
    cjk = sum(1 for c in text if "一" <= c <= "鿿")
    latin = sum(1 for c in text if c.isascii() and c.isalpha())
    if cjk == 0:
        return "en"
    if latin == 0:
        return "zh"
    return "zh" if cjk / (cjk + latin) >= 0.2 else "en"


# Injected at the END of the user turn (see build_messages). Small models obey a
# language directive placed here far more reliably than one in the system prompt.
LANG_INSTRUCTION = {
    "zh": "請務必用繁體中文回答。",
    "en": "Please answer in English.",
}


def build_context(retrieved: list[dict]) -> str:
    lines = []
    for i, item in enumerate(retrieved, start=1):
        lines.append(f"[{i}] {item['chunk']['text']}")
    return "\n".join(lines)


def build_messages(
    question: str,
    retrieved: list[dict],
    history: list[dict] | None = None,
) -> list[dict]:
    """OpenAI-style messages (handy if you ever call create_chat_completion).

    ``history`` is the prior conversation as plain ``{role, content}`` turns
    (raw question / raw answer text, WITHOUT each turn's spec context — old
    contexts must not bloat the window). It is inserted between the system
    prompt and the current user turn, so the model can resolve references like
    "那 BYH 呢？". The caller is responsible for trimming history to the token
    budget (see RagPipeline._fit_history).
    """
    context = build_context(retrieved)
    # Dynamic language directive appended at the very end of the user turn.
    lang = detect_lang(question)
    user = (
        f"【規格內容 / Spec context】\n{context}\n\n"
        f"【問題 / Question】\n{question}\n\n"
        f"{LANG_INSTRUCTION[lang]}"
    )
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user})
    return messages


def _render_qwen(messages: list[dict], enable_thinking: bool) -> str:
    parts = []
    for m in messages:
        parts.append(f"{IM_START}{m['role']}\n{m['content']}{IM_END}\n")
    parts.append(f"{IM_START}assistant\n")
    if not enable_thinking:
        parts.append(THINK_OFF)
    return "".join(parts)


def _render_llama3(messages: list[dict]) -> str:
    parts = ["<|begin_of_text|>"]
    for m in messages:
        parts.append(
            f"<|start_header_id|>{m['role']}<|end_header_id|>\n\n{m['content']}<|eot_id|>"
        )
    parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
    return "".join(parts)


def build_prompt(
    question: str,
    retrieved: list[dict],
    enable_thinking: bool | None = None,
    history: list[dict] | None = None,
) -> str:
    """Render the full prompt string fed to llama.cpp, per cfg.prompt_format."""
    messages = build_messages(question, retrieved, history=history)
    if cfg.prompt_format == "llama3":
        # Base Llama-3.2 has no thinking block; the header template is enough.
        return _render_llama3(messages)
    if enable_thinking is None:
        enable_thinking = cfg.enable_thinking
    return _render_qwen(messages, enable_thinking)
