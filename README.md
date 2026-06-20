# AORUS MASTER 16 AM6H — 規格問答 RAG

## 1. 簡介

純 Python 手寫的 RAG 問答系統,精準回答 **GIGABYTE AORUS MASTER 16 AM6H** 筆電的產品規格。支援**繁中／英文**混合提問、**串流輸出**、**多輪對話記憶**,並附一個單頁聊天網頁。設計目標是在 **≤4GB VRAM** 的環境穩定運作。

- **無框架**:不依賴 LangChain／LlamaIndex,chunking／retrieval／generation 全部手寫(`src/rag/`)。
- **推論引擎**:`llama.cpp`(`llama-cpp-python`),LLM 全層 GPU offload、Embedding Model跑 CPU,把顯存全留給 SLM,且**不需要 torch**。
- **雙模型 A/B**:`MODEL=qwen|llama` 一鍵切換,比較生成能力。
  - `qwen`(預設):`MaziyarPanahi/Qwen3-1.7B-GGUF`(Q4_K_M),ChatML、關閉 thinking。
  - `llama`:`bartowski/Llama-3.2-3B-Instruct-GGUF`(Q4_K_M),Llama-3 header 模板。
- **共用嵌入**:`Qwen/Qwen3-Embedding-0.6B-GGUF`,兩個模型共用 → 同一份索引,A/B 只比較「生成」一個變因。
- **環境/部署**:`uv` 管理相依;Docker(GPU)一鍵啟動。

| 硬性限制 | 做法 |
|---|---|
| ≤4GB VRAM | 1.7B Q4 權重約 1.1GB,全層上 GPU;n_ctx=8192 下實測 Qwen 2.95GB／Llama 3.76GB |
| 無框架 | `chunk.py` / `retrieve.py`(BM25+RRF) / `index.py`(NumPy cosine) / `prompt.py` / `llm.py` 全手寫 |
| uv 管理 | `pyproject.toml` + `uv sync`(僅一個 llama.cpp CUDA wheel,無 torch/transformers) |
| 串流輸出 | `llm.stream()` 逐 token yield → CLI 即時印、Server 走 SSE、網頁逐字顯示 |

---

## 2. 專案架構與資料流

```
規格 HTML ─scrape─▶ spec.jsonl(17 筆 K-V)─chunk+variants─▶ 62 塊 ─embed─▶ NumPy 向量索引
                                                                                    │
   問題 ─┬─ dense（向量 cosine）─┐                                                  │
        └─ lexical（BM25）──────┴─ RRF 融合 ─ top-7 ─┐                             │
                                                      └─ prompt(+對話記憶)─ llama.cpp ─▶ 串流答案
```

```
gigabyte-rag/
├── data/raw/am6h_spec.html      # 規格頁原始 HTML（唯一資料來源）+ variants.json（BZH/BYH/BXH）
├── data/processed/spec.jsonl    # 解析後的 K-V（產生物）
├── data/index/                  # embeddings.npy + chunks.jsonl（共用索引，產生物）
├── models/{qwen,llama,embedding}/*.gguf   # 權重（gitignore，啟動時自動下載）
├── src/rag/
│   ├── config.py     # 所有設定，環境變數可覆寫
│   ├── scrape.py     # 解析 spec-item-list → 結構化 K-V
│   ├── variants.py   # 併入 BZH/BYH/BXH 三個 GPU 版本的規格塊
│   ├── chunk.py      # 多粒度切塊（row / line）
│   ├── embed.py      # llama.cpp 嵌入封裝（Qwen3-Embedding GGUF, CPU）
│   ├── index.py      # NumPy 向量索引（cosine）
│   ├── retrieve.py   # 手寫 BM25 + dense + RRF 混合檢索
│   ├── prompt.py     # ChatML / llama3 模板 + 語言注入 + 對話記憶
│   ├── llm.py        # llama.cpp 串流 + TTFT/TPS 計時
│   └── pipeline.py   # 串起 retrieve → prompt → generate（含記憶裁切）
├── app/
│   ├── server.py     # FastAPI SSE 服務 + 對話記憶（Docker 入口）
│   └── frontend.html # 單頁聊天網頁（串流、markdown、記憶、新對話）
├── scripts/          # download_model / scrape / build_index / vram_report / entrypoint
├── eval/             # questions.jsonl + run_eval.py + results/<model>/benchmark.md
└── Dockerfile + docker-compose.yml
```

---

## 3. 研究方法(RAG 具體技術)

1. **解析(scrape.py)**:規格頁每筆是 `<ul class="spec-item-list">`,`spec-title` 為欄位名(繁中)、`spec-desc` 為值(多為英文,`<br>` 換行)。讀成乾淨的 Key→Value。BZH/BYH/BXH 三種 GPU 版本由 `variants.py` 併入(官網單頁只含一種配置)。

2. **多粒度切塊(chunk.py)**:結構化規格**不能用固定字數亂切**(會把「2560×1600」跟「顯示器」拆開)。因此:
   - **row 粒度**:一個分類一塊(適合「螢幕規格?」這種綜合題)。
   - **line 粒度**:連接埠/顯示器這種多行值,逐行各成一塊(適合「有沒有 Thunderbolt 5?」這種精確題)。
   - 每塊都寫進英文別名,讓英文問題也能命中繁中欄位。

3. **向量索引(index.py)**:資料量小(62 塊),不需 FAISS。向量單位正規化後存成 NumPy 矩陣,查詢時一次**矩陣相乘**即得對所有塊的 cosine 相似度,取 top-k。

4. **混合檢索(retrieve.py)**:規格題充滿須精確命中的字串(型號、240Hz、Thunderbolt 5),純語意向量易漏,故再加**手寫 BM25**(中英 tokenizer:英數成詞、每個中文字一 token),兩條排名用 **RRF(Reciprocal Rank Fusion)** 融合,免去分數尺度問題。`top_k=7` 讓高 IDF 的機型碼不會把真正分類擠出。

5. **Prompt(prompt.py)**:依模型用 ChatML(Qwen)或 Llama-3 header 模板。Qwen **關閉 thinking**(在 assistant 開頭預塞空 `<think></think>`,直接給答案、不浪費 token)。系統提示要求只依規格作答、找不到就明說「規格表未提供」。**語言控制採動態注入**:偵測問題語言後,把「請用繁體中文/Please answer in English」加在 **user prompt 最後一行**——小模型對此處指示的遵循度遠高於埋在 system prompt。

6. **生成 + 串流(llm.py)**:llama.cpp 載入 GGUF,逐 token 串流,同時記錄 **TTFT**(首 token 延遲)與 **TPS**(每秒 token 數)。

7. **多輪記憶(server.py + pipeline.py)**:伺服器端依 `session_id` 保存對話歷史(滾動視窗,最近 16 則)。每次回答前用模型自己的 tokenizer 量測,把 `n_ctx − max_tokens − 規格上下文 − 安全餘量` 的剩餘額度,**從最新往舊**塞入放得下的歷史(放不下就丟),保證不超出 context;追問時並把上一句問題併入檢索 query,讓「那 BYH 呢?」也能檢索到正確塊。不做摘要(省一次 LLM 呼叫)。

---

## 4. 實驗結果

評測集 `eval/questions.jsonl` 共 **26 題**(25 題可答的規格題 + 1 題刻意問規格外的價格/續航,測拒答),涵蓋繁中/英文、單一規格、跨型號比較、釐清與拒答。指標分量化(TTFT/TPS)與定性(檢索命中、答案正確、拒答、語言一致、`<think>` 洩漏)。

| 指標 | Qwen3-1.7B | Llama-3.2-3B-Instruct |
|---|---|---|
| TTFT 平均 | **0.072s** | 0.099s |
| TPS 平均 | **224.9 tok/s** | 185.3 tok/s |
| 檢索 Hit@7 | 24/25 | 24/25 |
| 答案正確率(≥70% 關鍵字) | 21/25 | 21/25 |
| 關鍵字覆蓋率 | 0.88 | **0.90** |
| 拒答正確率 | 1/1 | 1/1 |
| 語言一致 | 26/26 | 26/26 |
| `<think>` 洩漏 | 0 | 0 |
| 執行 VRAM(n_ctx=8192) | **2.95 GB** | 3.76 GB |

**分析**:檢索共用嵌入,兩者命中完全一致(含唯一漏失 q23)。答案品質打平(各 21/25);逐題檢視顯示兩者各有一處硬傷(Qwen 否認 q18 的 HDR 支援;Llama 在 q09 幻覺出「128GB」記憶體),其餘為對稱的細微遺漏。效能與資源 **Qwen 明顯較優**:TTFT 快約 28%、TPS 高約 21%、VRAM 少 0.8GB。

**結論**:此任務(單頁規格、4GB 目標)以 **Qwen3-1.7B 為預設**最合適——同等品質下更快更省、且這次零數字幻覺;**Llama-3.2-3B-Instruct 作為備援**,英文長題與多段推理更穩,語料變大時可換上(代價是 +0.8GB VRAM、約 20–28% 較慢)。完整逐題輸出見 `eval/results/<model>/benchmark.md`。

---

## 5. 程式運行方法

> 需求:Linux + NVIDIA GPU + NVIDIA Container Toolkit。容器啟動時自動 ① 下載 GGUF 模型 ② 解析 HTML 建索引 ③ 啟動服務;`models/`、`data/` 以 volume 掛載,只做一次。

### Docker(建議)

```bash
# 啟動（預設 Qwen），然後開瀏覽器 http://localhost:8000 使用聊天網頁
docker compose up -d --build rag

# 切換到 Llama-3.2-3B：① 直接指定服務，或 ② 在 .env 設 MODEL=llama 後重新 up
docker compose up -d --build rag-llama
```

聊天網頁(`http://localhost:8000`)支援串流逐字輸出、markdown 排版、範例提問、Enter 送出、多輪記憶與「＋ 新對話」重置,版面固定單頁不捲動。

API(SSE 串流):

```bash
curl -N "http://localhost:8000/chat?question=螢幕的更新率是多少？"
curl -N "http://localhost:8000/chat?question=Which GPU does it use?"
curl localhost:8000/healthz
```

評測、VRAM 量測(在容器內):

```bash
docker compose exec rag uv run --no-sync python eval/run_eval.py           # → eval/results/<model>/benchmark.md
docker compose exec rag uv run --no-sync python scripts/vram_report.py     # → eval/results/uesd_vram.json
```

> **切換模型**:單 GPU 同時只跑一個。`MODEL=qwen|llama` 自動選對 GGUF 來源、prompt 格式、`models/<model>/` 權重與 `eval/results/<model>/` 輸出;嵌入模型共用,索引只建一次。
> **設定**:預設(Qwen、公開模型)**不需 `.env`**;要切 Llama 或用 gated repo(`HF_TOKEN`)才 `cp .env.example .env` 填值。常用變數:`MODEL`、`N_CTX`(預設 8192)、`MAX_TOKENS`(1024)、`TOP_K`(7)、`N_GPU_LAYERS`(-1)。完整見 `src/rag/config.py`。

### 本機 uv(無 Docker 時)

```bash
uv sync
uv run python scripts/download_model.py   # 下載 GGUF → models/
uv run python scripts/scrape.py           # 解析 HTML → data/processed/spec.jsonl
uv run python scripts/build_index.py      # 切塊 + 嵌入 → data/index/
uv run uvicorn app.server:app --host 0.0.0.0 --port 8000   # 啟動服務，開 http://localhost:8000
```
