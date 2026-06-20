# AORUS MASTER 16 AM6H — RAG Benchmark (llama)

_Generated: 2026-06-20 15:51:24_

## Environment & Config

- Platform: `Linux-6.17.0-14-generic-x86_64-with-glibc2.35` | Python `3.11.15`
- GPU: `NVIDIA GeForce RTX 3080 Ti`
- MODEL: `llama` | prompt format: `llama3`
- LLM: `bartowski/Llama-3.2-3B-Instruct-GGUF` / `Llama-3.2-3B-Instruct-Q4_K_M.gguf` (n_ctx=8192, n_gpu_layers=-1, flash_attn=True)
- Thinking mode: `False` | Embedding: `Qwen3-Embedding-0.6B-Q8_0.gguf` (llama.cpp, CPU)
- Retrieval: hybrid=True, top_k=7, dense_k=10, lexical_k=10, rrf_k=60

## Quantitative

| Metric | avg | p50 | p95 |
|---|---|---|---|
| TTFT (s) | 0.099 | 0.101 | 0.116 |
| TPS (tok/s) | 185.26 | 187.165 | 190.89 |

## Qualitative

- Retrieval Hit@7 (Qs with expected category): **24/25**
- Answer correctness (answerable Qs, ≥70% keyword coverage): **21/25**
- Avg keyword coverage (answerable Qs): **0.9**
- Refusal correctness (out-of-scope Qs): **1/1**
- Language match: **26/26**
- `<think>` leaks (should be 0): **0**

## Per-question detail

| id | lang | type | hit | correct | cov | lang_ok | TTFT | TPS | answer |
|---|---|---|---|---|---|---|---|---|---|
| q01 | zh | spec | True | False | 0.67 | True | 0.1588 | 173.74 | 根據規格內容，AORUS MASTER 16 AM6H BZH 可以選擇的作業系統有：Windows 11 Home 和 UEFI Shell OS。 |
| q02 | zh | spec | True | True | 0.75 | True | 0.1051 | 181.0 | AORUS MASTER 16 AM6H BYH 搭載的中央處理器是 Intel® Core™ Ultra 9 Processor 275HX，核心數為 24 … |
| q03 | zh | comparison | True | False | 0.67 | True | 0.0597 | 190.21 | AORUS MASTER 16 AM6H BZH 和 BYH 的顯卡差在顯示記憶體容量上，BZH 有 24GB GDDR7，而 BYH 只有 16GB GDDR… |
| q04 | zh | comparison | True | True | 0.75 | True | 0.1002 | 180.75 | 根據規格內容，AORUS MASTER 16 AM6H BXH 顯示晶片（GPU / Graphics Card）的 Maximum Graphics Powe… |
| q05 | zh | comparison | True | True | 1.00 | True | 0.1014 | 187.28 | AORUS MASTER 16 AM6H 的 RTX 5090 版本的型號是 BZH。顯示記憶體（VRAM）為 24GB GDDR7。 |
| q06 | zh | comparison | True | True | 1.00 | True | 0.1046 | 182.2 | 根據規格內容，AORUS MASTER 16 AM6H BYH 顯示晶片（GPU / Graphics Card）為 NVIDIA® GeForce RTX™ … |
| q07 | zh | spec | True | True | 0.86 | True | 0.1076 | 181.65 | AORUS MASTER 16 AM6H BZH 的螢幕規格如下：  * 尺寸：16" 16:10 * 解析度：OLED WQXGA (2560×1600) *… |
| q08 | zh | spec | True | True | 1.00 | True | 0.1073 | 188.84 | AORUS MASTER 16 AM6H BYH 的螢幕支援以下顯示技術或認證：  * NVIDIA® G-SYNC® * NVIDIA® Advanced O… |
| q09 | zh | spec | True | False | 0.67 | True | 0.0934 | 173.98 | AORUS MASTER 16 AM6H BXH 的記憶體可擴充到 128GB DDR5 5600MHz，使用 2x SO-DIMM 磁片。 |
| q10 | zh | multi_spec | True | True | 1.00 | True | 0.0912 | 175.89 | AORUS MASTER 16 AM6H BZH 有 1x PCIe Gen5 M.2槽和 1x PCIe Gen4x4 M.2槽。 |
| q11 | zh | multi_spec | True | True | 1.00 | True | 0.1047 | 190.35 | AORUS MASTER 16 AM6H BYH 的左側有：1 x DC in、1 x RJ-45、1 x HDMI 2.1、1 x Type-A suppor… |
| q12 | zh | model_relation | True | True | 0.86 | True | 0.1185 | 187.35 | AORUS MASTER 16 AM6H 和 BZH、BYH、BXH 是同一台機身的不同規格型號（子系列），因為搭載不同的 NVIDIA GeForce RTX… |
| q13 | en | spec | True | True | 0.80 | True | 0.1036 | 187.16 | The AORUS MASTER 16 AM6H BZH uses the Intel Core Ultra 9 Processor, which has:  … |
| q14 | en | comparison | True | True | 1.00 | True | 0.1006 | 192.02 | The GPU and VRAM differences between AORUS MASTER 16 AM6H BZH and BYH are:  * GP… |
| q15 | en | comparison | True | True | 1.00 | True | 0.101 | 185.92 | The AORUS MASTER 16 AM6H BXH variant comes with the NVIDIA GeForce RTX 5070 Ti L… |
| q16 | en | comparison | True | True | 1.00 | True | 0.1008 | 184.66 | Yes, both the RTX 5090 and RTX 5080 versions of AORUS MASTER 16 have the same ma… |
| q17 | en | spec | True | True | 1.00 | True | 0.108 | 189.2 | The display specifications for the AORUS MASTER 16 AM6H BXH are as follows:  * S… |
| q18 | en | spec | True | True | 1.00 | True | 0.1059 | 180.38 | Yes, the AORUS MASTER 16 AM6H BZH display supports:  * NVIDIA G-SYNC * NVIDIA Ad… |
| q19 | en | spec | True | True | 1.00 | True | 0.0937 | 188.45 | The AORUS MASTER 16 AM6H BYH supports up to 64GB DDR5 memory. It uses 2x SO-DIMM… |
| q20 | en | multi_spec | True | True | 1.00 | True | 0.0673 | 191.07 | For the AORUS MASTER 16 AM6H BXH, the available storage slots are:  * 1x PCIe Ge… |
| q21 | en | multi_spec | True | True | 1.00 | True | 0.0777 | 185.52 | According to the spec context, on the AORUS MASTER 16 AM6H BZH, the left side ha… |
| q22 | en | multi_spec | True | True | 1.00 | True | 0.0932 | 189.73 | For the AORUS MASTER 16 AM6H BYH, the networking and wireless features are:  * L… |
| q23 | en | multi_spec | False | False | 0.50 | True | 0.1 | 187.17 | The AORUS MASTER 16 AM6H BXH includes:  * Webcam: FHD (1080p) IR Webcam * Microp… |
| q24 | en | clarification | True | True | 1.00 | True | 0.0744 | 188.44 | The AORUS MASTER 16 AM6H refers to a platform (chassis) model, not a fixed GPU c… |
| q25 | zh | clarification | True | True | 1.00 | True | 0.1017 | 187.79 | 不是。AORUS MASTER 16 AM6H 不是固定搭載 RTX 5090， 而是有三個具體規格型號（BZH、BYH、BXH），每個型號搭載不同的 NVID… |
| q26 | en | refusal | None | True | 1.00 | True | 0.1003 | 186.13 | Unfortunately, the official spec document does not provide the USD price and bat… |

## Full answers

### q01 (zh/spec) — hit=True correct=False cov=0.67

**Q:** AORUS MASTER 16 BZH 可以選哪些作業系統？

**A:** 根據規格內容，AORUS MASTER 16 AM6H BZH 可以選擇的作業系統有：Windows 11 Home 和 UEFI Shell OS。

### q02 (zh/spec) — hit=True correct=True cov=0.75

**Q:** AORUS MASTER 16 BYH 搭載的處理器是哪一顆？請包含核心數與最高時脈。

**A:** AORUS MASTER 16 AM6H BYH 搭載的中央處理器是 Intel® Core™ Ultra 9 Processor 275HX，核心數為 24 個核心、最高時脈為 5.4 GHz。

### q03 (zh/comparison) — hit=True correct=False cov=0.67

**Q:** AORUS MASTER 16 BZH 和 BYH 的顯卡差在哪？

**A:** AORUS MASTER 16 AM6H BZH 和 BYH 的顯卡差在顯示記憶體容量上，BZH 有 24GB GDDR7，而 BYH 只有 16GB GDDR7。

### q04 (zh/comparison) — hit=True correct=True cov=0.75

**Q:** AORUS MASTER 16 BXH 是不是 175W 顯卡？如果不是，正確的 Maximum Graphics Power 是多少？

**A:** 根據規格內容，AORUS MASTER 16 AM6H BXH 顯示晶片（GPU / Graphics Card）的 Maximum Graphics Power 是 140W，而不是 175W。

### q05 (zh/comparison) — hit=True correct=True cov=1.00

**Q:** 如果我想買 AORUS MASTER 16 的 RTX 5090 版本，對應的型號是哪個？VRAM 是多少？

**A:** AORUS MASTER 16 AM6H 的 RTX 5090 版本的型號是 BZH。顯示記憶體（VRAM）為 24GB GDDR7。

### q06 (zh/comparison) — hit=True correct=True cov=1.00

**Q:** AORUS MASTER 16 的 RTX 5080 版本和 RTX 5070 Ti 版本，顯卡記憶體與功耗各是多少？

**A:** 根據規格內容，AORUS MASTER 16 AM6H BYH 顯示晶片（GPU / Graphics Card）為 NVIDIA® GeForce RTX™ 5080 Laptop GPU，顯示記憶體為 16GB GDDR7，最大功耗為 175W。

另一方面，AORUS MASTER 16 AM6H BXH 顯示晶片（GPU / Graphics Card）為 NVIDIA® GeForce RTX™ 5070 Ti Laptop GPU，顯示記憶體為 12GB GDDR7，最大功耗為 140W。

### q07 (zh/spec) — hit=True correct=True cov=0.86

**Q:** AORUS MASTER 16 BZH 的螢幕規格是什麼？請包含尺寸、解析度、更新率和面板類型。

**A:** AORUS MASTER 16 AM6H BZH 的螢幕規格如下：

* 尺寸：16" 16:10
* 解析度：OLED WQXGA (2560×1600)
*更新率：240Hz
* 面板類型：OLED

### q08 (zh/spec) — hit=True correct=True cov=1.00

**Q:** AORUS MASTER 16 BYH 的螢幕有支援哪些顯示技術或認證？

**A:** AORUS MASTER 16 AM6H BYH 的螢幕支援以下顯示技術或認證：

* NVIDIA® G-SYNC®
* NVIDIA® Advanced Optimus
* VESA DisplayHDR True Black 500
* VESA ClearMR 10000
* Pantone® Validated
* TÜV Rheinland Low Blue Light
* Dolby Vision®

### q09 (zh/spec) — hit=True correct=False cov=0.67

**Q:** AORUS MASTER 16 BXH 最高可以擴充到多少記憶體？記憶體插槽是什麼配置？

**A:** AORUS MASTER 16 AM6H BXH 的記憶體可擴充到 128GB DDR5 5600MHz，使用 2x SO-DIMM 磁片。

### q10 (zh/multi_spec) — hit=True correct=True cov=1.00

**Q:** AORUS MASTER 16 BZH 有幾個 M.2 SSD 插槽？分別支援什麼 PCIe 規格？

**A:** AORUS MASTER 16 AM6H BZH 有 1x PCIe Gen5 M.2槽和 1x PCIe Gen4x4 M.2槽。

### q11 (zh/multi_spec) — hit=True correct=True cov=1.00

**Q:** AORUS MASTER 16 BYH 左右兩側有哪些連接埠？Thunderbolt 5 和 Thunderbolt 4 分別在哪一側？

**A:** AORUS MASTER 16 AM6H BYH 的左側有：1 x DC in、1 x RJ-45、1 x HDMI 2.1、1 x Type-A support USB3.2 Gen2 和 1 x Type-C with Thunderbolt™5 (支持 USB4、DisplayPort™ 2.1 和 Power Delivery 3.0)。而右側則有：1 x Type-A support USB3.2 Gen2、1 x Type-C with Thunderbolt™4 (支持 USB4、DisplayPort™ 1.4 和 Power Delivery 3.0)、1 x MicroSD (UHS-II) 和 1 x Audio Jack support mic / headphone combo。

### q12 (zh/model_relation) — hit=True correct=True cov=0.86

**Q:** AORUS MASTER 16 AM6H 和 BZH、BYH、BXH 是什麼關係？

**A:** AORUS MASTER 16 AM6H 和 BZH、BYH、BXH 是同一台機身的不同規格型號（子系列），因為搭載不同的 NVIDIA GeForce RTX 50 系列獨立顯示卡。

### q13 (en/spec) — hit=True correct=True cov=0.80

**Q:** What CPU does the AORUS MASTER 16 BZH use, including cache, maximum frequency, cores, and threads?

**A:** The AORUS MASTER 16 AM6H BZH uses the Intel Core Ultra 9 Processor, which has:

* Cache: 36MB
* Maximum frequency: up to 5.4 GHz
* Cores: 24 cores
* Threads: 24 threads

### q14 (en/comparison) — hit=True correct=True cov=1.00

**Q:** I am comparing the AORUS MASTER 16 BZH and BYH. What are the GPU and VRAM differences?

**A:** The GPU and VRAM differences between AORUS MASTER 16 AM6H BZH and BYH are:

* GPU: NVIDIA GeForce RTX 5090 Laptop GPU (BZH) vs. NVIDIA GeForce RTX 5080 Laptop GPU (BYH)
* VRAM: 24GB GDDR7 (BZH) vs. 16GB GDDR7 (BYH)

Note that the GPU power consumption and boost clock speeds are also different between the two models, but these are not differences in the GPU itself, but rather differences in the specific configuration.

### q15 (en/comparison) — hit=True correct=True cov=1.00

**Q:** Which AORUS MASTER 16 variant comes with the RTX 5070 Ti Laptop GPU, and what is its listed graphics power?

**A:** The AORUS MASTER 16 AM6H BXH variant comes with the NVIDIA GeForce RTX 5070 Ti Laptop GPU, and its listed graphics power is 140W Maximum Graphics Power with Dynamic Boost*.

### q16 (en/comparison) — hit=True correct=True cov=1.00

**Q:** Do the RTX 5090 and RTX 5080 versions of AORUS MASTER 16 have the same maximum graphics power?

**A:** Yes, both the RTX 5090 and RTX 5080 versions of AORUS MASTER 16 have the same maximum graphics power, which is 175W.

### q17 (en/spec) — hit=True correct=True cov=1.00

**Q:** What display specifications are listed for the AORUS MASTER 16 BXH, including size, aspect ratio, resolution, refresh rate, and response time?

**A:** The display specifications for the AORUS MASTER 16 AM6H BXH are as follows:

* Size: 16 inches
* Aspect ratio: 16:10
* Resolution: OLED WQXGA (2560×1600)
* Refresh rate: 240Hz
* Response time: 1ms
* DCIP-3: 100%
* Peak brightness: 500 nits
* Contrast ratio: 1,000,000:1
* Display technology: OLED
* NVIDIA G-SYNC
* NVIDIA Advanced Optimus
* VESA DisplayHDR True Black 500
* VESA ClearMR 10,000
* Pantone Validated
* TÜV Rheinland Low Blue Light
* Dolby Vision

### q18 (en/spec) — hit=True correct=True cov=1.00

**Q:** Does the AORUS MASTER 16 BZH display support G-SYNC, Advanced Optimus, HDR, and Dolby Vision?

**A:** Yes, the AORUS MASTER 16 AM6H BZH display supports:

* NVIDIA G-SYNC
* NVIDIA Advanced Optimus
* VESA DisplayHDR True Black 500
* Dolby Vision

### q19 (en/spec) — hit=True correct=True cov=1.00

**Q:** How much memory can the AORUS MASTER 16 BYH support, and what memory slot configuration does it use?

**A:** The AORUS MASTER 16 AM6H BYH supports up to 64GB DDR5 memory. It uses 2x SO-DIMM sockets for expansion.

### q20 (en/multi_spec) — hit=True correct=True cov=1.00

**Q:** For the AORUS MASTER 16 BXH, what storage slots are available and what is the maximum listed SSD capacity?

**A:** For the AORUS MASTER 16 AM6H BXH, the available storage slots are:

* 1x PCIe Gen5 M.2 slot
* 1x PCIe Gen4x4 M.2 slots

The maximum listed SSD capacity is up to 4TB PCIe NVMe M.2 SSD.

### q21 (en/multi_spec) — hit=True correct=True cov=1.00

**Q:** On the AORUS MASTER 16 BZH, which side has Thunderbolt 5 and which side has Thunderbolt 4?

**A:** According to the spec context, on the AORUS MASTER 16 AM6H BZH, the left side has Thunderbolt 5 and the right side has Thunderbolt 4.

### q22 (en/multi_spec) — hit=True correct=True cov=1.00

**Q:** What networking and wireless features are listed for the AORUS MASTER 16 BYH?

**A:** For the AORUS MASTER 16 AM6H BYH, the networking and wireless features are:

* LAN: 1G
* Bluetooth v5.4
* WIFI 7 (802.11be 2x2)

### q23 (en/multi_spec) — hit=False correct=False cov=0.50

**Q:** What webcam, microphone, and login security features does the AORUS MASTER 16 BXH include?

**A:** The AORUS MASTER 16 AM6H BXH includes:

* Webcam: FHD (1080p) IR Webcam
* Microphone: Build-in array Microphone
* Login security feature: Windows Hello Face Authentication

### q24 (en/clarification) — hit=True correct=True cov=1.00

**Q:** Does the AORUS MASTER 16 AM6H refer to one fixed GPU configuration, or do I need to choose between BZH, BYH, and BXH?

**A:** The AORUS MASTER 16 AM6H refers to a platform (chassis) model, not a fixed GPU configuration. You need to choose between BZH, BYH, and BXH at purchase, as each of these models comes with a different NVIDIA GeForce RTX 50 series discrete GPU.

### q25 (zh/clarification) — hit=True correct=True cov=1.00

**Q:** AORUS MASTER 16 AM6H 固定搭載 RTX 5090 嗎？

**A:** 不是。AORUS MASTER 16 AM6H 不是固定搭載 RTX 5090， 而是有三個具體規格型號（BZH、BYH、BXH），每個型號搭載不同的 NVIDIA GeForce RTX 50 系列顯示卡。

### q26 (en/refusal) — hit=None correct=True cov=1.00

**Q:** What is the official USD price and battery life in hours of the AORUS MASTER 16 BZH?

**A:** Unfortunately, the official spec document does not provide the USD price and battery life in hours of the AORUS MASTER 16 BZH.

