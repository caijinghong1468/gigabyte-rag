# AORUS MASTER 16 AM6H — RAG Benchmark (qwen)

_Generated: 2026-06-20 15:54:42_

## Environment & Config

- Platform: `Linux-6.17.0-14-generic-x86_64-with-glibc2.35` | Python `3.11.15`
- GPU: `NVIDIA GeForce RTX 3080 Ti`
- MODEL: `qwen` | prompt format: `qwen`
- LLM: `MaziyarPanahi/Qwen3-1.7B-GGUF` / `Qwen3-1.7B.Q4_K_M.gguf` (n_ctx=8192, n_gpu_layers=-1, flash_attn=True)
- Thinking mode: `False` | Embedding: `Qwen3-Embedding-0.6B-Q8_0.gguf` (llama.cpp, CPU)
- Retrieval: hybrid=True, top_k=7, dense_k=10, lexical_k=10, rrf_k=60

## Quantitative

| Metric | avg | p50 | p95 |
|---|---|---|---|
| TTFT (s) | 0.072 | 0.075 | 0.083 |
| TPS (tok/s) | 224.92 | 227.445 | 233.048 |

## Qualitative

- Retrieval Hit@7 (Qs with expected category): **24/25**
- Answer correctness (answerable Qs, ≥70% keyword coverage): **21/25**
- Avg keyword coverage (answerable Qs): **0.88**
- Refusal correctness (out-of-scope Qs): **1/1**
- Language match: **26/26**
- `<think>` leaks (should be 0): **0**

## Per-question detail

| id | lang | type | hit | correct | cov | lang_ok | TTFT | TPS | answer |
|---|---|---|---|---|---|---|---|---|---|
| q01 | zh | spec | True | False | 0.67 | True | 0.1045 | 214.0 | AORUS MASTER 16 BZH 可以選的作業系統為：  - Windows 11 Home - Windows 11 Pro (GIGABYTE 推荐用… |
| q02 | zh | spec | True | True | 0.75 | True | 0.0762 | 213.06 | AORUS MASTER 16 BYH 搭載的處理器是 Intel® Core™ Ultra 9 Processor 275HX，核心數為 24 個，最高時脈為… |
| q03 | zh | comparison | True | True | 1.00 | True | 0.0471 | 229.55 | AORUS MASTER 16 BZH 和 BYH 的顯卡差在以下方面：  1. **顯示晶片（GPU）**：      - BZH：NVIDIA® GeFor… |
| q04 | zh | comparison | True | True | 0.75 | True | 0.0754 | 221.55 | AORUS MASTER 16 BXH 不是 175W 顯卡。正確的 Maximum Graphics Power 是 140W。 |
| q05 | zh | comparison | True | True | 1.00 | True | 0.0747 | 222.79 | 若你想買 AORUS MASTER 16 的 RTX 5090 版本，對應的型號是 **BZH**。   VRAM 是 **24GB GDDR7**。 |
| q06 | zh | comparison | True | True | 0.75 | True | 0.0788 | 225.23 | AORUS MASTER 16 RTX 5080 版本的顯示記憶體為 **16GB GDDR7**，功耗為 **175W Maximum Graphics Po… |
| q07 | zh | spec | True | True | 0.71 | True | 0.0794 | 226.96 | AORUS MASTER 16 BZH 的螢幕規格為：   尺寸：16"   解析度：2560×1600（WQXGA）   更新率：240Hz   面板類型：O… |
| q08 | zh | spec | True | True | 0.86 | True | 0.0788 | 229.58 | AORUS MASTER 16 BYH 的螢幕支援以下顯示技術或認證：  - NVIDIA® G-SYNC® - VESA DisplayHDR True Bl… |
| q09 | zh | spec | True | True | 1.00 | True | 0.0651 | 230.84 | AORUS MASTER 16 BXH 最高可以擴充到 **64GB DDR5 5600MHz**，記憶體插槽為 **2x SO-DIMM sockets**。 |
| q10 | zh | multi_spec | True | False | 0.50 | True | 0.0641 | 207.09 | AORUS MASTER 16 BZH 有 **1個 PCIe Gen5 M.2 插槽**，並支援 **PCIe Gen5** 規格。 |
| q11 | zh | multi_spec | True | True | 1.00 | True | 0.0753 | 228.24 | AORUS MASTER 16 BYH 左右兩側的連接埠如下：  - 左側：     - 1 x DC in     - 1 x RJ-45     - 1 x… |
| q12 | zh | model_relation | True | True | 1.00 | True | 0.0831 | 206.71 | AORUS MASTER 16 AM6H 是這台旗艦 AI 電競筆電的機身模具總型號（Platform 平台型號）。BZH、BYH、BXH 則是同一台 AM6H… |
| q13 | en | spec | True | False | 0.60 | True | 0.0783 | 233.46 | The AORUS MASTER 16 BZH uses an Intel® Core™ Ultra 9 Processor 275HX with the fo… |
| q14 | en | comparison | True | True | 1.00 | True | 0.076 | 235.51 | The GPU and VRAM differences between the AORUS MASTER 16 BZH and BYH are as foll… |
| q15 | en | comparison | True | True | 1.00 | True | 0.0752 | 228.65 | The AORUS MASTER 16 BXH variant comes with the RTX 5070 Ti Laptop GPU. Its liste… |
| q16 | en | comparison | True | True | 1.00 | True | 0.0752 | 227.81 | The RTX 5090 and RTX 5080 versions of AORUS MASTER 16 have the same maximum grap… |
| q17 | en | spec | True | True | 1.00 | True | 0.0811 | 230.16 | The display specifications for the AORUS MASTER 16 BXH are as follows:   - **Siz… |
| q18 | en | spec | True | True | 1.00 | True | 0.0765 | 220.42 | The AORUS MASTER 16 BZH display supports G-SYNC, Advanced Optimus, and Dolby Vis… |
| q19 | en | spec | True | True | 1.00 | True | 0.0652 | 227.24 | The AORUS MASTER 16 BYH supports up to 64GB DDR5 5600MHz memory, and it uses 2x … |
| q20 | en | multi_spec | True | True | 1.00 | True | 0.0501 | 230.14 | For the AORUS MASTER 16 BXH, available storage slots are 1x PCIe Gen5 M.2 slot a… |
| q21 | en | multi_spec | True | True | 1.00 | True | 0.055 | 227.65 | On the AORUS MASTER 16 BZH, the left side supports Thunderbolt™ 5, and the right… |
| q22 | en | multi_spec | True | True | 1.00 | True | 0.0645 | 231.81 | The networking and wireless features listed for the AORUS MASTER 16 BYH are:   -… |
| q23 | en | multi_spec | False | False | 0.50 | True | 0.067 | 230.48 | The AORUS MASTER 16 BXH includes the following webcam, microphone, and login sec… |
| q24 | en | clarification | True | True | 0.88 | True | 0.055 | 222.67 | The AORUS MASTER 16 AM6H is not a fixed GPU configuration. You need to choose be… |
| q25 | zh | clarification | True | True | 1.00 | True | 0.0755 | 225.45 | AORUS MASTER 16 AM6H 不固定搭載 RTX 5090，購買時需在 BZH / BYH / BXH 之間擇一。不同型號僅在顯示晶片（GPU、顯示… |
| q26 | en | refusal | None | True | 1.00 | True | 0.0685 | 220.95 | The official USD price and battery life of the AORUS MASTER 16 BZH are not provi… |

## Full answers

### q01 (zh/spec) — hit=True correct=False cov=0.67

**Q:** AORUS MASTER 16 BZH 可以選哪些作業系統？

**A:** AORUS MASTER 16 BZH 可以選的作業系統為：

- Windows 11 Home
- Windows 11 Pro (GIGABYTE 推荐用于商业用途)

官方規格文件未提供相關內容。

### q02 (zh/spec) — hit=True correct=True cov=0.75

**Q:** AORUS MASTER 16 BYH 搭載的處理器是哪一顆？請包含核心數與最高時脈。

**A:** AORUS MASTER 16 BYH 搭載的處理器是 Intel® Core™ Ultra 9 Processor 275HX，核心數為 24 個，最高時脈為 5.4 GHz。

### q03 (zh/comparison) — hit=True correct=True cov=1.00

**Q:** AORUS MASTER 16 BZH 和 BYH 的顯卡差在哪？

**A:** AORUS MASTER 16 BZH 和 BYH 的顯卡差在以下方面：

1. **顯示晶片（GPU）**：  
   - BZH：NVIDIA® GeForce RTX™ 5090 Laptop GPU  
   - BYH：NVIDIA® GeForce RTX™ 5080 Laptop GPU  

2. **顯示記憶體（VRAM）**：  
   - BZH：24GB GDDR7  
   - BYH：16GB GDDR7  

3. **最大繪圖功耗**：  
   - BZH：175W Maximum Graphics Power with Dynamic Boost*  
   - BYH：175W Maximum Graphics Power with Dynamic Boost*  

4. **AI Boost 频率**：  
   - BZH：1797 MHz (1597 MHz Boost Clock + 200 MHz OC)*  
   - BYH：1902 MHz (1702 MHz Boost Clock + 200 MHz OC)*  

其他規格（如處理器、螢幕、記憶體、儲存、連接埠等）相同。

### q04 (zh/comparison) — hit=True correct=True cov=0.75

**Q:** AORUS MASTER 16 BXH 是不是 175W 顯卡？如果不是，正確的 Maximum Graphics Power 是多少？

**A:** AORUS MASTER 16 BXH 不是 175W 顯卡。正確的 Maximum Graphics Power 是 140W。

### q05 (zh/comparison) — hit=True correct=True cov=1.00

**Q:** 如果我想買 AORUS MASTER 16 的 RTX 5090 版本，對應的型號是哪個？VRAM 是多少？

**A:** 若你想買 AORUS MASTER 16 的 RTX 5090 版本，對應的型號是 **BZH**。  
VRAM 是 **24GB GDDR7**。

### q06 (zh/comparison) — hit=True correct=True cov=0.75

**Q:** AORUS MASTER 16 的 RTX 5080 版本和 RTX 5070 Ti 版本，顯卡記憶體與功耗各是多少？

**A:** AORUS MASTER 16 RTX 5080 版本的顯示記憶體為 **16GB GDDR7**，功耗為 **175W Maximum Graphics Power with Dynamic Boost***。

AORUS MASTER 16 RTX 5070 Ti 版本的顯示記憶體為 **12GB GDDR7**，功耗為 **140W Maximum Graphics Power with Dynamic Boost***。

### q07 (zh/spec) — hit=True correct=True cov=0.71

**Q:** AORUS MASTER 16 BZH 的螢幕規格是什麼？請包含尺寸、解析度、更新率和面板類型。

**A:** AORUS MASTER 16 BZH 的螢幕規格為：  
尺寸：16"  
解析度：2560×1600（WQXGA）  
更新率：240Hz  
面板類型：OLED

### q08 (zh/spec) — hit=True correct=True cov=0.86

**Q:** AORUS MASTER 16 BYH 的螢幕有支援哪些顯示技術或認證？

**A:** AORUS MASTER 16 BYH 的螢幕支援以下顯示技術或認證：

- NVIDIA® G-SYNC®
- VESA DisplayHDR True Black 500
- VESA ClearMR 10000
- Pantone® Validated
- TÜV Rheinland Low Blue Light
- Dolby Vision®

### q09 (zh/spec) — hit=True correct=True cov=1.00

**Q:** AORUS MASTER 16 BXH 最高可以擴充到多少記憶體？記憶體插槽是什麼配置？

**A:** AORUS MASTER 16 BXH 最高可以擴充到 **64GB DDR5 5600MHz**，記憶體插槽為 **2x SO-DIMM sockets**。

### q10 (zh/multi_spec) — hit=True correct=False cov=0.50

**Q:** AORUS MASTER 16 BZH 有幾個 M.2 SSD 插槽？分別支援什麼 PCIe 規格？

**A:** AORUS MASTER 16 BZH 有 **1個 PCIe Gen5 M.2 插槽**，並支援 **PCIe Gen5** 規格。

### q11 (zh/multi_spec) — hit=True correct=True cov=1.00

**Q:** AORUS MASTER 16 BYH 左右兩側有哪些連接埠？Thunderbolt 5 和 Thunderbolt 4 分別在哪一側？

**A:** AORUS MASTER 16 BYH 左右兩側的連接埠如下：

- 左側：  
  - 1 x DC in  
  - 1 x RJ-45  
  - 1 x HDMI 2.1  
  - 1 x Type-A support USB3.2 Gen2  
  - 1 x Type-C with Thunderbolt™5 (support USB4, DisplayPort™ 1.4 and Power Delivery 3.0)  

- 右側：  
  - 1 x Type-A support USB3.2 Gen2  
  - 1 x Type-C with Thunderbolt™4 (support USB4, DisplayPort™ 1.4 and Power Delivery 3.0)  
  - 1 x MicroSD (UHS-II)  
  - 1 x Audio Jack support mic / headphone combo

### q12 (zh/model_relation) — hit=True correct=True cov=1.00

**Q:** AORUS MASTER 16 AM6H 和 BZH、BYH、BXH 是什麼關係？

**A:** AORUS MASTER 16 AM6H 是這台旗艦 AI 電競筆電的機身模具總型號（Platform 平台型號）。BZH、BYH、BXH 則是同一台 AM6H 機身下、因為搭載不同 NVIDIA GeForce RTX 50 系列獨立顯示卡而區分出的三個具體規格型號（子系列）。三個型號除了顯示晶片（GPU、顯示記憶體、繪圖功耗）不同之外，其餘規格（處理器、螢幕、記憶體、儲存、連接埠等）完全相同。AM6H 並非固定搭載單一顯卡，購買時需在 BZH / BYH / BXH 之間擇一。

### q13 (en/spec) — hit=True correct=False cov=0.60

**Q:** What CPU does the AORUS MASTER 16 BZH use, including cache, maximum frequency, cores, and threads?

**A:** The AORUS MASTER 16 BZH uses an Intel® Core™ Ultra 9 Processor 275HX with the following specifications:
- Cache: 36MB
- Maximum Frequency: 5.4 GHz
- Cores: 24
- Threads: 24

### q14 (en/comparison) — hit=True correct=True cov=1.00

**Q:** I am comparing the AORUS MASTER 16 BZH and BYH. What are the GPU and VRAM differences?

**A:** The GPU and VRAM differences between the AORUS MASTER 16 BZH and BYH are as follows:

- **GPU**: 
  - BZH: NVIDIA® GeForce RTX™ 5090 Laptop GPU
  - BYH: NVIDIA® GeForce RTX™ 5080 Laptop GPU
- **VRAM**:
  - BZH: 24GB GDDR7
  - BYH: 16GB GDDR7

The GPU in the BZH model is the RTX 5090, while the GPU in the BYH model is the RTX 5080. The VRAM capacity is also different, with BZH offering 24GB GDDR7 and BYH offering 16GB GDDR7.

### q15 (en/comparison) — hit=True correct=True cov=1.00

**Q:** Which AORUS MASTER 16 variant comes with the RTX 5070 Ti Laptop GPU, and what is its listed graphics power?

**A:** The AORUS MASTER 16 BXH variant comes with the RTX 5070 Ti Laptop GPU. Its listed graphics power is 140W.

### q16 (en/comparison) — hit=True correct=True cov=1.00

**Q:** Do the RTX 5090 and RTX 5080 versions of AORUS MASTER 16 have the same maximum graphics power?

**A:** The RTX 5090 and RTX 5080 versions of AORUS MASTER 16 have the same maximum graphics power, which is 175W.

### q17 (en/spec) — hit=True correct=True cov=1.00

**Q:** What display specifications are listed for the AORUS MASTER 16 BXH, including size, aspect ratio, resolution, refresh rate, and response time?

**A:** The display specifications for the AORUS MASTER 16 BXH are as follows:  
- **Size**: 16"  
- **Aspect Ratio**: 16:10  
- **Resolution**: 2560×1600 (WQXGA)  
- **Refresh Rate**: 240Hz  
- **Response Time**: 1ms  
- **Display Technology**: OLED  
- **Color Depth**: 100% (DCIP-3)  
- **Brightness**: 500nits (peak)  
- **Contrast Ratio**: 1,000,000:1  
- **Certifications**: NVIDIA® G-SYNC®, NVIDIA® Advanced Optimus, VESA DisplayHDR True Black 500, VESA ClearMR 10000, Pantone® Validated, TÜV Rheinland Low Blue Light, Dolby Vision®

### q18 (en/spec) — hit=True correct=True cov=1.00

**Q:** Does the AORUS MASTER 16 BZH display support G-SYNC, Advanced Optimus, HDR, and Dolby Vision?

**A:** The AORUS MASTER 16 BZH display supports G-SYNC, Advanced Optimus, and Dolby Vision. It does not support HDR.

### q19 (en/spec) — hit=True correct=True cov=1.00

**Q:** How much memory can the AORUS MASTER 16 BYH support, and what memory slot configuration does it use?

**A:** The AORUS MASTER 16 BYH supports up to 64GB DDR5 5600MHz memory, and it uses 2x SO-DIMM sockets for expansion.

### q20 (en/multi_spec) — hit=True correct=True cov=1.00

**Q:** For the AORUS MASTER 16 BXH, what storage slots are available and what is the maximum listed SSD capacity?

**A:** For the AORUS MASTER 16 BXH, available storage slots are 1x PCIe Gen5 M.2 slot and 1x PCIe Gen4x4 M.2 slots. The maximum listed SSD capacity is up to 4TB PCIe NVMe™ M.2 SSD.

### q21 (en/multi_spec) — hit=True correct=True cov=1.00

**Q:** On the AORUS MASTER 16 BZH, which side has Thunderbolt 5 and which side has Thunderbolt 4?

**A:** On the AORUS MASTER 16 BZH, the left side supports Thunderbolt™ 5, and the right side supports Thunderbolt™ 4.

### q22 (en/multi_spec) — hit=True correct=True cov=1.00

**Q:** What networking and wireless features are listed for the AORUS MASTER 16 BYH?

**A:** The networking and wireless features listed for the AORUS MASTER 16 BYH are:  
- LAN: 1G  
- Bluetooth v5.4  
- WIFI 7 (802.11be 2x2)

### q23 (en/multi_spec) — hit=False correct=False cov=0.50

**Q:** What webcam, microphone, and login security features does the AORUS MASTER 16 BXH include?

**A:** The AORUS MASTER 16 BXH includes the following webcam, microphone, and login security features:

- Webcam: FHD (1080p) IR Webcam
- Microphone: Build-in array Microphone
- Login security features: Support Windows Hello Face Authentication

### q24 (en/clarification) — hit=True correct=True cov=0.88

**Q:** Does the AORUS MASTER 16 AM6H refer to one fixed GPU configuration, or do I need to choose between BZH, BYH, and BXH?

**A:** The AORUS MASTER 16 AM6H is not a fixed GPU configuration. You need to choose between BZH, BYH, and BXH based on the NVIDIA GeForce RTX 50 series discrete GPU you want to install.

### q25 (zh/clarification) — hit=True correct=True cov=1.00

**Q:** AORUS MASTER 16 AM6H 固定搭載 RTX 5090 嗎？

**A:** AORUS MASTER 16 AM6H 不固定搭載 RTX 5090，購買時需在 BZH / BYH / BXH 之間擇一。不同型號僅在顯示晶片（GPU、顯示記憶體、繪圖功耗）上有所差異，其他規格完全相同。

### q26 (en/refusal) — hit=None correct=True cov=1.00

**Q:** What is the official USD price and battery life in hours of the AORUS MASTER 16 BZH?

**A:** The official USD price and battery life of the AORUS MASTER 16 BZH are not provided in the spec context.

