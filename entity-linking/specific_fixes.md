# Diagnostic Report: Generic Harmonization Failure Modes in Sector-Specific Patent Data & Stage 2 Rectification Strategy

## Executive Overview

Applying the validated domain-agnostic harmonization pipeline (`paraphrase-multilingual-MiniLM-L12-v2`, equal weights $w = \frac{1}{3}$, unsupervised Otsu thresholding, and Louvain clustering) to the Quantum Computing sector dataset (`harmonization_lookup-2.csv`, comprising **4,109 raw applicant strings** and **3,375 unique cleaned names**) yields **1,933 harmonized clusters**.

Structural inspection shows a strongly bimodal quality distribution:

* **99% of clusters (`~1,915` entities)** are clean and cohesive ($\le 15$ distinct strings; $\text{median} = 1$, $90\text{th percentile} = 2$, $95\text{th percentile} = 3$).
* **~15–20 oversized clusters** suffer from transitive chain-merging driven by sector-specific vocabulary crowding, institutional naming templates, and mixed entity types.

Because the generic backbone performs accurately across the vast majority of the distribution, the optimal engineering design is to **preserve Stage 1 unchanged** and append a **targeted Stage 2 post-processing rectification layer** that operates exclusively on oversized clusters (`n_clean >= 6`).

---

## 1. Taxonomy of Residual Failure Modes

| Failure Mode | Empirical Examples (`harmonization_lookup-2.csv`) | Underlying Mathematical & Structural Mechanism |
| --- | --- | --- |
| **1. Cross-Sector Entity Collisions** | • `HARM_00009` (`18` names): merges `IBM`, `IBM DEUTSCHLAND`, `IBM JAPAN` with `BANK CHINA` and `HUAZHONG AGRICULTURAL UNIV`.<br>

<br>• `HARM_00007` (`86` names): merges `1QB INFORMATION TECH` with `CHENGDU UNIV INFORMATION TECH`.<br>

<br>• `HARM_00022` (`22` names): merges `TENCENT` with `RESEARCH INST HUAZHONG UNIV SCIENCE TECH IN SHENZHEN`.<br>

<br>• `HARM_00013` (`53` names): merges `DELFT UNIV TECH` with `CIQTEK HEFEI TECH`. | Shared generic modifiers (`INFORMATION TECH`, `SCIENCE TECH`, `SHENZHEN`) create bridge edges across distinct institutional classes (corporations, universities, public research institutes, and banks) when all entity types occupy the same similarity graph. |
| **2. Institutional & Legal Template Merging** | • `HARM_00005` (`76` names): merges `TSINGHUA UNIV`, `FUDAN UNIV`, `BEIHANG UNIV`, `HOKKAIDO UNIV`, `EWHA UNIV`.<br>

<br>• `HARM_00038` (`39` names): merges German/Swiss public universities (`ETH ZURICH`, `RWTH AACHEN`, `TU DRESDEN`, `TU MUNCHEN`, `FREIE UNIV BERLIN`).<br>

<br>• `HARM_00053` (`19` names): merges `UNIV ARIZONA`, `FLORIDA STATE UNIV`, `MICHIGAN STATE UNIV`, `OHIO STATE UNIV`.<br>

<br>• `HARM_00030` (`24` names): merges `UNIV TOKYO`, `SEOUL NATIONAL UNIV`, `AUSTRALIAN NATIONAL UNIV`. | Universities and public agencies rely on standardized structural templates (`<GEO> STATE UNIV`, `<GEO> NATIONAL UNIV`, `<UNIV> KORPERSCHAFT OFFENTLICHEN RECHTS`). Multilingual SBERT maps these templates tightly together ($\cos \approx 0.75\text{–}0.85$), while repeated legal/structural words inflate uniform Jaccard and character edit similarity above the Otsu cutoff despite different geographic proper nouns. |
| **3. Domain & Geographic Vocabulary Crowding** | • `HARM_00002` (`100` names): chains `ALPINE QUANTUM TECH`, `BOSON QUANTUM TECH`, `BEIJING QBOSON QUANTUM TECH`, `GUOKAI QIKE QUANTUM TECH`.<br>

<br>• `HARM_00001` (`25` names): merges `ORIGIN QUANTUM COMPUTING`, `ATOM COMPUTING`, `CAMBRIDGE QUANTUM COMPUTING`, `HORIZON QUANTUM COMPUTING`.<br>

<br>• `HARM_00004` (`58` names): merges `BEIJING BAIDU NETCOM SCIENCE TECH`, `BEIJING BOE TECH`, `BEIJING INST TECH`, `BEIJING ACADEMY QUANTUM INFORMATION SCIENCE`. | In a single-sector extract, domain keywords (`QUANTUM`, `COMPUTING`, `INFORMATION`, `TECH`) and major R&D hubs (`BEIJING`, `SHENZHEN`, `HEFEI`) act as high-degree hub tokens. Under uniform Jaccard, two unrelated startups sharing `"QUANTUM COMPUTING"` or `"BEIJING ... TECH"` receive high lexical and semantic overlap, allowing Louvain communities to chain across distinct brands. |
| **4. Residual Unlabeled Individual Inventors** | • `HARM_00126` (`16` names): merges `KIRICHENKO ALEXANDER`, `FISKE MICHAEL STEPHEN`, `COURY MICHAEL`, `BROWN STEPHEN L`. | When PATSTAT's `psn_sector` field is `UNKNOWN` or `NaN`, individual inventors slip past raw sector filters. Multilingual embeddings cluster personal name patterns (`FIRST LAST`) closely together, causing unrelated inventors sharing a common given name (`MICHAEL`, `STEPHEN`) to merge. |

---

## 2. Four-Step Stage 2 Rectification Architecture

To resolve all four failure modes without altering the validated Stage 1 hyperparameters or breaking legitimate multi-variant clusters, apply the following sequential rectification layer to any cluster containing `n_clean >= 6` distinct strings:

### Step 1: Deterministic Entity-Type Stratification (`psn_sector` + Regex Fallback)

* **Mechanism:** Assign every applicant record to one of four mutually exclusive classes—`company`, `university`, `gov_nonprofit`, or `individual`—prioritizing PATSTAT's `psn_sector` field and falling back to deterministic name rules (`UNIV`, `COLLEGE`, `ECOLE`, `HOCHSCHULE` $\rightarrow$ `university`; `INST`, `ACADEMY`, `CNRS`, `MINISTRY`, `HOSPITAL`, `NATIONAL` $\rightarrow$ `gov_nonprofit`; `INC`, `LTD`, `GMBH`, `CORP`, `CO`, `TECH` $\rightarrow$ `company`; 2–3 word personal name patterns without organizational markers $\rightarrow$ `individual`) when `psn_sector` is missing.
* **Impact:**
1. Removes residual unlabeled individuals (`HARM_00126`).
2. Partitions mixed clusters immediately so corporate entities (`IBM`, `TENCENT`, `1QB INFORMATION TECH`) can never merge with universities (`HUAZHONG UNIV`, `CHENGDU UNIV`) or public agencies.



### Step 2: Squared-IDF Token Weighting ($\text{idf}(t)^2$)

* **Mechanism:** Replace uniform token overlap with **squared inverse document frequency Jaccard**, computing token weights across the target sector corpus:

$$w(t) = \left[\log\left(\frac{N}{\text{df}(t)}\right)\right]^2$$



where $N$ is the total number of unique cleaned names and $\text{df}(t)$ is the document frequency of token $t$.
* **Impact:** Standard $\log(N/\text{df})$ leaves moderately frequent sector/city words (`QUANTUM`, `COMPUTING`, `BEIJING`, `TECH`, `UNIV`) with ~50% as much weight as unique brand roots. Squaring IDF compresses ubiquitous domain and geographic words toward zero while preserving high matching weight for genuine brand identifiers (`QBOSON`, `ORIGIN`, `CAMBRIDGE`, `BAIDU`, `BOE`).

### Step 3: Distinguishing-Core Guard for Institutional & Sector Templates

* **Mechanism:** Inside oversized sub-clusters (`n_clean >= 6`), strip high-frequency structural, legal, and domain boilerplate tokens (`UNIV`, `INST`, `TECH`, `SCIENCE`, `RESEARCH`, `NORMAL`, `NATIONAL`, `STATE`, `PUBLIC`, `ROYAL`, `KORPERSCHAFT`, `OFFENTLICHEN`, `RECHTS`, `QUANTUM`, `COMPUTING`, `INFORMATION`, `SYSTEMS`, `GROUP`, `GRID`, `POWER`, plus major administrative regions) to isolate each entity's **distinguishing core** (e.g., `TSINGHUA`, `ARIZONA`, `RWTH AACHEN`, `ATOM`, `ORIGIN`). Require either:
* **Shared distinguishing core token overlap** ($\text{core\_idf\_jaccard} \ge 0.35$), **OR**
* **High core character similarity** ($\text{core\_edit} \ge 0.80$, capturing genuine typos/spacing variants such as `ATOM` vs. `ATOMIC` or `QBOSON QUANTUM TECH` vs. `QBOSON QUANTUM TECHNOLOG`).


* **Impact:** Directly eliminates template merges (`ANHUI NORMAL UNIV` vs. `HUNAN NORMAL UNIV`; `UNIV ARIZONA` vs. `OHIO STATE UNIV`; `ETH ZURICH` vs. `TU MUNCHEN`) while keeping true spelling, word-order, and subsidiary variants unified.

### Step 4: Tightened Sub-Community Detection (`Louvain resolution = 1.35`)

* **Mechanism:** Re-run Louvain community detection with $\text{resolution} = 1.35$ on the rectified edge weights inside each oversized sub-cluster, leaving all small and medium Stage 1 clusters (`n_clean < 6`) untouched.
* **Impact:** Breaks multi-hop transitive bridges without over-fragmenting legitimate corporate families across the rest of the dataset.

---

## 3. Recommended Validation Protocol

Before finalizing the Stage 2 outputs for downstream patent analysis:

1. **Audit the Top 20 Largest Post-Rectification Clusters:** Verify that the largest remaining clusters correspond exclusively to genuine multinational patent filers with many legal/regional subsidiaries (e.g., `IBM`, `HONEYWELL`, `TOSHIBA`, `NTT`, `INTEL`, `MICROSOFT`, `FUJITSU`) rather than multi-firm or multi-university aggregates.
2. **Quantify Consolidation Stability:** Confirm that overall cluster count increases modestly (from `1,933` to roughly `~2,150–2,300` entities) as the ~15 oversized hairballs decompose, while the ~1,915 clean Stage 1 clusters remain 100% invariant.
3. 