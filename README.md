### 🧩 Interpreting Generative Models
<p align="center">

![Model](https://img.shields.io/badge/Model-GPT--Neo--125M-blue)
![Focus](https://img.shields.io/badge/Focus-Transformer%20Interpretability-red)
![Analysis](https://img.shields.io/badge/Analysis-Attention%20%26%20Hidden%20States-green)
![Architecture](https://img.shields.io/badge/Architecture-12%20Layers%20%C3%97%2012%20Heads-purple)
![Libraries](https://img.shields.io/badge/Libraries-PyTorch%20%7C%20Transformers-skyblue)

</p align="center">

This repository contains exploratory work focused on understanding and inspecting the internal behavior of generative language models during text generation.

Rather than treating models as black boxes, this project examines what internal signals (such as attention weights and hidden states) are available and how they relate to the generated output.

---

#### Goal:
Building intuition about how transformers process information by 
analyzing attention patterns and hidden state evolution.

---

#### Current Contents

#### `01_llm_interpretability_baseline.ipynb`
**Establishes foundation for transformer internals analysis**
- Loads GPT-Neo-125M model
- Generates text with attention and hidden state tracking
- Inspects attention tensor shapes and statistics across layers
- Analyzes hidden state evolution from input to output layers
- Provides baseline understanding of model internals

**Key observations:**
- Attention exhibits sharp focus (low mean, peaks ~1.0)
- Hidden state magnitudes grow in deeper layers
- Representation norms increase substantially from first to last layer

#### `02_attention_pattern_analysis.ipynb`
**Deep dive into attention head specialization**
- Classifies all 144 attention heads (12 layers × 12 heads)
- Identifies specialized head types: previous-token, first-token, focused, broad
- Visualizes attention patterns with heatmaps
- Tracks attention strategy evolution across network depth

**Key findings:**
- Strong head specialization (e.g., Layer 5 Head 8: 99.96% previous-token attention)
- Layer progression: scattered (early) -> sequential (middle) -> global (late)
- Layer 8 dominates first-token attention, suggesting aggregation role
- Clear computational strategy from broad exploration to focused output

#### `03_hidden_state_analysis.ipynb`
**Hidden state trajectory analysis across layers**
- Extracts token representation trajectories across all 13 layers (embedding + 12)
- Computes geometric metrics: L2 norms, cosine similarity, and Euclidean distance 
  between consecutive layers
- Analyzes pairwise layer similarity to identify representational clusters

**Key findings:**
- First token norm spikes dramatically in layers 6–8 (~10k), directly corroborating 
  the aggregation role identified in notebook 02
- Representation directions are highly stable through middle layers (cosine sim ~1.0) 
  but undergo sharp geometric reorganization at transitions 0->1 and 11->12
- Largest Euclidean shifts occur at boundary layers; middle layers (3–4, 7–9) show 
  minimal change (~100–150), confirming a stable mid-network processing plateau
- Layers 2–9 form a high-similarity block (0.91–1.00), while layers 0 and 12 are 
  geometrically isolated — confirming distinct input, processing, and output regimes
---

#### Model Details

- **Model:** EleutherAI/gpt-neo-125M
- **Architecture:** 12 layers, 12 attention heads per layer, 768 hidden dimensions
- **Parameters:** 125 million
- **Type:** Autoregressive transformer (GPT-style)

---

#### Status
Active development baseline, attention completed and hidden state analyses in progress.

---
