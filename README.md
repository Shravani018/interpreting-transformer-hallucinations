### 🧩interpreting-generative-models
<p align="center">
  <img src="https://img.shields.io/badge/Model-GPT--Neo--125M-blue" />
  <img src="https://img.shields.io/badge/Focus-Transformer%20Interpretability-red" />
  <img src="https://img.shields.io/badge/Analysis-Attention%20%26%20Hidden%20States-green" />
  <img src="https://img.shields.io/badge/Architecture-12%20Layers%20|%20144%20Heads-purple" />
  <img src="https://img.shields.io/badge/Libraries-PyTorch%20|%20Transformers-skyblue" />
</p>
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
- Layer progression: scattered (early) → sequential (middle) → global (late)
- Layer 8 dominates first-token attention, suggesting aggregation role
- Clear computational strategy from broad exploration to focused output
---

#### Model Details

- **Model:** EleutherAI/gpt-neo-125M
- **Architecture:** 12 layers, 12 attention heads per layer, 768 hidden dimensions
- **Parameters:** 125 million
- **Type:** Autoregressive transformer (GPT-style)

---

#### Status

Active development with completed baseline and attention analyses.

---
