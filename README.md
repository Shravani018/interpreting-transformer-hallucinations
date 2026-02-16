# 🧩 Interpreting Transformer Hallucinations

<p align="center">
  <img src="https://img.shields.io/badge/Model-GPT--Neo--125M-blue" alt="Model">
  <img src="https://img.shields.io/badge/Focus-Transformer%20Interpretability-red" alt="Focus">
  <img src="https://img.shields.io/badge/Analysis-Attention%20%7C%20Geometry%20%7C%20Attribution-yellow" alt="Analysis">
  <img src="https://img.shields.io/badge/Architecture-12%20Layers%20%C3%97%2012%20Heads-purple" alt="Architecture">
</p>

An exploration of how GPT-Neo-125M generates factual hallucinations, tracing the pathway from input tokens through attention mechanisms to incorrect outputs.

---

## Main Results

| Finding | Evidence |
|---------|----------|
| **Sparse Computation** | Attribution is highly concentrated in ~10 heads, with the remaining heads contributing minimally |
| **Circuit Structure** | L0_H2 (encoding) -> L6_H4 (aggregation) -> L11_H4 (output) |
| **Attention vs Causation** | Model looks at "Marie" (36%) but period/who cause output (52%) |
| **Geometric Stability** | Layers 2-11 share 0.86-1.00 similarity |
| **First-Token Hubs** | Layers 6-10 aggregate 79-85% attention on first token |

---

## Notebooks

### 01: Model Internals

Establishes foundation for transformer analysis.

- Loads GPT-Neo-125M with attention/hidden state tracking
- Generates text: "Marie Curie was born in the United States..."
- Confirms capture of 40 generation steps across 12 layers
- Validates tensor shapes and data extraction

**Key Stats**: 125M parameters, 12 layers, 768 hidden dimensions, 15 input → 40 generated tokens

---

### 02: Attention Pattern Analysis

Classifies all 144 attention heads by specialization.

- Computes entropy, previous-token attention, first-token attention per head
- Identifies layer-wise computational strategy evolution
- Visualizes attention patterns with heatmaps

**Key Findings**:
- 55 heads show strong first-token attention (>0.5)
- 52 heads highly focused (entropy <0.3)
- Layer 5: Peak previous-token attention (~0.39) for sequential pattern detection
- Layers 6-10: Dominant first-token hubs (0.79-0.85) for global aggregation
- Strategy progression: Broad context → Sequential → Global → Sharp output

---

### 03: Hidden State Trajectory Analysis

Tracks token representation evolution across layers.

- Extracts hidden states for all tokens across 13 layers
- Computes L2 norms, cosine similarity, Euclidean distances
- Analyzes layer-to-layer similarity matrices

**Key Findings**:
- Layer 0: Isolated (0.00-0.03 similarity) - raw embeddings
- Layers 2-11: High similarity (0.86-1.00) - stable processing regime
- Layer 12: Isolated (0.12-0.19) - output projection
- Norm spikes: First token reaches ~10k at Layers 6-8 (aggregation phase)
- Geometric stability: Cosine similarity ~1.0 through middle layers

---

### 04: Attribution Analysis

Identifies causal drivers of hallucination.

Compares three attribution methods:
1. Raw Attention (baseline)
2. Attention Rollout (accumulated flow)
3. Integrated Gradients (gradient-based causation)

**Key Findings**:
- Target token: "United" (position 16)
- Raw attention focuses on "Marie" (36%), "in" (24%)
- Integrated Gradients reveals period "." (29%), "who" (23%) actually cause output
- Top heads: L11_H4 (score=1.0), L0_H2 (0.99), L6_H4 (0.99)
- Divergence proves attention shows where model looks, not what causes output

**Output**: Saves `04_attribution_data.json` for visualization

---

### 05: Interactive Visualizations

Creates three interactive visualizations using attribution data.

**Sankey Flow Diagram**
- Traces input tokens → attention heads → output
- Shows "." and "who" flowing through L0_H2 → L6_H4 → L11_H4
- Interactive HTML with hover tooltips

**Head Attribution Heatmap**
- 12×12 matrix of all attention heads
- Reveals three hotspots: Layer 0, Layer 6, Layer 11
- Confirms 93% of heads contribute minimally

**3D Circuit Graph**
- Network visualization positioned by (layer, head, attribution_score)
- Shows vertical stratification of important heads
- Interactive 3D rotation

**Outputs**: `05_sankey_flow.html`, `05_head_heatmap.html`, `05_circuit_graph.html`

---

## Key Insights

### Hallucination follows a sparse, structured circuit.

The model generates "United States" through a specific pathway:
- **Input**: Period "." and "who" signal biographical context
- **Encoding (L0_H2)**: Structural cues encoded in early layer
- **Aggregation (L6_H4)**: First-token hub consolidates pattern onto "Marie"
- **Output (L11_H4)**: Pattern triggers nationality retrieval from training data
- **Result**: Wrong nationality generated

### Attention Misleads

Raw attention suggests "Marie" drives output (36% attention). Integrated Gradients reveals grammatical structure (period + relative clause) causes 52% of output. What models look at does not equal what causes outputs.

### Sparse Circuits Dominate

Attribution is highly concentrated in approximately 10 heads(7%), with the remaining heads contributing marginally. The remaining 93% are passengers. This suggests interventions like ablating L11_H4 could fix errors without retraining.

### In this case, structural tokens contribute more to the output than semantic tokens.

Sentence structure (punctuation, clauses) drives factual errors more reliably than semantic content. This suggests the error may arise from internal circuit structure rather than simple lack of factual exposure.

### Stable Geometry Enables Error Propagation

Layers 2-11 share stable geometry (0.86-1.00 similarity). This acts as a highway for structural signals to flow from input → middle → output. The hallucination leverages this stability.

---

## Methodology

### Analysis Pipeline

```
Input: "Marie Curie was a physicist who discovered radium."
         ↓
[01] Model Internals → attention + hidden states captured
         ↓
[02] Attention Analysis → First-token hubs in L6-10
         ↓
[03] Geometry Analysis → Stable manifold in L2-11
         ↓
[04] Attribution → "." and "who" cause "United"
         ↓
[05] Visualization → L0_H2 → L6_H4 → L11_H4 circuit
         ↓
Output: "...born in the United States..."
```

### Attribution Methods

**Raw Attention**: Baseline showing where model attends

**Attention Rollout**: Accumulated attention with residual connections across layers

**Integrated Gradients**: Gradient of output with respect to input, integrated along interpolation path (50 steps). Based on [Sundararajan et al., ICML 2017](https://arxiv.org/abs/1703.01365)

### Metrics

- **Entropy**: Shannon entropy of attention distribution (normalized)
- **Cosine Similarity**: Angular distance between hidden state vectors
- **L2 Norm**: Magnitude of token representations
- **Euclidean Distance**: Layer-to-layer representation shift
- **Spearman Correlation**: Rank agreement between attribution methods

---

## Model Architecture

- **Base**: EleutherAI/gpt-neo-125M
- **Type**: Autoregressive transformer
- **Layers**: 12
- **Heads per layer**: 12 (144 total)
- **Hidden dimensions**: 768
- **Parameters**: 125,198,592
- **Vocabulary**: 50,257 tokens
- **Max context**: 2048 tokens

---

## References

This project builds on findings from:

- [Quantifying Attention Flow in Transformers](https://arxiv.org/abs/2005.00928) (Abnar & Zuidema, ACL 2020) - Attention rollout methodology
- [In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) (Olsson et al., Anthropic 2022) - Circuit discovery in transformers
- [Axiomatic Attribution for Deep Networks](https://arxiv.org/abs/1703.01365) (Sundararajan et al., ICML 2017) - Integrated Gradients method
- [Attention is not Explanation](https://arxiv.org/abs/1902.10186) (Jain & Wallace, 2019) - Attention vs causation
- [Transformer Feed-Forward Layers Are Key-Value Memories](https://arxiv.org/abs/2012.14913) (Geva et al., 2021) - Middle layer representations

---
