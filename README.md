## interpreting-generative-models

This repository contains exploratory work focused on understanding and inspecting the internal behavior of generative language models during text generation.

Rather than treating models as black boxes, the project aims to examine what internal signals (such as attention weights and hidden states) are available and how they relate to the generated output.

---

## Goal

The primary goal of this project is to build intuition around how generative models produce explanations and multi-token outputs, and to establish a clear baseline for inspecting generation-time internals before attempting any form of attribution or interpretation.

The emphasis is on observation and analysis, not on proposing a finalized explanation method.

---

## Scope

This repository focuses on:
- autoregressive text generation,
- inspection of attention tensors and hidden states,
- simple, descriptive visualizations of generation behavior.

---

## Current contents

- `01_generation_internals.ipynb`  
  - loads a small causal language model,
  - generates a short response to a simple prompt,
  - inspects attention and hidden-state shapes

This notebook serves as a starting point and baseline.

---

## Status

Early exploratory stage.  
The repository is intended to evolve incrementally as additional analyses are added.

---

## Notes

- Attention-based signals are treated as descriptive indicators only.
- All results should be interpreted with caution.
- Simplicity and clarity are prioritized over completeness.
