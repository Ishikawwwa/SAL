# Steering-Aware LoRA for Disentangled Persona Control

This repository contains the research notebooks and experimental artifacts for my Bachelor's thesis:

> **User-Controlled Full-Personality AI Assistants: Extending Persona Vectors to Multi-Dimensional Personality Spaces**

The project investigates whether targeted LoRA fine-tuning can reshape an LLM's activation geometry so that persona steering directions become more separable, stable, and controllable at inference time.

The main result is a **Steering-Aware LoRA (SAL)** pipeline that combines language-modeling fine-tuning with an activation-space separation objective, followed by Contrastive Activation Addition (CAA) vector extraction.

## Research Question

Can reshaping the activation geometry of a language model through targeted LoRA fine-tuning improve the quality, stability, and separability of persona steering vectors?

The broader goal is to move beyond prompt-only persona control toward controllable, composable, and interpretable personality representations in LLMs.

## Main Idea

Persona steering is performed through **Contrastive Activation Addition (CAA)**:

1. Collect positive and contrastive examples for a target persona.
2. Extract residual-stream activations at a selected transformer layer.
3. Compute a normalized difference between positive and negative activation centroids.
4. Inject the resulting vector during generation with a controllable steering coefficient.

The central hypothesis of this project is that CAA vectors become more useful when the model is first fine-tuned to make persona representations geometrically distinct.

## Steering-Aware LoRA Pipeline

The final pipeline is implemented in:

```text
Final-LoRA-training-pipeline.ipynb
```

It performs the following stages:

1. **Persona dataset construction**
   - 78 diverse questions across work, relationships, health, philosophy, technology, finance, creativity, education, and daily life
   - 9 persona-specific answers per question
   - 702 labeled question-answer pairs in total

2. **LoRA adaptation**
   - Base model: `meta-llama/Llama-3.1-8B-Instruct`
   - 4-bit NF4 quantization
   - LoRA applied to attention projections: `q_proj`, `k_proj`, `v_proj`, and `o_proj`
   - Rank: 8
   - Trainable parameters: approximately 6.8M

3. **Two-phase optimization**
   - Phase 1: language-modeling loss only
   - Phase 2: language-modeling loss plus activation-space separation loss
   - The separation loss encourages persona activation centroids to become more distinct at a target residual-stream layer.

4. **CAA vector extraction**
   - Activations are collected across transformer layers.
   - Persona vectors are extracted contrastively from answer-token activations.
   - The best steering layer is selected per persona.

5. **Inference-time steering**
   - Persona vectors are injected into the residual stream during autoregressive generation.
   - Steering strength is controlled through coefficient `alpha`.

## Personas

The final experiment uses nine communication personas:

- Sarcastic
- Supportive
- Wise
- Energetic
- Angry
- Calm
- Formal
- Pessimistic
- Optimistic

Each persona is paired with a contrastive counterpart for CAA extraction and evaluation.

## Experimental Design

The final evaluation compares:

| Condition | Description |
|---|---|
| No steering | Base Llama-3.1-8B-Instruct without intervention |
| Prompt-only | Persona specified through a short system instruction |
| Frozen-model CAA | CAA vectors extracted from the untouched base model |
| LoRA-only | Persona LoRA fine-tuning without separation loss |
| SAL / LoRA-CAA | Two-phase LoRA training with separation loss, followed by CAA extraction |

The experiments assess activation-space separation, qualitative generation behavior, coherence, repetition, capitalization patterns, and sensitivity to steering strength.

## Key Findings

- SAL improved activation-space separation for all nine evaluated personas compared with frozen-model CAA.
- The mean improvement was approximately **2.6 separation units** across personas.
- The strongest observed improvement was for the **formal** persona, while **angry** persona steering was the least stable.
- The two-phase training schedule was important: language-modeling warm-up before the separation objective produced more stable training.
- A separation-loss weight of `0.25` provided the best balance between persona separation and language-modeling quality in the reported ablation.
- High-strength steering can degrade coherence, especially for aggressive or safety-adjacent personas.

## Repository Structure

```text
.
├── Final-LoRA-training-pipeline.ipynb
├── experiments-1.ipynb
├── experiments-3-llama_visualization.ipynb
├── experiments-4.ipynb
├── experiments-5.ipynb
├── experiments-8.ipynb
└── User-Controlled-Full-Personality-AI-Assistants-Thesis-Defense.pdf
```

### Final Artifact

| File | Purpose |
|---|---|
| `Final-LoRA-training-pipeline.ipynb` | Final thesis pipeline: dataset construction, two-phase Steering-Aware LoRA training, CAA extraction, evaluation, ablations, and inference-time persona steering |
| `User-Controlled-Full-Personality-AI-Assistants-Thesis-Defense.pdf` | Final thesis-defense presentation with methodology, setup, results, failure-mode analysis, and future work |

### Supporting Experiments

| File | Purpose |
|---|---|
| `experiments-1.ipynb` | Early activation-steering, residual-stream hooking, causal ablation, sentiment steering, and prototype-based steering experiments on Mistral-7B |
| `experiments-3-llama_visualization.ipynb` | Exploratory visualization and analysis notebook for Llama-based experiments |
| `experiments-4.ipynb` | Prototype-Based Dynamic Steering experiments, including clustering, adaptive prototype selection, multi-persona blending, and steering-strength analysis |
| `experiments-5.ipynb` | Additional prototype-steering experiments and evaluation runs |
| `experiments-8.ipynb` | Additional exploratory experiments developed during the project |

> The `experiments-*.ipynb` notebooks document the research path toward the final SAL pipeline. They may contain incomplete ideas, intermediate implementations, failed runs, exploratory metrics, and model-specific engineering work.

## Installation

The notebooks were developed in Google Colab and are intended to run on a CUDA-enabled environment.

Install the primary dependencies:

```bash
pip install -U \
  torch \
  transformers \
  accelerate \
  peft \
  bitsandbytes \
  datasets \
  tqdm \
  matplotlib \
  scikit-learn \
  scipy \
  ipywidgets
```

Depending on the notebook, you may also need:

```bash
pip install vaderSentiment einops
```

## Hardware

The final pipeline was designed to run on a GPU environment comparable to:

- NVIDIA L4 GPU
- Approximately 24 GB VRAM
- CUDA-enabled PyTorch
- 4-bit NF4 quantization for the Llama-3.1-8B-Instruct model

A Hugging Face account and access token may be required to download gated Meta Llama checkpoints.

## Running the Final Pipeline

1. Open `Final-LoRA-training-pipeline.ipynb` in Google Colab or Jupyter.
2. Configure Hugging Face authentication if required.
3. Install dependencies from the first notebook cells.
4. Select a CUDA runtime.
5. Run the notebook top to bottom.

The notebook creates directories for intermediate artifacts:

```text
steering_lora/
├── data/
├── checkpoints/
└── vectors/
```

## Important Notes

- Persona vectors are highly dependent on the base model, layer, dataset, prompt format, and steering coefficient.
- Stronger steering is not always better: high `alpha` values can cause repetition, incoherence, or behavior collapse.
- Aggressive or confrontational personas may interact with safety-related model representations and therefore require more conservative steering strengths.
- The included persona dataset is designed for research on controllable style and behavior, not for deployment as a general-purpose safety or alignment solution.
- Results from the exploratory notebooks should not be treated as final thesis results unless they are reproduced in the final SAL notebook.

## Limitations

This work has several limitations:

- The dataset is relatively small and manually authored.
- Evaluation relies primarily on activation-space separation, automated coherence proxies, and qualitative generation analysis.
- Persona composition was not fully solved.
- Cross-model transfer was not evaluated systematically.
- High-intensity personas remain difficult to steer reliably under safety-aligned post-training.

## Future Work

Potential extensions include:

- Safety-aware separation losses that reduce interference with refusal-related directions
- Multi-persona composition in a shared activation subspace
- Sparse Autoencoder-based decomposition of persona features
- Larger datasets and broader human evaluation
- Cross-model and cross-scale transfer experiments
- Adaptive steering policies that select layers and coefficients dynamically


## License

This repository is shared for research and educational purposes.

Please ensure that your use of the underlying language models complies with their respective licenses and acceptable-use policies.
