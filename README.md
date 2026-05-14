
## S(U)LM: Evaluation of Semantic Preservation in Multi-Agent Pipelines for UML

This project evaluates semantic preservation across small language models in a multi-agent pipeline that generates and interprets UML.

The main pipeline is: text -> use case -> UML -> text -> comparison with the original text. The goal is to experiment with different model combinations to find which preserves meaning best while minimizing resource usage.

## Repository Structure

- **models/**: model wrappers and utilities grouped by task (examples: [models/uml_generation/Qwen2_5_3B_Instruct.py](models/uml_generation/Qwen2_5_3B_Instruct.py), [models/requirement_extraction/Qwen2_5_5B.py](models/requirement_extraction/Qwen2_5_5B.py)). More models will be added over time.
- **agents/**: orchestration agents that call models, apply prompts, and perform post-processing (example: [agents/UML/UML.py](agents/UML/UML.py)).
- **prompts/**: prompt templates used by agents and models; edit these JSON files to change behavior without touching code (example: [prompts/UML.json](prompts/UML.json)).
- **config.toml**: initial input text and basic pipeline options.
- **main.py**: pipeline orchestrator that runs the full flow using configured agents and models.

## What each folder contains (summary)

- `models/`: wrappers, tokenization and generation logic for each model family.
- `agents/`: higher-level functions that run model steps and validate outputs.
- `prompts/`: JSON templates for system/user instructions.
- `main.py`: ties everything together and runs experiments.

## Pipeline Flow

1. Input text (use case / initial description) — configured in `config.toml`.
2. Use case extraction (`models/requirement_extraction`).
3. UML generation from the extracted use case (`models/uml_generation`).
4. UML interpretation back into text (`models/uml_interpretation`).
5. Semantic comparison between original text and reconstructed text (`models/sentence_transformers`).

The objective is to measure which model combinations produce the most semantically faithful outputs and which consume fewer resources (tokens, time).

## Metrics and Future Extensions

- Sentence comparison currently uses `sentence-transformers` (for example `all-MiniLM-L6-v2`). The pipeline converts the original and reconstructed texts into embeddings and computes cosine similarity (score between 0 and 1; higher means more similar). Euclidean distance or thresholding can also be used.
- We plan to add more metrics such as token usage, inference time, and aggregated scores exportable for analysis.
- `config.toml` will be extended to allow selecting per-stage models (e.g., a specific model for use-case extraction and another for UML generation) and other runtime options.

## How to Run

1. Create and activate the environment (Conda recommended):

```bash
conda create -n amaspput python=3.10 -y
conda activate amaspput
pip install -r requirements.txt

# (Optional) Install PyTorch with CUDA if you have an NVIDIA GPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

2. Run the pipeline:

```powershell
python .\main.py
```

This executes the pipeline with the currently configured agents and models.

## Tips for Experimentation

- Edit prompts in `prompts/` to change behavior without touching code.
- Add new model wrappers under `models/` and integrate them via `agents/` or `main.py`.
- When `config.toml` supports it, select different models per pipeline stage to compare combinations.



