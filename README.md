
## S-U-LM: Evaluation of Semantic Preservation in Multi-Agent Pipelines for UML With Slm

This project evaluates semantic preservation across small language models in a multi-agent pipeline that generates and interprets UML.

The main pipeline is: text -> use case -> UML -> text -> comparison with the original text. The goal is to experiment with different model combinations to find which preserves meaning best while minimizing resource usage.

## Repository Structure

- **models/**: model wrappers and utilities grouped by task (examples: [models/uml_generation/Qwen2_5_3B_Instruct.py](models/uml_generation/Qwen2_5_3B_Instruct.py), [models/requirement_extraction/Qwen2_5_5B.py](models/requirement_extraction/Qwen2_5_5B.py)). More models can be added over time.
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

## Model Combinations (Config-Driven)

The pipeline runs all possible combinations between the models listed in `config.toml`:

- `[model_use_case].ModelUseCase` defines the list of models for use-case extraction.
- `[UML].ModelUML` defines the list of models for UML generation.

Execution uses the Cartesian product between both lists.

Example:

- 4 use-case models x 4 UML models = 16 combinations
- 2 use-case models x 3 UML models = 6 combinations

To test fewer combinations, just reduce the lists in `config.toml`.

## Metrics and Outputs

- Sentence comparison uses `sentence-transformers` (for example `all-MiniLM-L6-v2`) and computes semantic similarity with cosine-based metrics.
- One run generates:
	- a consolidated CSV in `runs/` with one row per model combination
	- a detailed execution log in `logs/` with captured prints and stage details
- The CSV includes model pair metadata, stage durations, CPU/GPU usage snapshots, semantic score, prompts, and hardware info.

## How to Run

1. Create and activate the environment (Conda recommended):

```bash
conda create -n sulm python=3.10 -y
conda activate sulm
pip install -r requirements.txt

# (Optional) Install PyTorch with CUDA if you have an NVIDIA GPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

2. Configure models in `config.toml`.

You can provide one model or multiple models in each list.

Example (full 4x4):

```toml
[model_use_case]
ModelUseCase = ["Qwen25_5B", "QwenInstruct", "Llama3_2_3B_Instruct", "Gemma2B"]

[UML]
ModelUML = ["Qwen25_5B", "QwenInstruct", "Llama3_2_3B_Instruct", "Gemma2B"]
```

3. Run the pipeline:

```powershell
python .\main.py
```

This executes all configured model combinations.

Note on API keys / `.env`:

`Llama3_2_3B_Instruct` and `Gemma2B` require a Hugging Face token.

Create a `.env` file in the project root (the repo includes `.env.example`) and define:

```dotenv
HF_TOKEN=your_huggingface_token_here
```

Without this token, those authenticated models may fail to load.

## Analyze Results with Streamlit

After generating CSV files in `runs/`, open the analysis dashboard:

```powershell
streamlit run .\streamlit_app.py
```

The dashboard supports:

- loading the latest CSV automatically (or manual upload)
- filters by combination, success status, and semantic score range
- views for semantic score, stage timing, CPU/GPU/VRAM usage
- charts such as semantic vs consumption and semantic vs consumption vs time
- download of filtered data for further analysis

## Tips for Experimentation

- Edit prompts in `prompts/` to change behavior without touching code.
- Add new model wrappers under `models/` and register them in the corresponding agent MODEL_MAP.
- Start small (e.g., 2x2) to validate outputs and runtime, then scale to 4x4.
- Use `logs/` to investigate failures and `runs/` for aggregate comparisons.



