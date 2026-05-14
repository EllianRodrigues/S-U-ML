import json
import logging
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import logging as transformers_logging

# Reduce transformer verbosity
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
transformers_logging.set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)


class QwenInstruct:
    """Simple wrapper to run an instruction-tuned Qwen model for UML generation."""

    def __init__(self, model_name: str = "qwen/qwen2.5-3b-instruct", device: str = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading model {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )

        self.model.to(self.device)
        self.model.eval()

        self.prompt_template = self._load_prompt_template()

        print(f"Model loaded successfully on device: {self.device}")

    def _load_prompt_template(self) -> str:
        prompt_path = Path(__file__).resolve().parents[2] / "prompts" / "UML.json"
        with prompt_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)

        prompt = payload.get("prompt", "").strip()
        if not prompt:
            raise ValueError("Missing 'prompt' in prompts/UML.json")
        if "{text}" not in prompt:
            raise ValueError("Prompt template must contain '{text}' placeholder")
        return prompt

    def generateUML(self, text: str, max_new_tokens: int = 128) -> str:
        if not text or not isinstance(text, str):
            raise ValueError("Invalid input text")

        prompt = self.prompt_template.replace("{text}", text)

        # Tokenize and move to device
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generation: beam search for more stable output
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                num_beams=4,
                early_stopping=True,
                repetition_penalty=1.1,
                no_repeat_ngram_size=3,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # If model echoes the prompt, strip it
        if result.startswith(prompt):
            result = result[len(prompt) :]

        return result.strip()

    def close(self):
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "tokenizer"):
            del self.tokenizer
        if self.device == "cuda":
            torch.cuda.empty_cache()

        print("\nModel QwenInstruct unloaded and memory freed")

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
