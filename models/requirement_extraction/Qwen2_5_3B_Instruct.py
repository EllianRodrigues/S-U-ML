import json
import logging
import os
from pathlib import Path
import re

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
    """Simple wrapper to run an instruction-tuned Qwen model for use-case extraction."""

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
        prompt_path = Path(__file__).resolve().parents[2] / "prompts" / "use_case.json"
        with prompt_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)

        prompt = payload.get("extract_use_case_prompt", "").strip()
        if not prompt:
            raise ValueError("Missing 'extract_use_case_prompt' in prompts/use_case.json")
        if "{text}" not in prompt:
            raise ValueError("Prompt template must contain '{text}' placeholder")
        return prompt

    def extract_use_case(self, text: str, max_new_tokens: int = 256) -> str:
        if not text or not isinstance(text, str):
            raise ValueError("Invalid input text")

        prompt = self._build_chat_prompt(text)

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        input_length = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.3,
                top_p=0.8,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_tokens = outputs[0][input_length:]
        result = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        cleaned = self._clean_use_case_text(result)

        if not cleaned:
            return self._fallback_use_case(text)

        return cleaned

    def _clean_use_case_text(self, raw_output: str) -> str:
        candidate = raw_output.strip()
        candidate = re.sub(r"^(assistant|user|system)\s*[:\-\n\s]*", "", candidate, flags=re.I)
        candidate = re.sub(r"<\/?UseCase>", "", candidate, flags=re.I)
        candidate = re.sub(r"Use Case:\s*", "", candidate, flags=re.I)
        candidate = re.sub(r"System Description:\s*", "", candidate, flags=re.I)
        return candidate.strip()

    def _fallback_use_case(self, source_text: str) -> str:
        text = source_text.strip().rstrip(".")
        if not text:
            return ""

        if text and text[-1] not in ".!?":
            text = f"{text}."

        return text

    def _build_chat_prompt(self, text: str) -> str:
        system_prompt = self.prompt_template.split("<SystemDescription>", 1)[0].strip()
        user_prompt = f"<SystemDescription>\n{text.strip()}\n</SystemDescription>"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        return f"System: {system_prompt}\nUser: {user_prompt}\nAssistant:"

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