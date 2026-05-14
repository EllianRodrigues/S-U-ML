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


class Phi35MiniInstruct:
    """Wrapper to transform UML descriptions into use case text."""

    def __init__(self, model_name: str = "microsoft/Phi-3.5-mini-instruct", device: str = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading model {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map=self.device, 
        )

        self.model.to(self.device)
        self.model.eval()

        self.prompt_template = self._load_prompt_template()

        print(f"Model loaded successfully on device: {self.device}")

    def _load_prompt_template(self) -> str:
        prompt_path = Path(__file__).resolve().parents[2] / "prompts" / "uml_to_text.json"
        with prompt_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)

        prompt = payload.get("prompt", "").strip()
        if not prompt:
            raise ValueError("Missing 'prompt' in prompts/uml_to_text.json")
        if "{text}" not in prompt:
            raise ValueError("Prompt template must contain '{text}' placeholder")
        return prompt

    def _build_chat_prompt(self, text: str) -> str:
        prompt_head, prompt_tail = self.prompt_template.split("{text}", 1)
        system_prompt = prompt_head.split("<UML>", 1)[0].strip()
        user_prompt = f"<UML>\n{text}\n</UML>{prompt_tail}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt.strip()},
        ]

        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        return f"{system_prompt}\n\nUser: {user_prompt.strip()}\nAssistant:"

    def generate_uml_to_text(self, text: str, max_new_tokens: int = 256) -> str:
            if not text or not isinstance(text, str):
                raise ValueError("Invalid input text")

            prompt = self._build_chat_prompt(text)

            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False, 
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            # Fatiamos para pegar apenas os tokens gerados
            generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

            result = self.tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True
            ).strip()
            
            
            return result

    def generateUML(self, text: str, max_new_tokens: int = 64) -> str:
        return self.generate_uml_to_text(text, max_new_tokens=max_new_tokens)

    def close(self):
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "tokenizer"):
            del self.tokenizer
        if self.device == "cuda":
            torch.cuda.empty_cache()

        print("\nModel Phi35MiniInstruct unloaded and memory freed")

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass