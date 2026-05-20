import logging
import os
import json
from pathlib import Path
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import logging as transformers_logging
from dotenv import load_dotenv

# Reduce verbosity
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
transformers_logging.set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

class Gemma2B:
    """
    Wrapper for google/gemma-2b to generate UML diagrams.
    """

    def __init__(self, model_name: str = "google/gemma-2b", device: str = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # load env and token
        load_dotenv()
        self.hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN") or os.getenv("LLAMA_API_KEY")

        self.prompt_template = self._load_prompt_template()

        print(f"Loading model {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=self.hf_token)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map=self.device,
            token=self.hf_token,
        )
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded successfully on device: {self.device}")

    def generateUML(self, text: str, max_length: int = 256) -> str:
        if not text or not isinstance(text, str):
            raise ValueError("Invalid input text")

        prompt = self._build_chat_prompt(text)

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        input_length = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                do_sample=True,
                temperature=0.3,
                top_p=0.8,
                pad_token_id=self.tokenizer.eos_token_id,
            )

            generated_tokens = outputs[0][input_length:]
            result = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

            cleaned = self._extract_or_build_plantuml(result, text)

            lc = cleaned.lower()
            instr_like = any(kw in lc for kw in ("start with", "end with", "do not output", "only output"))
            has_block = "@startuml" in lc and "@enduml" in lc

            if (not has_block) or instr_like:
                try:
                    with torch.no_grad():
                        outputs2 = self.model.generate(
                            **inputs,
                            max_new_tokens=max_length * 2,
                            do_sample=True,
                            temperature=0.7,
                            top_p=0.9,
                            pad_token_id=self.tokenizer.eos_token_id,
                        )

                    raw2 = self.tokenizer.decode(outputs2[0][input_length:], skip_special_tokens=True).strip()
                    cleaned2 = self._extract_or_build_plantuml(raw2, text)
                    if self._is_valid_plantuml(cleaned2):
                        return cleaned2.strip()
                except Exception as e:
                    print(f"DEBUG: retry generation failed: {e}")

            return cleaned.strip()

    def _extract_or_build_plantuml(self, raw_output: str, source_text: str) -> str:
        candidate = raw_output.strip()
        start_match = re.search(r"@startuml", candidate, flags=re.I)
        end_match = re.search(r"@enduml", candidate, flags=re.I)

        if start_match:
            start_idx = start_match.start()
            if end_match and end_match.end() > start_idx:
                candidate = candidate[start_idx:end_match.end()]
            else:
                candidate = candidate[start_idx:]

        candidate = re.sub(r'^(assistant|user|system)\s*[:\-\n\s]*', '', candidate, flags=re.I)
        candidate = re.sub(r'<\/?UML>', '', candidate, flags=re.I)
        candidate = re.sub(r'PlantUML:\s*', '', candidate, flags=re.I)
        candidate = candidate.strip()

        if self._is_valid_plantuml(candidate):
            return candidate

        return self._build_fallback_plantuml(source_text)

    def _is_valid_plantuml(self, candidate: str) -> bool:
        lowered = candidate.lower().strip()
        if not lowered.startswith("@startuml"):
            return False
        if "@enduml" not in lowered:
            return False

        inner = lowered.split("@startuml", 1)[1].split("@enduml", 1)[0].strip()
        if not inner:
            return False

        if "and end with" in inner or "start with" in inner:
            return False

        meaningful_markers = ("actor ", "class ", "rectangle ", "-->", "->", "<--", "<->", "usecase", "use case")
        return any(marker in inner for marker in meaningful_markers)

    def _build_fallback_plantuml(self, source_text: str) -> str:
        text = source_text.lower()

        if "doctor" in text or "doctors" in text:
            actor_name = "Doctor"
        elif "patient" in text or "patients" in text:
            actor_name = "Patient"
        elif "user" in text or "users" in text:
            actor_name = "User"
        elif "admin" in text or "administrator" in text:
            actor_name = "Admin"
        else:
            actor_name = "User"

        if "monitor" in text and "health" in text:
            class_name = "HealthMonitoringSystem"
            relation = "monitor"
        elif "alert" in text:
            class_name = "AlertSystem"
            relation = "alert"
        elif "register" in text:
            class_name = "RegistrationSystem"
            relation = "register"
        elif "manage" in text:
            class_name = "ManagementSystem"
            relation = "manage"
        else:
            class_name = "SystemModule"
            relation = "use"

        return (
            "@startuml\n"
            f"actor {actor_name}\n"
            "rectangle System {\n"
            f"  class {class_name}\n"
            "}\n"
            f"{actor_name} --> {class_name} : {relation}\n"
            "@enduml"
        )

    def _build_chat_prompt(self, text: str) -> str:
        system_prompt = self.prompt_template.split("<UseCase>", 1)[0].strip()
        user_prompt = f"<UseCase>\n{text.strip()}\n</UseCase>"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        if getattr(self.tokenizer, "chat_template", None):
            try:
                return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                pass

        return f"System: {system_prompt}\nUser: {user_prompt}\nAssistant:"

    def _load_prompt_template(self) -> str:
        prompt_path = Path(__file__).resolve().parents[2] / "prompts" / "UML.json"
        with prompt_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)

        prompt = payload.get("prompt", "").strip()
        if not prompt:
            raise ValueError("Missing 'prompt' in prompts/UML.json")
        if "{text}" not in prompt:
            raise ValueError("Prompt template must contain '{text}' placeholder")

        return prompt

    def close(self):
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "tokenizer"):
            del self.tokenizer

        if self.device == "cuda":
            torch.cuda.empty_cache()

        print("\nModel Gemma2B unloaded and memory freed")

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass