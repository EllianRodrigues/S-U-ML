import logging
import os
import json
from pathlib import Path
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import logging as transformers_logging
from dotenv import load_dotenv

# Remove warnings Hugging Face
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Remove logs transformers
transformers_logging.set_verbosity_error()

# Remove logs gerais
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

class Llama3_2_3B_Instruct:
    """
    Class to manage Llama 3.2 3B Instruct model from Hugging Face.
    Performs extraction of requirements and use cases from text.
    """

    def __init__(self, model_name: str = "meta-llama/Llama-3.2-3B-Instruct", device: str = None):
        """
        Initializes the Llama 3.2 3B Instruct model.

        Args:
            model_name (str): Model name from Hugging Face
            device (str): Device to load the model ('cuda' or 'cpu')
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # load env vars (.env)
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
        self.model.eval()
        print(f"Model loaded successfully on device: {self.device}")

    def extract_use_case(self, text: str, max_length: int = 256) -> str:
        """
        Extracts a use case from the provided text.

        Args:
            text (str): Input text for analysis
            max_length (int): Maximum length of generated response

        Returns:
            str: Extracted use case from text
        """
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
            cleaned = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

            if 'Use Case:' in cleaned:
                cleaned = cleaned.split('Use Case:')[-1].strip()

            cleaned = re.sub(r'^(assistant|user|system)\s*[:\-\n\s]*', '', cleaned, flags=re.I)
            cleaned = re.sub(r'System Description:\s*', '', cleaned, flags=re.I)
            cleaned = re.sub(r'Description:\s*', '', cleaned, flags=re.I)

            return cleaned

    def _build_chat_prompt(self, text: str) -> str:
        prompt_head, prompt_tail = self.prompt_template.split("{text}", 1)
        system_prompt = prompt_head.split("System Description:", 1)[0].strip()
        user_prompt = f"System Description:\n{text.strip()}{prompt_tail}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt.strip()},
        ]

        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        return f"System: {system_prompt}\nUser: {user_prompt.strip()}\nAssistant:"
    
    def _load_prompt_template(self) -> str:
        """
        Loads prompt template from prompts/use_case.json.

        Returns:
            str: Prompt template with {text} placeholder.
        """
        prompt_path = Path(__file__).resolve().parents[2] / "prompts" / "use_case.json"
        with prompt_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)

        prompt = payload.get("extract_use_case_prompt", "").strip()
        if not prompt:
            raise ValueError("Missing 'extract_use_case_prompt' in prompts/use_case.json")
        if "{text}" not in prompt:
            raise ValueError("Prompt template must contain '{text}' placeholder")

        return prompt

    def close(self):
        """
        Frees model memory and cleans up resources.
        """
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "tokenizer"):
            del self.tokenizer

        if self.device == "cuda":
            torch.cuda.empty_cache()

        print("\nModel Llama3_2_3B_Instruct unloaded and memory freed")

    def __del__(self):
        """Ensures resources are freed when the object is destroyed."""
        try:
            self.close()
        except Exception:
            pass
