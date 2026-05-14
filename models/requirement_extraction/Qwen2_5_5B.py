import logging
import os
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import logging as transformers_logging

# Remove warnings Hugging Face
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Remove logs transformers
transformers_logging.set_verbosity_error()

# Remove logs gerais
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

class Qwen25_5B:
    """
    Class to manage Qwen 2.5 5B model from Hugging Face.
    Performs extraction of requirements and use cases from text.
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B", device: str = None):
        """
        Initializes the Qwen 2.5 5B model.

        Args:
            model_name (str): Model name from Hugging Face
            device (str): Device to load the model ('cuda' or 'cpu')
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.prompt_template = self._load_prompt_template()

        print(f"Loading model {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map=self.device,
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

        prompt = self.prompt_template.format(text=text)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        use_case = result.replace(prompt, "").strip()

        return use_case
    
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

        print("\nModel unloaded and memory freed")

    def __del__(self):
        """Ensures resources are freed when the object is destroyed."""
        self.close()
