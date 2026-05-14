import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


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

        print(f"Loading model {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
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

        prompt = f"""Analyze the following text and extract the main use case(s):

Text: {text}

Use Case:"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )

        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the prompt from response
        use_case = result.replace(prompt, "").strip()

        return use_case

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

        print("Model unloaded and memory freed")

    def __del__(self):
        """Ensures resources are freed when the object is destroyed."""
        self.close()
