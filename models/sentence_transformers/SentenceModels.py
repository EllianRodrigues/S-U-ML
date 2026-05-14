from pathlib import Path
import json

import torch
from sentence_transformers import SentenceTransformer, util


class SentenceModels:
    """Compare two texts using sentence embeddings."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading sentence model {model_name}...")
        self.model = SentenceTransformer(model_name, device=self.device)
        print(f"Sentence model loaded successfully on device: {self.device}")

    def compare_texts(self, initial_text: str, final_text: str) -> dict:
        if not initial_text or not isinstance(initial_text, str):
            raise ValueError("Invalid initial text")
        if not final_text or not isinstance(final_text, str):
            raise ValueError("Invalid final text")

        embeddings = self.model.encode(
            [initial_text, final_text],
            convert_to_tensor=True,
            normalize_embeddings=True,
        )

        similarity = float(util.cos_sim(embeddings[0], embeddings[1]).item())
        distance = 1.0 - similarity

        return {
            "initial_text": initial_text,
            "final_text": final_text,
            "similarity": similarity,
            "distance": distance,
            "match": similarity >= 0.75,
        }

    def print_comparison(self, initial_text: str, final_text: str) -> dict:
        result = self.compare_texts(initial_text, final_text)
        print("\nComparison result:")
        print(f"Similarity: {result['similarity']:.4f}")
        print(f"Distance: {result['distance']:.4f}")
        print(f"Match: {result['match']}")
        return result

    def close(self):
        if hasattr(self, "model"):
            del self.model

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
