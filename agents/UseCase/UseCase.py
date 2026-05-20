from models.requirement_extraction.Qwen2_5_5B import Qwen25_5B
from models.requirement_extraction.Qwen2_5_3B_Instruct import QwenInstruct
from models.requirement_extraction.Llama3_2_3B_Instruct import Llama3_2_3B_Instruct
import tomli as tomllib

with open("./config.toml", "rb") as file:
    text = tomllib.load(file)["use_case"]["input_text"]

MODEL_MAP = {
	"Qwen25_5B": Qwen25_5B,
	"QwenInstruct": QwenInstruct,
	"Llama3_2_3B_Instruct": Llama3_2_3B_Instruct,
}

def run_use_case(text: str = text, model_name: str = "Qwen25_5B"):
	if model_name not in MODEL_MAP:
		raise ValueError(f"Model '{model_name}' not found. Available models: {list(MODEL_MAP.keys())}")
	
	model_cls = MODEL_MAP[model_name]
	model = model_cls()

	print(f"\nInput Text:\n{text}\n")

	try:
		use_case = model.extract_use_case(text)
		print(f"Extracted Use Case:\n{use_case}")
		return use_case
	except Exception as e:
		print(f"Error extracting use case: {e}")
		return None
