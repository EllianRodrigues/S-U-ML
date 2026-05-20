from models.uml_generation.Qwen2_5_3B_Instruct import QwenInstruct
from models.uml_generation.Qwen2_5_5B import Qwen25_5B
from models.uml_generation.Llama3_2_3B_Instruct import Llama3_2_3B_Instruct

MODEL_MAP = {
    "QwenInstruct": QwenInstruct,
    "Qwen25_5B": Qwen25_5B,
    "Llama3_2_3B_Instruct": Llama3_2_3B_Instruct,
}

def run_uml(use_case: str, model_name: str = "QwenInstruct"):
	if model_name not in MODEL_MAP:
		raise ValueError(f"Model '{model_name}' not found. Available models: {list(MODEL_MAP.keys())}")
	
	model_cls = MODEL_MAP[model_name]
	model = model_cls()

	print(f"\nInput Use Case:\n{use_case}\n")

	try:
		result = model.generateUML(use_case)
		print(f"Generated UML:\n{result}")
		return result
	except Exception as e:
		print(f"Error generating UML: {e}")
		return None
