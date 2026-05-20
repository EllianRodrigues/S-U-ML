from models.uml_generation.Qwen2_5_3B_Instruct import QwenInstruct

# Mapa de nomes de modelos para classes
MODEL_MAP = {
    "QwenInstruct": QwenInstruct
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
