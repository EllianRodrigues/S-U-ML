from models.uml_generation.Qwen2_5_3B_Instruct import QwenInstruct


def run_uml(use_case: str, model_cls=QwenInstruct):
	
	model = model_cls()

	print(f"\nInput Use Case:\n{use_case}\n")

	try:
		result = model.generateUML(use_case)
		print(f"Generated UML:\n{result}")
		return result
	except Exception as e:
		print(f"Error generating UML: {e}")
		return None
