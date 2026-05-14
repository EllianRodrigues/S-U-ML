from models.uml_interpretation.Phi3_5_Mini_Instruct import Phi35MiniInstruct


def run_uml_to_use_case(uml_text: str, model_cls=Phi35MiniInstruct):
	model = model_cls()

	print(f"\nInput UML:\n{uml_text}\n")

	try:
		result = model.generate_use_case(uml_text)
		print(f"Generated Use Case:\n{result}")
		return result
	except Exception as e:
		print(f"Error generating use case: {e}")
		return None
