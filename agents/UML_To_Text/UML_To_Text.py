from models.uml_interpretation.Phi3_5_Mini_Instruct import Phi35MiniInstruct


def run_uml_to_text(uml_text: str, model_cls=Phi35MiniInstruct):
	model = model_cls()

	print(f"\nInput UML:\n{uml_text}\n")

	try:
		result = model.generate_uml_to_text(uml_text)
		print(f"Generated Text:\n{result}")
		return result
	except Exception as e:
		print(f"Error generating text: {e}")
		return None
