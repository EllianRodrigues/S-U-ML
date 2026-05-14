from models.requirement_extraction.Qwen2_5_5B import Qwen25_5B
import tomli as tomllib

with open("./config.toml", "rb") as file:
    text = tomllib.load(file)["use_case"]["input_text"]

def run_use_case(text: str = text, model_cls=Qwen25_5B):
	
	model = model_cls()

	print(f"\nInput Text:\n{text}\n")

	try:
		use_case = model.extract_use_case(text)
		print(f"Extracted Use Case:\n{use_case}")
		return use_case
	except Exception as e:
		print(f"Error extracting use case: {e}")
		return None
