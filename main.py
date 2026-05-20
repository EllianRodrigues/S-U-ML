from agents.UseCase.UseCase import run_use_case
from agents.UML.UML import run_uml
from agents.UML_To_Text.UML_To_Text import run_uml_to_text
from models.sentence_transformers.SentenceModels import SentenceModels
import tomli as tomllib

with open("./config.toml", "rb") as file:
    config = tomllib.load(file)
    text = config["use_case"]["input_text"]
    model_use_case = config["model_use_case"]["ModelUseCase"][0]
    model_uml = config["UML"]["ModelUML"][0]

def main():

    use_case = run_use_case(text, model_use_case)
    uml = run_uml(use_case, model_uml)
    final_text = run_uml_to_text(uml)

    if use_case and final_text:
        comparator = SentenceModels()
        comparator.print_comparison(use_case, final_text)

if __name__ == "__main__":
    main()
