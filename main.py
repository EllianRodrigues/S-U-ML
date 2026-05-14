from agents.UseCase.UseCase import run_use_case
from agents.UML.UML import run_uml
from agents.UML_To_Text.UML_To_Text import run_uml_to_text
from models.sentence_transformers.SentenceModels import SentenceModels

def main():

    use_case = run_use_case()
    uml = run_uml(use_case)
    final_text = run_uml_to_text(uml)

    if use_case and final_text:
        comparator = SentenceModels()
        comparator.print_comparison(use_case, final_text)

if __name__ == "__main__":
    main()
