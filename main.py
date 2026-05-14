from agents.UseCase.UseCase import run_use_case
from agents.UML.UML import run_uml
from agents.UML_To_UseCase.UML_To_UseCase import run_uml_to_use_case
from models.sentence_transformers.SentenceModels import SentenceModels

def main():

    use_case = run_use_case()
    uml = run_uml(use_case)
    final_use_case = run_uml_to_use_case(uml)

    if use_case and final_use_case:
        comparator = SentenceModels()
        comparator.print_comparison(use_case, final_use_case)

if __name__ == "__main__":
    main()
