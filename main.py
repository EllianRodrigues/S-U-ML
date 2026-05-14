from agents.UseCase.UseCase import run_use_case
from agents.UML.UML import run_uml
from agents.UML_To_UseCase.UML_To_UseCase import run_uml_to_use_case

def main():

    use_case = run_use_case()
    uml = run_uml(use_case)
    run_uml_to_use_case(uml)

if __name__ == "__main__":
    main()
