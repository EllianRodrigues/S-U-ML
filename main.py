from agents.UseCase.UseCase import run_use_case
from agents.UML.UML import run_uml

def main():

    use_case = run_use_case()
    run_uml(use_case)

if __name__ == "__main__":
    main()
