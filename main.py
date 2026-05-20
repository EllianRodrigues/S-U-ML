from agents.UseCase.UseCase import run_use_case
from agents.UML.UML import run_uml
from agents.UML_To_Text.UML_To_Text import run_uml_to_text
from models.sentence_transformers.SentenceModels import SentenceModels
from utils.PipelineLogger import PipelineLogger
import tomli as tomllib


def _ensure_list(value):
    if isinstance(value, list):
        return value
    return [value]


with open("./config.toml", "rb") as file:
    config = tomllib.load(file)
    text = config["use_case"]["input_text"]
    model_use_case_list = _ensure_list(config["model_use_case"]["ModelUseCase"])
    model_uml_list = _ensure_list(config["UML"]["ModelUML"])


def main():
    logger = PipelineLogger(runs_dir="./runs", logs_dir="./logs")
    with logger.capture_output():
        logger.load_prompts(prompts_dir="./prompts")

        logger.set_config({
            "input_text": text,
            "model_use_case_options": model_use_case_list,
            "model_uml_options": model_uml_list,
            "model_uml_to_text": "Phi3_5_Mini_Instruct",
        })

        total_combinations = len(model_use_case_list) * len(model_uml_list)
        combo_index = 0

        for use_case_model in model_use_case_list:
            for uml_model in model_uml_list:
                combo_index += 1
                logger.announce_combo(combo_index, total_combinations, use_case_model, uml_model)
                logger.start_combo(combo_index, total_combinations, use_case_model, uml_model)

                logger.start_stage("use_case_extraction")
                use_case = run_use_case(text, use_case_model)
                logger.end_stage(
                    "use_case_extraction",
                    success=use_case is not None,
                    output=use_case,
                    error=None if use_case else "Failed to extract use case",
                )

                if use_case:
                    logger.start_stage("uml_generation")
                    uml = run_uml(use_case, uml_model)
                    logger.end_stage(
                        "uml_generation",
                        success=uml is not None,
                        output=uml,
                        error=None if uml else "Failed to generate UML",
                    )
                else:
                    uml = None

                if uml:
                    logger.start_stage("uml_to_text")
                    final_text = run_uml_to_text(uml)
                    logger.end_stage(
                        "uml_to_text",
                        success=final_text is not None,
                        output=final_text,
                        error=None if final_text else "Failed to convert UML to text",
                    )
                else:
                    final_text = None

                if use_case and final_text:
                    logger.start_stage("semantic_comparison")
                    comparator = SentenceModels()
                    comparison_result = comparator.compare_texts(use_case, final_text)
                    semantic_score = comparison_result.get("similarity", None)

                    logger.end_stage(
                        "semantic_comparison",
                        success=True,
                        output=f"Similarity score: {semantic_score}",
                    )

                    logger.set_final_comparison(semantic_score=semantic_score, comparison_data=comparison_result)

                logger.finalize_combo()

        csv_path = logger.export_csv()
        logger._log(f"Metrics exported to: {csv_path}")

    logger.print_summary()


if __name__ == "__main__":
    main()
