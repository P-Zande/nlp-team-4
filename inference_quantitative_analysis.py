"""Run a trained model on the entire SciQ test set for qualitative analysis."""

import os

from datasets import load_dataset
from utils import do_inference, format_row_with_answer, load_model_for_inference, parse_output


def main() -> None:
    """Run main function."""
    model, tokenizer = load_model_for_inference(os.environ.get("HF_TOKEN"), "./model")

    dataset = load_dataset("allenai/sciq")

    with open("datasets/inference_quantitative_analysis.tsv", "w") as f:
        f.write("id\tcase\tsupport\tanswer\tquestion\tdistractor1\tdistractor2\tdistractor3\n")
        for idx, sample in enumerate(dataset["test"]):
            if sample["support"] != "":
                # True answer
                line = (
                    f'{idx}\tground_truth\t'
                    f'{sample["support"]}\t'
                    f'{sample["correct_answer"]}\t'
                    f'{sample["question"]}\t'
                    f'{sample["distractor1"]}\t'
                    f'{sample["distractor2"]}\t'
                    f'{sample["distractor3"]}\n'
                )
                print(line)
                f.write(line)

                # Answer given
                outputs = do_inference(model, tokenizer, format_row_with_answer(sample)["formatted"])
                split_outputs = parse_output(idx, "answer_given", outputs)
                if split_outputs:
                    print(split_outputs)
                    f.write(split_outputs)
                else:
                    print(outputs)
                    print("Line not properly formatted. Skipping...")


if __name__ == "__main__":
    main()
