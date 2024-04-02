"""Training loop for the model."""

import pandas as pd
from scipy.stats import ttest_rel


def main() -> None:
    """Run main function."""
    complete_df = pd.read_csv(
        "datasets/sheets_qualitative_analysis_paired.tsv", sep="\t", index_col=0
    )  # Removed one buggy line
    answer_given = complete_df[complete_df["case"] == "answer_given"]
    no_answer = complete_df[complete_df["case"] == "no_answer"]
    print(complete_df)
    print(answer_given)
    print(no_answer)

    print(
        ttest_rel(
            answer_given["Does the question make sense wrt the given context?"].astype(float),
            no_answer["Does the question make sense wrt the given context?"].astype(float),
        )
    )
    print(
        ttest_rel(
            answer_given["Does the answer belong to the question?"].astype(float),
            no_answer["Does the answer belong to the question?"].astype(float),
        )
    )
    print(
        ttest_rel(
            answer_given["Are the distractors plausible?"].astype(float),
            no_answer["Are the distractors plausible?"].astype(float),
        )
    )


if __name__ == "__main__":
    main()
