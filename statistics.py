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

    answer_given1 = answer_given["Does the question make sense wrt the given context?"].astype(float)
    no_answer1 = no_answer["Does the question make sense wrt the given context?"].astype(float)

    answer_given2 = answer_given["Does the answer belong to the question?"].astype(float)
    no_answer2 = no_answer["Does the answer belong to the question?"].astype(float)

    answer_given3 = answer_given["Are the distractors plausible?"].astype(float)
    no_answer3 = no_answer["Are the distractors plausible?"].astype(float)

    print("\nQuestions (answer given v.s. no answer)")
    print(answer_given1.describe())
    print(no_answer1.describe())
    print(
        ttest_rel(
            answer_given1,
            no_answer1,
        )
    )

    print("\nAnswers (answer given v.s. no answer)")
    print(answer_given2.describe())
    print(no_answer2.describe())
    print(
        ttest_rel(
            answer_given2,
            no_answer2,
        )
    )

    print("\nDistractors (answer given v.s. no answer)")
    print(answer_given3.describe())
    print(no_answer3.describe())
    print(
        ttest_rel(
            answer_given3,
            no_answer3,
        )
    )


if __name__ == "__main__":
    main()
