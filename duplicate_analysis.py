import pandas as pd

cases = ["ground_truth", "answer_given", "no_answer"]

# Read the data from the tsv file
def read_data(filepath):
    data = pd.read_csv(filepath, sep='\t')
    newData = data[["case", "answer", "distractor1", "distractor2", "distractor3"]]

    return newData

# count the answer-distractor and distractor-distractor duplicates per case
def count_duplicates(data, case):
    answerDuplicate = 0
    distractorDuplicate = 0
    countRow = data.shape[0]

    # Turn each dataframe row into a list and loop over the dataframe
    for i in range(countRow):
        rowList = data.loc[i, :].values.flatten().tolist()
        if rowList[0] == case:
            # Count answer-distractor duplicates
            if (rowList[1] == rowList[2]) or (rowList[1] == rowList[3]) or (rowList[1] == rowList[4]):
                answerDuplicate += 1
            
            # # Count distractor-distractor duplicates
            if (rowList[2] == rowList[3]) or (rowList[2] == rowList[4]) or (rowList[3] == rowList[4]):
                distractorDuplicate += 1
    
    return countRow, answerDuplicate, distractorDuplicate

# print the amount of duplicates
def print_results(case, row, answer, distractor):
    print("Case: " + case)
    print("Amount of rows in the dataset: " + str(row))
    print("Amount of answer duplicates in the distractors: " + str(answer))
    print("Amount of distractor duplicates in the distractors: " + str(distractor))
    print()


# Get the data and extract the answer and distractors per case
data = read_data("datasets/inference_quantitative_analysis.tsv")

for case in cases:
    countRow, answerDuplicate, distractorDuplicate = count_duplicates(data, case)
    print_results(case, countRow, answerDuplicate, distractorDuplicate)

# Get the data and extract the answer and distractors per case
data = read_data("datasets/sheets_qualitative_analysis_paired.tsv") 

for case in cases:
    countRow, answerDuplicate, distractorDuplicate = count_duplicates(data, case)
    print_results(case, countRow, answerDuplicate, distractorDuplicate)