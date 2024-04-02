import pandas as pd
from collections import Counter


def count_duplicates(data):
    answerDuplicate = 0
    distractorDuplicate = 0
    countRow = data.shape[0]

    # Turn each dataframe row into a list and loop over the dataframe
    for i in range(countRow):
        rowList = data.loc[i, :].values.flatten().tolist()
        # Count answer-distractor duplicates
        if (rowList[0] == rowList[1]) or (rowList[0] == rowList[2]) or (rowList[0] == rowList[3]):
            answerDuplicate += 1
        
        # # Count distractor-distractor duplicates
        if (rowList[1] == rowList[2]) or (rowList[1] == rowList[3]) or (rowList[2] == rowList[3]):
            distractorDuplicate += 1
    
    return countRow, answerDuplicate, distractorDuplicate


# Get the data and extract the answer and distractors
data = pd.read_csv("datasets/inference_quantitative_analysis.tsv", sep='\t')
newData = data[["answer", "distractor1", "distractor2", "distractor3"]]
countRow, answerDuplicate, distractorDuplicate = count_duplicates(newData)

# print the amount of duplicates
print("Amount of rows in the dataset: " + str(countRow))
print("Amount of answer duplicates in the distractors: " + str(answerDuplicate))
print("Amount of distractor duplicates in the distractors: " + str(distractorDuplicate))
print()


# Get the data and extract the answer and distractors
data2 = pd.read_csv("datasets/sheets_qualitative_analysis_paired.tsv", sep='\t')
newData2 = data2[["answer", "distractor1", "distractor2", "distractor3"]]
countRow, answerDuplicate, distractorDuplicate = count_duplicates(newData2)

# print the amount of duplicates
print("Amount of rows in the dataset: " + str(countRow))
print("Amount of answer duplicates in the distractors: " + str(answerDuplicate))
print("Amount of distractor duplicates in the distractors: " + str(distractorDuplicate))