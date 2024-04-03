import pandas as pd
from datasets import load_metric

# bleu_metric = load_metric("bleu")

# Load the dataset from a TSV file
def load_data():
    print("Loading data...")
    data = pd.read_csv('../datasets/inference_quantitative_analysis.tsv', sep='\t')

    return data

def split_data(data):
    print("Processing data...")

    # Define an empty list to store the questions
    org_qs = []
    gen_qs = []
    ctxt = []

    # Iterate over the rows in the dataset, keeping track of each row and its index
    for index, row in data.iterrows():
        # Locate the rows where the case is 'ground_truth'
        if row['case'] == 'ground_truth':
            # Look at the next row of the data (if it exists), check if the case of this next row is 'answer given'
            if index + 1 < len(data) and data.loc[index + 1, 'case'] == 'answer_given':
                # If the previous condition hold, save the original question to the extr_qs list
                org_qs.append(data.loc[index, 'question'])
        # If we found a generated question, save the question and the context used for it
        if row['case'] == 'answer_given':
            gen_qs.append(row['question'])
            ctxt.append(row['support'])
            

    # Translate the lists to pandas series
    contexts = pd.Series(ctxt)
    org_questions = pd.Series(org_qs)
    gen_questions = pd.Series(gen_qs)

    return contexts, org_questions, gen_questions

# Calculate the bleu similarity between two texts
# def bleu_score(reference, generated):
#     return bleu_metric.compute(predictions=generated, references=reference)

# Calculate the jaccard similarity between two texts
def jaccard_similarity(reference, generated):
    similarities = []
    
    for index, question in reference.items():
        # Transform the questions into sets
        set_ref = set(question.lower().split())
        set_gen = set(generated[index].lower().split())

        # Calculate the intersection and union of the sets
        intersection = len(set_ref.intersection(set_gen))
        union = len(set_ref.intersection(set_gen))

        # Calculate and append the jaccard similarity to the list of similarities
        jacc_sim = intersection / union if union != 0 else 0
        similarities.append(jacc_sim)

    return similarities

# Load the dataset
data = load_data()

# Split the data to extract the context and questions
contexts, org_questions, gen_questions = split_data(data)


print("Calculating similarities...")

##############################
###  Jaccard similarities  ###
##############################

# Calculate the similarities between contexts and given questions, to use as reference
org_context_sim_list = jaccard_similarity(contexts, org_questions)

# Calculate the similarities between contexts and the corresponding generated questions
gen_context_sim_list = jaccard_similarity(contexts, gen_questions)

# Calculate the similarities between original questions and generated questions
q_sim_list = jaccard_similarity(org_questions, gen_questions)

df = pd.DataFrame(q_sim_list)
print(df.describe())

##############################
###       Bleu score       ###
##############################

# bleu_score_q = bleu_score(org_questions, gen_questions)