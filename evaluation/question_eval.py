import pandas as pd
import evaluate
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

# Calculate the jaccard similarity between two texts
def jaccard_similarity(reference, generated):
    similarities = []
    
    for index, question in reference.items():
        # Transform the questions into sets
        set_ref = set(question.lower().split())
        set_gen = set(generated[index].lower().split())

        # Calculate the intersection and union of the sets
        intersection = len(set_ref.intersection(set_gen))
        union = len(set_ref.union(set_gen))

        # Calculate and append the jaccard similarity to the list of similarities
        jacc_sim = float(intersection / union) if union != 0 else 0
        similarities.append(jacc_sim)

    return similarities

def plot_scores(metric, ref_scores, scores):
    mean_scores = np.mean(scores)
    mean_ref_scores = np.mean(ref_scores)
    
    plt.figure(figsize=(12,6))

    plt.axvline(mean_ref_scores, color='blue', linestyle='--')
    plt.axvline(mean_scores, color='orange', linestyle='--')

    plt.xlabel(metric, fontsize='large')
    plt.ylabel('Density', fontsize='large')
    plt.title('Distribution of ' + metric + ' Values', fontsize='x-large')
    plt.legend(fontsize='large')

    plt.savefig(fname="similarity_distributions.png")
    plt.show()

# Calculate the bleu similarity between two texts
def bleu_score(reference, generated):
    return bleu.compute(predictions=generated, references=reference)

def rouge_score(reference, generated):
    return rouge.compute(predictions=generated, references=reference)

# Load the dataset
data = load_data()

# Split the data to extract the context and questions
contexts, org_questions, gen_questions = split_data(data)

print("questions:\n", org_questions)


print("Calculating similarities...")

##############################
###   Jaccard similarity   ###
##############################

# Calculate the similarities between contexts and given questions, to use as reference
org_context_sim_list = jaccard_similarity(contexts, org_questions)

df_org = pd.DataFrame(org_context_sim_list)
print("Similarity between context and original questions:\n", df_org.describe())

# Calculate the similarities between contexts and the corresponding generated questions
gen_context_sim_list = jaccard_similarity(contexts, gen_questions)

df_gen = pd.DataFrame(gen_context_sim_list)
print("Similarity between context and generated questions:\n", df_gen.describe())

# Plot the above calculated similarities
plt.figure(figsize=(10, 6))
sns.kdeplot(org_context_sim_list, label='Ground truth', cut=0, fill=True)
sns.kdeplot(gen_context_sim_list, label='Generated', cut=0, fill=True)

plt.axvline(np.mean(org_context_sim_list), linestyle='--', color='blue')
plt.axvline(np.mean(gen_context_sim_list), linestyle='--', color='orange')

plt.xlabel('Jaccard Similarity')
plt.ylabel('Frequency')
plt.title('Jaccard Similarity Distribution between context and questions')
plt.legend()
plt.savefig(fname="org_vs_gen_sim_context_question.png")
plt.show()

# Calculate the similarities between original questions and generated questions
jacc_score_q = jaccard_similarity(org_questions, gen_questions)

df_q = pd.DataFrame(jacc_score_q)
print("Similarity between original and generated questions:\n", df_q.describe())

# Plot the above calculated similarities
plt.figure(figsize=(10, 6))
sns.kdeplot(jacc_score_q, cut=0, fill=True)

plt.axvline(np.mean(jacc_score_q), linestyle='--', color='blue')

plt.xlabel('Jaccard Similarity')
plt.ylabel('Frequency')
plt.title('Jaccard Similarity Distribution between original and generated questions')
plt.savefig(fname="org_vs_gen_question.png")
plt.show()



##############################
###       Bleu score       ###
##############################

bleu = evaluate.load("bleu")

org_list = [[q] for q in org_questions]

bleu_scores = bleu_score(org_list, gen_questions.to_list())
print(bleu_scores)


##############################
###       Rouge score      ###
##############################

rouge = evaluate.load("rouge")

rouge_scores = rouge_score(org_questions.to_list(), gen_questions.to_list())
print(rouge_scores)


##############################
###      Meteor score      ###
##############################

meteor = evaluate.load("meteor")

meteor_scores = meteor.compute(predictions=gen_questions.to_list(), references=org_questions.to_list())
print(meteor_scores)