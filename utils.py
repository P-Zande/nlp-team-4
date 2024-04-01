import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM


def format_row(row):
    row["formatted"] = f"{row['support']}\n{row['correct_answer']}\n{row['question']}\n{row['distractor1']}\n{row['distractor2']}\n{row['distractor3']}<eos>"
    return row

def format_row_no_answer(row):
    support = row['support'].replace('\n', ' ')
    row["formatted"] = f"{support}\n"
    return row


def format_row_with_answer(row):
    support = row['support'].replace('\n', ' ')
    correct_answer = row['correct_answer'].replace('\n', ' ')
    row["formatted"] = f"{support}\n{correct_answer}\n"
    return row

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model_for_inference(hf_token, model_directory):
    model_id = "google/gemma-2b"
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    base_model = AutoModelForCausalLM.from_pretrained(model_id, token=hf_token)
    model = PeftModel.from_pretrained(base_model, model_directory)
    model = model.merge_and_unload()
    model = model.to(get_device())
    return model, tokenizer

def do_inference(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt").to(get_device())
    outputs = model.generate(**inputs, max_new_tokens=100)
    decoded_outputs = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded_outputs

def parse_output(idx, type, output):
    split_outputs = output.split("\n")
    if len(split_outputs) == 6:
        return f'{idx}\t{type}\t{split_outputs[0]}\t{split_outputs[1]}\t{split_outputs[2]}\t{split_outputs[3]}\t{split_outputs[4]}\t{split_outputs[5]}\n'
    else:
        return ""