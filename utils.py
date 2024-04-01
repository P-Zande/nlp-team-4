"""Helper functions."""

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def format_row(row: dict[str, str]) -> dict[str, str]:
    """Format the row for training."""
    row["formatted"] = (
        f"{row['support']}\n{row['correct_answer']}\n{row['question']}\n{row['distractor1']}\n{row['distractor2']}\n{row['distractor3']}<eos>"
    )
    return row


def format_row_no_answer(row: dict[str, str]) -> dict[str, str]:
    """Format the row without the answer."""
    support = row["support"].replace("\n", " ")
    row["formatted"] = f"{support}\n"
    return row


def format_row_with_answer(row: dict[str, str]) -> dict[str, str]:
    """Format the row with the answer."""
    support = row["support"].replace("\n", " ")
    correct_answer = row["correct_answer"].replace("\n", " ")
    row["formatted"] = f"{support}\n{correct_answer}\n"
    return row


def get_device() -> str:
    """Get the device to run the model on."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model_for_inference(hf_token: str, model_directory: str) -> tuple[PeftModel, AutoTokenizer]:
    """Load the model and tokenizer for inference."""
    model_id = "google/gemma-2b"
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    base_model = AutoModelForCausalLM.from_pretrained(model_id, token=hf_token)
    model = PeftModel.from_pretrained(base_model, model_directory)
    model = model.merge_and_unload()
    model = model.to(get_device())
    return model, tokenizer


def do_inference(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, text: str) -> str:
    """Run causal LM inference on the given text."""
    inputs = tokenizer(text, return_tensors="pt").to(get_device())
    outputs = model.generate(**inputs, max_new_tokens=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def parse_output(idx: int, answer_type: str, output: str) -> str:
    """Parse the decoded model output into a formatted tab-seperated string."""
    split_outputs = output.split("\n")
    if len(split_outputs) == 6:
        return (
            f"{idx}\t{answer_type}\t"
            f"{split_outputs[0]}\t"
            f"{split_outputs[1]}\t"
            f"{split_outputs[2]}\t"
            f"{split_outputs[3]}\t"
            f"{split_outputs[4]}\t"
            f"{split_outputs[5]}\n"
        )
    return ""
