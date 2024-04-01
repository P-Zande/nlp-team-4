import os

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer


from utils import format_row


def main():

    model_id = "google/gemma-2b"

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.environ.get("HF_TOKEN"))
    model = AutoModelForCausalLM.from_pretrained(model_id, token=os.environ.get("HF_TOKEN"), device_map="cuda")

    dataset = load_dataset("allenai/sciq")
    dataset = dataset.map(lambda sample: format_row(sample), batched=False, remove_columns=["support", "correct_answer", "question", "distractor1", "distractor2", "distractor3"])
    dataset = dataset.map(lambda sample: tokenizer(sample["formatted"]), batched=False)

    lora_config = LoraConfig(
        r=8,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        args=TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=2,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=10,
            output_dir="outputs",
            num_train_epochs=2,
            save_strategy="steps",
            evaluation_strategy="steps",
            save_steps=500,
            eval_steps=500,
        ),
        peft_config=lora_config,
        dataset_text_field="formatted",
    )
    trainer.train()


if __name__ == '__main__':
    main()

