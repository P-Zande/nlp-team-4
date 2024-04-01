"""Generate 128-length lines from the test set for `detect-pretrain`."""

from datasets import load_dataset
import json



def main():


    dataset = load_dataset("allenai/sciq")

    with open('lines.jsonl', 'w') as f:
        for row in dataset["test"]:
            support = row["support"]
            if len(support) > 127:
                f.write(json.dumps({"input": support[:127]})+ "\n")



if __name__ == '__main__':
    main()

