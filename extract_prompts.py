from datasets import load_dataset
import pandas as pd

dataset = load_dataset("allenai/IF_multi_constraints_upto5", split="train")

prompts = []

for example in dataset:
    text = example["messages"][0]["content"]
    constraints = example["constraint"].split('\t')
    stripped_text = text
    for constraint in constraints:
        stripped_text = stripped_text.replace(constraint, "")
    stripped_text = stripped_text.strip()
    prompts.append(stripped_text)

pd.DataFrame(prompts, columns=["prompt"]).to_json("data/prompts.jsonl", lines=True, orient="records")