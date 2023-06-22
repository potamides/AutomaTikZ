#!/usr/bin/env python

from datetime import datetime

from datasets import concatenate_datasets, disable_caching, load_dataset

from transformers import set_seed

MODELS = [
    "chatgpt",
    "gpt4"
]

def is_test_candidate(ex, cutoff=datetime(2022, 12, 1)):
    """
    Returns True for human-generated examples newer than llama
    """
    return not ex['origin'] in MODELS and ex['date'] >= cutoff

def train_test_split(dataset):
    if len(cand := dataset.filter(is_test_candidate)):
        cand = cand.add_column("labels", cand.class_encode_column("origin")['origin']).class_encode_column("labels")
        remainder, test = cand.train_test_split(test_size=500, stratify_by_column="labels").values()

        no_cand = dataset.filter(lambda ex: not is_test_candidate(ex))
        train = concatenate_datasets([no_cand, remainder.remove_columns("labels")])

        return train, test.remove_columns("labels")
    return dataset, cand

if __name__ == "__main__":
    set_seed(0)
    disable_caching()

    text2tikz = load_dataset("text2tikz", split="train")
    train, test = train_test_split(text2tikz)

    train.to_parquet("text2tikz-train.parquet")
    test.to_parquet("text2tikz-test.parquet")
