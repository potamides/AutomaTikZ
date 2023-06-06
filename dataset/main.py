#!/usr/bin/env python

from datasets import load_dataset

# generate and save dataset
text2tikz = load_dataset("text2tikz")
#newer_than_llama = text2tikz.filter(lambda ex: ex['date'] >= datetime(2022, 12, 1, 0, 0))
text2tikz['train'].to_parquet("text2tikz.parquet")
