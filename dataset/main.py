#!/usr/bin/env python

from datasets import load_dataset

# generate and save dataset
x = load_dataset("text2tikz")
x['train'].to_json("text2tikz.json")
