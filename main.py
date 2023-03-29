#!/usr/bin/env python

from datasets import load_dataset

x = load_dataset("text2tikz")
x['train'].to_json("tikz.json")
x['train'].to_csv("tikz.csv")
