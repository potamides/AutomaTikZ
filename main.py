#!/usr/bin/env python

from datasets import load_dataset

x = load_dataset("tikz")
x['train'].to_json("tikz.json")
x['train'].to_csv("tikz.csv")

#from tikz.loaders.tex_stackexchange_com import TeXExchangeParser
#
#for x in TeXExchangeParser("/home/amnifilius/.cache/huggingface/datasets/downloads/1afc592b8daf99f2418f61737ee6288d8788ad7e2e82e81276e587ae5131b10d").load():
#    pass
