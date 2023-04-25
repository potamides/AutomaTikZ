#!/usr/bin/env -S torchrun --nproc_per_node gpu
from argparse import ArgumentParser
from os.path import join

from datasets import load_dataset
from transformers import set_seed
from transformers.utils.logging import (
    enable_explicit_format,
    set_verbosity_debug,
    set_verbosity_info,
)

from scidraw import train

def parse_args():
    argument_parser = ArgumentParser(
        description="Fine-tune language models for text2tikz"
    )
    [f"llama-{size}" for size in ["7b", "13b", "30b", "65b"]]
    [f"t5-{size}" for size in ["base", "small", "large"]]
    argument_parser.add_argument(
        "--model",
        default="llama-7b",
        choices=(
            [f"llama-{size}" for size in ["7b", "13b", "30b", "65b"]] +
            [f"t5-{size}" for size in ["base", "small", "large"]]
        ),
        help="specify which language model to fine-tune",
    )
    argument_parser.add_argument(
        "--output",
        default="models",
        help="directory where to write the model files",
    )
    argument_parser.add_argument(
        "--dataset",
        required=True,
        help="path to the text2tikz dataset (in json format)",
    )
    argument_parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="use gradient checkpointing",
    )
    argument_parser.add_argument(
        "--debug",
        action="store_true",
        help="perform a test run on debug verbosity",
    )

    return argument_parser.parse_args()

if __name__ == "__main__":
    set_verbosity_info()
    enable_explicit_format()
    set_seed(0)

    args = parse_args()

    if args.debug:
        set_verbosity_debug()
        args.output = join(args.output, "debug")

    name, size = args.model.split("-")
    model, tokenizer = getattr(train, name).load(size=size)
    model, tokenizer = getattr(train, name).train(
        model=model,
        tokenizer=tokenizer,
        gradient_checkpointing=args.gradient_checkpointing,
        output_dir=join(args.output, model.config.name_or_path),
        dataset=load_dataset("json", data_files=args.dataset, split="train")
    )