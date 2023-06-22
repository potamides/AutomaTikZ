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
    argument_parser.add_argument(
        "--model",
        default="llama-7b",
        choices=(
            [f"{model}-{size}" for size in ["7b", "13b", "30b", "65b"] for model in ["llama", "clima"]] +
            [f"t5-{size}" for size in ["base", "small", "large"]]
        ),
        help="specify which language model to fine-tune",
    )
    argument_parser.add_argument(
        "--output",
        default="models/tikz",
        help="directory where to write the model files",
    )
    argument_parser.add_argument(
        "--projector",
        help="url or path to a pretrained projector for clip soft prompts for multimodal models"
    )
    argument_parser.add_argument(
        "--clip_only",
        action="store_true",
        help="condition only on clip soft prompts for multimodal models"
    )
    argument_parser.add_argument(
        "--dataset",
        required=True,
        help="path to the text2tikz dataset (in parquet format)",
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

    args, load_args, train_args = parse_args(), dict(), dict()
    name, size = args.model.split("-")

    if args.debug:
        set_verbosity_debug()
        args.output = join(args.output, "debug")

    if name == "clima":
        assert args.projector, "CLiMA needs a pretrained adapter before fine-tuning!"
        load_args['pretrain_mm_mlp_adapter'] = args.projector
        train_args['clip_only'] = args.clip_only

    model, tokenizer = getattr(train, name).load(size=size, **load_args)
    model, tokenizer = getattr(train, name).train(
        model=model,
        tokenizer=tokenizer,
        gradient_checkpointing=args.gradient_checkpointing,
        output_dir=join(args.output, model.config.name_or_path),
        dataset=load_dataset("parquet", data_files=args.dataset, split="train"),
        **train_args
    )
