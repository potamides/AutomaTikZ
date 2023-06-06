#!/usr/bin/env -S torchrun --nproc_per_node gpu
from argparse import ArgumentParser
from os.path import isfile, join

from datasets import DownloadManager, load_dataset
from transformers import set_seed
from transformers.utils.logging import (
    enable_explicit_format,
    set_verbosity_debug,
    set_verbosity_info,
)

from scidraw.train.clima import pretrain, load

DATASET_URL = "https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K/raw/main/chat.json"
IMAGES_URL = "https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K/resolve/main/images.zip"

def parse_args():
    argument_parser = ArgumentParser(
        description="Pretrain projection layer for CLiMA model."
    )
    argument_parser.add_argument(
        "--size",
        default="7b",
        choices= ["7b", "13b", "30b", "65b"],
        help="Specify which model size to use.",
    )
    argument_parser.add_argument(
        "--output",
        default="models/projector",
        help="directory where to write the model files",
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

def preprocess(example, path_to_images):
    example['caption'] = example['conversations'][-1]['value']
    example['image'] = join(path_to_images, example['image'])
    assert isfile(example['image'])

    return example

if __name__ == "__main__":
    set_verbosity_info()
    enable_explicit_format()
    set_seed(0)

    args = parse_args()

    if args.debug:
        set_verbosity_debug()
        args.output = join(args.output, "debug")

    dataset = load_dataset("json", data_files=DATASET_URL, split="train")
    train_data = dataset.map(
        preprocess,
        remove_columns=['conversations'],
        fn_kwargs={  # pyright: ignore
            "path_to_images": DownloadManager().download_and_extract(IMAGES_URL),
        }
    )

    model, tokenizer = load(size=args.size)

    model, tokenizer = pretrain(
        model=model,
        tokenizer=tokenizer,
        output_dir=join(args.output, model.config.name_or_path),
        dataset=train_data,
    )
