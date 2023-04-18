#!/usr/bin/env -S torchrun --nproc_per_node gpu
from argparse import ArgumentParser
from os.path import join

from datasets import load_dataset
from torch.cuda import current_device, is_available as has_cuda
from transformers import (
    Text2TextGenerationPipeline as T2TGP,
    TextGenerationPipeline as TGP,
    set_seed,
)
from transformers.utils.logging import (
    enable_explicit_format,
    set_verbosity_debug,
    set_verbosity_info,
)

import train

set_verbosity_info()
enable_explicit_format()
set_seed(0)

def parse_args():
    argument_parser = ArgumentParser(
        description="Fine-tune language models for poetry generation"
    )
    argument_parser.add_argument(
        "--model",
        default="llama",
        choices=["llama", "t5"],
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
        "--debug",
        action="store_true",
        help="perform a test run on debug verbosity",
    )

    return argument_parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    if args.debug:
        set_verbosity_debug()
        args.output = join(args.output, "debug")

    model, tokenizer = getattr(train, args.model).load()
    model, tokenizer = getattr(train, args.model).train(
        model=model,
        tokenizer=tokenizer,
        output_dir=join(args.output, model.config.name_or_path),
        dataset=load_dataset("json", data_files=args.dataset, split="train")
    )

    def generate(instruction):
        enc_dec = model.config.is_encoder_decoder
        gen_kwargs = dict(
            temperature=0.2,
            #top_p=0.9,
            num_beams=1,
            max_length=1024,
            do_sample=True,
            clean_up_tokenization_spaces=True,
        )
        pipeline = (T2TGP if enc_dec else TGP)(
            model=model,
            tokenizer=tokenizer,
            device=current_device() if has_cuda() else -1
        )
        return pipeline( # pyright: ignore
            instruction + ("" if enc_dec else tokenizer.sep_token),
            **(gen_kwargs | ({} if enc_dec else dict(return_full_text=False)))
        )[0]["generated_text"].strip()
