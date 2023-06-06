#!/usr/bin/env -S python -i
from argparse import ArgumentParser
from os.path import isfile
from sys import flags

from PIL import UnidentifiedImageError
from datasets import DownloadManager as DL
from peft import PeftModel
import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PretrainedConfig,
    set_seed,
)
from transformers.utils.hub import get_file_from_repo, is_remote_url

from scidraw import train
from scidraw.infer import TikZGenerator
from scidraw.model.clima import register
from scidraw.util import merge_and_unload

try:
    import readline # patches input()
except:
    pass #readline not available

def load(path):
    register()
    for AutoModel in [AutoModelForSeq2SeqLM, AutoModelForCausalLM]:
        try:
            model = AutoModel.from_pretrained(path)
            tokenizer = AutoTokenizer.from_pretrained(path)
            return model, tokenizer
        except EnvironmentError:
            pass

    if (is_remote:=is_remote_url(path)) or isfile(path): # treat it as a pretrained mm projector
        hidden_size = len(torch.load(DL().download(path) if is_remote else path)['model.mm_projector.weight'])
        size_dict = {
            4096: "7b",
            5120: "13b",
            6656: "30b",
            8192: "65b"
        }
        model, tokenizer = train.clima.load(pretrain_mm_mlp_adapter=path, size=size_dict[hidden_size])
    elif conf_file := get_file_from_repo(path, "adapter_config.json"): # local folder or on huggingface hub
        conf = PretrainedConfig.get_config_dict(conf_file)[0]
        base_model = conf["base_model_name_or_path"]
        model_type = conf.get("model_type", AutoConfig.from_pretrained(base_model).model_type)
        model, tokenizer = getattr(train, model_type).load(base_model=base_model)
        model = merge_and_unload(PeftModel.from_pretrained(
            model,
            path,
            torch_dtype=model.config.torch_dtype,
        ))
    else:
        raise ValueError(f"Cannot load model from {path}.")

    return model.eval(), tokenizer

def parse_args():
    argument_parser = ArgumentParser(
        description="Inference helper for fine-tuned models."
    )
    argument_parser.add_argument(
        "--path",
        required=True,
        help="path or url to projector weights or directory where to find models/adapters (local or hub)",
    )
    return argument_parser.parse_args()

if __name__ == "__main__":
    set_seed(0)
    generate = TikZGenerator(*load(parse_args().path))
    desc = dict(
        caption="the caption",
        image="optional input fed into CLIP, defaults to the caption (can be a Pillow Image, a URI to an image, or a caption)"
    )

    if flags.interactive:
        print("generate(*args, **kwargs):", str(TikZGenerator.generate.__doc__).strip())
    else:
        print("Starting a REPL for generating TikZ. Arguments:", *[f"\t{k}: {v}" for k, v in desc.items()], sep="\n")
        while True:
            try:
                caption = input("Caption: ")
                image = input("Image (optional): ") if generate.processor else None
            except (KeyboardInterrupt, EOFError):
                break
            try:
                generate(caption=caption, image=image or None)
            except KeyboardInterrupt:
                pass
            except UnidentifiedImageError:
                print("Error: Cannot identify image file!")
