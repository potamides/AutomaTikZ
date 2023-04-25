#!/usr/bin/env -S python -i
from argparse import ArgumentParser
from os import listdir
from os.path import isdir, join

from peft import PeftModel
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer, PretrainedConfig
from transformers.utils.logging import enable_explicit_format, set_verbosity_info

from scidraw import train
from scidraw.infer import TikZGenerator
from scidraw.infer.llama import merge_and_unload

def load(path):
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(path)
        tokenizer = AutoTokenizer.from_pretrained(path)
        return model, tokenizer
    except EnvironmentError:
        pass

    if isdir(path) and (conf := "adapter_config.json") in listdir(path):
        base_model = PretrainedConfig.get_config_dict(join(path, conf))[0]["base_model_name_or_path"]
        model_type = AutoConfig.from_pretrained(base_model).model_type
        model, tokenizer = getattr(train, model_type).load(base_model=base_model)
        model = merge_and_unload(PeftModel.from_pretrained(
            model,
            path,
            torch_dtype=model.config.torch_dtype,
        )).eval()

        return model, tokenizer
    else:
        raise ValueError(f"Cannot load model from {path}.")


def parse_args():
    argument_parser = ArgumentParser(
        description="Inference helper for fine-tuned models (best run with python -i)"
    )
    argument_parser.add_argument(
        "--path",
        required=True,
        help="directory where to find models/adapters (local or hub)",
    )
    return argument_parser.parse_args()

if __name__ == "__main__":
    set_verbosity_info()
    enable_explicit_format()

    generate = TikZGenerator(*load(parse_args().path))
