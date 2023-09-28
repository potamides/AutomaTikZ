from os.path import isfile

from datasets import DownloadManager as DL
from peft import PeftModel
import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PretrainedConfig,
)
from transformers.utils.hub import get_file_from_repo, is_remote_url

from .. import train
from ..model.clima import register
from ..util import merge_and_unload

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
        hidden_size = len(torch.load(DL().download(path) if is_remote else path)['model.mm_projector.weight']) # type: ignore
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

    return model.eval(), tokenizer # type: ignore
