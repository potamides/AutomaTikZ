from os.path import isfile

from datasets import DownloadManager as DL
from peft import PeftModel
import torch
from transformers import (
    AutoConfig,
    PretrainedConfig,
)
from transformers.utils.hub import get_file_from_repo, is_remote_url

from .. import train
from ..model.clima import register
from ..util import merge_and_unload, temporary_change_attributes

def load(path, **kwargs):
    register()
    if (is_remote:=is_remote_url(path)) or isfile(path): # treat it as a pretrained mm projector
        hidden_size = len(torch.load(DL().download(path) if is_remote else path)['model.mm_projector.weight']) # type: ignore
        size_dict = {
            4096: "7b",
            5120: "13b",
            6656: "30b",
            8192: "65b"
        }
        model, tokenizer = train.clima.load(pretrain_mm_mlp_adapter=path, size=size_dict[hidden_size], model_kwargs=kwargs)
    elif conf_file := get_file_from_repo(path, "adapter_config.json"): # local folder or on huggingface hub
        conf = PretrainedConfig.get_config_dict(conf_file)[0]
        base_model = conf["base_model_name_or_path"]
        model_type = conf.get("model_type", AutoConfig.from_pretrained(base_model).model_type)
        model, tokenizer = getattr(train, model_type).load(base_model=base_model, model_kwargs=kwargs)
        # hack to load adapter weights into ram before merging which saves gpu memory
        with temporary_change_attributes(torch.cuda, is_available=lambda: False):
            model = merge_and_unload(PeftModel.from_pretrained(
                model,
                path,
                torch_dtype=model.config.torch_dtype,
                **kwargs
            ))
    else:
        raise ValueError(f"Cannot load model from {path}.")

    return model.eval(), tokenizer # type: ignore
