from datasets import DownloadManager
import torch
from transformers import (
    Blip2Config,
    Blip2ForConditionalGeneration,
    Blip2Processor,
    BlipImageProcessor,
    LlamaConfig,
    LlamaForCausalLM,
)

from .llama import load as load_llama, temporary_change_attributes

KEYS_TO_MODIFY_MAPPING = {
    ".positional_embedding":".embeddings.position_embedding.weight",
    ".token_embedding":".embeddings.token_embedding",
    "text.text_projection":"text_projection.weight",
    ".ln_final":".final_layer_norm",
    "text.":"text_model.",
}

# exact clip model is EVA-CLIP/EVA01_CLIP_g_14_psz14_s11B.pt
def load(base_model="decapoda-research/llama-{size}-hf", size="7b", model_kwargs={}):
    assert size in ["7b", "13b"]

    #import os
    #base_model = os.getenv('HOME') + "/vicuna-{size}"
    llama_name = base_model.format(size=size)
    blip_name = "Salesforce/blip2-flan-t5-xxl"
    linear_name = "https://drive.google.com/uc?id={id}".format(
        id="1RY9jV0dyqLX-o38LrumkKRh6Jtaop58R" if size == "7b" else "1a4zLvaiDBr-36pasffmgpvH5P7CKmpze"
    )

    blipconf = Blip2Config.from_pretrained(
        pretrained_model_name_or_path="Salesforce/blip2-flan-t5-xxl",
        use_decoder_only_language_model=True,
        text_config=LlamaConfig.from_pretrained(
            pretrained_model_name_or_path=llama_name,
            architectures=[
                LlamaForCausalLM.__name__
            ]
        )
    )

    # quickfix for false-positive warnigns for missing and unexpected keys
    with temporary_change_attributes(Blip2ForConditionalGeneration,
        _keys_to_ignore_on_load_missing=["language_model.*"],
        _keys_to_ignore_on_load_unexpected=["language_model.*"],
    ):
        # use low_cpu_mem_usage as a quickfix for this bug: https://github.com/huggingface/transformers/issues/22563
        blip_model = Blip2ForConditionalGeneration.from_pretrained(blip_name,
            config=blipconf,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            **model_kwargs
        )

    blip_model.language_model, llama_tokenizer = load_llama(base_model, size, add_bos_token=False, model_kwargs=model_kwargs) # type: ignore
    linear = torch.load(DownloadManager().download(linear_name))['model'] # type: ignore
    blip_model.language_projection.load_state_dict(dict( # type: ignore
        weight=linear['llama_proj.weight'],
        bias=linear['llama_proj.bias']
    ))

    blip_processor = Blip2Processor(
        image_processor=BlipImageProcessor.from_pretrained(blip_name),
        tokenizer=llama_tokenizer
    )

    return blip_model, blip_processor

def train(*args, **kwargs):
    raise NotImplementedError
