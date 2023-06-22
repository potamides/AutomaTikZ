from contextlib import contextmanager
from copy import deepcopy
import os
from typing import List

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
import transformers
from transformers import AddedToken
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import logging

from ..util import prepare_model_for_training

logger = logging.get_logger("transformers")

@contextmanager
def temporary_change_attributes(something, **kwargs):
    previous_values = {k: getattr(something, k) for k in kwargs}
    for k, v in kwargs.items():
        setattr(something, k, v)
    try:
        yield
    finally:
        for k, v in previous_values.items():
            setattr(something, k, v)

def load(base_model="decapoda-research/llama-{size}-hf", size="7b", base_class=LlamaForCausalLM, **tokenizer_kwargs):
    base_model = base_model.format(size=size)
    token = lambda s: AddedToken(s, lstrip=False, rstrip=False)
    model = base_class.from_pretrained(base_model,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        torch_dtype=torch.float16
    )
    tokenizer = LlamaTokenizer.from_pretrained(base_model,
        model_max_length=1024, # 1536
        unk_token=token("<unk>"),
        bos_token=token("<s>"),
        eos_token=token("</s>"),
        pad_token=token("<unk>"), # same as unk_token
        sep_token=token("<0x1D>"), # ascii group separator
        padding_side="right", # Note: only for training, need to change to "left" for batched inference
        **tokenizer_kwargs
    )

    return model, tokenizer

def preprocess(examples, tokenizer, train_on_inputs=False, clip_only=False, num_patches=0, min_len=32):
    """Construct model inputs and tokenize them"""
    min_len = min_len + num_patches
    patch_prefix = num_patches * tokenizer.mask_token if num_patches else ""

    if clip_only:
        assert num_patches, "When only using CLIP to process inputs the model needs to be multimodal!"

    def tokenize(texts, add_bos_token=True, add_eos_token=False, add_sep_token=False):
        with temporary_change_attributes(tokenizer, add_bos_token=add_bos_token, add_eos_token=add_eos_token):
            result = tokenizer(texts)
            if add_sep_token:
                for input_ids, attention_mask in zip(result["input_ids"], result["attention_mask"]):
                    input_ids.append(tokenizer.sep_token_id)
                    attention_mask.append(1)
            result["labels"] = deepcopy(result["input_ids"])

        return result

    def try_truncate(ids, max_len):
        while len(ids) > max_len and not len(ids) <= min_len:
            for idx in reversed(range(len(ids))):
                # make sure to not remove special tokens
                if ids[idx] not in tokenizer.all_special_ids:
                    ids.pop(idx)
                    break
            else:
                break
        return ids

    captions = tokenize([patch_prefix + ("" if clip_only else caption) for caption in examples['caption']], add_sep_token=True)
    codesnippets = tokenize(examples['code'], add_bos_token=False, add_eos_token=True)

    if not train_on_inputs:
        captions["labels"] = [[-100] * len(labels) for labels in captions["labels"]]

    for key, val in codesnippets.items():
        for instruction_ids, code_ids in zip(captions[key], val):
            # try to truncate caption, when len(caption) + len(code) > tokenizer.model_max_length
            try_truncate(instruction_ids, tokenizer.model_max_length - len(code_ids)).extend(code_ids)

    return captions

# https://github.com/tloen/alpaca-lora#official-weights
def train(
    output_dir: str,
    model,
    tokenizer,
    dataset,
    overwrite=False,
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 1,
    num_epochs: int = 10,
    learning_rate: float = 3e-4,
    gradient_checkpointing = False,
    # lora hyperparams
    lora_r: int = 64,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [ # defaults to all linear layers of llama
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        'up_proj',
        'down_proj',
        'gate_proj'
    ],
    full_finetune_modules: List[str] = [
        "embed_tokens",
        "lm_head"
    ],
    # llm hyperparams
    train_on_inputs: bool = False,  # if False, masks out inputs in loss
    group_by_length: bool = False,  # faster when True, but produces an odd training loss curve
):
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    gradient_accumulation_steps = batch_size // micro_batch_size
    if ddp := world_size != 1:
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        modules_to_save=full_finetune_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(prepare_model_for_training(
        model=model,
        modules_to_save=full_finetune_modules,
        use_gradient_checkpointing=gradient_checkpointing),
        peft_config=config
   )

    last_checkpoint = None
    if os.path.isdir(output_dir) and not overwrite:
        last_checkpoint = get_last_checkpoint(output_dir)
        if last_checkpoint is None and len(os.listdir(output_dir)) > 0:
            raise ValueError(
                f"Output directory ({output_dir}) already exists and is not empty. "
                "Use `overwrite` to overcome."
            )
        elif last_checkpoint is not None:
            # Check the available weights and load them
            checkpoint_name = os.path.join(
                last_checkpoint, "pytorch_model.bin"
            )  # Full checkpoint
            if not os.path.exists(checkpoint_name):
                checkpoint_name = os.path.join(
                    last_checkpoint, "adapter_model.bin"
                )  # only LoRA model - LoRA config above has to fit
                last_checkpoint = (
                    False  # So the trainer won't try loading its state
                )
            # The two files above have a different name depending on how they were saved, but are actually the same.
            if os.path.exists(checkpoint_name):
                logger.info(
                    f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                    "the `output_dir` or add `overwrite` to train from scratch."
                )
                adapters_weights = torch.load(checkpoint_name)
                model = set_peft_model_state_dict(model, adapters_weights)

    train_data = dataset.map(
        preprocess,
        batched=True,
        remove_columns=dataset.column_names,
        fn_kwargs=dict(  # pyright: ignore
            tokenizer=tokenizer,
            train_on_inputs=train_on_inputs
        )
    )
    logger.info(f"Dataset size before filtering out too long examples: {len(train_data)}")
    train_data = train_data.filter(lambda example: len(example['input_ids']) <= tokenizer.model_max_length)
    logger.info(f"Dataset size after filtering out too long examples: {len(train_data)}")

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            #bf16=True,
            #tf32=True,
            logging_steps=10,
            optim="adamw_torch",
            save_strategy="epoch",
            output_dir=output_dir,
            save_total_limit=1,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))

    model = torch.compile(model)
    trainer.train(resume_from_checkpoint=last_checkpoint)

    # undo float casting to be able to maintain correct name of lm_head weights
    if type(model.lm_head).__name__ == "CastOutputToFloat":
        model.base_model.model.lm_head = model.lm_head[0]

    model.save_pretrained(output_dir)
    trainer.save_state()

    return model, tokenizer
