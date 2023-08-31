import os
from random import Random
from types import SimpleNamespace
from typing import Dict, List

from datasets import DownloadManager
from peft import LoraConfig, get_peft_model
import torch
from torch.random import initial_seed
from torch.utils.data import Dataset
from transformers import AddedToken, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import logging
from transformers.utils.hub import is_remote_url

from ...model.clima import ClimaConfig, ClimaForCausalLM
from ...util import PeftTrainer, prepare_model_for_training, save_peft_model
from ..llama import load as llama_load, preprocess
from .pretrain import DataCollatorForSupervisedDataset

logger = logging.get_logger("transformers")

def load(vision_tower="laion/CLIP-ViT-H-14-laion2B-s32B-b79K", pretrain_mm_mlp_adapter=None, *args, **kwargs):
    model, tokenizer =  llama_load(
        *args,
        base_class=ClimaForCausalLM,
        mask_token=AddedToken("<0x1A>" , lstrip=False, rstrip=False),
        **kwargs
    )

    if pretrain_mm_mlp_adapter and is_remote_url(pretrain_mm_mlp_adapter):
        pretrain_mm_mlp_adapter = DownloadManager().download(pretrain_mm_mlp_adapter)

    model.config.model_type = ClimaConfig.model_type # type: ignore
    processor, _ = model.get_model().initialize_vision_modules( # type: ignore
        vision_tower=vision_tower,
        mask_token_id=tokenizer.mask_token_id,
        pretrain_mm_mlp_adapter=pretrain_mm_mlp_adapter
    ).values()

    return model, SimpleNamespace(text=tokenizer, image=processor)

class LazySupervisedMultiModalDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, dataset, tokenizer, train_on_inputs=False, image_patches=1):
        super(LazySupervisedMultiModalDataset, self).__init__()

        self.tokenizer = tokenizer
        self.dataset = dataset
        self.image_patches = image_patches
        self.train_on_inputs = train_on_inputs
        # do not generate the same sequence of random numbers on each worker
        self.random = Random(initial_seed() + int(os.getenv("RANK", 0)))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i) -> Dict[str, torch.Tensor | Dict[str, torch.Tensor]]:
        assert isinstance(i, int)
        item = self.dataset[i]

        # FIXME: __getitem__ should be pure, so randomly selecting the modality
        # is kind of bad
        if self.random.randint(0, 1):
            image = self.tokenizer.image(text=item['caption'], return_tensors="pt", truncation=True)
        else:
            #image = Image.open(BytesIO(item['image']['bytes']))
            image = self.tokenizer.image(images=item['image'], return_tensors='pt')

        return dict(
            input_ids=torch.LongTensor(item["input_ids"]),
            labels=torch.LongTensor(item["labels"]),
            image={k: v[0] for k, v in image.items()}
        )

def train(
    output_dir: str,
    model,
    tokenizer,
    dataset,
    overwrite=False,
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 1,
    num_epochs: int = 12,
    learning_rate: float = 5e-4,
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
        "mm_projector",
        "lm_head"
    ],
    # llm hyperparams
    train_on_inputs: bool = False,  # if False, masks out inputs in loss
    group_by_length: bool = False,  # faster when True, but produces an odd training loss curve
    clip_only: bool = False, # use only the soft prompt of clip for conditioning
):

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    gradient_accumulation_steps = batch_size // micro_batch_size
    if ddp := world_size != 1:
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    def prepare_dataset(dataset):
        dataset = dataset.map(
            preprocess,
            batched=True,
            fn_kwargs=dict(  # pyright: ignore
                tokenizer=tokenizer.text,
                train_on_inputs=train_on_inputs,
                num_patches=(num_patches:=model.get_model().vision_tower[0].config.num_patches), # type: ignore
                clip_only=clip_only
            )
        )
        logger.info(f"Dataset size before filtering out too long examples: {len(dataset)}")
        dataset = dataset.filter(lambda example: len(example['input_ids']) <= tokenizer.text.model_max_length)
        logger.info(f"Dataset size after filtering out too long examples: {len(dataset)}")
        return LazySupervisedMultiModalDataset(
            tokenizer=tokenizer,
            dataset=dataset,
            train_on_inputs=train_on_inputs,
            image_patches=num_patches
        )

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
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `output_dir` or add `overwrite` to train from scratch."
            )

    trainer = PeftTrainer(
        model=model,
        train_dataset=prepare_dataset(dataset),
        args=TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_ratio=0.03,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            #bf16=True,
            #tf32=True,
            logging_steps=10,
            lr_scheduler_type="cosine",
            optim="adamw_torch",
            save_strategy="epoch",
            output_dir=output_dir,
            save_total_limit=1,
            ddp_find_unused_parameters=False if ddp else None,
            remove_unused_columns=False,
            group_by_length=group_by_length,
        ),
        data_collator=DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    )

    model.config.use_cache = False
    model = torch.compile(model)

    trainer.train(resume_from_checkpoint=last_checkpoint)
    save_peft_model(model, output_dir) # type: ignore
    trainer.save_state()

    return model, tokenizer
