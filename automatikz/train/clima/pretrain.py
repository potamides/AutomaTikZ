# Adopted from https://github.com/haotian-liu/LLaVA/blob/main/llava/train/train.py. Below is the original copyright:
# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import copy
from dataclasses import dataclass
import os
from typing import Dict, List, Sequence

from PIL import Image
import torch
from torch.utils.data import Dataset
import transformers

from ...util import prepare_model_for_training

IGNORE_INDEX = -100

def preprocess(
    texts: str | List[str],
    tokenizer,
    train_on_inputs,
):
    input_ids = tokenizer(
        texts,
        return_tensors="pt",
        max_length=tokenizer.model_max_length,
        truncation=True,
    )

    input_ids['labels'] = copy.deepcopy(input_ids['input_ids'])

    if not train_on_inputs:
        for label_ids in input_ids['labels']:
            for idx, label_id in enumerate(label_ids):
                if label_id == tokenizer.mask_token_id != label_ids[idx + 1]:
                    label_ids[idx] = IGNORE_INDEX
                    break
                label_ids[idx] = IGNORE_INDEX

    return input_ids


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, dataset, tokenizer, train_on_inputs=False, image_patches=1):
        super(LazySupervisedDataset, self).__init__()

        self.tokenizer = tokenizer
        self.dataset = dataset
        self.image_patches = image_patches
        self.train_on_inputs = train_on_inputs

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        assert isinstance(i, int)
        item = self.dataset[i]
        image = Image.open(item['image'])
        image = self.tokenizer.image(image, return_tensors='pt')['pixel_values']
        data_dict = preprocess(
            texts=(self.image_patches * self.tokenizer.text.mask_token) + item['caption'],
            tokenizer=self.tokenizer.text,
            train_on_inputs=self.train_on_inputs
        )
        return dict(
            input_ids=data_dict["input_ids"][0],
            labels=data_dict["labels"][0],
            image=image[0]
        )


@dataclass
class DataCollatorForSupervisedDataset():
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
            for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.text.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
             batch_first=True,
             padding_value=IGNORE_INDEX)
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.text.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(isinstance(x, torch.Tensor) and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        return batch

def train(
    output_dir: str,
    model,
    tokenizer,
    dataset,
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 1,
    num_epochs: int = 1,
    learning_rate: float = 2e-3,
    gradient_checkpointing = False,
    full_finetune_modules: List[str] = [
        "mm_projector",
    ],
    # llm hyperparams
    train_on_inputs: bool = False,  # if False, masks out inputs in loss
    group_by_length: bool = False,  # faster when True, but produces an odd training loss curve
):

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    gradient_accumulation_steps = batch_size // micro_batch_size
    if ddp := world_size != 1:
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    dataset = LazySupervisedDataset(
        tokenizer=tokenizer,
        dataset=dataset,
        train_on_inputs=train_on_inputs,
        image_patches=model.get_model().vision_tower[0].config.num_patches
    )

    model = prepare_model_for_training(
        model=model,
        modules_to_save=full_finetune_modules,
        use_gradient_checkpointing=gradient_checkpointing
    )
    model.enable_input_require_grads()
    for name, param in model.named_parameters():
        if any(module in name for module in full_finetune_modules):
            param.requires_grad = True

    trainer = transformers.Trainer(
        model=model,
        train_dataset=dataset,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_ratio=0.03,
            weight_decay=0,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            #bf16=True,
            #tf32=True,
            logging_steps=10,
            optim="adamw_torch",
            save_strategy="no",
            output_dir=output_dir,
            ddp_find_unused_parameters=False if ddp else None,
            remove_unused_columns=False,
            group_by_length=group_by_length,
        ),
        data_collator=DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    )

    model.config.use_cache = False
    model = torch.compile(model)
    trainer.train()

    # undo float casting to be able to maintain correct name of lm_head weights
    if type(model.lm_head).__name__ == "CastOutputToFloat":
        model.lm_head = model.lm_head[0]

    model.save_pretrained(
        output_dir,
        state_dict={
            k.split(".", 1)[-1]: v
            for k, v in model.state_dict().items()
            if any(key_match in k for key_match in full_finetune_modules)
        },
    )
    trainer.save_state()

    return model, tokenizer
