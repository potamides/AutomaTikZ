from itertools import chain
import os

from transformers import T5ForConditionalGeneration, RobertaTokenizer
from transformers import (
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import logging

logger = logging.get_logger("transformers")

def load(base_model="Salesforce/codet5-large"):
    model = T5ForConditionalGeneration.from_pretrained(base_model)
    tokenizer = RobertaTokenizer.from_pretrained(base_model, model_max_length=1024) #VERY_LARGE_INTEGER)
    # https://stackoverflow.com/a/72305836
    #tokenizer.add_tokens([AddedToken("\n", normalized=False), "{", "}", "\\", "^", "`", "~"])
    #model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer

def train(
    output_dir: str,
    model,
    tokenizer,
    dataset,
    overwrite=False,
    # training hyperparams
    batch_size: int = 8,
    micro_batch_size: int = 2,
    num_epochs: int = 30,
    learning_rate: float = 3e-4,
    # llm hyperparams
    group_by_length: bool = False,  # faster when True, but produces an odd training loss curve
):
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    gradient_accumulation_steps = batch_size // micro_batch_size
    if ddp := world_size != 1:
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Detecting last checkpoint.
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

    def preprocess_function(examples):
        instructions = list(chain.from_iterable(examples['instructions']))
        # repeat snippets as often as we instructions we have for them
        codesnippets = [tikz for tikz in examples['code'] for _ in range(len(examples['instructions'][0]))]

        model_inputs = tokenizer(instructions)#, truncation=True)
        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(text_target=codesnippets)#, truncation=True)
        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names,
    )
    logger.info(f"Dataset size before filtering out too long examples: {len(dataset)}")
    dataset = dataset.filter(lambda example: max(len(example['input_ids']), len(example['labels'])) <= tokenizer.model_max_length)
    logger.info(f"Dataset size after filtering out too long examples: {len(dataset)}")

    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=Seq2SeqTrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            #gradient_checkpointing=True,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            #fp16=True,
            #tf32=True,
            bf16=True,
            logging_steps=10,
            optim="adamw_torch",
            save_strategy="epoch",
            output_dir=output_dir,
            save_total_limit=1,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
        ),
        data_collator=DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            pad_to_multiple_of=8,
            return_tensors="pt",
            padding=True
        ),
    )

    # Training
    trainer.train(resume_from_checkpoint=last_checkpoint)
    trainer.save_model()
    trainer.save_state()

    return model, tokenizer
