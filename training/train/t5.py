import os

from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import (
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import logging

from . import GlobalBatchSeq2SeqTrainingArguments

logger = logging.get_logger("transformers")

def load(base_model="Salesforce/codet5-large"):
    model = T5ForConditionalGeneration.from_pretrained(base_model)
    tokenizer = T5Tokenizer.from_pretrained(base_model, model_max_length=1024) #VERY_LARGE_INTEGER)
    # https://stackoverflow.com/a/72305836
    #tokenizer.add_tokens([AddedToken("\n", normalized=False), "{", "}", "\\", "^", "`", "~"])
    #model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer

class GlobalBatchSeq2SeqTrainingArguments(Seq2SeqTrainingArguments):
    """
    TrainingArguments class which evenly distributes batch_size on available
    GPUs under distributed training (DistributedDataParallel). Normal
    TrainingArguments use same batch_size on each GPU. (see
    https://discuss.pytorch.org/t/should-we-split-batch-size-according-to-ngpu-per-node-when-distributeddataparallel/72769/15)
    This should also work for DataParallel which does splitting on its own (see
    https://discuss.pytorch.org/t/a-question-concerning-batchsize-and-multiple-gpus-in-pytorch/33767).
    Additionally, batch_size is scaled according to gradient accumulation
    steps.
    """

    def __init__(
        self,
        global_train_batch_size=8,
        global_eval_batch_size=8,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.global_train_batch_size = global_train_batch_size
        self.global_eval_batch_size = global_eval_batch_size
        self.per_device_train_batch_size = self._scale_batch_size(
            global_train_batch_size
        )
        self.per_device_eval_batch_size = self._scale_batch_size(global_eval_batch_size)
        if self.world_size > 1:
            logger.info(f"Dividing batches equally on {self.world_size} processes.")

    def _scale_batch_size(self, batch_size) -> int:
        scaled_batch_size, remainder = divmod(
            batch_size,
            self.world_size * self.gradient_accumulation_steps,
        )
        if remainder != 0:
            raise ValueError(
                "`batch_size` must be divisible by number of processes times gradient accumulation steps."
            )
        return scaled_batch_size


def train(output_dir, model, tokenizer, dataset, overwrite=False):
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
        inputs = [f"{title.strip()}\n\n{desc.strip()}".strip() for title, desc in zip(examples["title"], examples["description"])]
        targets = examples["code"]

        model_inputs = tokenizer(inputs)#, truncation=True)
        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(text_target=targets)#, truncation=True)
        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names,
    )
    logging.info(f"Dataset size before filtering out too long examples: {len(dataset)}")
    dataset = dataset.filter(lambda example: max(len(example['input_ids']), len(example['labels'])) <= tokenizer.model_max_length)
    logging.info(f"Dataset size after filtering out too long examples: {len(dataset)}")

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=tokenizer.pad_token_id,
    )

    training_args = GlobalBatchSeq2SeqTrainingArguments(
        optim="adamw_torch",
        bf16=True,
        ddp_find_unused_parameters=False,
        gradient_accumulation_steps=2,
        global_train_batch_size=8,
        num_train_epochs=30,
        save_total_limit=1,
        output_dir=output_dir
    )

    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training
    trainer.train(resume_from_checkpoint=last_checkpoint)
    trainer.save_model()  # Saves the tokenizer too for easy upload
    trainer.save_state()

    return model, tokenizer
