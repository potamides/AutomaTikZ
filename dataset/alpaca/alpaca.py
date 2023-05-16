from typing import List
import warnings

import torch
from torch.cuda import current_device, is_available as has_cuda
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers.pipelines.text_generation import TextGenerationPipeline

# https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py
PROMPTS = {
    "input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

class Alpaca():
    def __init__(self, bs=1, model="chavinlo/alpaca-13b", prefix=None):
        self.pipeline = TextGenerationPipeline(
            batch_size=bs,
            model=LlamaForCausalLM.from_pretrained(model, torch_dtype=torch.float16),
            tokenizer=LlamaTokenizer.from_pretrained(model, padding_side="left"),
            device=current_device() if has_cuda() else -1
        )
        self.pipeline.model = torch.compile(self.pipeline.model)
        self.prefix = prefix if prefix else ""

    def __call__(self, *args, **kwargs):
        return self.generate(*args, **kwargs)

    def _gen_prompts(self, instructions, inputs):
        prompts = list()
        for instruction, input in zip(instructions, inputs or [None] * len(instructions)):
            if input is None:
                prompts.append(PROMPTS["no_input"].format(instruction=instruction) + self.prefix)
            else:
                prompts.append(PROMPTS["input"].format(instruction=instruction, input=input) + self.prefix)

        return prompts

    def generate(self, instructions: str | List[str], inputs: str | List[str] = None):
        assert inputs is None or type(instructions) == type(inputs)
        prompts = self._gen_prompts(
                [instructions] if isinstance(instructions, str) else instructions,
                [inputs] if isinstance(inputs, str) else inputs
        )

        # catch false-positive warnings on current transformers master
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # hyperparameters from https://github.com/tatsu-lab/stanford_alpaca/issues/35
            completions = [cmp[0]["generated_text"] for cmp in self.pipeline(
                prompts,
                temperature=0.7,
                top_p=0.9,
                num_beams=1,
                max_new_tokens=600,
                do_sample=True,
                return_full_text=False,
                eos_token_id=self.pipeline.tokenizer.eos_token_id,
                pad_token_id=self.pipeline.tokenizer.pad_token_id,
                # hack to suppress XML output
                begin_suppress_tokens=[self.pipeline.tokenizer.convert_tokens_to_ids("<")]
            )]

        completions = [self.prefix + completion for completion in completions]
        return completions[0] if isinstance(instructions, str) else completions
