from abc import ABC, abstractmethod
from functools import cached_property
from typing import List
import warnings

import torch
from torch.cuda import current_device, is_available as has_cuda
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers.pipelines.text_generation import TextGenerationPipeline

# https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py
ALPACA_PROMPTS = {
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

WIZARD_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions. "
    "USER: {data} ASSISTANT:"
)

class ChatBot(ABC):
    def __init__(self, model, bs=1, prefix=None, **tokenizer_kwargs):
        self.tokenizer = LlamaTokenizer.from_pretrained(model, padding_side="left", **tokenizer_kwargs)
        self.prefix = prefix if prefix else ""
        self.model_name = model
        self.bs = bs

    @cached_property
    def pipeline(self):
        pipeline = TextGenerationPipeline(
            batch_size=self.bs,
            model=LlamaForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float16),
            tokenizer=self.tokenizer,
            device=current_device() if has_cuda() else -1
        )
        pipeline.model = torch.compile(self.pipeline.model)

        return pipeline

    def __call__(self, *args, **kwargs):
        return self.generate(*args, **kwargs)

    @abstractmethod
    def _gen_prompts(self):
        """Construction of prompts is model specific"""

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
                prefix=self.tokenizer.bos_token,
                temperature=1,
                top_p=0.95,
                top_k=40,
                num_beams=1,
                max_length=self.tokenizer.model_max_length,
                do_sample=True,
                return_full_text=False,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                # hack to suppress XML output
                begin_suppress_tokens=[self.tokenizer.convert_tokens_to_ids("<")]
            )]

        completions = [self.prefix + completion for completion in completions]
        return completions[0] if isinstance(instructions, str) else completions

class WizardLM(ChatBot):
    def __init__(self, *args, model="TheBloke/WizardLM-13B-1.0-fp16", **kwargs):
        super().__init__(*args, model=model, **kwargs)

    def _gen_prompts(self, instructions, inputs):
        prompts = list()
        for instruction, input_ in zip(instructions, inputs or [None] * len(instructions)):
            if input_ is None:
                prompts.append(WIZARD_PROMPT.format(data=instruction) + " " + self.prefix.lstrip())
            else:
                prompts.append(WIZARD_PROMPT.format(data="\n\n".join([instruction, input_])) + " " + self.prefix.lstrip())

        return prompts

# alpaca implementation below is untested
class Alpaca(ChatBot):
    def __init__(self, *args, model="chavinlo/alpaca-13b", **kwargs):
        super().__init__(*args, model=model, **kwargs)

    def _gen_prompts(self, instructions, inputs):
        prompts = list()
        for instruction, input in zip(instructions, inputs or [None] * len(instructions)):
            if input is None:
                prompts.append(ALPACA_PROMPTS["no_input"].format(instruction=instruction) + self.prefix)
            else:
                prompts.append(ALPACA_PROMPTS["input"].format(instruction=instruction, input=input) + self.prefix)

        return prompts
