"""
Combination of various sources of tikz descriptions with aligned code.
"""

from itertools import chain, islice
from re import sub

from alpaca import Alpaca
from datasets import Features, Sequence, Value, builder
from datasets.info import DatasetInfo
from datasets.splits import Split, SplitGenerator
from datasets.utils import logging
from pyarrow import Table

from text2tikz.loaders import (
    janosh_tikz,
    petarv_tikz,
    tex_stackexchange_com,
    texample_net,
    tikz_net,
)

logger = logging.get_logger("transformers")

PROMPT_TEMPLATES = {
    "example": "Rephrase the following description of a TikZ picture into a detailed instruction. Do not leave out any information. The instruction will be given to a language model as a prompt to generate the corresponding TikZ picture. {formulation}",
    "stack_exchange": "Rephrase the following Stack Exchange question about TikZ into a detailed instruction. Do not leave out any information but ignore references to example images or code snippets. The instruction will be given to a language model as a prompt to generate the corresponding TikZ picture. {formulation}"
}

FORMULATIONS = {
    "polite": "Formulate the instruction in detail as a polite request.",
    "imperative": "Formulate the instruction in detail using imperative speech."
}

def batched(iterable, n):
    it = iter(iterable)
    while (batch := tuple(islice(it, n))):
        yield batch

class TikZConfig(builder.BuilderConfig):
    """BuilderConfig for TikZ."""

    def __init__(self, *args, bs=8, **kwargs):
        super().__init__(*args, **kwargs)
        self.bs = bs
        self.alpaca = Alpaca(bs=bs)
        self.data_urls = {
            "PetarV-/TikZ": "https://github.com/PetarV-/TikZ/archive/refs/heads/master.zip",
            "janosh/tikz": "https://github.com/janosh/tikz/archive/refs/heads/main.zip",
            "tex.stackexchange.com": "https://archive.org/download/stackexchange/tex.stackexchange.com.7z/Posts.xml",
        }
        self.generators = {
            "PetarV-/TikZ": petarv_tikz.load,
            "janosh/tikz": janosh_tikz.load,
            "texample.net": texample_net.load,
            "tikz.net": tikz_net.load,
            "pgfplots.net": tikz_net.load, # tikz.net downloader also works for this site
            "tex.stackexchange.com": tex_stackexchange_com.TeXExchangeParser
        }
        self.prompts = {
            "PetarV-/TikZ": PROMPT_TEMPLATES["example"],
            "janosh/tikz": PROMPT_TEMPLATES["example"],
            "texample.net": PROMPT_TEMPLATES["example"],
            "tikz.net": PROMPT_TEMPLATES["example"],
            "pgfplots.net": PROMPT_TEMPLATES["example"],
            "tex.stackexchange.com": PROMPT_TEMPLATES["stack_exchange"]
        }

        for source, prompt in self.prompts.items():
            self.prompts[source] = [prompt.format(formulation=f) for f in FORMULATIONS.values()]


class TikZ(builder.ArrowBasedBuilder):
    """A TikZ corpus."""

    BUILDER_CONFIG_CLASS = TikZConfig

    def _info(self):
        features = {
            "title": Value("string"),
            "description": Value("string"),
            "instructions": Sequence(Value("string"), 2),
            "code": Value("string"),
            "origin": Value("string"),
        }

        return DatasetInfo(
            description=str(__doc__),
            features=Features(features),
            supervised_keys=("description", "code"),
        )
    def _split_generators(self, dl_manager):
        urls_to_download = self.config.data_urls
        downloaded_files = dl_manager.download_and_extract(urls_to_download)

        return [
            SplitGenerator(
                name=str(Split.TRAIN), gen_kwargs={"datasets": downloaded_files}
            ),
        ]

    def _clean(self, example):
        for key, text in example.items():
            example[key] = sub(r"\r\n|\r", r"\n", text).strip() # normalize newlines
        example["code"] = sub(r"^(%.*\n)*", "", example["code"]) # remove leading comments
        return example

    def _truncate(self, description):
        """
        Descriptions can be longer than 512 tokens, which leads to OOM errors. So we
        truncate them to the half of the model maximum length to leave enough
        space for generating text.
        """
        tokenizer = self.config.alpaca.pipeline.tokenizer
        return tokenizer.decode(
            tokenizer(description.strip(),
            truncation=True,
            add_special_tokens=False,
            max_length=tokenizer.model_max_length//2)['input_ids']
        ).strip()


    def _generate_tables(self, datasets):
        loaders = list()

        for name, load in self.config.generators.items():
            logger.debug("Processing examples from '%s'.", name)
            match name:
                case "PetarV-/TikZ" | "janosh/tikz": loader = load(datasets[name])
                case "tex.stackexchange.com": loader = load(datasets[name]).load()
                case "texample.net" | "tikz.net": loader = load()
                case "pgfplots.net": loader = load(base_url=f"https://{name}")
                case _: raise ValueError(f'Source "{name}" not known!')

            loaders.append(map(lambda item, name=name: self._clean(item) | {"origin": name}, loader))

        for idx, examples in enumerate(batched(chain(*loaders), self.config.bs)):
            instructions = [instr for example in examples for instr in self.config.prompts[example["origin"]]]
            inputs = ["\n\n".join((ex['title'], self._truncate(ex['description']))).strip() for ex in examples for _ in range(2)]

            for example, instructions in zip(examples, batched(self.config.alpaca(instructions, inputs), len(FORMULATIONS))):
                example['instructions'] = instructions

            yield idx, Table.from_pylist(examples)
