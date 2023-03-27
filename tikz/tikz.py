"""
Combination of various sources of tikz descriptions with aligned code.
"""

from itertools import chain
from re import sub

from datasets import Features, Value, builder
from datasets.info import DatasetInfo
from datasets.splits import Split, SplitGenerator
from datasets.utils import logging

from tikz.loaders import (
    janosh_tikz,
    petarv_tikz,
    tex_stackexchange_com,
    texample_net,
    tikz_net,
)

logger = logging.get_logger("transformers")


class TikZConfig(builder.BuilderConfig):
    """BuilderConfig for TikZ."""

    def __init__(self, *args, scrape=True, **kwargs):
        super().__init__(*args, name="scraped" if scrape else "preprocessed", **kwargs)
        if scrape:
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
        else:
            raise ValueError("Not yet supported!") # TODO


class TikZ(builder.GeneratorBasedBuilder):
    """A TikZ corpus."""

    BUILDER_CONFIG_CLASS = TikZConfig

    def _info(self):
        features = {
            "title": Value("string"),
            "description": Value("string"),
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

    def _generate_examples(self, datasets):
        loaders = list()

        for name, load in self.config.generators.items():
            logger.debug("Processing examples from '%s'.", name)
            match name:
                case "PetarV-/TikZ" | "janosh/tikz": loader = load(datasets[name])
                case "tex.stackexchange.com": loader = load(datasets[name]).load()
                case "texample.net" | "tikz.net": loader = load()
                case "pgfplots.net": loader = load(base_url=f"https://{name}")
                case _: raise ValueError(f'Source "{name}" not known!')

            loaders.append(map(lambda item, name=name: (name, item), loader))

        for idx, (origin, example) in enumerate(chain(*loaders)):
            example['origin'] = origin
            yield idx, self._clean(example)
