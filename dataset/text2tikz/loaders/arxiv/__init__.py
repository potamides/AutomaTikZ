from os import listdir
from os.path import join
from .finder import TikzFinder
from .. import load_silent

def load(directories, min_caption_length=16):
    paper_files = [join(directory, paper) for directory in directories for paper in listdir(directory)]

    for paper in load_silent("json", data_files=paper_files, split="train"):
        try:
            for tikz in TikzFinder(tex=paper['text']).find(): # type: ignore
                if len(tikz.caption) > min_caption_length:
                    yield {
                        "caption": tikz.caption,
                        "code": tikz.code,
                        "date": paper['meta']['timestamp'], # type: ignore
                        "uri": paper['meta']['url'] # type: ignore
                    }
        except AssertionError:
            continue
