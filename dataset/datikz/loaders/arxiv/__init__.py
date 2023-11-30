from multiprocessing import Pool
from os import listdir
from os.path import isdir, join

from .. import load_silent
from .finder import TikzFinder

def _load_worker(paper):
    found = list()
    try:
        for tikz in TikzFinder(tex=paper['text']).find(): # type: ignore
            found.append({
                "caption": tikz.caption,
                "code": tikz.code,
                "date": paper['meta']['timestamp'], # type: ignore
                "uri": paper['meta']['url'] # type: ignore
            })
    except (AssertionError, RecursionError): # FIXME: where does the recursion error come from?
        pass
    return found

def expand(directories_or_files):
    files = list()
    for file in directories_or_files:
        if isdir(file):
            files.extend([join(file, paper) for paper in listdir(file)])
        else:
            files.append(file)
    return files

def load(directories, bs=1):
    files = expand(directories)
    with Pool(bs) as p:
        for results in p.imap_unordered(_load_worker, load_silent("json", data_files=files, split="train")):
            yield from results
