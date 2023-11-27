from os import listdir
from os.path import join
from .finder import TikzFinder
from multiprocessing import Pool
from .. import load_silent

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

def load(directories, bs=1):
    paper_files = [join(directory, paper) for directory in directories for paper in listdir(directory)]
    with Pool(bs) as p:
        for results in p.imap_unordered(_load_worker, load_silent("json", data_files=paper_files, split="train")):
            yield from results
