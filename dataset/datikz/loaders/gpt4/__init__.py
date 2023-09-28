from datetime import datetime
from importlib import import_module
from importlib.resources import files

from .. import load_silent

REPO = "potamides/AutomaTikZ"
TIKZ_DATA = str(files(import_module(__name__)) / "gpt4.json")

def load():
    dataset = load_silent("json", data_files=TIKZ_DATA, split="train")

    for idx, item in enumerate(dataset, 1):
        yield {
            "caption": item['caption'],
            "code": item['code'],
            "date": datetime.utcfromtimestamp(item['date']/1000),
            "uri": f"https://github.com/{REPO}/blob/main/dataset/datikz/loaders/gpt4/gpt4.json#L{idx}"
        }
