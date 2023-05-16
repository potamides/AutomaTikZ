from datetime import datetime
from glob import glob
from json import load as jload
from os.path import dirname, join, relpath, sep
from urllib.parse import quote
from urllib.request import urlopen

from yaml import Loader, load as yload

REPO = "janosh/tikz"
CREATED = datetime.strptime(jload(urlopen(f"https://api.github.com/repos/{REPO}"))['created_at'], "%Y-%m-%dT%H:%M:%SZ")

def load(filepath):
    globpath = join(filepath, "tikz-*/assets/*/*.{ext}")
    for tex, yml in zip(glob(globpath.format(ext="tex")), glob(globpath.format(ext="yml"))):
        with open(tex, 'r') as f, open(yml, "r") as g:
            code = f.read().strip()
            yaml = yload(g.read(), Loader)

        yield {
            "caption": (yaml['description'] or yaml['title'] or "").strip(),
            "code": code,
            "date": CREATED,
            "uri": join(f"https://github.com/{REPO}/blob/main", quote(relpath(dirname(tex), filepath).split(sep, 1)[1]))
        }
