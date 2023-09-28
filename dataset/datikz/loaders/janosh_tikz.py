from glob import glob
from os.path import dirname, join, relpath, sep
from urllib.parse import quote

from yaml import Loader, load as yload

from . import get_creation_time

REPO = "janosh/tikz"
CREATED = get_creation_time(REPO)

def load(directory):
    globpath = join(directory, "tikz-*/assets/*/*.{ext}")
    for tex, yml in zip(glob(globpath.format(ext="tex")), glob(globpath.format(ext="yml"))):
        with open(tex, 'r') as f, open(yml, "r") as g:
            code = f.read().strip()
            yaml = yload(g.read(), Loader)

        yield {
            "caption": (yaml['description'] or yaml['title'] or "").strip(),
            "code": code,
            "date": CREATED,
            "uri": join(f"https://github.com/{REPO}/blob/main", quote(relpath(dirname(tex), directory).split(sep, 1)[1]))
        }
