from glob import glob
from os.path import join

from yaml import Loader, load as yload


def load(filepath):
    globpath = join(filepath, "tikz-*/assets/*/*.{ext}")
    for tex, yml in zip(glob(globpath.format(ext="tex")), glob(globpath.format(ext="yml"))):
        with open(tex, 'r') as f, open(yml, "r") as g:
            code = f.read().strip()
            yaml = yload(g.read(), Loader)

        yield {
            "title": (yaml['title'] or "").strip(),
            "description": (yaml['description'] or "").strip(),
            "code": code
        }
