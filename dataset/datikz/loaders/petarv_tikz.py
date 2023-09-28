from glob import glob
from os.path import dirname, join, relpath, sep
from urllib.parse import quote

from bs4 import BeautifulSoup
from markdown import markdown

from . import get_creation_time

REPO = "PetarV-/TikZ"
CREATED = get_creation_time(REPO)

def load(directory):
    for file in glob(join(directory, "TikZ-*/*/*.tex")):
        with open(file, 'r') as f, open(join(dirname(file), "README.md"), "r") as g:
            soup = BeautifulSoup(markdown(g.read()), 'html.parser')
            descr = soup.select_one('h2:-soup-contains("Notes")').find_next('p').text
            code = f.read().strip()

        yield {
            "caption": descr or soup.h1.text,
            "code": code,
            "date": CREATED,
            "uri": join(f"https://github.com/{REPO}/blob/master", quote(relpath(dirname(file), directory).split(sep, 1)[1]))
        }
