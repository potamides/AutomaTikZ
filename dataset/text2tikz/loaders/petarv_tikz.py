from datetime import datetime
from glob import glob
from json import load as jload
from os.path import dirname, join, relpath, sep
from urllib.parse import quote
from urllib.request import urlopen

from bs4 import BeautifulSoup
from markdown import markdown

REPO = "PetarV-/TikZ"
CREATED = datetime.strptime(jload(urlopen(f"https://api.github.com/repos/{REPO}"))['created_at'], "%Y-%m-%dT%H:%M:%SZ")

def load(filepath):
    for file in glob(join(filepath, "TikZ-*/*/*.tex")):
        with open(file, 'r') as f, open(join(dirname(file), "README.md"), "r") as g:
            soup = BeautifulSoup(markdown(g.read()), 'html.parser')
            descr = soup.select_one('h2:-soup-contains("Notes")').find_next('p').text
            code = f.read().strip()

        yield {
            "caption": descr or soup.h1.text,
            "code": code,
            "date": CREATED,
            "uri": join(f"https://github.com/{REPO}/blob/master", quote(relpath(dirname(file), filepath).split(sep, 1)[1]))
        }
