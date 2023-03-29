from glob import glob
from os.path import dirname, join
from markdown import markdown
from bs4 import BeautifulSoup

def load(filepath):
    for file in glob(join(filepath, "TikZ-*/*/*.tex")):
        with open(file, 'r') as f, open(join(dirname(file), "README.md"), "r") as g:
            soup = BeautifulSoup(markdown(g.read()), 'html.parser')
            descr = soup.select_one('h2:-soup-contains("Notes")').find_next('p').text
            code = f.read().strip()

        yield {
            "title": soup.h1.text,
            "description": descr,
            "code": code
        }
