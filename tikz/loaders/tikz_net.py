from itertools import count
from json import load as jload
from urllib.request import urlopen

from bs4 import BeautifulSoup

def load(base_url="https://tikz.net"):
    for page in count(0):
        example_list = f"{base_url}?infinity=scrolling&action=infinite_scroll&page={page}"
        json = jload(urlopen(example_list))

        if json['type'] == "empty":
            break

        for article in BeautifulSoup(json['html'], 'html.parser').find_all('article'):
            soup = BeautifulSoup(urlopen(article.a.get('href')), 'html.parser')
            content = soup.find_all('div', attrs={"class": "entry-content"})

            if content and (text := content[0].text.strip()):
                desc, delim, tikz = text.rpartition(r"\documentclass")
                tikz = "".join(tikz.rpartition(r"\end{document}")[:2])

                # remove any comments before \documentclass
                cleaned_desc = ""
                for idx, line in enumerate((rev_desc := list(reversed(desc.strip().split("\n"))))):
                    if line.strip() and not line.startswith("%"):
                        cleaned_desc = "\n".join(reversed(rev_desc[idx:])).strip()
                        break
                if cleaned_desc and tikz:
                    yield {
                        "title": soup.h1.text,
                        "description": cleaned_desc,
                        "code": delim + tikz
                    }
