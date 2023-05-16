from datetime import datetime
from itertools import count
from json import load as jload
from string import punctuation
from urllib.request import urlopen

from bs4 import BeautifulSoup

def load(base_url="https://tikz.net"):
    for page in count(0):
        example_list = f"{base_url}?infinity=scrolling&action=infinite_scroll&page={page}"
        json = jload(urlopen(example_list))

        if json['type'] == "empty":
            break

        for article in BeautifulSoup(json['html'], 'html.parser').find_all('article'):
            soup = BeautifulSoup(urlopen(uri := article.a.get('href')), 'html.parser')
            content = soup.find('div', attrs={"class": "entry-content"})

            title = soup.h1.text.strip()
            date = soup.find("time", attrs={"itemprop": "datePublished"})['datetime']

            if content:
                for pre in content.find_all("pre"):
                    desc, tikz = "", pre.text
                    for s in pre.previous_siblings:
                        if s.name == "pre": break
                        desc += s.text

                    desc = desc.replace("Edit and compile if you like:", "").strip()
                    if desc.strip():
                        if title.lower() not in desc.lower():
                            desc = f"{title}. {desc}"
                            if desc[-1] not in punctuation:
                                desc += "."
                    else:
                        desc = title

                    if tikz:
                        yield {
                            "caption": " ".join(desc.split()),
                            "code": tikz.strip(),
                            "date": datetime.fromisoformat(date),
                            "uri": uri
                        }
