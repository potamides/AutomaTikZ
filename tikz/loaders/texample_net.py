from ssl import create_default_context
from urllib.parse import urljoin as join
from urllib.request import urlopen

from bs4 import BeautifulSoup

base_url = 'https://texample.net'
example_list = join(base_url, '/tikz/examples/all/list')

def cleanup(text):
    lines, filtered = text.split("\n"), list()

    for line in lines:
        if not (line.lower().startswith("author:") or "http://" in line or "https://" in line):
            filtered.append(line.strip())

    return "\n".join(filtered).strip()

def load():
    context = create_default_context()
    context.set_ciphers('DEFAULT:!DH') # fixes ssl.SSLError: [SSL: DH_KEY_TOO_SMALL] dh key too small (_ssl.c:997)

    soup = BeautifulSoup(urlopen(example_list, context=context), 'html.parser')
    content = soup.find(id="content-wrapper").find("ul")

    for link in content.find_all("a"):
        example = BeautifulSoup(urlopen(join(base_url, link.get('href')), context=context), 'html.parser')

        title = link.text.strip()
        description = cleanup(example.find_all('div', attrs={"class": "example-description"})[0].text.strip())
        tikz = example.pre.text.strip()

        yield {
            "title": title,
            "description": description,
            "code": tikz
        }
