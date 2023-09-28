from datetime import datetime
from ssl import VerifyMode, create_default_context
from string import punctuation
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
    # certificate of server expired
    context.check_hostname = False
    context.verify_mode = VerifyMode.CERT_NONE

    soup = BeautifulSoup(urlopen(example_list, context=context), 'html.parser')
    content = soup.find(id="content-wrapper").find("ul")

    for link in content.find_all("a"):
        example = BeautifulSoup(urlopen(uri := join(base_url, link.get('href')), context=context), 'html.parser')

        title = link.text.strip()
        description = cleanup(example.find_all('div', attrs={"class": "example-description"})[0].text.strip())
        date = example.find('div', attrs={"class": "pubinfo"}).text.split()[1]
        tikz = example.pre.text.strip()

        if description:
            if title.lower() not in description.lower():
                description = f"{title}. {description}"
                if description[-1] not in punctuation:
                    description += "."
        else:
            description = title

        yield {
            "caption": " ".join(description.split()),
            "code": tikz,
            "date": datetime.strptime(date, "%Y-%m-%d"),
            "uri": uri
        }
