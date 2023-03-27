from collections import defaultdict
from string import punctuation
import xml.etree.ElementTree as etree

from bs4 import BeautifulSoup
from spacy.lang.en import English

# This implementation is based on https://github.com/EleutherAI/stackexchange-dataset
# Information about database schema at https://meta.stackexchange.com/a/2678

def is_question(elem_attribs):
    if elem_attribs["PostTypeId"] is not None:
        if elem_attribs["PostTypeId"] == "1":
            return True
    return False

def is_answer(elem_attribs):
    if elem_attribs["PostTypeId"] is not None:
        if elem_attribs["PostTypeId"] == "2":
            return True
    return False

def is_accepted_answer(a_attribs, q_attribs):
    assert is_question(q_attribs), "Must be a question to have an accepted answer"
    assert is_answer(a_attribs), "Must be an answer to be an accepted answer"
    if q_attribs["AcceptedAnswerId"] is not None:
        if q_attribs["AcceptedAnswerId"] == a_attribs["Id"]:
            return True
    else:
        return False

def has_answers(elem_attribs):
    assert is_question(elem_attribs), "Must be a question to have answers"
    if elem_attribs["AnswerCount"] is not None:
        if int(elem_attribs["AnswerCount"]):
            return True
    return False

def trim_attribs(elem_attribs, attrib_type="question"):
    """deletes non-useful data from attribs dict for questions / answers, returns remaining"""
    if attrib_type == "question":
        to_keep = ['Id', 'Body', 'Title', 'Tags', 'AnswerCount', 'AcceptedAnswerId', 'PostTypeId']
        to_delete = [x for x in elem_attribs.keys() if x not in to_keep]
        [elem_attribs.pop(x, None) for x in to_delete]
        elem_attribs["ParsedAnswers"] = 0
        elem_attribs["Answers"] = {}
    elif attrib_type == "answer":
        to_keep = ['Id', 'Body', 'Score']
        new_dict = {}
        for item in to_keep:
            new_dict[item] = elem_attribs[item]
        return new_dict
    else:
        raise Exception('Unrecognized attribute type - please specify either question or answer')

def extract_code(soup):
    # first search for whole documents and then tikzpictures only
    for patterns in [[r"\documentclass", r"\end{document"], [r"\begin{tikzpicture}", r"\end{tikzpicture"]]:
        # last code snippet first
        for formatted in reversed(soup.find_all("pre")):
            if code := formatted.code:
                if all(pattern in code.text for pattern in patterns):
                    return code
    return False

nlp = English()
nlp.add_pipe('sentencizer')
def extract_description(soup):
    bla = repr(soup)
    if target := soup.find(['pre', 'img']):
        # delete everything after <img> and <pre>
        for e in target.find_all_next():
            e.extract()
        # if img and pre are inside <p> with text delete the text
        if (para := target.find_parent("p")) and para.text:
            para.extract()
        # else look at last sentence before, if it ends in ":", "," or "" delete it
        elif para := next(iter(p for p in target.find_all_previous("p") if p is not para), None):
            text = para.text.strip()
            if text and (text.endswith((":", ",")) or not text.endswith(tuple(punctuation))):
                doc = nlp(text)
                end_pos = list(doc.sents)[-1].start
                para.string = doc[:end_pos].text
                if not para.string.strip():
                    para.extract()
        target.extract()

    return soup

class TeXExchangeParser():
    def __init__(self, xml_path, min_score=1, tags=["tikz-pgf"]):
        self.xml_path = xml_path
        # dict to save questions
        self.questions = defaultdict(lambda: None, {})
        # min_score required to parse an answer
        self.min_score = min_score
        # acceptable tags of questions
        self.tags = tags

    def load(self):
        for _, elem in etree.iterparse(self.xml_path, events=('end',)):
            if elem.tag == "row":
                attribs = defaultdict(lambda: None, elem.attrib)
                if is_question(attribs):
                    if has_answers(attribs) and all(f"<{tag}>" in attribs.get("Tags", "") for tag in self.tags):
                        trim_attribs(attribs, "question")
                        self.questions[attribs["Id"]] = attribs
                    else:
                        # if the question has wrongs tags or no answers, discard it
                        continue
                elif is_answer(attribs):
                    # if is accepted answer, append answer Body to relevant questions "AcceptedAnswer" field
                    # if the answer's score > min_score
                    # append the answer to the relevant question's OtherAnswers dict
                    self.add_answer(attribs)
                    yield from self.check_complete(attribs)
                elem.clear()

    def is_above_threshold(self, a_attribs):
        assert is_answer(a_attribs), "Must be an answer to be above threshold"
        if a_attribs["Score"] is not None:
            if int(a_attribs["Score"]) >= self.min_score:
                return True
        return False

    def add_answer(self, a_attribs):
        assert is_answer(a_attribs), "Must be an answer to add to parent"
        if a_attribs is not None and self.questions[a_attribs["ParentId"]] is not None:
            if is_accepted_answer(a_attribs, self.questions[a_attribs["ParentId"]]):
                self.questions[a_attribs["ParentId"]]["Answers"][a_attribs["Id"]] = trim_attribs(a_attribs, "answer")
                self.questions[a_attribs["ParentId"]]["ParsedAnswers"] += 1
            elif self.is_above_threshold(a_attribs):
                if a_attribs["Id"] is not None:
                    parent = self.questions[a_attribs["ParentId"]]
                    if parent is not None:
                        self.questions[a_attribs["ParentId"]]["Answers"][a_attribs["Id"]] = trim_attribs(a_attribs, "answer")
                        self.questions[a_attribs["ParentId"]]["ParsedAnswers"] += 1
                else:
                    self.questions[a_attribs["ParentId"]]["ParsedAnswers"] += 1
            else:
                self.questions[a_attribs["ParentId"]]["ParsedAnswers"] += 1

    def check_complete(self, a_attribs):
        keys_to_del = []
        parent = self.questions[a_attribs["ParentId"]]

        if a_attribs is not None and parent is not None:
            if parent["AnswerCount"] is not None and parent["ParsedAnswers"] is not None:
                if int(parent["ParsedAnswers"]) == int(parent['AnswerCount']):
                    keys_to_del.append(a_attribs["ParentId"])

                    if parent["Answers"] is not None and len(parent["Answers"]) > 0:
                        title = parent["Title"] or ""
                        markup = BeautifulSoup(parent["Body"] or "", "html.parser")

                        if description := extract_description(markup).text.strip():
                            for a in parent["Answers"].values():
                                if code := extract_code(BeautifulSoup(a["Body"], "html.parser")):
                                    yield {
                                        "title": title,
                                        "description": description,
                                        "code": code.text
                                    }
        for key in keys_to_del:
            self.questions.pop(key, None)
