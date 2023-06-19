from spacy_llm.tasks.util import parsing
from typing import Iterable


def get_offsets(txt, substring):
    offsets = parsing.find_substrings(txt, [substring])
    if not offsets:
        offsets = parsing.find_substrings(txt, substring.split(" "))     
    return offsets


def offset_to_string(offset_array):
    s = ""
    for offset_tuple in offset_array:                            
        offset_string = str(offset_tuple[0]) + " " + str(offset_tuple[0])
        if not s:
            s = offset_string
        else:
            s = s + ";" + offset_string
    return s