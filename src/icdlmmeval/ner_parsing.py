from spacy_llm.tasks.util import parsing
from typing import Iterable


def get_offsets(txt, substring):
    offsets = parsing.find_substrings(txt, [substring])
    # if not offsets:
    #     offsets = parsing.find_substrings(txt, substring.split(" "))
    return offsets

def find_icd_phrase_offsets(record):
    main_term_start, main_term_end = map(int, record['offsets'].split())
    icd_phrase_start_in_main_term = record['icd_phrase'].find(record['main_term'])

    icd_phrase_start = main_term_start + icd_phrase_start_in_main_term
    icd_phrase_end = icd_phrase_start + len(record['icd_phrase'])

    return f"{icd_phrase_start} {icd_phrase_end}"



def offset_to_string(offset_array):
    s = ""
    for offset_tuple in offset_array:                            
        offset_string = str(offset_tuple[0]) + " " + str(offset_tuple[1])
        if not s:
            s = offset_string
        else:
            s = s + ";" + offset_string
    return s