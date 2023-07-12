import re


def get_sentences(txt):
    sents = re.split("(\\.\s)", txt)
    sents = [sent for sent in sents if len(sent.strip()) > 2]
    return sents

def get_sentences_offsets(txt, sentences):
    return [txt.index(sent) for sent in sentences]

def find_offset_index(offsets, input_offset):
    for i, offset in enumerate(offsets):
        if offset > input_offset:
            return i - 1
    return len(offsets) - 1

def add_html(sent, start, end):
    sent = sent[:end] + "</main>" + sent[end:]
    sent = sent[:start] + "<main>" + sent[start:]
    return sent


def get_surrounding_items(lst, index, n):
    start = max(0, index - n)
    end = min(len(lst), index + n + 1)
    return lst[start:end]

def add_html_offset(txt, offsets, types):
    offsets_start = [int(offset.split(" ")[0]) for offset in offsets]
    offsets_end = [int(offset.split(" ")[1]) for offset in offsets]

    tagged_txt = ""
    for idx, char in enumerate(txt):
        if idx in offsets_start:
            index = offsets_start.index(idx)
            offsets_index=offsets[index]
            icd_type=types[index][0]
            # tagged_txt += f"<main id=\"{index}\" offsets=\"{offsets_index}\" type=\"{icd_type}\">"
            tagged_txt += f"<main id=\"{index}\" type=\"{icd_type}\">"
        tagged_txt += char

        if idx + 1 in offsets_end:
            tagged_txt += "</main>"

    return tagged_txt