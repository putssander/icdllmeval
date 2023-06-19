from .embedding_lookup import EmbeddingLookup
from .icd_prompts import IcdPrompts
from icdlmmeval import ner_parsing

import logging

class LLMIcdCoding():

    def __init__(self, model_name, path_diagnoses_db, path_procedures_db):

        print("load prompting module")
        self.icd_prompts = IcdPrompts(model_name=model_name)
        print("load embeddings")
        self.embedding_lookup = EmbeddingLookup(path_diagnoses_db=path_diagnoses_db, path_procedures_db=path_procedures_db)

    def code_txt(self, txt,  examples=[], examples_ranking=[], k=50):
        substrings_json = self.icd_prompts.extract_substrings(txt, examples=examples)

        icd_codes = []
        for substring in substrings_json['procedures']:
            docs = self.embedding_lookup.search_procedure(substring=substring, k=k)
            code = self.icd_prompts.select_code(substring=substring, docs=docs, examples=examples_ranking)
            offsets = ner_parsing.get_offsets(txt, substring=substring)
            offsets_string = ner_parsing.offset_to_string(offset_array=offsets)
            
            item = {}
            item["SUBSTRING"] = substring
            item["CODE"] = code
            item["OFFSETS"] = offsets_string
            item["TYPE"] = "PROCEDURE"
            
        icd_codes.append(item)
        
        for substring in substrings_json['diagnoses']:
            docs = self.embedding_lookup.search_diagnose(substring=substring, k=k)
            code = self.icd_prompts.select_code(substring=substring, docs=docs, examples=examples_ranking)
            offsets = ner_parsing.get_offsets(txt, substring=substring)
            offsets_string = ner_parsing.offset_to_string(offset_array=offsets)
            
            item = {}
            item["SUBSTRING"] = substring
            item["CODE"] = code
            item["OFFSETS"] = offsets_string
            item["TYPE"] = "DIAGNOSE"
            
            icd_codes.append(item)
        
        return icd_codes
