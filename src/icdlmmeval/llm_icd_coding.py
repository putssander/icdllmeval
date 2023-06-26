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

        raw = {}
        substrings_json = self.icd_prompts.extract_substrings(txt, examples=examples)
        raw['substrings_json'] = substrings_json
        icd_info = self.icd_prompts.prompt_icd_item_info(txt, substrings_json)
        raw['icd_info'] = icd_info

        icd_codes = []
        for item in icd_info.diagnoses:
           
            code = self.code_diagnose(item, examples_ranking)

            offsets = ner_parsing.get_offsets(txt, substring=item.original_phrase)
            for offsets_item in offsets:
                offsets_string = ner_parsing.offset_to_string(offset_array=[offsets_item])
    
                codi_item = {}
                codi_item["SUBSTRING"] = item.original_phrase
                codi_item["CODE"] = code
                codi_item["OFFSETS"] = offsets_string
                codi_item["TYPE"] = "DIAGNOSE"
            
                icd_codes.append(codi_item)
        
        for item in icd_info.procedures:

            code = self.code_procedure(item, examples_ranking)

            offsets = ner_parsing.get_offsets(txt, substring=item.original_phrase)
            for offsets_item in offsets:
                offsets_string = ner_parsing.offset_to_string(offset_array=[offsets_item])

                codi_item = {}
                codi_item["SUBSTRING"] = item.original_phrase
                codi_item["CODE"] = code
                codi_item["OFFSETS"] = offsets_string
                codi_item["TYPE"] = "PROCEDURE"
            
                icd_codes.append(codi_item)
        
        return icd_codes


    def code_diagnose(self, item, examples_ranking):

        original_phrase = item.original_phrase
        if item.icd_code_description_es:
            substring = item.icd_code_description_es
        else:
            substring = item.original_phrase

        json_docs_descr = self.embedding_lookup.docs_to_json(self.embedding_lookup.search_diagnose(substring=substring))

        clean_item = {}
        clean_item['original_phrase'] = original_phrase
        clean_item['context'] = item.text_snippets

        # GPT-4 suggestion
        # clean_item['suggestions_item_descr'] = item.icd_code_description_es
        # clean_item['suggestions_item_code'] = item.icd_code

        # embedding suggestions based on GPT-4 code description
        clean_item['suggestions_list_descr'] = json_docs_descr

        code_gpt = self.icd_prompts.select_code(clean_item)
        # code_gpt = self.icd_prompts.select_code(item=item, examples=examples_ranking)

        return code_gpt

    


    def code_procedure(self, item, examples_ranking):

        original_phrase = item.original_phrase
        if item.icd_code_description_es:
            substring = item.icd_code_description_es
        else:
            substring = item.original_phrase


        json_docs_descr = self.embedding_lookup.docs_to_json(self.embedding_lookup.search_procedure(substring=substring))

        # do not suggest GPT-4 coded ICD-10 procedures, they are incorrect
        clean_item = {}
        clean_item['original_phrase'] = original_phrase
        clean_item['context'] = item.text_snippets

        # embedding suggestions based on GPT-4 code description
        clean_item['suggestions_list_descr'] = json_docs_descr

        code_gpt = self.icd_prompts.select_code(clean_item)
        # code_gpt = self.icd_prompts.select_code(item=item, examples=examples_ranking)
        return code_gpt

