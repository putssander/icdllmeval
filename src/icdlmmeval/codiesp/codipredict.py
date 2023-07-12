import configparser
import logging
from icdlmmeval.embedding_lookup import EmbeddingLookup
from icdlmmeval.icd_prompts import IcdPrompts
from icdlmmeval import ner_parsing
from icdlmmeval.codiesp.codiformat import CodiFormat
from icdlmmeval.codiesp.prompt_examples import PromptExamples

import logging

FORMAT = '%(asctime)s %(message)s'

logging.basicConfig(filename='/home/jovyan/work/icdllmeval/notebooks/llm.log', encoding='utf-8', level=logging.INFO, filemode='w', format=FORMAT)
logging.info('test')


class CodiPredict():

    def __init__(self, model_name):
        
        config = configparser.ConfigParser()
        config.read('./../resources/config.ini')
        self.config = config

        self.path_codiesp = config["codiesp"]['data']

        print("load prompting module")
        self.icd_prompts = IcdPrompts(model_name=model_name)
        print("load embeddings")
        self.embedding_lookup = EmbeddingLookup()

        self.codiformat = CodiFormat()
        self.prompt_examples = PromptExamples()
        self.description_example = self.prompt_examples.get_prompt_description_example(file_number=11, context_size = self.config["descriptions"]["context_size"])
        self.examples_ranking=[]
    
    
    def predict_icd_x(self, df_main_terms, split, file_names):

        for idx, file_name in file_names:
            icd_codes = []

            descriptions_file = self.prompt_code_descriptions(split=split, df_main_terms=df_main_terms, file_name=file_name)
            for item in descriptions_file["diagnoses"]:
           
                code = self.code_diagnose(item, self.examples_ranking)

                offsets = ner_parsing.find_icd_phrase_offsets(item)
                for offsets_item in offsets:
                    offsets_string = ner_parsing.offset_to_string(offset_array=[offsets_item])
        
                    codi_item = {}
                    codi_item["FILE"] = file_name
                    codi_item["SUBSTRING"] = item.icd_phrase
                    codi_item["CODE"] = code
                    codi_item["OFFSETS"] = offsets_string
                    codi_item["TYPE"] = "DIAGNOSE"
            
            for item in descriptions_file["procedures"]:
           
                code = self.code_procedure(item, self.examples_ranking)

                offsets = ner_parsing.find_icd_phrase_offsets(item)
                for offsets_item in offsets:
                    offsets_string = ner_parsing.offset_to_string(offset_array=[offsets_item])
        
                    codi_item = {}
                    codi_item["FILE"] = file_name
                    codi_item["SUBSTRING"] = item.icd_phrase
                    codi_item["CODE"] = code
                    codi_item["OFFSETS"] = offsets_string
                    codi_item["TYPE"] = "DIAGNOSE"
                    
            icd_codes.append(codi_item)




    def prompt_code_descriptions(self, split, df_main_terms, file_name):
        chunk_size = self.config["descriptions"]["prompt_size"]
        context_size = self.config["descriptions"]["context_size"]

        prompt_file = self.codiformat.get_description_prompt(split=split, df_main_x=df_main_terms, file_name=file_name, n=context_size)
        prompts = self.codiformat.get_description_prompt_chunk(prompt=prompt_file, chunk_size=chunk_size)
        descriptions = {"diagnoses": [], "procedures": []}
        for prompt in prompts:
            output = self.icd_prompts.prompt_icd_code_description_from_main_terms(example=self.description_example, main_terms=prompt)
            descriptions["diagnoses"].extend(output["diagnoses"])
            descriptions["procedures"].extend(output["procedures"])
        return descriptions



    def code_diagnose(self, item, examples_ranking):

        icd_phrase = item.icd_phrase
        if item.icd_code_description_es:
            substring = item.icd_code_description_es
        else:
            substring = item.icd_phrase

        json_docs_descr = self.embedding_lookup.docs_to_json(self.embedding_lookup.search_diagnose(substring=substring))

        clean_item = {}
        clean_item['icd_phrase'] = icd_phrase
        clean_item['context'] = item.context

        # GPT-4 suggestion
        # clean_item['suggestions_item_descr'] = item.icd_code_description_es
        # clean_item['suggestions_item_code'] = item.icd_code

        # embedding suggestions based on GPT-4 code description
        clean_item['suggestions_list_descr'] = json_docs_descr

        code_gpt = self.icd_prompts.select_code(clean_item)
        # code_gpt = self.icd_prompts.select_code(item=item, examples=examples_ranking)

        return code_gpt

    


    def code_procedure(self, item, examples_ranking):

        icd_phrase = item.icd_phrase
        if item.icd_code_description_es:
            substring = item.icd_code_description_es
        else:
            substring = item.icd_phrase

        json_docs_descr = self.embedding_lookup.docs_to_json(self.embedding_lookup.search_procedure(substring=substring))

        # do not suggest GPT-4 coded ICD-10 procedures, they are incorrect
        clean_item = {}
        clean_item['icd_phrase'] = icd_phrase
        clean_item['context'] = item.context

        # embedding suggestions based on GPT-4 code description
        # clean_item['suggestions_list_descr'] = json_docs_descr

        code_gpt = self.icd_prompts.select_code(clean_item)
        # code_gpt = self.icd_prompts.select_code(item=item, examples=examples_ranking)
        return code_gpt