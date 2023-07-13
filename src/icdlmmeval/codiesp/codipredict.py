import configparser
import logging
from icdlmmeval.embedding_lookup import EmbeddingLookup
from icdlmmeval.icd_prompts import IcdPrompts
from icdlmmeval import ner_parsing
from icdlmmeval.codiesp.codiformat import CodiFormat
from icdlmmeval.codiesp.prompt_examples import PromptExamples
import pandas as pd
import logging
import json

FORMAT = '%(asctime)s %(message)s'

logging.basicConfig(filename='/home/jovyan/work/icdllmeval/notebooks/llm.log', encoding='utf-8', level=logging.INFO, filemode='w', format=FORMAT)
logging.info('test')


class CodiPredict():

    def __init__(self, model_name, load_dicts=True):
        
        config = configparser.ConfigParser()
        config.read('./../resources/config.ini')
        self.config = config

        self.path_codiesp = config["codiesp"]['data']

        print("load prompting module")
        self.icd_prompts = IcdPrompts(model_name=model_name)
        if load_dicts:
            print("load embeddings")
            self.embedding_lookup = EmbeddingLookup()

        self.codiformat = CodiFormat()
        self.prompt_examples = PromptExamples()
        self.description_example = self.prompt_examples.get_prompt_description_example_txt(file_number=11, context_size = int(self.config["descriptions"]["example_context_size"]))
        print("EXAMPLE DESCRIPTION")
        print(self.description_example)
        self.examples_ranking=[]
    
    
    def predict_icd_descriptions(self, df_main_terms, split, file_names):

        json_out = open(self.config["descriptions"]["json_file"], "w")
        json_out.writelines("")
        json_out.close()

        raw = []
        for idx, file_name in enumerate(file_names):

            prompt, output = self.prompt_code_descriptions(split=split, df_main_terms=df_main_terms, file_name=file_name)
            out = {"file": file_name, "prompt": prompt, "output": output}

            json_out = open(self.config["descriptions"]["json_file"], "a")
            json_out.writelines(json.dumps(out, ensure_ascii=False) + "\n")
            json_out.close()


    def code_using_embeddings(self, json_file):
        df_descr = pd.read_json(json_file, lines=True)
        icd_codes = []
        for idx, row in df_descr.iterrows():
            file_name = row["file"]
            output = json.loads(row["output"])
            
            for item in output["diagnoses"]:
           
                code = self.code_diagnose(item, self.examples_ranking)

                offsets = [ner_parsing.find_icd_phrase_offsets(item)]
                for offsets_item in offsets:
                    offsets_string = ner_parsing.offset_to_string(offset_array=[offsets_item])
        
                    codi_item = {}
                    codi_item["FILE"] = file_name
                    codi_item["SUBSTRING"] = item["icd_phrase"]
                    codi_item["CODE"] = code
                    codi_item["OFFSETS"] = offsets_string
                    codi_item["TYPE"] = "DIAGNOSE"

                    icd_codes.append(codi_item)

            if "procedures" in output:
                for item in output["procedures"]:
            
                    code = self.code_procedure(item, self.examples_ranking)

                    offsets = [ner_parsing.find_icd_phrase_offsets(item)]
                    for offsets_item in offsets:
                        offsets_string = ner_parsing.offset_to_string(offset_array=[offsets_item])
            
                        codi_item = {}
                        codi_item["FILE"] = file_name
                        codi_item["SUBSTRING"] = item["icd_phrase"]
                        codi_item["CODE"] = code
                        codi_item["OFFSETS"] = offsets_string
                        codi_item["TYPE"] = "DIAGNOSE"
                        
                        icd_codes.append(codi_item)
        return icd_codes

    def prompt_code_descriptions(self, split, df_main_terms, file_name):
        prompt = self.codiformat.get_description_prompt_txt(split=split, df_main_x=df_main_terms, file_name=file_name)
        output = self.icd_prompts.prompt_icd_code_description_from_main_terms(example=self.description_example, main_terms=prompt)
        return prompt, output



    def code_diagnose(self, item, examples_ranking):

        icd_phrase = item["icd_phrase"]
        if item["icd_code_description_es"]:
            substring = item["icd_code_description_es"]
        else:
            substring = item["icd_phrase"]

        json_docs_descr = self.embedding_lookup.docs_to_json(self.embedding_lookup.search_diagnose(substring=substring))

        clean_item = {}
        clean_item['icd_phrase'] = icd_phrase
        clean_item['context'] = item["context"]

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
    

    def eval_descriptions(self, split, descriptions_file):
        df_descriptions = pd.read_json(path_or_buf=descriptions_file, lines=True)
        
        for idx, row in df_descriptions.iterrows():
            json_file_descriptions = json.loads(row["descriptions"])


    # def get_description_codes(file_name, json_file_descriptions):
        
    #     for description in json_file_descriptions:

