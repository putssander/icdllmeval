import pandas as pd
import configparser
from icdlmmeval.icd_lookup import IcdLookup
from icdlmmeval.codiesp.codiformat import CodiFormat


from icdlmmeval import text_util

class PromptExamples:
    
    def __init__(self):
        config = configparser.ConfigParser()
        config.read('./../resources/config.ini')
        path_mapping = config["mappings"]["main-mapping"]

        self.train_main_x = pd.read_excel(f"{path_mapping}/train_main_x.xlsx")
        self.files = self.train_main_x["FILE"].to_list()
        
        # self.path_codiesp = path_codiesp
        self.icdlookup = IcdLookup()
        self.codiformat = CodiFormat()
        

    def get_description_prompt(self, item_output):
        item_prompt = {}
        item_prompt["main_term"] = item_output["main_term"]
        item_prompt["offsets"] = item_output["offsets"]
        item_prompt["context"] = item_output["context"]
        return item_prompt

    def get_prompt_diagnose_description_example(self, file_number=0, n=2):
        file_name = self.files[file_number]
        df_select = self.train_main_x[self.train_main_x["FILE"] == file_name]
        txt = self.codiformat.get_text('train', file_name)
        sentences = text_util.get_sentences(txt)
        offsets = text_util.get_sentences_offsets(txt, sentences)
        diagnoses_response = []

        for idx, row, in df_select.iterrows():
            term_offsets = self.codiformat.get_term_offsets(row["MAIN_OFFSETS"])
            sent_index = text_util.find_offset_index(offsets, term_offsets[0])
            sent_offset = offsets[sent_index]
            
            start = term_offsets[0] - sent_offset
            end = term_offsets[1] - sent_offset
            
            replace_sent_list = sentences.copy()
            replace_sent_list[sent_index] = text_util.add_html(sent=replace_sent_list[sent_index], start=start, end=end)
            context = ". ".join(text_util.get_surrounding_items(replace_sent_list, sent_index, n))
            
            item_output = {}
            item_output["main_term"] = row["MAIN_SUBSTRING"].strip()
            item_output["offsets"] = row["MAIN_OFFSETS"]
            item_output["context"] = context
            item_output["code"] = row["CODE"]
            item_output["substring"] = row["SUBSTRING"]
            if row["TYPE"] == CodiFormat.DIAGNOSTICO:
                item_output["descr_en"] = self.icdlookup.get_diangose_description_en(row["CODE"])
                item_output["desc_es"] =  self.icdlookup.get_diangose_description_es(row["CODE"])
                diagnoses_response.append(item_output)

        diagnoses_prompt = [self.get_description_prompt(item) for item in diagnoses_response]
        
        example = {}
        example["prompt"] = diagnoses_prompt
        example["output"] = diagnoses_response

        return example
