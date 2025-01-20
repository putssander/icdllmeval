import pandas as pd
import json
from icdlmmeval import util
from icdlmmeval import util_text


import re
import configparser
import os


DIAGNOSTICO = "DIAGNOSTICO"
PROCEDIMIENTO = "PROCEDIMIENTO"


class CodiFormat:

    header_X = ["FILE","TYPE", "CODE", "SUBSTRING", "OFFSETS"]
    header_X_eval = ["FILE", "OFFSETS", "TYPE", "CODE"]
    header_D_P = ["FILE","CODE"]
    DIAGNOSTICO = "DIAGNOSTICO"
    PROCEDIMIENTO = "PROCEDIMIENTO"   

    def __init__(self, path_codiesp=None):
        if not path_codiesp:
            config = configparser.ConfigParser()
            CONFIG_PATH = os.path.join(os.path.dirname(__file__), '../../../resources/config.ini')
            config.read(CONFIG_PATH)
            path_codiesp = config["codiesp"]['data']
        self.path_codiesp = path_codiesp

    def get_text(self, split, id):
        fp = open(f'{self.path_codiesp}/{split}/text_files/{id}.txt', 'r')
        txt = fp.read()
        fp.close()
        return txt


    def get_json_md(self, json_object):
        json_string = json.dumps(json_object, indent=2, ensure_ascii=False)
        return f"```json\n{json_string}\n```"


    def get_json_x_substrings(self, df, id):        
        select = df[df["FILE"] == id]
        # select = select.filter(["TYPE", "SUBSTRING"], axis=1)
        # json_object = select.to_json(force_ascii=False, orient='records')   

        json_object = {}
        json_object["diagnoses"] = select[select["TYPE"] == DIAGNOSTICO]["SUBSTRING"].tolist()
        json_object["procedures"] = select[select["TYPE"] == PROCEDIMIENTO]["SUBSTRING"].tolist()
        return json_object
    
    def get_df_p(self, split):
        path = f"{self.path_codiesp}/{split}/{split}P.tsv"
        return pd.read_csv(path, delimiter="\t", names=self.header_D_P)
    
    def get_df_d(self, split):
        path = f"{self.path_codiesp}/{split}/{split}D.tsv"
        return pd.read_csv(path, delimiter="\t", names=self.header_D_P)

    def get_df_x(self, split):
        path_x = f"{self.path_codiesp}/{split}/{split}X.tsv"
        return pd.read_csv(path_x, delimiter="\t", names=self.header_X)

    def get_df_x_path(self, path_x):
        return pd.read_csv(path_x, delimiter="\t", names=self.header_X)
    
    def write_df_csv(self, path, df):
        df.to_csv(path, sep="\t", index=False, header=False)


    def get_path_d_gold(self, split):
        return f"{self.path_codiesp}/{split}/{split}D.tsv"    
    
    def get_path_p_gold(self, split):
        return f"{self.path_codiesp}/{split}/{split}P.tsv"

    def get_path_x_gold(self, split):
        return str(f"{self.path_codiesp}/{split}/{split}X.tsv")


    def format_codiesp(self, codes, file):
        for code in codes:
            code["FILE"] = file
            code["TYPE"] = code["TYPE"].replace("DIAGNOSE", DIAGNOSTICO).replace("PROCEDURE", PROCEDIMIENTO)
        return codes
    
    def write_codiesp_eval(self, predictions, split, path_pred, path_gold, path_pred_raw):
                # write temp result 

        df_pred = pd.DataFrame().from_records(predictions)
        df_pred.to_csv(path_pred_raw, sep ='\t', index=False, header=False)


        pred_header = ["clinical_case","pos_pred","label_pred", "code"]

        df = pd.DataFrame()
        df["clinical_case"] = df_pred["FILE"]
        df["pos_pred"] = df_pred["OFFSETS"]
        df["label_pred"] = df_pred["TYPE"]
        df["code"] = df_pred["CODE"]

        df.to_csv(path_pred, sep ='\t', index=False, header=False)

        df_split_x = self.get_df_x(split)
        df_split_x = df_split_x[df_split_x["FILE"].isin(df_pred["FILE"].unique())]

        df_split_x.to_csv(path_gold, sep ='\t', index=False, header=False)


    def get_substring_offsets(self, offsets):
        if ";" in offsets:
            offsets_list = offsets.split(";")
        else:
            offsets_list = [offsets]
        return offsets_list

    def get_term_offsets(self, offset_string):
        start = int(str(offset_string).split(" ")[0])
        end = int(str(offset_string).split(" ")[1])
        return (start, end)


    def get_description_prompt(self, split, df_main_x, file_name, n):

        df_select = df_main_x[df_main_x["FILE"] == file_name]
        df_select_diagnoses = df_select[df_select["TYPE"] == CodiFormat.DIAGNOSTICO]

        txt = self.get_text(split, file_name)
        sentences = util_text.get_sentences(txt)
        offsets = util_text.get_sentences_offsets(txt, sentences)
        diagnoses_prompt = []
        for idx, row, in df_select_diagnoses.iterrows():
            item_output = {}
            item_output["main_term"] = row["MAIN_SUBSTRING"].strip()
            item_output["offsets"] = row["MAIN_OFFSETS"]
            item_output["context"] = self.get_context(row["MAIN_OFFSETS"], offsets, sentences, n)
            diagnoses_prompt.append(item_output)
                   
        prompt = {"diagnoses" :diagnoses_prompt, "procedures": []}
        return prompt
    
    
    def get_description_prompt_chunk(self, prompt, chunk_size):

        diagnoses = prompt["diagnoses"]
        diagnoses_chunked = util.chunk_list(diagnoses, chunk_size)
        prompts_diagnoses = [{"diagnoses": diagnoses, "procedures": []} for diagnoses in diagnoses_chunked]
         
        procedures = prompt["procedures"]
        procedures_chunked = util.chunk_list(procedures, chunk_size)
        prompts_procedures = [{"diagnoses": [], "procedures": procedures} for procedures in procedures_chunked]

        return prompts_diagnoses + prompts_procedures


    def get_predicted_entities(self, df_ner, file_name):
        df_select = df_ner[df_ner["file"] == file_name]
        entities_list = []
    
        for idx, row in df_select.iterrows():
            offset = row["offset"]
            entities = json.loads(row["entities"])
            for entity in entities:
                entity["start"] = entity["start"] + offset
                entity["end"] = entity["end"] + offset
                entities_list.append(entity)
        return entities_list
    
    def get_description_prompt_txt_entities(self, txt, entities):
        ner_offsets = []
        ner_types = []
        for entity in entities:
            entity_offset = " ".join([str(entity["start"]), str(entity["end"])])
            ner_offsets.append(entity_offset)
            if entity["entity_group"] == "D":
                entity_type = DIAGNOSTICO
            else:
                entity_type = PROCEDIMIENTO
            ner_types.append(entity_type)
        prompt = util_text.add_html_offset(txt, ner_offsets, ner_types)
        return prompt

    def get_context(self, offset_string, offsets, sentences, n, tag="main"):
            term_offsets = self.get_term_offsets(offset_string)
            sent_index = util_text.find_offset_index(offsets, term_offsets[0])
            sent_offset = offsets[sent_index]
            
            start = term_offsets[0] - sent_offset
            end = term_offsets[1] - sent_offset
            
            replace_sent_list = sentences.copy()
            replace_sent_list[sent_index] = util_text.add_html(sent=replace_sent_list[sent_index], start=start, end=end, tag=tag)
            context = ". ".join(util_text.get_surrounding_items(replace_sent_list, sent_index, n))
            return context

        
    def get_description_prompt_substring(self, txt, row, idx=0, n=1):
        prompt = {}
        offsets_substring = row["OFFSETS"]

        offsets = self.get_substring_offsets(offsets_substring)

        sentences = util_text.get_sentences(txt)
        sentence_offsets = util_text.get_sentences_offsets(txt, sentences)

        context = []
        for term_offset in offsets:
            context.append(self.get_context(term_offset, sentence_offsets, sentences, n, tag="icd_phrase"))
        
        prompt["id"] = str(idx)
        prompt["icd_phrase"] = row["SUBSTRING"]
        prompt["context"] = " ".join(context)
        prompt["type"] = row["TYPE"]

        return prompt
    
    def get_description_prompt_select(self, txt, row, idx=0, n=1, hits=[]):
        prompt = self.get_description_prompt_substring(txt, row, idx=0, n=1)
        prompt["hits"] = hits
        return prompt