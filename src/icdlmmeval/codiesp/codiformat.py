import pandas as pd
import json
from icdlmmeval import ner_parsing

DIAGNOSTICO = "DIAGNOSTICO"
PROCEDIMIENTO = "PROCEDIMIENTO"


class CodiFormat:

    header_X = ["FILE","TYPE", "CODE", "SUBSTRING", "OFFSETS"]      


    def __init__(self, path_codiesp):
        self.path_codiesp = path_codiesp

    def get_text(self, split, id):
        fp = open(f'{self.path_codiesp}/final_dataset_v4_to_publish/{split}/text_files/{id}.txt', 'r')
        txt = fp.read()
        fp.close()
        return txt


    def get_json_md(self, json_object):
        json_string = json.dumps(json_object, indent=2, ensure_ascii=False)
        return f"```json\n{json_string}\n```"


    def get_json_x_substrings(self, df, id):        
        select = df[df["FILE"] == id]        
        json_object = {}
        json_object["diagnoses"] = select[select["TYPE"] == DIAGNOSTICO]["SUBSTRING"].tolist()
        json_object["procedures"] = select[select["TYPE"] == PROCEDIMIENTO]["SUBSTRING"].tolist()
        return json_object
    

    def get_df_x(self, split):
        path_x = f"{self.path_codiesp}/final_dataset_v4_to_publish/{split}/{split}X.tsv"
        return pd.read_csv(path_x, delimiter="\t", names=self.header_X)


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
