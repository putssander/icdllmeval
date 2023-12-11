import pandas as pd
import configparser
from icdlmmeval.icd_lookup import IcdLookup
from icdlmmeval.codiesp.codiformat import CodiFormat
from icdlmmeval.icd_prompts import IcdItemNer, IcdListNer

from icdlmmeval import util_text
import os

examples_ranking = [
  {
    "question": """[Document(page_content='Reimplantación de ovario, izquierdo, abordaje abierto', metadata={'code': '0UM10ZZ'}), Document(page_content='Destrucción de pelvis renal, izquierda, abordaje orificio natural o artificial', metadata={'code': '0T547ZZ'}), Document(page_content='Destrucción de ovario, izquierdo, abordaje abierto', metadata={'code': '0U510ZZ'}), Document(page_content='Extirpación en ovario, izquierdo, abordaje percutáneo', metadata={'code': '0UC13ZZ'}), Document(page_content='Resección de tendón tronco, lado izquierdo, abordaje abierto', metadata={'code': '0LTB0ZZ'}), Document(page_content='Amputación de interpelviabdominal, izquierda, abordaje abierto', metadata={'code': '0Y630ZZ'}), Document(page_content='Resección de riñón, izquierdo, abordaje abierto', metadata={'code': '0TT10ZZ'}), Document(page_content='Resección de intestino grueso, izquierdo, abordaje orificio natural o artificial', metadata={'code': '0DTG7ZZ'}), Document(page_content='Resección de pelvis renal, izquierda, abordaje abierto', metadata={'code': '0TT40ZZ'}), Document(page_content='Extirpación en ovario, izquierdo, abordaje endoscópico percutáneo', metadata={'code': '0UC14ZZ'}), Document(page_content='Resección de trompa de eustaquio, izquierda, abordaje orificio natural o artificial', metadata={'code': '09TG7ZZ'}), Document(page_content='Resección de uréter, izquierdo, abordaje abierto', metadata={'code': '0TT70ZZ'}), Document(page_content='Extirpación en riñón, izquierdo, abordaje orificio natural o artificial', metadata={'code': '0TC17ZZ'}), Document(page_content='Extirpación en vena renal, izquierda, abordaje abierto', metadata={'code': '06CB0ZZ'}), Document(page_content='Resección de testículo, izquierdo, abordaje abierto', metadata={'code': '0VTB0ZZ'}), Document(page_content='Extirpación en rótula, izquierda, abordaje abierto', metadata={'code': '0QCF0ZZ'}), Document(page_content='Extirpación en pelvis renal, izquierda, abordaje endoscópico percutáneo', metadata={'code': '0TC44ZZ'}), Document(page_content='Extirpación en rótula, izquierda, abordaje endoscópico percutáneo', metadata={'code': '0QCF4ZZ'})]
    """,   
    "answer": """
    ```json
{"code": "0VTB0ZZ", "listed": true, "reasoning": "This response code was not suggested but listed"})
```
"""
  }
  ]

def isNaNorNone(num):
    if not num:
        return True
    return num != num

class PromptExamples:
    
    def __init__(self):
        config = configparser.ConfigParser()
        CONFIG_PATH = os.path.join(os.path.dirname(__file__), '../../../resources/config.ini')
        config.read(CONFIG_PATH)
        self.config = config

        path_mapping = config["main"]["annotation"]

        # self.path_codiesp = path_codiesp
        self.icdlookup = IcdLookup()
        self.codiformat = CodiFormat()
        
        # use dev for examples
        self.dev_main_x = pd.read_excel(f"{path_mapping}/main_x_dev.xlsx")
        self.dev_x = self.codiformat.get_df_x("dev")
        
        self.files = self.dev_main_x["FILE"].unique()


    def get_description_prompt(self, item_output):
        item_prompt = {}
        item_prompt["main_term"] = item_output["main_term"]
        item_prompt["offsets"] = item_output["offsets"]
        item_prompt["context"] = item_output["context"]
        return item_prompt

    def get_prompt_description_example_txt(self, file_number=0, selected_codes=[]):
        context_size = int(self.config["descriptions"]["example_context_size"])
        file_name = self.files[file_number]
        df_select = self.dev_main_x[self.dev_main_x["FILE"] == file_name]

        txt = self.codiformat.get_text('dev', file_name)
        sentences = util_text.get_sentences(txt)
        offsets = util_text.get_sentences_offsets(txt, sentences)
        output_diagnoses = []
        output_procedures = []

        for idx, row, in df_select.iterrows():
            if selected_codes and row["CODE"] not in selected_codes:
                continue

            print("load example code: ", row["CODE"])
            item_idx = idx
            item_main_term = str(row["MAIN_SUBSTRING"]).strip()
            item_offsets = str(row["MAIN_OFFSETS"])
            item_context = self.codiformat.get_context(str(row["MAIN_OFFSETS"]), offsets, sentences, context_size)
            item_icd_phrase= row["SUBSTRING"]

            if row["TYPE"] == CodiFormat.DIAGNOSTICO:
                item_icd_description_en = self.icdlookup.get_diagnose_description_en(row["CODE"])
                item_icd_description_es =  self.icdlookup.get_diagnose_description_es(row["CODE"])
                item_output = IcdItemNer(id=item_idx, main_term=item_main_term, offsets=item_offsets, context=item_context, icd_phrase=item_icd_phrase, 
                                         icd_description_en=item_icd_description_en, icd_description_es=item_icd_description_es)
                output_diagnoses.append(item_output)
            else:
                item_icd_description_en = self.icdlookup.get_procedure_description_en(row["CODE"])
                item_icd_description_es =  self.icdlookup.get_procedure_description_es(row["CODE"])
                item_output = IcdItemNer(id=item_idx, main_term=item_main_term, offsets=item_offsets, context=item_context, icd_phrase=item_icd_phrase, 
                                 icd_description_en=item_icd_description_en, icd_description_es=item_icd_description_es)
                output_procedures.append(item_output)

        prompt = util_text.add_html_offset(txt, df_select["MAIN_OFFSETS"].to_list(), df_select["TYPE"].to_list())

        example = {}
        example["prompt"] = prompt
        example["output"] = IcdListNer(diagnoses= output_diagnoses, procedures=output_procedures)

        return example
    


    def get_prompt_description_example_substrings_txt(self, file_number=0, context_size=1, selected_codes=[], max_examples=5):
        file_name = self.files[file_number]
        df_select = self.dev_x[self.dev_x["FILE"] == file_name]
        txt = self.codiformat.get_text(split="dev", id=file_name)
        
        prompt_items = []
        output_items = []
        found_procudure = False

        for idx, row, in df_select.iterrows():
            if selected_codes and row["CODE"] not in selected_codes:
                continue
            prompt_item = self.codiformat.get_description_prompt_substring(txt=txt, row=row, n=context_size)
            prompt_item["id"] = len(prompt_items)
            
            output_item = {}
            output_item["id"] = len(prompt_items)

            if row["TYPE"] == CodiFormat.DIAGNOSTICO:
                output_item["description_en"] = self.icdlookup.get_diangose_description_en(row["CODE"])
                output_item["description_es"] = self.icdlookup.get_diangose_description_es(row["CODE"])
            else:
                output_item["description_en"] = self.icdlookup.get_procedure_description_en(row["CODE"])
                output_item["description_es"] = self.icdlookup.get_procedure_description_es(row["CODE"])

            if not isNaNorNone(output_item["description_es"]) and not isNaNorNone(output_item["description_en"]):
                if output_item["description_es"] != output_item["description_en"]:
                    prompt_items.append(prompt_item)
                    output_items.append(output_item)
                    if row["TYPE"] == CodiFormat.PROCEDIMIENTO:
                        found_procudure = True
            
            if len(prompt_items) > max_examples and found_procudure:
                break
                

        return {"prompt": prompt_items, "output": output_items}