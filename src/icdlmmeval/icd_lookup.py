import pandas as pd
from bs4 import BeautifulSoup
import configparser
import re


header_icd10_codes = ["LINE"]
header_icd10_codes_es = ["CODE", "DESCRIPTION", "DESCRIPTION_L"]


class IcdLookup:


    def __init__(self):
        config = configparser.ConfigParser()
        config.read('./../resources/config.ini')

        path_codiesp_eval = config['codiesp']['eval']
        path_icd_10_cm_codes = config['gov.cms.icd']['path_icd_10_cm_codes']
        path_icd_10_pcs_codes = config['gov.cms.icd']['path_icd_10_pcs_codes']
        path_icd_10_cm_code_tables = config['gov.cms.icd']['path_icd_10_cm_code_tables']
        path_icd_10_pcm_code_tables = config['gov.cms.icd']['path_icd_10_pcm_code_tables']

        print("read path_icd_10_cm_codes")
        df_icd10cm_codes = pd.read_csv(f"{path_icd_10_cm_codes}/2020 Code Descriptions/icd10cm_codes_2020.txt", names=header_icd10_codes, sep="\t")
        df_icd10cm_codes["CODE"] = df_icd10cm_codes["LINE"].apply(lambda x: x[:7].strip())
        df_icd10cm_codes["DESCRIPTION"] = df_icd10cm_codes['LINE'].astype(str).str[8:]

        self.df_icd10cm_codes_dict = dict(zip(df_icd10cm_codes.CODE, df_icd10cm_codes.DESCRIPTION))
        print(len(self.df_icd10cm_codes_dict))
        print("read path_icd_10_pcs_codes")
        df_icd10pcs_codes = pd.read_csv(f"{path_icd_10_pcs_codes}/icd10pcs_codes_2020.txt", names=header_icd10_codes, sep='\t')
        df_icd10pcs_codes["CODE"] = df_icd10pcs_codes["LINE"].apply(lambda x: x[:7].strip())
        df_icd10pcs_codes["DESCRIPTION"] = df_icd10pcs_codes['LINE'].astype(str).str[8:]
        self.df_icd10pcs_codes_dict = dict(zip(df_icd10pcs_codes.CODE, df_icd10pcs_codes.DESCRIPTION))
        print(len(self.df_icd10pcs_codes_dict))

        print("read codiesp_codes")
        df_icd10cm_codes_es = pd.read_csv(f"{path_codiesp_eval}/codiesp_codes/codiesp-D_codes.tsv", names=header_icd10_codes_es, sep="\t")
        self.df_icd10cm_codes_dict_es = dict(zip(df_icd10cm_codes_es.CODE, df_icd10cm_codes_es.DESCRIPTION))

        df_icd10pcs_codes = pd.read_csv(f"{path_codiesp_eval}/codiesp_codes/codiesp-P_codes.tsv", names=header_icd10_codes_es, sep="\t")
        self.df_icd10pcs_codes_dict_es = dict(zip(df_icd10pcs_codes.CODE, df_icd10pcs_codes.DESCRIPTION))

        print("read icd10cm_index")
        with open(f'{path_icd_10_cm_code_tables}/2020 Table and Index/icd10cm_index_2020.xml', 'r') as f:
            data = f.read()
        self.bs_data_diagnoses = BeautifulSoup(data, "xml")
        print("read icd10cm_neoplasm")
        with open(f'{path_icd_10_cm_code_tables}/2020 Table and Index/icd10cm_neoplasm_2020.xml', 'r') as f:
            data = f.read()
        self.bs_data_neoplasm = BeautifulSoup(data, "xml")
        print("read icd10pcs_index")
        with open(f'{path_icd_10_pcm_code_tables}/PCS_2020/icd10pcs_index_2020.xml', 'r') as f:
            data = f.read()
        self.bs_data_pcs = BeautifulSoup(data, "xml")
        print("lookup dictonaries loaded")



    def get_diangose_description_en(self, code):
        code = code.replace(".", "").upper()
        if code in self.df_icd10cm_codes_dict:
            return self.df_icd10cm_codes_dict[code]
        else:
            return None 

   

    def get_diangose_description_es(self, code):
        code = code.upper()
        if code in self.df_icd10cm_codes_dict_es:
            return self.df_icd10cm_codes_dict_es[code]
        else:
            return None 
 

    def get_procedure_description_en(self, code):
        code = code.replace(".", "").upper()
        if code in self.df_icd10pcs_codes_dict_es:
            return self.df_icd10pcs_codes_dict_es[code]
        else:
            return None 

    def get_procedure_description_es(self, code):
        code = code.upper()
        if code in self.df_icd10pcs_codes_dict_es:
            return self.df_icd10pcs_codes_dict_es[code]
        else:
            return None  
        

    def get_main_terms(self, code):
        code = code.upper()
        if re.match(r'^C|D', code):
            items = self.bs_data_neoplasm.find_all(lambda tag: tag.name == "cell" and code in tag.text)
        else:
            items = self.bs_data_diagnoses.find_all(lambda tag: tag.name == "code" and code in tag.text)
        result = [self.get_main_term(item) for item in items]
        return {code: result}


    def get_main_terms_pcs(self, code):
        code = code.upper()
        items = self.bs_data_pcs.find_all(lambda tag: tag.name == "code" and code in tag.text)
        result = [self.get_main_term(item) for item in items]
        return {code: result}
    
 
    def get_main_term(self, item):
        parent = item.parent
        children = []
        while(parent):
            children.insert(0, parent.title.text)
            if parent.name == "mainTerm":
                return parent.title.text, children          
            else:
                parent = parent.parent
        return None, children
    
       
    def get_main_terms_list(self, code):
        code = code.upper()
        if re.match(r'^C|D', code):
            return ["Neoplasm, neoplastic"]
        item = self.get_main_terms(code)
        code_item = item[code.upper()]
        main_set = set()
        for main in code_item:
            main_set.add(main[0])
        main_list = list(main_set)
        main_list.sort()
        return main_list

    def get_main_terms_list_pcs(self, code):
        code = code.upper()
        item = self.get_main_terms_pcs(code)
        code_item = item[code.upper()]
        main_set = set()
        for main in code_item:
            main_set.add(main[0])
        main_list = list(main_set)
        main_list.sort()
        return main_list
    

    def get_cm_main_terms(self):
        main_terms = self.bs_data_diagnoses.find_all(lambda tag: tag.name == "mainTerm")
        titles = [item.title.text for item in main_terms]
        return titles
    
    def get_pcs_main_terms(self):
        main_terms = self.bs_data_pcs.find_all(lambda tag: tag.name == "mainTerm")
        titles = [item.title.text for item in main_terms]
        return titles
    

    def get_cm_main_terms_entry(self, main_term):
        entries = self.bs_data_diagnoses.find_all(lambda tag: tag.name == "mainTerm" and main_term in tag.title.text)
        return entries
    
    def get_pcs_main_terms_entry(self, main_term):
        entries = self.bs_data_pcs.find_all(lambda tag: tag.name == "mainTerm" and main_term in tag.title.text)
        return entries