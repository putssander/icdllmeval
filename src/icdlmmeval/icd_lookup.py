import pandas as pd
from bs4 import BeautifulSoup
import configparser


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
        self.bs_data = BeautifulSoup(data, "xml")
        print("lookup dictonaries loaded")



    def get_diangose_description_en(self, code):
        code = code.replace(".", "").upper()
        print(code)
        if code in self.df_icd10cm_codes_dict:
            return self.df_icd10cm_codes_dict[code]
        else:
            return None 

   

    def get_diangose_description_es(self, code):
        code = code.upper()
        print(code)
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
        items = self.bs_data.find_all(lambda tag: tag.name == "code" and code in tag.text)
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