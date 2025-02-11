from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
import json
import re
import configparser
from icdlmmeval.codiesp.codiformat import CodiFormat
import os

class EmbeddingLookup:

    def __init__(self):

        embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')

        CONFIG_PATH = os.path.join(os.path.dirname(__file__), '../../resources/config.ini')
        config = configparser.ConfigParser()
        config.read(CONFIG_PATH)

        path_diagnoses_db = config["vectorstore"]['diagnoses_db']
        path_procedures_db= config["vectorstore"]['procedures_db']

        self.vectordb_diag = Chroma(persist_directory=path_diagnoses_db, embedding_function=embeddings)
        self.vectordb_proc = Chroma(persist_directory=path_procedures_db, embedding_function=embeddings)
        print('vector stores loaded')


    def search_diagnose(self, substring, k=50):
        return self.vectordb_diag.similarity_search(substring, k=k)

    def search_procedure(self, substring, k=50):
        return self.vectordb_proc.similarity_search(substring, k=k)


    def docs_to_json(self, docs):
        json_array = []
        for doc in docs:
            jdoc = {}
            jdoc['code'] = doc.metadata['code']
            jdoc["descr"] = doc.page_content
            json_array.append(jdoc)
        return json.dumps(json_array, ensure_ascii=False)
    
    def get_code_index(self, docs_json, code):
        code = code.replace("X", ".")
        for idx, doc in enumerate(json.loads(docs_json)):
            current_code = doc['code']
            if re.search(code, current_code, re.IGNORECASE):
                return idx
        else:
            return -1
  
    def get_code_index_parent(self, docs_json, code, type):
        code = code.replace("X", ".")
        if type == CodiFormat.DIAGNOSTICO:
            code = code[:3]
        else:
            code = code[:4]
        for idx, doc in enumerate(json.loads(docs_json)):
            current_code = doc['code']
            if re.search(code, current_code, re.IGNORECASE):
                return idx
        else:
            return -1
