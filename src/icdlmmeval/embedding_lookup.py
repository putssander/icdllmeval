from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
import json
import re
import configparser

class EmbeddingLookup:

    def __init__(self):

        embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')

        config = configparser.ConfigParser()
        config.read('./../resources/config.ini')
        path_diagnoses_db = config["vectorstore"]['diagnoses_db']
        path_procedures_db= config["vectorstore"]['procedures_db']

        self.vectordb_diag = Chroma(persist_directory=path_diagnoses_db, embedding_function=embeddings)
        self.vectordb_proc = Chroma(persist_directory=path_procedures_db, embedding_function=embeddings)
        print('vector stores loaded')


    def search_diagnose(self, substring, icd_phrase, k=25):
        subsearch = self.vectordb_diag.similarity_search(substring, k=k)
        phrasesearch = self.vectordb_diag.similarity_search(icd_phrase, k=k)
        return subsearch + phrasesearch

    def search_procedure(self, substring, icd_phrase, k=25):
        subsearch = self.vectordb_proc.similarity_search(substring, k=k)
        phrasesearch = self.vectordb_proc.similarity_search(icd_phrase, k=k)
        return subsearch + phrasesearch

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

            
