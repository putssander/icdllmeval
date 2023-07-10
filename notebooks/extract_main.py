import sys, os
import pandas as pd
import json
import openai


sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'src'))
print(sys.path)

from icdlmmeval.codiesp.codiformat import CodiFormat
from icdlmmeval.icd_prompts import IcdPrompts



def get_terms(s):
    return s.split(" ")

def get_len(terms):
    return len(terms)

def dump_main_terms(data):
    with open("main-terms-script.json", "w") as outfile:
        outfile.write(data)


codiformat = CodiFormat(path_codiesp='/home/jovyan/work/icdllmeval/notebooks/codiesp')
df_train_x = codiformat.get_df_x("train")
df_dev_x = codiformat.get_df_x("dev")
df_test_x = codiformat.get_df_x("test")

df_x = pd.concat([df_train_x, df_dev_x, df_test_x])

df_x["SUBSTRING_LOWER"] = df_x["SUBSTRING"].str.lower()
df_x = df_x.drop_duplicates(subset=['SUBSTRING_LOWER'])
df_x["TERMS"] = df_x["SUBSTRING_LOWER"].apply(get_terms)
df_x["TERMS_LEN"] = df_x["TERMS"].apply(get_len)

df_multiple_main_terms = df_x[df_x["TERMS_LEN"] > 1]
df_multiple_main_terms_diag = df_multiple_main_terms[df_multiple_main_terms["TYPE"] == "DIAGNOSTICO"]
df_multiple_main_terms_proc = df_multiple_main_terms[df_multiple_main_terms["TYPE"] == "PROCEDIMIENTO"]
print(len(df_multiple_main_terms))
print(len(df_multiple_main_terms_diag))
print(len(df_multiple_main_terms_proc))

os.environ["OPENAI_API_KEY"] = "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"
openai.api_key = os.getenv("OPENAI_API_KEY") or "OPENAI_API_KEY"
openai.Engine.list()  # check we have authenticated

# model_name = "gpt-3.5-turbo-0613"
model_name = "gpt-4"
icd_prompts = IcdPrompts(model_name=model_name)

f = open('main-term-examples.json', "r")
examples = json.loads(f.read())
substrings = df_multiple_main_terms_diag["SUBSTRING"].to_list()
batch_size = 50
batch_results = []
print('batches', len(substrings)/batch_size)
for i in range(0, len(substrings), batch_size):
    print(f'batch {i}'.format(i=i))
    batch = substrings[i:i+batch_size]
    batch_results.append(icd_prompts.prompt_main_terms(examples, batch))
    dump_main_terms(batch_results)