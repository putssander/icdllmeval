import os
import configparser
import pandas as pd
import json
import sys, os

sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'src'))
from tqdm import tqdm
from icdlmmeval import util_text, util
print(sys.path)

from tqdm import tqdm

import openai
import langchain

from icdlmmeval.codiesp.codiformat import CodiFormat
from icdlmmeval.codiesp.prompt_examples import PromptExamples
from icdlmmeval.codiesp import eval
from icdlmmeval.ner_main_predict import NerMainPredict
from icdlmmeval.embedding_lookup import EmbeddingLookup
from icdlmmeval.icd_prompts import IcdPrompts
from icdlmmeval.icd_prompts import IcdListNer

from icdlmmeval import ner_parsing
from icdlmmeval.llm_icd_coding import LLMIcdCoding
from langchain.output_parsers import PydanticOutputParser
from langchain.load.dump import dumps


def get_example(file_number):
    prompt_examples = PromptExamples()
    ## warning dev file 29 is selected, as most files include procedure codes with missing decriptions
    description_example = prompt_examples.get_prompt_description_example_txt(file_number=file_number)
    print("\n** PROMPT EXAMPLE ***")
    print(description_example["prompt"])
    print(json.dumps(description_example["output"].dict(), indent=4, ensure_ascii=False)) #pydantic v1 is using model.dict(), v2 is using model.dumps()
    return description_example


def get_main_code_descriptions(icd_prompts, codiformat, split, selected_files, df_ner=pd.DataFrame(), fp_out=None, max_length=None, description_example=None):
    
    encoding = util_text.get_encoding(llm_model_name)
    file_code_descriptions = []
    
    if df_ner.empty:
        ner = NerMainPredict()
    else:
        ner = None

    for file_name in tqdm(selected_files):
        txt = codiformat.get_text(split, file_name)

        if not df_ner.empty:
            main_terms = codiformat.get_predicted_entities(df_ner, file_name)
        else:
            main_terms = ner.classify(txt)
        
        ner_html = codiformat.get_description_prompt_txt_entities(txt, main_terms)
        
        if max_length:
            sections = util_text.merge_sections(util_text.get_sections(ner_html, max_length, encoding),max_length, encoding)
        else:
            sections = [ner_html]

        for idx, section_html in enumerate(sections):
            descriptions = icd_prompts.prompt_icd_code_description_from_main_terms(example=description_example, main_terms=section_html)
        
            file_result = {}
            file_result["file"] = file_name
            file_result["text"] = txt
            file_result["section"] = idx
            file_result["ner_html"] = section_html
            file_result["descriptions"] = descriptions
            file_code_descriptions.append(file_result)
            
            df_file_code_descriptions = pd.DataFrame().from_records(file_code_descriptions)
            df_file_code_descriptions.to_excel(fp_out)
    return file_code_descriptions


def predict_code_from_descriptions_file_df(llm_icd_coding, codiformat, df_descriptions, fp_out):
    code_results = []
    for idx, row in df_descriptions.iterrows():
        print(idx)
        file_name = row["file"]
        descriptions = util.clean_md(row["descriptions"])
        print(f"file={file_name} descriptions={descriptions}")
        if "diagnoses" in descriptions:
            for description in tqdm(descriptions["diagnoses"]):
                code_result = llm_icd_coding.predict_code_from_description(file_name=file_name, description=description, code_type=codiformat.DIAGNOSTICO)
                code_results.append(code_result)
        if "procedures" in descriptions:
            for description in tqdm(descriptions["procedures"]):
                code_result = llm_icd_coding.predict_code_from_description(file_name=file_name, description=description, code_type=codiformat.PROCEDIMIENTO)        
                code_results.append(code_result)
        df_codes = pd.DataFrame.from_records(code_results)
        df_codes.to_excel(fp_out) # write directly to excel to prevent loss of data
    return code_results



def code_descriptions():
    df_descriptions = pd.read_excel(f"/home/jovyan/work/icdllmeval/resources/gpt-descriptions/file-descriptions-{split}.xlsx")

    # file1 = df_descriptions["file"].to_list()[0]
    # df_descriptions = df_descriptions[df_descriptions["file"] == file1]
    fp_out = f"/home/jovyan/work/icdllmeval/resources/gpt-codes/predicted-codes-{split}.xlsx"
    code_results = predict_code_from_descriptions_file_df(df_descriptions=df_descriptions, fp_out=fp_out)


def main(split="test", llm_model_name="gpt-4", max_sequence_length=False, split_begin_index=None, split_end_index=None):
    config = configparser.ConfigParser()
    config.read('./../resources/config.ini')

    print("load split and selection")
    codiformat = CodiFormat()
    df_gold_x = codiformat.get_df_x(split)
    print(df_gold_x.head())
    if split_begin_index or split_end_index:
        selected_files = df_gold_x["FILE"].unique()[split_begin_index:split_end_index]
    else:
        selected_files = df_gold_x["FILE"].unique()
    print(selected_files)
    print(len(selected_files))

    print("set example")
    description_example = get_example(29)

    print("load prompts")
    icd_prompts = IcdPrompts(model_name=llm_model_name)
    icd_prompts.set_model(llm_model_name)

    # output file for PROMPT 1 & 2
    descriptions_out = f"/home/jovyan/work/icdllmeval/resources/gpt-descriptions/file-descriptions-{split}-{split_begin_index}-{split_end_index}.xlsx"

    # PROMPT 1 & 2
    # df_ner = pd.read_excel(f'/home/jovyan/work/icdllmeval/resources/main-pred/entities-ner-{split}.xlsx')
    # get_main_code_descriptions(icd_prompts=icd_prompts, codiformat=codiformat, split=split, selected_files=selected_files, df_ner=df_ner, fp_out=descriptions_out, description_example=description_example)

    # PROMPT 3: Select the correct code by quering the vector database and using GPT-4 to select the best matching code from the results
    df_descriptions = pd.read_excel(descriptions_out)
    llm_icd_coding = LLMIcdCoding(model_name=llm_model_name)
    fp_out = f"/home/jovyan/work/icdllmeval/resources/gpt-codes/predicted-codes-{split}-{split_begin_index}-{split_end_index}.xlsx"
    predict_code_from_descriptions_file_df(llm_icd_coding=llm_icd_coding, codiformat=codiformat, df_descriptions=df_descriptions, fp_out=fp_out)



if __name__ == "__main__":
    openai.api_key="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"
    os.environ["OPENAI_API_KEY"] = "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"

    split = "test"
    # llm_model_name = "gpt-3.5-turbo"
    llm_model_name = "gpt-4"
    llm_model_name = "gpt-4-1106-preview"
    max_sequence_length = 1024 # gpt4 limit is 8k, answer often longer than question, gpt-4-turbo limit is 128k of which 4k output
    max_sequence_length = None
    split_begin_index = 0
    split_end_index = None
    split_begin_index = None
    split_end_index = None

    main(split, llm_model_name, max_sequence_length, split_begin_index, split_end_index)

