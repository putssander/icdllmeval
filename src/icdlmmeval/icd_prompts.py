from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate, AIMessagePromptTemplate
from langchain.chat_models import ChatOpenAI

import logging
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator
from typing import List
import json
import tiktoken


FORMAT = '%(asctime)s %(message)s'

logging.basicConfig(filename='/home/jovyan/work/icdllmeval/notebooks/prompts.log', encoding='utf-8', level=logging.INFO, filemode='w', format=FORMAT)
logging.info('test')


    # main_term: str = Field(description="translate the 'original phrase' to the standardized 'main term' to locate the medical code in the alphabetic icd-index")


class IcdItem(BaseModel):
    original_phrase: str = Field(description="original phrase")
    text_snippets: str = Field(description="extract (multiple) text snippets from source text containing all information and context for the ICD coding of 'original phrase' (mimLength=64; maxLength=128)")
    icd_code: str = Field(description="ICD code")
    icd_code_description: str = Field(description="ICD code description")
    icd_code_description_es: str = Field(description="Translated ICD code description in Spanish")

# PROMPT 1&2 object
class IcdItemNer(BaseModel):

    id: str = Field(description="Unique identifier for the main tag.")
    main_term: str = Field(description="ICD primary/main/lead term identified by NER in the text.")
    offsets: str = Field(description="Character positions in the source text indicating the start and end of the main term.")
    context: str = Field(description="A snippet of text providing context around the main term.")
    icd_phrase: str = Field(description="A string capturing the main term along with its pertinent descriptive elements, offering concise textual evidence for the ICD code, without including entire sentences.")
    icd_description_en: str = Field(description="Most specific ICD-10 code description in English.")
    icd_description_es: str = Field(description="Most specific ICD-10 code description in Spanish.")

class IcdList(BaseModel):
    procedures: List[IcdItem] = Field(description="list of icd diagnose items")
    diagnoses: List[IcdItem] = Field(description="list of icd procedure items")

# PROMPT 1&2 object
class IcdListNer(BaseModel):
    diagnoses: List[IcdItemNer] = Field(description="list of icd diagnose items (main type=\"D\")")
    procedures: List[IcdItemNer] = Field(description="list of icd procedure items (main type=\"P\")")

class TermItem(BaseModel):
    original_phrase: str = Field(description="original phrase'")
    main_term: List[str] = Field(description="preferably a single term for the main condition, no anatomical sites, laterality, aetiology, severity, type or other details ")

class TermList(BaseModel):
    main_terms: List[TermItem] = Field(description="list extracted main items")

class IcdPhraseDescriptionPrompt(BaseModel):
    id: str = Field(description="id of the icd item")
    context: str = Field(description="context")
    icd_phrase: str = Field(description="textual evidence for the medical code")

class IcdPhraseDescriptionOutput(BaseModel):
    id: str = Field(description="id of the icd item")
    icd_description_en: str = Field(description="Official ICD-10 code description for the icd_phrase within its context")
    icd_description_es: str = Field(description="Translated ICD-10 code description in Spanish")

class IcdPhraseDescriptionList(BaseModel):
    descriptions: List[IcdPhraseDescriptionOutput] = Field(description="output list of code descriptions")


class IcdPrompts():

    def __init__(self, model_name = "gpt-4"):
        self.output_parser_substrings = self.get_output_parser_substrings()
        self.format_instructions_substrings = self.get_format_instructions_substrings(self.output_parser_substrings )
        self.output_parser_select = self.get_output_parser_select()
        self.output_parser_select_simple = self.get_output_parser_select_simple()

        self.set_model(model_name=model_name)

    def get_output_parser_substrings(self):
        response_schemas = [
            ResponseSchema(name="procedures", description="list of substrings for coding ICD procedures"),
            ResponseSchema(name="diagnoses", description="list of substrings for coding ICD diagnoses"),
        ]
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        return output_parser


    def get_format_instructions_substrings(self, output_parser):
        format_instructions = output_parser.get_format_instructions().replace(
            '"procedures": string', '"procedures": ["strings"]'
        ).replace(
            '"diagnoses": string', '"diagnoses": ["strings"]'
        )
        return format_instructions


    def get_output_parser_select(self):
        response_schemas_code = [
            ResponseSchema(name="code_listed", description="best listed ICD-10 code"),
            ResponseSchema(name="code_suggestion", description="suggested ICD-10 code"),
            ResponseSchema(name="listed", description="was the correct code listed (boolean)"),
            ResponseSchema(name="reasoning", description="explanation of final code assigned"),
            ResponseSchema(name="code_assigned", description="final code to assigned, empty if no code should be assigned"),
            ResponseSchema(name="confidence", description="correctly ICD code_assigned confidence (probablity from 0-1)")
        ]
        output_parser_code = StructuredOutputParser.from_response_schemas(response_schemas_code)
        return output_parser_code
    

    def get_output_parser_select_simple(self):
        response_schemas_code = [
            ResponseSchema(name="selected_code", description="best listed ICD-10 code"),
        ]
        output_parser_code = StructuredOutputParser.from_response_schemas(response_schemas_code)
        return output_parser_code
    

    def extract_substrings(self, txt, examples):
        format_instructions = self.format_instructions_substrings

        behaviour_instructions = "You are an expert in medical ICD coding, teaching peers the highest level of coding possible."
        coding_instructions = "Extract procedures and diagnoses substrings for an input text as best you can. The substrings should form the basis for ICD-10 coding. Focus on recall over precision, repeat substrings with multiple occurences."
        system_message_prompt = SystemMessagePromptTemplate.from_template("{behaviour_instructions} {coding_instructions} {format_instructions}\n")
        example_human = HumanMessagePromptTemplate.from_template("{question}", additional_kwargs={"name": "example_user"})
        example_ai = AIMessagePromptTemplate.from_template("{answer}", additional_kwargs={"name": "example_assistant"})
        human_message_prompt = HumanMessagePromptTemplate.from_template("{text}")

        chat_prompt = ChatPromptTemplate(
            messages=[system_message_prompt, example_human, example_ai, human_message_prompt], 
            input_variables=["question", "answer", "text"],
            partial_variables={"behaviour_instructions": behaviour_instructions, "coding_instructions": coding_instructions, "format_instructions": format_instructions,}
        )
        _input = chat_prompt.format_prompt(question=examples[0]['question'], answer=examples[0]['answer'], text=txt, format_instructions=format_instructions)
        logging.info(_input.to_messages())
        output = self.chat(_input.to_messages())
        logging.info(output.content)
        json_substrings = self.output_parser_substrings.parse(output.content)
        return json_substrings

    # PROMPT 3
    def select_code(self, item):

        format_instructions = self.output_parser_select.get_format_instructions()
        # coding_instructions = "Your task is to determine the most specific and exact ICD-10 code that aligns directly with the provided icd_phrase. While the 'context' field offers additional information, your primary focus should be on pinpointing the code that most accurately and specifically represents the icd_phrase itself. In cases where the ideal code is not listed, please provide a precise alternative in the 'code_suggestion' field, ensuring compliance with the format instructions."
        # behaviour_instructions = "You are an expert in medical ICD coding, teaching peers the highest level of coding possible."
        # system_message_prompt = SystemMessagePromptTemplate.from_template("{behaviour_instructions} {coding_instructions} {format_instructions}\n")
        # human_message_prompt = HumanMessagePromptTemplate.from_template("item: {item}. If you think the correct code is not in the list, provide your best code suggestion, always follow the format instructions.")

        behaviour_instructions = (
            "You are an expert in medical ICD coding, responsible for teaching peers the highest level "
            "of coding accuracy and specificity."
        )
        coding_instructions = (
            "Your primary task is to identify the most specific and exact ICD-10 code that aligns "
            "directly with the provided icd_phrase. Use the additional information in the 'context' field "
            "to better understand the scenario, but ensure your code selection precisely matches the icd_phrase."
        )

        system_message_prompt = SystemMessagePromptTemplate.from_template(
            "{behaviour_instructions} {coding_instructions} {format_instructions}\n"
        )
        human_message_prompt = HumanMessagePromptTemplate.from_template(
            "Consider this item: {item}. Should the correct code not be in the list, kindly provide your best "
            "code suggestion, following the format instructions meticulously."
        )


        chat_prompt = ChatPromptTemplate(
            # messages=[system_message_prompt, example_human, example_ai, human_message_prompt], 
            messages=[system_message_prompt, human_message_prompt], 
            input_variables=["item"],
            partial_variables={"behaviour_instructions": behaviour_instructions, "coding_instructions": coding_instructions, "format_instructions": format_instructions,}
        )
        # _input = chat_prompt.format_prompt(question=examples[0]['question'], answer=examples[0]['answer'], item=item)
        _input = chat_prompt.format_prompt(item=item)
        logging.info(_input.to_messages())
        output = self.chat(_input.to_messages())
        logging.info(output.content)
        json_substrings = self.output_parser_select.parse(output.content)      
        return json_substrings
    

    def select_code_simple(self, item):

        format_instructions = self.output_parser_select_simple.get_format_instructions()
        coding_instructions = "Select the best ICD-10 code for the icd_phrase from the listed hits."
        behaviour_instructions = "You are an expert in medical ICD coding, teaching peers the highest level of coding possible."
        system_message_prompt = SystemMessagePromptTemplate.from_template("{behaviour_instructions} {coding_instructions} {format_instructions}\n")
        human_message_prompt = HumanMessagePromptTemplate.from_template("item: {item}. Select the best ICD-10 code for the icd_phrase from the listed hits. Always follow the format instructions.")

        chat_prompt = ChatPromptTemplate(
            messages=[system_message_prompt, human_message_prompt], 
            input_variables=["item"],
            partial_variables={"behaviour_instructions": behaviour_instructions, "coding_instructions": coding_instructions, "format_instructions": format_instructions,}
        )
        _input = chat_prompt.format_prompt(item=item)
        logging.info(_input.to_messages())
        output = self.chat(_input.to_messages())
        logging.info(output.content)
        json_substrings = self.output_parser_select_simple.parse(output.content)      
        return json_substrings


    def prompt_icd_item_info(self, txt, substrings):

        # Set up a parser + inject instructions into the prompt template.
        parser = PydanticOutputParser(pydantic_object=IcdList)

        format_instructions = parser.get_format_instructions()
        # coding_instructions = "What is the correct ICD-10 code for the substring. If you think the correct code is not listed, provide your best code suggestion in the json field 'code', always follow the format instructions."
        behaviour_instructions = "You are an expert in medical ICD coding, teaching peers the highest level of coding possible."
        system_message_prompt = SystemMessagePromptTemplate.from_template("{behaviour_instructions} {format_instructions}\n")
        human_message_prompt = HumanMessagePromptTemplate.from_template("For each of items in the lists, add additional information in Spanish to preform an medical code lookup for '{substring}', use {format_instructions} in markdown! Extract from the following text {txt}. The number of listed items of the output should match the input.")

        chat_prompt = ChatPromptTemplate(
            messages=[system_message_prompt,human_message_prompt], 
            input_variables=["substring", "txt"],
            partial_variables={"behaviour_instructions": behaviour_instructions, "format_instructions": format_instructions,}
        )
        _input = chat_prompt.format_prompt( substring=substrings, txt=txt)
        logging.info(_input.to_messages())
        output = self.chat(_input.to_messages())
        logging.info(output.content)
        print(output.content)
        try:
            json_substrings = parser.parse(output.content)
            print(json_substrings)
        except Exception as e: 
            logging.error(e)            
        return json_substrings
    

    def prompt_main_terms(self, examples, unfiltered_substrings):

        # Set up a parser + inject instructions into the prompt template.
        parser = PydanticOutputParser(pydantic_object=TermList)

        format_instructions = parser.get_format_instructions()
        # coding_instructions = "What is the correct ICD-10 code for the substring. If you think the correct code is not listed, provide your best code suggestion in the json field 'code', always follow the format instructions."
        behaviour_instructions = "You are an expert in medical ICD coding, teaching peers the highest level of coding possible."
        system_message_prompt = SystemMessagePromptTemplate.from_template("{behaviour_instructions} {format_instructions}\n")
        example_human = HumanMessagePromptTemplate.from_template("{question}", additional_kwargs={"name": "example_user"})
        example_ai = AIMessagePromptTemplate.from_template("{answer}", additional_kwargs={"name": "example_assistant"})
        human_message_prompt = HumanMessagePromptTemplate.from_template("For each item in the lists, extract a single or composed main term, closely follow the format instructions {format_instructions} in markdown and the given above example! Extract for the following list {unfiltered_substrings}")

        chat_prompt = ChatPromptTemplate(
            messages=[system_message_prompt,example_human, example_ai, human_message_prompt], 
            input_variables=["question", "answer", "unfiltered_substrings"],
            partial_variables={"behaviour_instructions": behaviour_instructions, "format_instructions": format_instructions,}
        )
        _input = chat_prompt.format_prompt(question=json.dumps(examples['question']), answer=json.dumps(examples['answer']), unfiltered_substrings=unfiltered_substrings)
        logging.info(_input.to_messages())
        output = self.chat(_input.to_messages())
        logging.info(output.content)
        print(output.content)
        try:
            json_substrings = json.loads(output.content)
            print(json_substrings)
        except Exception as e: 
            logging.error(e)            
        return json_substrings
    
    # prompt-1
    def prompt_icd_description_from_main_terms(self, txt, main_terms):

        # Set up a parser + inject instructions into the prompt template.
        parser = PydanticOutputParser(pydantic_object=IcdList)

        format_instructions = parser.get_format_instructions()
        behaviour_instructions = "You are an expert in medical ICD coding, teaching peers the highest level of coding possible."
        system_message_prompt = SystemMessagePromptTemplate.from_template("{behaviour_instructions} {format_instructions}\n")
        human_message_prompt = HumanMessagePromptTemplate.from_template("For each item in the lists, extract and add additional information in Spanish to preform an medical code lookup. The list '{main_terms}', use {format_instructions} in markdown!")

        chat_prompt = ChatPromptTemplate(
            messages=[system_message_prompt,human_message_prompt], 
            input_variables=["substring", "txt"],
            partial_variables={"behaviour_instructions": behaviour_instructions, "format_instructions": format_instructions,}
        )
        _input = chat_prompt.format_prompt( substring=main_terms, txt=txt)
        logging.info(_input.to_messages())
        output = self.chat(_input.to_messages())
        logging.info(output.content)
        print(output.content)
        try:
            json_substrings = parser.parse(output.content)
            print(json_substrings)
        except Exception as e: 
            logging.error(e)            
        return json_substrings
    
    # prompt-1 and 2 as used on test set
    def prompt_icd_code_description_from_main_terms(self, example, main_terms):

        # Set up a parser + inject instructions into the prompt template.
        parser = PydanticOutputParser(pydantic_object=IcdListNer)

        format_instructions = parser.get_format_instructions()
        behaviour_instructions = (
            "You are an expert in medical ICD coding, responsible for teaching peers the highest level "
            "of coding accuracy and specificity."
        )
        
        system_message_prompt = SystemMessagePromptTemplate.from_template("{behaviour_instructions}\n")
        example_human = HumanMessagePromptTemplate.from_template("{question}", additional_kwargs={})
        example_ai = AIMessagePromptTemplate.from_template("{answer}", additional_kwargs={})
        human_message_prompt = HumanMessagePromptTemplate.from_template("For each tag main in the HTML, provide the additional information as shown in the example above. The HTML '{main_terms}', {format_instructions} in markdown!")

        chat_prompt = ChatPromptTemplate(
            messages=[system_message_prompt, example_human, example_ai, human_message_prompt], 
            input_variables=["main_terms", "question", "answer"],
            partial_variables={"behaviour_instructions": behaviour_instructions, "format_instructions": format_instructions,}
        )
        _input = chat_prompt.format_prompt(
            main_terms=json.dumps(main_terms,  ensure_ascii=False), 
            question=json.dumps(example["prompt"], ensure_ascii=False), 
            answer=json.dumps(example["output"].dict(),  ensure_ascii=False)
            )
        logging.info(_input.to_messages())
        output = self.chat(_input.to_messages())
        logging.info(output.content)          
        return output.content
    
    def set_model(self, model_name, temperature = 0.0):
        self.chat = ChatOpenAI(model_name=model_name, temperature=temperature)
        self.encoding = tiktoken.encoding_for_model(model_name)


    def prompt_icd_description_from_icd_phrase(self, examples, icd_phrases_list):

        # Set up a parser + inject instructions into the prompt template.
        parser = PydanticOutputParser(pydantic_object=IcdPhraseDescriptionList)

        format_instructions = parser.get_format_instructions()
        # coding_instructions = "What is the correct ICD-10 code for the substring. If you think the correct code is not listed, provide your best code suggestion in the json field 'code', always follow the format instructions."
        behaviour_instructions = "You are an expert in medical ICD coding, teaching peers the highest level of coding possible."
        system_message_prompt = SystemMessagePromptTemplate.from_template("{behaviour_instructions} {format_instructions}\n")
        example_human = HumanMessagePromptTemplate.from_template("{question}", additional_kwargs={"name": "example_user"})
        example_ai = AIMessagePromptTemplate.from_template("{answer}", additional_kwargs={"name": "example_assistant"})
        human_message_prompt = HumanMessagePromptTemplate.from_template("For each item in the lists, add the ICD-10 code description, closely follow the format instructions {format_instructions} in markdown and the given above example! Extract for the following list {prompt_list}")

        chat_prompt = ChatPromptTemplate(
            messages=[system_message_prompt, example_human, example_ai, human_message_prompt], 
            input_variables=["question", "answer", "prompt_list"],
            partial_variables={"behaviour_instructions": behaviour_instructions, "format_instructions": format_instructions,}
        )
        _input = chat_prompt.format_prompt(
            question=json.dumps(examples['prompt'], ensure_ascii=False), 
            answer=json.dumps(examples['output'], ensure_ascii=False), 
            prompt_list=json.dumps(icd_phrases_list, ensure_ascii=False)
        )
        logging.info(_input.to_messages())
        output = self.chat(_input.to_messages())
        logging.info(output.content)
        print(output.content)
        try:
            json_substrings = json.loads(output.content)
            print(json_substrings)
        except Exception as e: 
            logging.error(e)            
        return json_substrings
