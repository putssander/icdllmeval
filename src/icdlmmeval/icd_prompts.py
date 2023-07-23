from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate, AIMessagePromptTemplate
from langchain.chat_models import ChatOpenAI

import logging
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator
from typing import List
import json


    # main_term: str = Field(description="translate the 'original phrase' to the standardized 'main term' to locate the medical code in the alphabetic icd-index")


class IcdItem(BaseModel):
    original_phrase: str = Field(description="original phrase")
    text_snippets: str = Field(description="extract (multiple) text snippets from source text containing all information and context for the ICD coding of 'original phrase' (mimLength=64; maxLength=128)")
    icd_code: str = Field(description="ICD code")
    icd_code_description: str = Field(description="ICD code description")
    icd_code_description_es: str = Field(description="Translated ICD code description in Spanish")


class IcdItemNer(BaseModel):
    id: str = Field(description="id of main tag")
    main_term: str = Field(description="main term")
    offsets: str = Field(description="main term offsets in src txt")
    context: str = Field(description="context for main term")
    icd_phrase: str = Field(description="substring for ICD coding containing the information to choose the specific ICD-10 code")
    icd_code_lookup_terms_en: str = Field(description="The icd_phrase into standardized key terms used for an ICD dictionary lookup")
    icd_code_lookup_terms_es: str = Field(description="Translation of icd_code_lookup_terms_en in Spanish")

# Define your desired data structure.
class IcdList(BaseModel):
    procedures: List[IcdItem] = Field(description="list of icd diagnose items")
    diagnoses: List[IcdItem] = Field(description="list of icd procedure items")

# Define your desired data structure.
class IcdListNer(BaseModel):
    diagnoses: List[IcdItemNer] = Field(description="list of icd procedure items (main type=\"D\")")
    procedures: List[IcdItemNer] = Field(description="list of icd diagnose items (main type=\"P\")")


class TermItem(BaseModel):
    original_phrase: str = Field(description="original phrase'")
    main_term: List[str] = Field(description="preferably a single term for the main condition, no anatomical sites, laterality, aetiology, severity, type or other details ")

# Define your desired data structure.
class TermList(BaseModel):
    main_terms: List[TermItem] = Field(description="list extracted main items")




examples_context = [
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

class IcdPrompts():

    def __init__(self, model_name = "gpt-4"):
        self.output_parser_substrings = self.get_output_parser_substrings()
        self.format_instructions_substrings = self.get_format_instructions_substrings(self.output_parser_substrings )
        self.output_parser_select = self.get_output_parser_select()
        # model_name = "gpt-3.5-turbo"
        self.temperature = 0.0
        self.chat = ChatOpenAI(model_name=model_name, temperature=self.temperature)

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
            ResponseSchema(name="code", description="correct ICD-10 code"),
            ResponseSchema(name="listed", description="was the correct code listed (boolean)"),
            ResponseSchema(name="reasoning", description="explain why the your code is the correct one")
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


    def select_code(self, item):

        format_instructions = self.output_parser_select.get_format_instructions()
        coding_instructions = "What is the correct ICD-10 code for the icd_phrase. If you think the correct code is not listed, provide your best code suggestion in the json field 'code', always follow the format instructions."
        behaviour_instructions = "You are an expert in medical ICD coding, teaching peers the highest level of coding possible."
        system_message_prompt = SystemMessagePromptTemplate.from_template("{behaviour_instructions} {coding_instructions} {format_instructions}\n")
        # example_human = HumanMessagePromptTemplate.from_template("{question}", additional_kwargs={"name": "example_user"})
        # example_ai = AIMessagePromptTemplate.from_template("{answer}", additional_kwargs={"name": "example_assistant"})
        human_message_prompt = HumanMessagePromptTemplate.from_template("item: {item}. If you think the correct code is not in the list, provide your best code suggestion, always follow the format instructions.")

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
        try:
            json_substrings = self.output_parser_select.parse(output.content)
            code = json_substrings["code"]
        except Exception as e: 
            logging.error(e)            
            code = "-NA-"
        return code
    

    def prompt_icd_item_info(self, txt, substrings):

        # Set up a parser + inject instructions into the prompt template.
        parser = PydanticOutputParser(pydantic_object=IcdList)

        format_instructions = parser.get_format_instructions()
        # coding_instructions = "What is the correct ICD-10 code for the substring. If you think the correct code is not listed, provide your best code suggestion in the json field 'code', always follow the format instructions."
        behaviour_instructions = "You are an expert in medical ICD coding, teaching peers the highest level of coding possible."
        system_message_prompt = SystemMessagePromptTemplate.from_template("{behaviour_instructions} {format_instructions}\n")
        human_message_prompt = HumanMessagePromptTemplate.from_template("For each of items in the lists, add additional information in Spanish to preform an medical code lookup for '{substring}', use {format_instructions} in markdown! Extract from the following text {txt}. The number of list items of the output should match the input, ")

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
    
    def prompt_icd_description_from_main_terms(self, txt, main_terms):

        # Set up a parser + inject instructions into the prompt template.
        parser = PydanticOutputParser(pydantic_object=IcdList)

        format_instructions = parser.get_format_instructions()
        # coding_instructions = "What is the correct ICD-10 code for the substring. If you think the correct code is not listed, provide your best code suggestion in the json field 'code', always follow the format instructions."
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
    
    def prompt_icd_code_description_from_main_terms(self, example, main_terms):

        # Set up a parser + inject instructions into the prompt template.
        parser = PydanticOutputParser(pydantic_object=IcdListNer)

        format_instructions = parser.get_format_instructions()
        # coding_instructions = "What is the correct ICD-10 code for the substring. If you think the correct code is not listed, provide your best code suggestion in the json field 'code', always follow the format instructions."
        behaviour_instructions = "You are an expert in medical ICD coding, teaching peers the highest level of coding possible."
        system_message_prompt = SystemMessagePromptTemplate.from_template("{behaviour_instructions} {format_instructions}\n")
        example_human = HumanMessagePromptTemplate.from_template("{question}", additional_kwargs={"name": "example_user"})
        example_ai = AIMessagePromptTemplate.from_template("{answer}", additional_kwargs={"name": "example_assistant"})

        human_message_prompt = HumanMessagePromptTemplate.from_template("For each item in the lists, provide the additional information as shown in the example above. The list '{main_terms}', use {format_instructions} in markdown!")

        chat_prompt = ChatPromptTemplate(
            messages=[system_message_prompt,example_human,example_ai, human_message_prompt], 
            input_variables=["main_terms", "question", "answer"],
            partial_variables={"behaviour_instructions": behaviour_instructions, "format_instructions": format_instructions,}
        )
        _input = chat_prompt.format_prompt(
            main_terms=json.dumps(main_terms,  ensure_ascii=False), 
            question=json.dumps(example["prompt"], ensure_ascii=False), 
            answer=json.dumps(example["output"],  ensure_ascii=False)
            )
        logging.info(_input.to_messages())
        output = self.chat(_input.to_messages())
        logging.info(output.content)
        print(output.content)
        try:
            json_substrings = parser.parse(output.content)
            print(json_substrings)
        except Exception as e: 
            logging.error(e)            
        return output.content
    
    def set_model(self, model_name):
        self.chat = ChatOpenAI(model_name=model_name, temperature=self.temperature)
