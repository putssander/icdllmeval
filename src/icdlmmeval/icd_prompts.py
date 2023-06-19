from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate, AIMessagePromptTemplate
from langchain.chat_models import ChatOpenAI

import logging
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator
from typing import List




class IcdItem(BaseModel):
    original_phrase: str = Field(description="original phrase")
    context_snippets: str = Field(description="extract (multiple) text snippets from source text containing all information and context for the ICD coding of 'original phrase' (maxLength=64; maxLength=128)")
    # main_term: str = Field(description="translate the 'original phrase' to the standardized 'main term' to locate the medical code in the alphabetic icd-index")
    code_description: str = Field(description="translate the 'original phrase' and 'context_snippets' to a standarized formal ICD like code description, clearly indicating the nature, location, and other relevant details")

# Define your desired data structure.
class IcdList(BaseModel):
    procedures: List[IcdItem] = Field(description="list of icd diagonse items")
    diagnoses: List[IcdItem] = Field(description="list of icd procedure items")


class IcdPrompts():

    def __init__(self, model_name = "gpt-4"):
        self.output_parser_substrings = self.get_output_parser_substrings()
        self.format_instructions_substrings = self.get_format_instructions_substrings(self.output_parser_substrings )
        self.output_parser_select = self.get_output_parser_select()
        # model_name = "gpt-3.5-turbo"
        temperature = 0.0
        self.chat = ChatOpenAI(model_name=model_name, temperature=temperature)

    def get_output_parser_substrings(self):
        response_schemas = [
            ResponseSchema(name="procedures", description="list of substrings for coding ICD procedures"),
            ResponseSchema(name="diagnoses", description="list of substrings for coding ICD diagnoses")
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

        behaviour_instructions = "You are a senior professional medical ICD coder, teaching peers the highest level of coding possible."
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


    def select_code(self, substring, docs, examples):

        format_instructions = self.output_parser_select.get_format_instructions()
        coding_instructions = "What is the correct ICD-10 code for the substring. If you think the correct code is not listed, provide your best code suggestion in the json field 'code', always follow the format instructions."
        behaviour_instructions = "You are a senior professional medical ICD coder, teaching peers the highest level of coding possible."
        system_message_prompt = SystemMessagePromptTemplate.from_template("{behaviour_instructions} {coding_instructions} {format_instructions}\n")
        example_human = HumanMessagePromptTemplate.from_template("{question}", additional_kwargs={"name": "example_user"})
        example_ai = AIMessagePromptTemplate.from_template("{answer}", additional_kwargs={"name": "example_assistant"})
        human_message_prompt = HumanMessagePromptTemplate.from_template("substring: '{substring}', suggestions: {docs}. If you think the correct code is not in the list, provide your best code suggestion, always follow the format instructions.")

        chat_prompt = ChatPromptTemplate(
            messages=[system_message_prompt, example_human, example_ai, human_message_prompt], 
            input_variables=["question", "answer","substring", "docs"],
            partial_variables={"behaviour_instructions": behaviour_instructions, "coding_instructions": coding_instructions, "format_instructions": format_instructions,}
        )
        _input = chat_prompt.format_prompt(question=examples[0]['question'], answer=examples[0]['answer'], substring=substring, docs=docs)
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
        behaviour_instructions = "You are a senior professional medical ICD coder, teaching peers the highest level of coding possible."
        system_message_prompt = SystemMessagePromptTemplate.from_template("{behaviour_instructions} {format_instructions}\n")
        human_message_prompt = HumanMessagePromptTemplate.from_template("For each of the extracted substrings add additional information in Spanish to preform an medical code lookup for '{substring}' use {format_instructions} in markdown! Extract from the following text {txt}")

        chat_prompt = ChatPromptTemplate(
            messages=[system_message_prompt,human_message_prompt], 
            input_variables=["substring"],
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