from dotenv import load_dotenv
from openai import OpenAI
import json
load_dotenv()

openai_client = OpenAI()


# PROMPT THAT WE ARE TESTING
TEST_PROMPT = """
You are a geography expert. You must return valid JSON. 

You will be asked about the capitals of countries. 

You must provide a one word response with the capital of the country. 

You must return the response in the following JSON format:

{"answer": "Paris"}
"""

with open('test_examples.json', 'r') as f:
    TEST_EXAMPLES = json.load(f)

def call_llm(prompt, input):

    response = openai_client.responses.create(
        model="gpt-4.1",
        instructions=prompt,
        input=input
    )
    return response.output_text

def convert_response_to_json(response):
    return json.loads(response)

def test_prompt(prompt, examples):
    score = 0
    for example in examples:
        llm_response = call_llm(prompt, example["input"])
        json_response = convert_response_to_json(llm_response)
        answer = json_response.get("answer", "")
        if answer == example.get("output", ""):
            score += 1
    return score


print(test_prompt(TEST_PROMPT, TEST_EXAMPLES))




