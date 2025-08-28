from dotenv import load_dotenv
from openai import OpenAI
import json
load_dotenv()

openai_client = OpenAI()
# goal - test a prompt by giving it 10 examples and giving a score on 10 for how many it gets right 
# 


TEST_PROMPT = """
You are a geography expert. You must return valid JSON. 

You will be asked about the capitals of countries. 

You must provide a one word response with the capital of the country. 

You must return the response in the following JSON format:

{"answer": "Paris"}
"""

TEST_EXAMPLES = [
    {
        "input": "What is the capital of France?",
        "output": "Paris"
    },
    {
        "input": "What is the capital of Germany?",
        "output": "Berlin"
    },
]

def call_llm(prompt, input):

    response = openai_client.responses.create(
        model="gpt-4.1",
        instructions=prompt,
        input=input
    )
    return response.output_text

# create a function that will test the prompt by giving it the test examples and giving a score on 10 for how many it gets right 

def convert_response_to_json(response):
    return json.loads(response)

def test_prompt(prompt, examples):
    score = 0
    for example in examples:
        llm_response = call_llm(prompt, example["input"])
        json_response = convert_response_to_json(llm_response)
        answer = json_response.get("answer", "")
        breakpoint()
        if answer == example["output"]:
            score += 1
    return score



print(test_prompt(TEST_PROMPT, TEST_EXAMPLES))




