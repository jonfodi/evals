from abc import ABC, abstractmethod
from enum import Enum
import json
from openai import OpenAI
import anthropic
import dotenv

dotenv.load_dotenv()

class ModelProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"

class LLMProvider(ABC):
    @abstractmethod
    def call(self, prompt, input_text):
        pass

class OpenAIProvider(LLMProvider):
    def __init__(self):
        self.client = OpenAI()
    
    def call(self, prompt, input_text):
        response = self.client.responses.create(
            model="gpt-4.1",
            instructions=prompt,
            input=input_text
        )
        return response.output_text

class AnthropicProvider(LLMProvider):
    def __init__(self):
        self.client = anthropic.Anthropic()
    
    def call(self, prompt, input_text):
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": input_text}],
            system=[{"text": prompt, "type": "text"}]
        )
        return response.content[0].text

def get_provider(model_provider: ModelProvider) -> LLMProvider:
    if model_provider == ModelProvider.OPENAI:
        return OpenAIProvider()
    elif model_provider == ModelProvider.ANTHROPIC:
        return AnthropicProvider()

def convert_response_to_json(response):
    return json.loads(response)

def test_prompt(prompt, examples, model_provider: ModelProvider):
    """Test a single prompt against examples and return percentage score."""
    provider = get_provider(model_provider)
    score = 0
    for example in examples:
        llm_response = provider.call(prompt, example["input"])
        json_response = convert_response_to_json(llm_response)
        answer = json_response.get("answer", "")
        if answer == example.get("output", ""):
            score += 1
    percentage = (score / len(examples)) * 100
    return percentage

def test_prompts(prompts: list[str], examples, model_provider: ModelProvider):
    """
    Test multiple prompts and return percentage scores.
    
    Args:
        prompts: List of prompt strings
        examples: List of test examples
        model_provider: ModelProvider enum
    
    Returns:
        Dictionary mapping prompt index to percentage score
    """
    percentages = {}
    for i, prompt in enumerate(prompts):
        percentage = test_prompt(prompt, examples, model_provider)
        percentages[i] = percentage
        print(f"Prompt {i}: {percentage:.1f}%")
    
    return percentages

TEST_PROMPT = """
You are a geography expert. You must return valid JSON. 

You will be asked about the capitals of countries. 

You must provide a one word response with the capital of the country. 

You must return the response in the following JSON format:

{"answer": "Paris"}
"""

with open('test_examples.json', 'r') as f:
    TEST_EXAMPLES = json.load(f)

if __name__ == "__main__":
    # Test single prompt (now requires list)
    print("Testing single prompt:")
    result = test_prompts([TEST_PROMPT_1], TEST_EXAMPLES, ModelProvider.OPENAI)
    print(f"OpenAI result: {result}")
    
    result = test_prompts([TEST_PROMPT_1], TEST_EXAMPLES, ModelProvider.ANTHROPIC)
    print(f"Anthropic result: {result}")
    
    # Test multiple prompts
    print("Testing multiple prompts:")
    results = test_prompts([TEST_PROMPT_1, TEST_PROMPT_2], TEST_EXAMPLES, ModelProvider.OPENAI)
    print(f"Multiple prompts results: {results}")