from abc import ABC, abstractmethod
from enum import Enum

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
   else:
       raise ValueError(f"Unknown provider: {model_provider}")

def test_prompt(prompt, examples, model_provider: ModelProvider):
   provider = get_provider(model_provider)
   score = 0
   for example in examples:
       llm_response = provider.call(prompt, example["input"])
       json_response = convert_response_to_json(llm_response)
       answer = json_response.get("answer", "")
       if answer == example.get("output", ""):
           score += 1
   return score

# Usage
print(test_prompt(TEST_PROMPT, TEST_EXAMPLES, ModelProvider.OPENAI))
print(test_prompt(TEST_PROMPT, TEST_EXAMPLES, ModelProvider.ANTHROPIC))