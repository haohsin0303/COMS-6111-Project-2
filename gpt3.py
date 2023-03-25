import openai
import spacy

class Gpt3:
    def __init__(self, api_key, temperature):
        print("API KEY:", api_key)
        openai.api_key = api_key
        self.temperature = temperature

    def extract_relation(self, input_text, entity_1, entity_2):

        # Construct a prompt with the extracted entities
        # input_text = "Bill Gates founded Microsoft in 1975. He is currently a philanthropist and supports various causes."
        prompt = f"{input_text}\nWhat is the relationship between {entity_1} and {entity_2}?"

        # Call the GPT-3 API to generate a response based on the prompt
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=self.temperature,
        )

        # Extract the generated text from the API response
        response_text = response.choices[0].text

        # Print the extracted relation
        print(f"The relation between {entity_1} and {entity_2} is: {response_text}")
