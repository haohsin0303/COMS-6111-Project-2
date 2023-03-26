import openai
import spacy

class Gpt3:
    def __init__(self, api_key, temperature, target_relation):
        print("API KEY:", api_key)
        openai.api_key = api_key
        self.temperature = temperature
        self.target_relation = target_relation
    
    def predict(self, sentence, candidate_pairs):
        predictions = set()
        for candidate_pair in candidate_pairs:
            entity_1 = candidate_pair["subj"][0]
            entity_2 = candidate_pair["obj"][0]
            relation = self.extract_relation(sentence, entity_1, entity_2)
            if relation == "True":
                predictions.add((entity_1, entity_2))

        return predictions

    def extract_relation(self, input_text, entity_1, entity_2):

        # Construct a prompt with the extracted entities
        prompt = ""
        if self.target_relation == 1:
            prompt = "{input_text}\n\
                According to this sentence, is {entity_2} the school {entity_1} attended? Answer only True or False.".format(
                    input_text=input_text,
                    entity_1=entity_1,
                    entity_2=entity_2,
                )
        elif self.target_relation == 2:
            prompt = "{input_text}\n\
                According to this sentence, does {entity_1} work for {entity_2}? Answer only True or False.".format(
                    input_text=input_text,
                    entity_1=entity_1,
                    entity_2=entity_2,
                )
        elif self.target_relation == 3:
            prompt = "{input_text}\n\
                According to this sentence, does {entity_1} live in {entity_2}? Answer only True or False.".format(
                    input_text=input_text,
                    entity_1=entity_1,
                    entity_2=entity_2,
                )
        elif self.target_relation == 4:
            prompt = "{input_text}\n\
                According to this sentence, is {entity_1} the top member employee of {entity_2}? Answer only True or False.".format(
                    input_text=input_text,
                    entity_1=entity_1,
                    entity_2=entity_2,
                )

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
        print("extracted:", entity_1, entity_2, response_text)
        return response_text.strip()
