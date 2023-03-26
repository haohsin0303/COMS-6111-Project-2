import openai
import time
import json

class Gpt3:
    def __init__(self, api_key, open_ai_model, temperature, target_relation):
        openai.api_key = api_key
        self.open_ai_model = open_ai_model
        self.temperature = temperature
        self.target_relation = target_relation

    def predict(self, input_text):

        time.sleep(1.5)

        # Construct a prompt with the extracted entities
        prompt = ""

        if self.target_relation == 1:
            # prompt = 'Extract the Schools_Attended relationships between people and schools attended from the given sentence. Have the subject be the person, and the object be the official name of the school, university, college, or education institution they were the student there. Output a list of tuples. If there is no tuple, return an empty list.\n\n\
            #     Output: [("SUBJECT FULL NAME", "Schools_Attended", "SCHOOL NAME")]\n\
            #     Example: [("Jeff Bezos", "Schools_Attended", "Princeton University")]\n\
            #     Example: [("Mark Zuckerberg", "Schools_Attended", "Harvard University")]\n\n\
            #     Sentence: ' + input_text + '\n'

            file1 = open("query.txt","r")
            prompt = "".join(file1.readlines())
            print("prompt", prompt)
            
            # print("prompt:", prompt)
        # elif self.target_relation == 2:
        #     prompt = 'Given a sentence, extract all subjects that work for an organization. Return each instance as a JSON. \n\

        #             input_text=input_text,
        #             entity_1=entity_1,
        #             entity_2=entity_2,
        #         )
        # elif self.target_relation == 3:
        #     prompt = ""
        # elif self.target_relation == 4:
        #     prompt = ""

        # Call the GPT-3 API to generate a response based on the prompt
        response = openai.Completion.create(
            model=self.open_ai_model,
            prompt=prompt,
            max_tokens=1024,
            temperature=self.temperature,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        # Extract the generated text from the API response
        response_text = response['choices'][0]['text'].replace("\n", "").replace("\t", "").strip()

        # print("response_text:", response_text)

        # # if len(response_text) > 0:
        # json_response = json.loads(response_text)
        # return json_response["response"]
        response_text = response_text.replace('Answer: ', '')

        # Remove redundant text before "["
        if "[" in response_text:
            idx = response_text.index("[")
            response_text = response_text[idx:]
    
        return response_text
        # "[subject: Zuckerberg, object: Harvard University], [subject: Eduardo Saverin, object: Harvard University], [subject: Andrew McCollum, object: Harvard University], [subject: Dustin Moskovitz, object: Harvard University], [subject: Chris Hughes, object: Harvard University]"


