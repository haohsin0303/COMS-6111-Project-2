import openai
import time
from spacy_help_functions import create_entity_pairs


class Gpt3:

    def __init__(self, api_key, open_ai_model, temperature, target_relation, query):
        openai.api_key = api_key
        self.open_ai_model = open_ai_model
        self.temperature = temperature
        self.target_relation = target_relation
        self.query = query
    
    def extract_relations(self, doc, X, R, entities_of_interest=None):

        # Declare annotation counts
        sentences_with_annotations = 0
        overall_relations = 0
        non_duplicated_relations = 0

        for index, sentence in enumerate(doc.sents):

            # Create and iterate through the entity pairs
            candidate_pairs = []
            entity_pairs = create_entity_pairs(sentence, entities_of_interest)
            for ep in entity_pairs:
                # Append to the list the relations that satisfy the criteria corresponding to the target relation's entity types
                # Add the swapped version as well
                if R == 1 and ((ep[1][1] == "PERSON" and ep[2][1] == "ORGANIZATION") or (ep[2][1] == "PERSON" and ep[1][1] == "ORGANIZATION")):
                    candidate_pairs.append({"tokens": ep[0], "subj": ep[1], "obj": ep[2]})
                    candidate_pairs.append({"tokens": ep[0], "subj": ep[2], "obj": ep[1]})
                elif R == 2 and ((ep[1][1] == "PERSON" and ep[2][1] == "ORGANIZATION") or (ep[2][1] == "PERSON" and ep[1][1] == "ORGANIZATION")):
                    candidate_pairs.append({"tokens": ep[0], "subj": ep[1], "obj": ep[2]})
                    candidate_pairs.append({"tokens": ep[0], "subj": ep[2], "obj": ep[1]})
                elif R == 3 and ((ep[1][1] == "PERSON" and ep[2][1] in ["LOCATION", "CITY", "STATE_OR_PROVINCE", "COUNTRY"]) or (ep[2][1] == "PERSON" and ep[1][1] in ["LOCATION", "CITY", "STATE_OR_PROVINCE", "COUNTRY"])):
                    candidate_pairs.append({"tokens": ep[0], "subj": ep[1], "obj": ep[2]})
                    candidate_pairs.append({"tokens": ep[0], "subj": ep[2], "obj": ep[1]})
                elif R == 4 and ((ep[1][1] == "ORGANIZATION" and ep[2][1] == "PERSON") or (ep[2][1] == "ORGANIZATION" and ep[1][1] == "PERSON")):
                    candidate_pairs.append({"tokens": ep[0], "subj": ep[1], "obj": ep[2]})
                    candidate_pairs.append({"tokens": ep[0], "subj": ep[2], "obj": ep[1]})

            if (index + 1) % 5 == 0:
                print("\tProcessed {count} / {total} sentences".format(count=index+1, total=len(list(doc.sents))))

            # Skip if nothing was appended
            if len(candidate_pairs) == 0:
                continue

            # Pass into the GPT-3 API the sentence that passed the validation checks
            relation_preds = self.predict(str(sentence).strip())

            # If we get predictions pack, add count
            if len(relation_preds) > 0:
                sentences_with_annotations += 1

            for prediction in eval(relation_preds):
                print("\n\t\t=== Extracted Relation ===")
                print("\t\tSentence:", sentence)
                print("\t\tSubject: {subject} ; Object: {object} ;".format(subject=prediction[0], object=prediction[2]))

                overall_relations += 1

                # Check if duplicates exist in set of extracted tuples
                if (1.0, prediction[0], prediction[2]) in X:
                    print("\t\tDuplicate. Ignoring this.")
                else:
                    X.add((1.0, prediction[0], prediction[2]))
                    non_duplicated_relations += 1
                    print("\t\tAdding to set of extracted relations")
                
                print("\t\t==========")

                overall_relations += 1
        
        print("\tExtracted annotations for  {sentences_with_annotations}  out of total  {total_sentences}  sentences".format(sentences_with_annotations=sentences_with_annotations, total_sentences=len(list(doc.sents))))
        print("\tRelations extracted from this website: {non_duplicated_relations} (Overall: {overall_relations})".format(non_duplicated_relations=non_duplicated_relations, overall_relations=overall_relations))

        return X

    def predict(self, input_text):

        # Used to prevent Rate limit exceeding
        time.sleep(1.5)

        # Construct a prompt with the extracted entities
        prompt = ""

        if self.target_relation == 1:
             prompt = """
            Given a sentence, return a list of tuples representing people and the school/university/college/education institution that they attended.  Use the Schools_Attended relation to extract these tuples, with the person as the subject and the school's name as the object. If no tuple exists, return an empty list. If a solution exists, only output a tuple(s). Ensure that the subject and the object are listed together in the sentence that the subject is an actual person, and the object is an actual school or educational institution.
            The format of the result should follow this format: [(<PERSON'S FULL NAME>, "Schools_Attended", <ORGANIZATION NAME>)]
            Example Sentence: 'Mark Zuckerberg, Doctor of Laws Mark Zuckerberg, who founded Facebook in 2004, is the featured speaker at the Afternoon Program of Harvardâ€™s 366th Commencement'\
            Example Result: [("Mark Zuckerberg", "Schools_Attended", "Harvard")]
            Sentence: {input_text}
            """.format(input_text=input_text)

        elif self.target_relation == 2:
            prompt = """
            Given a sentence, return a list of tuples representing people and the organizations that they work for.  Use the Work_For relation to extract these tuples, with the person as the subject and the organization's name as the object. If no tuple exists, return an empty list. If a solution exists, only output a tuple(s). Ensure that the subject and the object are listed together in the sentence that the subject is an actual person, and the object is an actual organization.
            The format of the result should follow this format: [(<PERSON'S FULL NAME>, "Work_For", <ORGANIZATION NAME>)]
            Example Sentence: 'In 2013, Pichai added Android to the list of Google products that he oversaw.'
            Example Result: [("Sundar Pichai", "Work_For", "Google")]
            Sentence: {input_text}
            """.format(input_text=input_text)

        elif self.target_relation == 3:
            prompt = """
            Given a sentence, return a list of tuples representing people and the location, city, state, province, or country they live or have lived in. Use the Lived_In relation to extract these tuples, with the person as the subject and the location, city, state, province, or country name as the object. If no tuple exists, return an empty list. If a solution exists, only output a tuple(s). Ensure that the subject and the object are listed together in the sentence that the subject is an actual person, and the object is an actual location.
            The format of the result should follow this format: [(<PERSON'S FULL NAME>, "Live_In", <LOCATION NAME>)]
            Example Sentence: 'U.S. women's national team winger Megan Rapinoe grew up in Redding'
            Example Result: [("Megan Rapinoe", "Live_In", "Redding")]
            Sentence: {input_text}
            """.format(input_text=input_text)

        elif self.target_relation == 4:
            prompt = """
            Given a sentence, return a list of tuples representing an organization and the person who is a top employee of that organization. Use the Top_Member_Employees relation to extract these tuples, with the organization as the subject and the person as the object. If no tuple exists, return an empty list. If a solution exists, only output a tuple(s). Ensure that the subject and the object are listed together in the sentence that the subject is an actual organization, and the object is an actual person.
            The format of the result should follow this format: [(<ORGANIZATION NAME>, "Top_Member_Employees", <PERSON'S FULL NAME>)]
            Example Sentence: 'However, he dropped out in his junior year to found Microsoft with his friend Paul Allen.'
            Example Result: [("Microsoft", "Top_Member_Employees", "Bill Gates")]
            Sentence: {input_text}
            """.format(input_text=input_text)

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

        # Remove redundant text before "["
        if "[" in response_text:
            idx = response_text.index("[")
            response_text = response_text[idx:]
    
        return response_text
