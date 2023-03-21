import sys
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from textwrap import dedent
from bs4 import BeautifulSoup
import requests
import spacy
from spanbert import SpanBERT
from spacy_help_functions import get_entities, create_entity_pairs

spanbert = SpanBERT("./pretrained_spanbert")  

# Global Variables
API_KEY = None
ENGINE_KEY = None
# sk-t14ovJEbrUjc4eimt0BvT3BlbkFJZS1hOo97fZYj317oO2Z
OPENAI_KEY = None
OPEN_API_MODEL = 'text-davinci-003'
TEMPERATURE = 0.2
ITERATION_COUNT = 0
EXTRACTION_METHOD = None
CALCULATED_PRECISION = -1
USER_QUERY = ""
NEW_QUERY_TERMS = []
X = set() # set of extracted tuples
R, T, K = 0,0,0
Q = []
accepted_extraction_methods = {"-spanbert", "-gpt3"}
nlp = spacy.load("en_core_web_lg")
entities_of_interest = []

possible_relations = ["Schools_Attended", "Work_For", "Live_In", "Top_Member_Employees"]


# Define the required named entity types for each relation
required_entity_types = {
    "per:employee_of": {"subject": "PERSON", "object": "ORGANIZATION"},
    "org:top_members/employees": {"subject": "ORGANIZATION", "object": "PERSON"},
    "per:schools_attended": {"subject": "PERSON", "object": "ORGANIZATION"},
    "per:live_in": {"subject": "PERSON", "object": ["LOCATION", "CITY", "STATE_OR_PROVINCE", "COUNTRY"]}
}


def write_parameters():
    """
    Prints to the console the valid parameters provided by the 
    user when the program is launched
    """

    print(dedent("""
    Parameters:
    Client key      = {api_key}
    Engine key      = {engine_key}
    OpenAI key      = {openai_key}
    Method          = {extraction_method}
    Relation        = {target_relation}
    Threshold       = {target_threshold}
    Query           = {user_query}
    # of Tuples     = {num_of_tuples}
    Google Search Results:
    ======================""".format(api_key=API_KEY, engine_key=ENGINE_KEY, openai_key=OPENAI_KEY,
                                     extraction_method = EXTRACTION_METHOD, target_relation=possible_relations[int(R)-1],
                                     target_threshold=T, user_query = Q, num_of_tuples=K)
    ))


def get_google_search_results():
    """
    Initiates google querying process by taking the initial query
    and optionally appending the initial query to the new augmented query terms
    Returns if number of search results is less than 10 and throws exception if API or Engine key are invalid.
    """

    global ITERATION_COUNT

    service = build("customsearch", "v1", developerKey=API_KEY)
    querying = True
    while querying:
        write_parameters()

        try:
            res = (
            service.cse()
            .list(
                q=Q,
                cx=ENGINE_KEY,
            )
            .execute()
            )

            if (len(res["items"]) < 10):
                print("Your query produced less than 10 results")
                querying = False
                return
            else:
                print("=========== Iteration: {count} - Query: {query} ===========\n".format(count=ITERATION_COUNT, query=Q))
                querying = parse_search_results(res)
                ITERATION_COUNT += 1
        except HttpError:
            print("API key or Engine key not valid. Please pass a valid API and Engine key.")
            querying = False

def parse_search_results(res):

    visited_urls = set()
    for count, item in enumerate(res['items']):
        # Extracts url
        result_url = item.get('link', 'none')
        visited_urls.add(result_url)
    
    for url in visited_urls:
        # try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # raises an exception for 4xx or 5xx status codes
        page_content = response.content

        # Extract the actual plain text from the webpage using Beautiful Soup.
        soup = BeautifulSoup(page_content, "html.parser")
        resulting_plain_text = " ".join(soup.get_text())
        print("Fetching text from url ...")

        # Truncate the text to its first 10,000 characters (for efficiency) and discard the rest.
        if len(resulting_plain_text) > 10000:
            print("Trimming webpage content from {resulting_text_length} to 10000 characters".format(resulting_text_length=len(resulting_plain_text)))
            resulting_plain_text = resulting_plain_text[:10000]
        print("Webpage length (num characters): {text_length}".format(text_length=len(resulting_plain_text)))
        doc = nlp(resulting_plain_text)
        print("Annotating the webpage using spacy...")

        # Split the text into sentences
        # sentences = [s.text.strip() for s in doc.sents]

        # # Extract named entities from each sentence
        # entities = []
        # for s in doc.sents:
        #     entities.extend([(e.text, e.label_) for e in s.ents])
            
        print("Extracted {num_of_sentences} sentences. Processing each sentence one by one to check for presence of right pair of named entity types; if so, will run the second pipeline ...".format(num_of_sentences=len(doc.sents)))
        if EXTRACTION_METHOD == "-spanbert":
            spanbertExtraction(doc)
        else:
            gpt3Extraction(doc)


        # except (requests.exceptions.RequestException, ValueError):
        #     #Skip url if we cannot retrieve the webpage
        #     print("{URL} cannot be retrieved successfully ".format(URL=url))
        #     continue

        # except requests.exceptions.Timeout:
        #     # Skip url if timed out
        #     continue
    
def filter_entities_of_interest():
    global entities_of_interest

    # Filter entities of interest based on target relation
    if R == 1:
        entities_of_interest = ["PERSON", "ORGANIZATION"]
    elif R == 2:
        entities_of_interest = ["PERSON", "ORGANIZATION"]
    elif R == 3:
        entities_of_interest = ["PERSON", "LOCATION", "CITY", "STATE_OR_PROVINCE", "COUNTRY"]
    elif R == 4:
        entities_of_interest = ["ORGANIZATION", "PERSON"]

def spanbertExtraction(doc):

    filter_entities_of_interest()
    
    for sentence in doc.sents:
        print("\n\nProcessing sentence: {}".format(sentence))
        print("Tokenized sentence: {}".format([token.text for token in sentence]))
        ents = get_entities(sentence, entities_of_interest)
        print("spaCy extracted entities: {}".format(ents))

        # create entity pairs
        candidate_pairs = []
        sentence_entity_pairs = create_entity_pairs(sentence, entities_of_interest)
        for ep in sentence_entity_pairs:
            if R == 1 and ep[1][1] == "ORGANIZATION" and ep[2][1] == "PERSON": # Schools_Attended: Subject=PERSON, Object=ORGANIZATION
                candidate_pairs.append({"tokens": ep[0], "subj": ep[1], "obj": ep[2]})
            elif R == 2 and ep[1][1] == "PERSON" and ep[2][1] == "ORGANIZATION": # Work_For: Subject=PERSON, Object=ORGANIZATION
                candidate_pairs.append({"tokens": ep[0], "subj": ep[1], "obj": ep[2]})
            elif R == 3 and ep[1][1] == "PERSON" and ep[2][1] in ["LOCATION", "CITY", "STATE_OR_PROVINCE", "COUNTRY"]: # Live_In: Subject=PERSON, Object=LOCATION/CITY/STATE_OR_PROVINCE/COUNTRY
                candidate_pairs.append({"tokens": ep[0], "subj": ep[1], "obj": ep[2]})
            elif R == 4 and ep[1][1] == "ORGANIZATION" and ep[2][1] == "PERSON": # Top_Member_Employees: Subject=ORGANIZATION, Object=PERSON
                candidate_pairs.append({"tokens": ep[0], "subj": ep[1], "obj": ep[2]})

        # Classify Relations for all Candidate Entity Pairs using SpanBERT
        candidate_pairs = [p for p in candidate_pairs if not p["subj"][1] in ["DATE", "LOCATION"]]  # ignore subject entities with date/location type
        print("Candidate entity pairs:")
        for p in candidate_pairs:
            print("Subject: {}\tObject: {}".format(p["subj"][0:2], p["obj"][0:2]))
        print("Applying SpanBERT for each of the {} candidate pairs. This should take some time...".format(len(candidate_pairs)))

        if len(candidate_pairs) == 0:
            continue

        print("not skipped", len(candidate_pairs))
        
        relation_preds = spanbert.predict(candidate_pairs)  # get predictions: list of (relation, confidence) pairs

        # Print Extracted Relations
        print("\nExtracted relations:")
        for ex, pred in list(zip(candidate_pairs, relation_preds)):
            print("ex", ex)
            print("pred", pred)
            relation = pred[0]
            if relation in ['per:schools_attended', 'per:employee_of', 'per:cities_of_residence', 'org:top_members/employees']:
                print("\tSubject: {}\tObject: {}\tRelation: {}\tConfidence: {:.2f}".format(ex["subj"][0], ex["obj"][0], relation, pred[1]))



    


def gpt3Extraction():
    pass

def main():
    """
    Starting point of program that parses the terminal arguments
    and verifies that arguments are valid. 
    Once arguments are deemed valid, the querying function will be called
    """

    # Format Required: [-spanbert|-gpt3] <google api key> <google engine id> <openai secret key> <r> <t> <q> <k>
    global API_KEY, ENGINE_KEY, OPENAI_KEY, EXTRACTION_METHOD, R, T, Q, K

    terminal_arguments = sys.argv[1:]
    # Return if the number of arguments provided is incorrect
    if (len(terminal_arguments) != 8):
        # print(terminal_arguments)
        print("Format must be <-spanbert|-gpt3> <API Key> <Engine Key> <openai secret key> <r=[1,4]>, <t=[0,1]>, <Seed Query> <k > 0>")
        return
    
    EXTRACTION_METHOD = terminal_arguments[0]
    API_KEY = terminal_arguments[1]
    ENGINE_KEY = terminal_arguments[2]
    OPENAI_KEY = terminal_arguments[3]

    if (not (EXTRACTION_METHOD in accepted_extraction_methods )):
        print("Extraction method must be either '-spanbert' or '-gpt3'")
        return
    
    R = terminal_arguments[4]
    if (not (isinstance(eval(R), int) and 1 <= eval(R) <= 4)):
        print("R must be an integer between 1 and 4")
        return
    R = int(R)
    
    T = terminal_arguments[5]
    if (not ((T.isdigit() or isinstance(eval(T), float)) and 0 <= eval(T) <= 1)):
        print("T (Extraction confidence threshold) must be a real number between 0 and 1")
        return
    
    Q = terminal_arguments[6]
    if (not(len(Q) > 0)):
        print("Seed Query must be filled with a list of words")
        return
    
    K = terminal_arguments[7]
    if (not (eval(K) > 0)):
        print("K must be greater than 0")
        return

    get_google_search_results()


if __name__ == "__main__":
    main()