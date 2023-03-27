import re
import requests
import spacy
import sys
from bs4 import BeautifulSoup
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from gpt3 import Gpt3
from spacy_help_functions import extract_relations
from textwrap import dedent

sys.path.append('.../SpanBERT')
import spanbert.SpanBERT


# Global Variables
API_KEY = None
ENGINE_KEY = None
OPENAI_KEY = None
OPEN_API_MODEL = 'text-davinci-003'
TEMPERATURE = 0.1
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
tuples_used_for_query = set()

possible_relations = ["Schools_Attended", "Work_For", "Live_In", "Top_Member_Employees"]

relation_names = {
    1: "per:schools_attended",
    2: "per:employee_of",
    3: ["per:countries_of_residence", "per:cities_of_residence", "per:stateorprovinces_of_residence"],
    4: "org:top_members/employees",
}

required_entity_types = {
    "per:employee_of": "Work_For",
    "org:top_members/employees": "Top_Member_Employees",
    "per:schools_attended": "Schools_Attended",
    "per:live_in": "Live_In"
}

live_in_tuples = {
    "per:countries_of_residence": set(),
    "per:cities_of_residence": set(),
    "per:stateorprovinces_of_residence":set(),
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
    
    if (ITERATION_COUNT >= 1):
        print("Total # of iterations = {count}".format(count=ITERATION_COUNT))


def parse_search_results(res):

    # initialize spanbert if the extraction method matches
    spanbert = SpanBERT("./SpanBERT/pretrained_spanbert") if EXTRACTION_METHOD == "-spanbert" else None

    visited_urls = set()
    for _, item in enumerate(res['items']):
        # Extracts url
        result_url = item.get('link', 'none')
        visited_urls.add(result_url)

    visited_urls = set()
    visited_urls.add("https://en.wikipedia.org/wiki/Bill_Gates")
    for url_count, url in enumerate(visited_urls):
        # try:
        print("URL ( {curr_url} / {total_num_of_urls}): {link}".format(curr_url=url_count+1, total_num_of_urls=len(visited_urls), link=url))
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # raises an exception for 4xx or 5xx status codes
        page_content = response.content

        # Extract the actual plain text from the webpage using Beautiful Soup.
        soup = BeautifulSoup(page_content, "html.parser")
        
        # Removing redundant newlines and some whitespace characters, according to https://edstem.org/us/courses/34785/discussion/2831362
        preprocessed_text = "".join(soup.text)
        resulting_plain_text = re.sub(u'\xa0', ' ', preprocessed_text) 
        resulting_plain_text = re.sub('\t+', ' ', resulting_plain_text) 
        resulting_plain_text = re.sub('\n+', ' ', resulting_plain_text) 
        resulting_plain_text = re.sub(' +', ' ', resulting_plain_text) 
        resulting_plain_text = resulting_plain_text.replace('\u200b', '')

        print("\tFetching text from url ...")

        # Truncate the text to its first 10,000 characters (for efficiency) and discard the rest.
        if len(resulting_plain_text) > 10000:
            print("\tTrimming webpage content from {resulting_text_length} to 10000 characters".format(resulting_text_length=len(resulting_plain_text)))
            resulting_plain_text = resulting_plain_text[:10000]
        print("\tWebpage length (num characters): {text_length}".format(text_length=len(resulting_plain_text)))
        print("\tAnnotating the webpage using spacy...")
        
        doc = nlp(resulting_plain_text)

        print("\tExtracted {num_of_sentences} sentences. Processing each sentence one by one to check for presence of right pair of named entity types; if so, will run the second pipeline ...".format(num_of_sentences=len(list(doc.sents))))
        if EXTRACTION_METHOD == "-spanbert":
            spanbertExtraction(doc, spanbert)
        else:
            gpt3Extraction(doc)


        # except (requests.exceptions.RequestException, ValueError):
        #     #Skip url if we cannot retrieve the webpage
        #     print("\t{URL} cannot be retrieved successfully. The URL will be skipped".format(URL=url))
        #     continue

        # except requests.exceptions.Timeout:
        #     # Skip url if timed out
        #     print("\t{URL} repsonse timed out. The URL will be skipped".format(URL=url))
        #     continue
        
        # except:
        #     continue
    
    if (EXTRACTION_METHOD == "-spanbert" and R == 3):
        for relation_key in live_in_tuples:
            size_of_live_in_relation_tuples = len(live_in_tuples[relation_key])
            if ( size_of_live_in_relation_tuples > 0 ):
                print("================== ALL RELATIONS for {relation_name} ( {relations_length} ) =================".format(relation_name=relation_key, relations_length=size_of_live_in_relation_tuples))
                print_results(live_in_tuples[relation_key])
    else:
        if (EXTRACTION_METHOD == "-spanbert"):
            print("================== ALL RELATIONS for {relation_name} ( {relations_length} ) =================".format(relation_name=relation_names[R], relations_length=len(X)))
        else:
            print("================== ALL RELATIONS for {relation_name} ( {relations_length} ) =================".format(relation_name=possible_relations[int(R)-1], relations_length=len(X)))
        print_results()


    # If X contains at least k tuples
    if len(X) >= int(K):
        return False

    # Otherwise,
    else:
        found_y_tuple = False
        # select from X a tuple y
        for y in X:
            # y has not been used for querying yet
            if y not in tuples_used_for_query:
                if EXTRACTION_METHOD == "-spanbert":
                    unused_tuples = X - (tuples_used_for_query)
                    # y has an extraction confidence that is highest among the tuples in X that
                    # have not been used for querying.
                    max_confidence = max(unused_tuples, key=lambda x: x[0])[0]
                    if y[0] == max_confidence:
                        found_y_tuple = True
                        # Create query Q from tuple y
                        createNewQuery(y)
                        tuples_used_for_query.add(y)
                        break
                else:
                    found_y_tuple = True
                    # Create new query q for -gpt3
                    createNewQuery(y)
                    tuples_used_for_query.add(y)

        # If no such y tuple exists, then stop
        if (not(found_y_tuple)):
            print("ISE has stalled before retrieving k high-confidence tuples")
            return False

        # Else, continue querying
        return True
    

def print_results(live_in_tuples=None):
    if EXTRACTION_METHOD == "-spanbert":
        if (R == 3):
            topKTuples = get_TopK_tuples(live_in_tuples)
        else:
            topKTuples = get_TopK_tuples()
        for t in topKTuples:
            print("Confidence: {confidence} \t| Subject: {subject} \t| Object: {object}".format(confidence=t[0], subject=t[1], object=t[2]))
    else:
        for t in X:
            print("Subject: {subject} \t| Object: {object}".format(subject=t[1], object=t[2]))
    


def createNewQuery(y):
    """
    Create a query q from tuple y by just concatenating the 
    attribute values together,
    """
    global Q
    Q = y[1] + " " + y[2]

    
def get_TopK_tuples(relation_tuples = X):
    """
    Returns tuples sorted in decreasing order by extraction codeince,
    together with the extraction confidence of each tuple.
    """

    sorted_result= sorted(relation_tuples, key=lambda x: x[0], reverse=True)
    return sorted_result
    # return sorted_X[:K]


def filter_entities_of_interest():
    """
    Based on the desired relation R, we extract the relation's entities
    """

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


def spanbertExtraction(doc, spanbert):

    global X
     
    filter_entities_of_interest()

    X = extract_relations(doc, spanbert, X, R, relation_names[R], live_in_tuples, entities_of_interest, float(T))


def gpt3Extraction(doc):

    global X

    gpt3 = Gpt3(OPENAI_KEY, OPEN_API_MODEL, TEMPERATURE, R, Q)

    filter_entities_of_interest()

    X = gpt3.extract_relations(doc, X, R, entities_of_interest)


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
    
    R = eval(terminal_arguments[4])
    if (not (isinstance(R, int) and 1 <= R <= 4)):
        print("R must be an integer between 1 and 4")
        return
    R = int(R)
    
    T = eval(terminal_arguments[5])
    if (not ((isinstance(T, int) or isinstance(T, float)) and 0 <= T <= 1)):
        print("T (Extraction confidence threshold) must be a real number between 0 and 1")
        return
    
    Q = terminal_arguments[6]
    if (not(len(Q) > 0)):
        print("Seed Query must be filled with a list of words")
        return
    
    K = eval(terminal_arguments[7])
    if (not (K > 0)):
        print("K must be greater than 0")
        return

    get_google_search_results()


if __name__ == "__main__":
    main()
