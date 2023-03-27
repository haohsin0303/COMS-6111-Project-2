# COMS-6111-Project-2

This project is an implementation of an information retrieval system that aims to improve the search results returned by Google Search Engine API by exploiting user-provided relevance feedback. The goal of the project is to help users find more relevant search results and refine their queries.

## Team Members
- Christopher Asfour: cra2139
- Nina Hsu: hh2961

## Relevant Files
- gpt3.py
- main.py
- spacy_help_functions.py
- README.md

## How to run
1. Install the necessary Python packages using pip:
```
pip3 install beautifulsoup4
pip3 install -U pip setuptools wheel
pip3 install -U spacy
python3 -m spacy download en_core_web_lg
pip3 install openai
```

2. Install SpanBERT
```
git clone https://github.com/zackhuiiiii/SpanBERT
cd SpanBERT
pip3 install -r requirements.txt
bash download_finetuned.sh
```

3. Move the project into SpanBERT repository
```
cd ..
mv proj2 ./SpanBERT/
```

3. Run the program using the following command:
```
python3 main.py [-spanbert|-gpt3] <API Key> <Engine Key> <OPENAI Key> <r=[1,4]>, <t=[0,1]>, <Seed Query> <k > 0>
```
    - [-spanbert|-gpt3]: An argument that indicates whether we are using SpanBERT (-spanbert) or GPT-3 (-gpt3) for the extraction process.
    - API Key: The Google Custom Search Engine JSON API Key.
    - Engine Key: The Google Custom Search Engine ID.
    - OpenAI Key: The OpenAI key used for the GPT-3 API to extract relations from text documents.
    - r=[1,4] : An integer between 1 and 4, indicating the relation to extract - 1 is for Schools_Attended, 2 is for Work_For, 3 is for Live_In, and 4 is for Top_Member_Employees.
    - t=[0,1] : A real number between 0 and 1, indicating the "extraction confidence threshold," which is the minimum extraction confidence that we request for the tuples in the output; t is ignored if we are using -gpt3
    - Seed Query: A list of words in double quotes corresponding to a plausible tuple for the relation to extract (e.g., "bill gates microsoft" for relation Work_For)
    - k > 0: An integer greater than 0, indicating the number of tuples that we request in the output

## General Description

The project's internal design is composed of several modules that are responsible for different aspects of the iterative set expansion process. The main components of the project are:

1. `main.py`: This file is used to run the whole program. It performs all validation checks when user enters the program parameters, conducts the google search requests with the parameters, delegates whether spanbert or gpt3 extraction is to be performed by calling the respective methods. It is also mainly responsible for filtering out text from the BeautifulSoup output and printing the main output to the console.
2. `gpt3.py`: This file is responsible for validating sentences and formatting the prompt text to be used for the GPT-3 API to extract relations. A class called GPT3 is created in this file and instantiated in `main.py`. When gpt3 extraction is initiated, we first validate that all sentences checking to see if the extracted entities are corresponding to the entity types of the relation we are targeting in the program. After the validation is passed, we insert the sentence into the custom prompt text, which will be discussed in the Detailed Description. 
3. `spacy_help_functions.py`: This file is mainly based off of the file provided in https://github.com/zackhuiiiii/SpanBERT. We modified the extract_relations() function to handle the 4 possible target relations along with their corresponding allowed entity types. The provided print statements in the function were also modified to match the reference implementation for Project 2.

## Detailed Description

After terminal parameter validation checks have passed and querying google with the initial seed query has successfully completed, we first extract the unique URLs found from the search results and then iterate through each of the unique URLs. 
We perform an HTML request with the given URL and handle possible timeouts or response error codes through a `try-except` block. If an exception is found, we simply continue.

Next, we utilize BeautifulSoup to extract the actual plain text and use the `re` library to remove unnecessary characters. As stated in the project description, we trim the resulting plain text to 10,000 characters if the text exceeds such number of characters.

Then, we use the spacy language model and applies its NLP pipeline to the input text, which generates the processed Doc object.
This Doc object will serve as a parameter that will be applied for SpanBERT (spanbertExtraction(doc)) and GPT3 (gpt3Extraction(doc)) extractions

Note: We initialize a SpanBERT model using the pretrained weights located in the "./pretrained_spanbert" directory, if the value of the EXTRACTION_METHOD variable is equal to "-spanbert"; otherwise, it sets the variable spanbert to None; we won't be needing spanbert for GPT3


## SpanBERT
    In SpanBERT, we first filter the entities of interest depending on the target relation we specified in the terminal parameters. <br />
    Then, we call extract_methods() function, which stems from the `spacy_help_functions.py` file. We pass in the Doc object, the SpanBERT model, the set of extracted tuples X, the integer R specifying the target relation, the real name of the target relation R (i.e.  1: "per:schools_attended"), the filtered entities of interest, and the confidence threshold T. <br />

    In the extract_methods() function, we iterate through the sentences in the Doc object and call the predefined helper function create_entity_pairs(), which returns a list of extracted entity pairs for a given sentence. <br />
    We iterate through the entity pairs and perform an important check: We check to see if the second element of the entity pair (entity1) is equal to the target relation's subject entity and if the third element of the entity pair (entity2) is equal to the target relation's object entity. If the check passes, we add the token, subject, and object to the list of "correct" examples. 
    NOTE: Since subject and object entities might appear in either order in a sentence, we also add to the list the swapped version of the subject and object values. 
    If no correct examples are found for the given sentence, we skip the sentence and move on. 
    Otherwise, we use the SpanBERT model (stored in the variable spanbert) to predict the outputs for the input examples. <br />

    Now, we begin the process of adding tuples to the set of extracted tuples X. First, if we see the relation is not the same as the target relation, we skip. 
    We then check if the predicted confidence for the relation exceeds a certain threshold (conf). If it does, it checks if the extracted relation already exists in a set of non-duplicated relations.
    If it does not, it adds the extracted relation to the set X and prints a message indicating that it has been added. 
    If it does, it compares the confidence of the existing relation to the confidence of the new relation and keeps the relation with the higher confidence.
    If the confidence of the new relation is below the threshold, it is ignored.<br />

    Finally, the code prints the number of extracted relations, the number of sentences with annotations, and the total number of sentences. 
    
    The function finally returns the set of extracted relations X, which is redefined in `main.py`. 

### GPT3
    In GPT3, we first instantiate the custom class GPT3. The object takes the OpenAI key, the OpenAI model we will use (text-davinci-003), the temperature, the integer R, and the seed query Q. 
    Then, similar to SpanBERT, we filter the entities of interest depending on the target relation we specified in the terminal parameters. <br />
    Now, we call extract_methods() method, which stems from the custom class GPT3. The functionality is very similar to the function defined for the SpanBERT extraction method, but we perform less operations. We still perform the same important check of determining whether the entity pair correctly corresponds with the target relation's entity type. 
    Nevertheless, we pass in the sentences that were successfully validated as having correct entity types into the GPT-3 API, along with the vital prompt text. We do not need to check for level of confidence like SpanBERT.

    We attempt to have our prompt text be informative enough for GPT-3 API to understand what the extraction goal is for the respective target relation. 
    The prompt text starts with 










## Google Custom Search Engine Credentials
- JSON API Key: AIzaSyBr5aenBL0VfH55raQJUMSYiOmdkspmzPY
- Search Engine ID: 089e480ae5f6ce283

