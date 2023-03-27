# COMS-6111-Project-2

This project is about information extraction on the web, in other words, extracting "structured" information that is embedded in natural language text on the web. We use spaCy to process the documents and use SpanBERT and GPT3-API to extract relations in the sentences.

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

3. Copy all the project files into SpanBERT repository
```
cd ..
cp ./proj2/main.py ./SpanBERT/
cp ./proj2/gpt3.py ./SpanBERT/
cp ./proj2/spacy_help_functions.py ./SpanBERT/
cd SpanBERT/
```

3. Run the program in SpanBERT repository using the following command: ```
python3 main.py [-spanbert|-gpt3] <API Key> <Engine Key> <OPENAI Key> <r=[1,4]>, <t=[0,1]>, <Seed Query> <k > 0>```

- [-spanbert|-gpt3]: An argument that indicates whether we are using SpanBERT (-spanbert) or GPT-3 (-gpt3) for the extraction process.
- API Key: The Google Custom Search Engine JSON API Key.
- Engine Key: The Google Custom Search Engine ID.
- OpenAI Key: The OpenAI key used for the GPT-3 API to extract relations from text documents.
- r=[1,4] : An integer between 1 and 4, indicating the relation to extract - 1 is for Schools_Attended, 2 is for Work_For, 3 is for Live_In, and 4 is for Top_Member_Employees.
- t=[0,1] : A real number between 0 and 1, indicating the "extraction confidence threshold," which is the minimum extraction confidence that we request for the tuples in the output; t is ignored if we are using -gpt3
- Seed Query: A list of words in double quotes corresponding to a plausible tuple for the relation to extract (e.g., "bill gates microsoft" for relation Work_For)
- k > 0: An integer greater than 0, indicating the number of tuples that we request in the output

## General Description

The project's internal design comprises several modules responsible for different aspects of the iterative set expansion process. The main components of the project are:

1. `main.py`: This file is used to run the whole program. It performs all validation checks when user enters the program parameters, conducts the google search requests with the parameters, delegates whether spanbert or gpt3 extraction is to be performed by calling the respective methods. It is also mainly responsible for filtering out text from the BeautifulSoup output and printing the main output to the console.
2. `gpt3.py`: This file is responsible for validating sentences and formatting the prompt text to be used for the GPT-3 API to extract relations. A class called GPT3 is created in this file and instantiated in `main.py`. When gpt3 extraction is initiated, we first validate that all sentences checking to see if the extracted entities are corresponding to the entity types of the relation we are targeting in the program. After the validation is passed, we insert the sentence into the custom prompt text, which will be discussed in the Detailed Description. 
3. `spacy_help_functions.py`: This file is mainly based on the file provided at https://github.com/zackhuiiiii/SpanBERT. We modified the `extract_relations()` function to handle the 4 possible target relations along with their corresponding allowed entity types. The provided print statements in the function were also modified to match the reference implementation for Project 2.

## Detailed Description

After terminal parameter validation checks have passed and querying Google with the initial seed query has successfully completed, we first extract the unique URLs found from the search results and then iterate through each of the unique URLs. 
We perform an HTML request with the given URL and handle possible timeouts or response error codes through a `try-except` block. If an exception is found, we simply continue.

Next, we utilize BeautifulSoup to extract the actual plain text and use the `re` library to remove unnecessary characters. As stated in the project description, we trim the resulting plain text to 10,000 characters if the text exceeds such number of characters.

Then, we use the spaCy language model and apply its NLP pipeline to the input text, which generates the processed Doc object.
This Doc object will serve as a parameter that will be applied for SpanBERT (`spanbertExtraction(doc)`) and GPT3 (`gpt3Extraction(doc)`) extractions. We discuss further below in detail about how we do each respective extraction method. 

<i>Note: We initialize a SpanBERT model using the pre-trained weights located in the "./pretrained_spanbert" directory, if the value of the EXTRACTION_METHOD variable is equal to "-spanbert"; otherwise, it sets the variable spanbert to None; we won't be needing spanbert for GPT3.</i>

After the extraction method is complete, we receive a newly updated set of extracted tuples X. 
We first check a specific condition before we handle the general cases:
    - We first check whether we used the SpanBERT extraction method on the Live_In target relation. The reason is because Live_In involves `"per:countries_of_residence", "per:cities_of_residence", or "per:stateorprovinces_of_residence"`; any sentence can be classified with any of these three specific relation types, so there is a possibility we may get in our final output a mixture of tuple results with `per:countries_of_residence` relation type AND `per:cities_of_residence` relation type. We iterate through all the relations and print their respective outputs.

If we are not dealing with this specific scenario, we process the outputs as expected. 

To determine the querying status for the next iteration, we first check if X contains at least k tuples. If it does, we immediately stop querying. Otherwise, we select from X a tuple y such that (1) y has not been used for querying yet and (2) if -spanbert is specified, y has an extraction confidence that is highest among the tuples in X that have not yet been used for querying. We create a query q from tuple y by just concatenating the subject and object values together. If no such y tuple exists, then stop. We print "ISE has "stalled" before retrieving k high-confidence tuples". 

### SpanBERT
In SpanBERT, we first filter the entities of interest depending on the target relation we specified in the terminal parameters. <br />
Then, we call `extract_relations()` function, which stems from the `spacy_help_functions.py` file. We pass in the Doc object, the SpanBERT model, the set of extracted tuples X, the integer R specifying the target relation, the real name of the target relation R (e.g. 1: "per:schools_attended"), the filtered entities of interest, and the confidence threshold T. <br />

In the `extract_relations()` function, we iterate through the sentences in the Doc object and call the predefined helper function create_entity_pairs(), which returns a list of extracted entity pairs for a given sentence. <br />
We iterate through the entity pairs and perform an important check: We check to see if the second element of the entity pair (entity1) is equal to the target relation's subject entity and if the third element of the entity pair (entity2) is equal to the target relation's object entity. If the check passes, we add the token, subject, and object to the list of "correct" examples. 
<i> NOTE: Since subject and object entities might appear in either order in a sentence, we also add the swapped version of the subject and object values to the list. 
If no correct examples are found for the given sentence, we skip the sentence and move on. </i>
Otherwise, we use the SpanBERT model (stored in the variable spanbert) to predict the outputs for the input examples. <br />

Now, we begin the process of adding tuples to the set of extracted tuples X. First, if we see the relation is not the same as the target relation, we skip. 
We then check if the predicted confidence for the relation exceeds a certain threshold (conf). If it does, it checks if the extracted relation already exists in a set of non-duplicated relations.
If it does not, it adds the extracted relation to the set X and prints a message indicating that it has been added. 
If it does, it compares the confidence of the existing relation to the confidence of the new one and keeps it with higher confidence.
If the confidence of the new relation is below the threshold, it is ignored.<br />

Finally, the code prints the number of extracted relations, the number of sentences with annotations, and the total number of sentences. 
    
The function returns the set of extracted relations X, which is redefined in `main.py`. 

### GPT3
In GPT3, we first instantiate the custom class GPT3. The object takes the OpenAI key, the OpenAI model we will use (text-davinci-003), the temperature, the integer R, and the seed query Q. 
Then, similar to SpanBERT, we filter the entities of interest depending on the target relation we specified in the terminal parameters. <br />
Now, we call `extract_relations()` method, which stems from the custom class GPT3. The functionality is very similar to the function defined for the SpanBERT extraction method, but we perform less operations. We still perform the same important check of determining whether the entity pair correctly corresponds with the target relation's entity type. 
Nevertheless, we pass in the successfully validated sentences as having correct entity types into the GPT-3 API, along with the vital prompt text. We do not need to check for level of confidence like SpanBERT.

We attempt to have our prompt text be informative enough for GPT-3 API to understand the extraction goal for the respective target relation.

We can provide a general summary of the prompt text by saying it tries to extract information from a sentence about people and the organizations they work for, by looking for the `[Schools_Attended | Work_For | Live_In | Top_Member_Employees]` relation between the subject (a person) and the object (an organization) and returns a list of tuples in a specific format (e.g. `[(person's name, [Schools_Attended | Work_For | Live_In | Top_Member_Employees] , organization's name)]`). If no such tuple exists, an empty list is returned, and the program tries to ensure that the subject and object are mentioned in the same sentence and represent the correct types. 

<b><i>NOTE: The output for GPT-3 API can sometimes be inconsistent and out of our control. The sentences passed into GPT-3 do pass the validation checks applied using SpanBERT but can be incorrectly classified as a valid relation. We have done our best to create a descriptive prompt text with provided example sentences and example outputs for GPT-3 API in order to produce relevant extracted tuples.</b></i>

Finally, like SpanBERT, the function returns the updated set of extracted relations X.

## Credentials
- JSON API Key: AIzaSyBr5aenBL0VfH55raQJUMSYiOmdkspmzPY
- Search Engine ID: 089e480ae5f6ce283
- OpenAI Secret Key: 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
