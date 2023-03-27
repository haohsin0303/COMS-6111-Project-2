from collections import defaultdict


spacy2bert = { 
        "ORG": "ORGANIZATION",
        "PERSON": "PERSON",
        "GPE": "LOCATION", 
        "LOC": "LOCATION",
        "DATE": "DATE"
        }

bert2spacy = {
        "ORGANIZATION": "ORG",
        "PERSON": "PERSON",
        "LOCATION": "LOC",
        "CITY": "GPE",
        "COUNTRY": "GPE",
        "STATE_OR_PROVINCE": "GPE",
        "DATE": "DATE"
        }


def get_entities(sentence, entities_of_interest):
    return [(e.text, spacy2bert[e.label_]) for e in sentence.ents if e.label_ in spacy2bert]


def extract_relations(doc, spanbert, X, R, target_relation, live_in_tuples, entities_of_interest=None, conf=0.7):
    res = defaultdict(int)

    # Declare annotation counts
    sentences_with_annotations = 0
    overall_relations = 0
    non_duplicated_relations = 0

    # Iterate through each of the doc object sentences
    for index, sentence in enumerate(doc.sents):

        # Create and iterate through the entity pairs
        entity_pairs = create_entity_pairs(sentence, entities_of_interest)
        examples = []
        for ep in entity_pairs:
            # Append to the list the relations that satisfy the criteria corresponding to the target relation's entity types
            # Add the swapped version as well
            if R == 1 and ((ep[1][1] == "PERSON" and ep[2][1] == "ORGANIZATION") or (ep[2][1] == "PERSON" and ep[1][1] == "ORGANIZATION")):
                examples.append({"tokens": ep[0], "subj": ep[1], "obj": ep[2]})
                examples.append({"tokens": ep[0], "subj": ep[2], "obj": ep[1]})
            elif R == 2 and ((ep[1][1] == "PERSON" and ep[2][1] == "ORGANIZATION") or (ep[2][1] == "PERSON" and ep[1][1] == "ORGANIZATION")):
                examples.append({"tokens": ep[0], "subj": ep[1], "obj": ep[2]})
                examples.append({"tokens": ep[0], "subj": ep[2], "obj": ep[1]})
            elif R == 3 and ((ep[1][1] == "PERSON" and ep[2][1] in ["LOCATION", "CITY", "STATE_OR_PROVINCE", "COUNTRY"]) or (ep[2][1] == "PERSON" and ep[1][1] in ["LOCATION", "CITY", "STATE_OR_PROVINCE", "COUNTRY"])):
                examples.append({"tokens": ep[0], "subj": ep[1], "obj": ep[2]})
                examples.append({"tokens": ep[0], "subj": ep[2], "obj": ep[1]})
            elif R == 4 and ((ep[1][1] == "ORGANIZATION" and ep[2][1] == "PERSON") or (ep[2][1] == "ORGANIZATION" and ep[1][1] == "PERSON")):
                examples.append({"tokens": ep[0], "subj": ep[1], "obj": ep[2]})
                examples.append({"tokens": ep[0], "subj": ep[2], "obj": ep[1]})

        # ignore subject entities with date/location type
        examples = [p for p in examples if not p["subj"][1] in ["DATE", "LOCATION"]]

        if (index + 1) % 5 == 0:
            print("\tProcessed {count} / {total} sentences".format(count=index+1, total=len(list(doc.sents))))

        # Skip if nothing was appended
        if len(examples) == 0:
            continue

        sentence_has_annotation = 0

        preds = spanbert.predict(examples)
        for ex, pred in list(zip(examples, preds)):
            relation = pred[0]
            # We skip if the relation is not equal to the target relation or if Live_In tuple is used, the relation is not valid
            if (R != 3 and relation != target_relation) or (R == 3 and relation not in live_in_tuples.keys()):
                continue
            sentence_has_annotation = 1
            overall_relations += 1
            print("\n\t\t=== Extracted Relation ===")
            print("\t\tInput Tokens: {}".format(ex['tokens']))
            subj = ex["subj"][0]
            obj = ex["obj"][0]
            confidence = pred[1]
            print("\t\tOutput Confidence: {:.3f} ; Subject: {}  ; Object: {}".format(confidence, subj, obj))
            # If confidence exceeds the input threshold
            if confidence > conf:
                # Check if the confidence we have already added is lower than current confidence
                if res[(subj, relation, obj)] < confidence:
                    # Remove the low confidence tuple and add the higher confidence tuple
                    old_confidence = res[(subj, relation, obj)]
                    res[(subj, relation, obj)] = confidence
                    non_duplicated_relations += 1
                    X.discard((old_confidence, subj, obj))
                    X.add((float(confidence), subj, obj))
                    # Repeat as well if we have a Live_In relation
                    if (R == 3 and relation in live_in_tuples.keys()):
                        live_in_tuples[relation].discard((old_confidence, subj, obj))
                        live_in_tuples[relation].add((float(confidence), subj, obj))
                    print("\t\tAdding to set of extracted relations")
                else:
                    print("\t\tDuplicate with lower confidence than existing record. Ignoring this.")
            else:
                print("\t\tConfidence is lower than threshold confidence. Ignoring this.")
            print("\t\t==========")

        sentences_with_annotations += sentence_has_annotation
    
    print("\tExtracted annotations for  {sentences_with_annotations}  out of total  {total_sentences}  sentences".format(sentences_with_annotations=sentences_with_annotations, total_sentences=len(list(doc.sents))))
    print("\tRelations extracted from this website: {non_duplicated_relations} (Overall: {overall_relations})".format(non_duplicated_relations=non_duplicated_relations, overall_relations=overall_relations))

    # Return the updated set of extracted tuples X
    return X


def create_entity_pairs(sents_doc, entities_of_interest, window_size=40):
    '''
    Input: a spaCy Sentence object and a list of entities of interest
    Output: list of extracted entity pairs: (text, entity1, entity2)
    '''

    if entities_of_interest is not None:
        entities_of_interest = {bert2spacy[b] for b in entities_of_interest}
    ents = sents_doc.ents # get entities for given sentence

    length_doc = len(sents_doc)
    entity_pairs = []
    for i in range(len(ents)):
        e1 = ents[i]
        if entities_of_interest is not None and e1.label_ not in entities_of_interest:
            continue

        for j in range(1, len(ents) - i):
            e2 = ents[i + j]
            if entities_of_interest is not None and e2.label_ not in entities_of_interest:
                continue
            if e1.text.lower() == e2.text.lower(): # make sure e1 != e2
                continue

            if (1 <= (e2.start - e1.end) <= window_size):

                punc_token = False
                start = e1.start - 1 - sents_doc.start
                if start > 0:
                    while not punc_token:
                        punc_token = sents_doc[start].is_punct
                        start -= 1
                        if start < 0:
                            break
                    left_r = start + 2 if start > 0 else 0
                else:
                    left_r = 0

                # Find end of sentence
                punc_token = False
                start = e2.end - sents_doc.start
                if start < length_doc:
                    while not punc_token:
                        punc_token = sents_doc[start].is_punct
                        start += 1
                        if start == length_doc:
                            break
                    right_r = start if start < length_doc else length_doc
                else:
                    right_r = length_doc

                if (right_r - left_r) > window_size: # sentence should not be longer than window_size
                    continue

                x = [token.text for token in sents_doc[left_r:right_r]]
                gap = sents_doc.start + left_r
                e1_info = (e1.text, spacy2bert[e1.label_], (e1.start - gap, e1.end - gap - 1))
                e2_info = (e2.text, spacy2bert[e2.label_], (e2.start - gap, e2.end - gap - 1))
                if e1.start == e1.end:
                    assert x[e1.start-gap] == e1.text, "{}, {}".format(e1_info, x)
                if e2.start == e2.end:
                    assert x[e2.start-gap] == e2.text, "{}, {}".format(e2_info, x)
                entity_pairs.append((x, e1_info, e2_info))

    return entity_pairs
