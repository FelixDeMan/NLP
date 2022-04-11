# Implement linguistic analyses using spacy
# Run them on data/preprocessed/train/sentences.txt
import spacy
import os
from collections import Counter


if __name__ == '__main__':
    nlp = spacy.load('en_core_web_sm')

    # test_input = "I have an awesome cat. It's sitting on the mat that I bought yesterday."
    # # Let's run the NLP pipeline on our test input
    # doc = nlp(test_input)
    # print(doc)
    # for token in doc:
    #     print("Text = ", token.text," lemma = ", token.lemma_," pos = ", token.pos_, " tag= " ,token.tag_)
    # print(spacy.explain("VBD"))
    # first_token = doc[0]
    # print(dir(first_token))

    test_input = "Processing raw text intelligently is difficult: most words are rare, and it’s common for words that look completely different to mean almost the same thing. The same words in a different order can mean something completely different. Even splitting text into useful word-like units can be difficult in many languages. While it’s possible to solve some problems starting from only the raw characters, it’s usually better to use linguistic knowledge to add useful information. That’s exactly what spaCy is designed to do: you put in raw text, and get back a Doc object, that comes with a variety of annotations."
    # Let's run the NLP pipeline on our test input
    doc = nlp(test_input)

    word_frequencies = Counter()

    for sentence in doc.sents:
        words = []
        for token in sentence:
            # Let's filter out punctuation
            if not token.is_punct:
                words.append(token.text)
        word_frequencies.update(words)

    print(word_frequencies)
    num_tokens = len(doc)
    num_words = sum(word_frequencies.values())
    num_types = len(word_frequencies.keys())

    print(num_tokens, num_words, num_types)
   # COOL




