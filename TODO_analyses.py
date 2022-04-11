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
    filename = "data/preprocessed/train/sentences.txt"
    with open(filename, 'r', encoding='utf-8') as f:
        newline_break =""
        for readline in f:
            line_strip = readline.rstrip()
            newline_break+=line_strip
    print(newline_break)
    input = newline_break

    # Let's run the NLP pipeline on our test input
    doc = nlp(input)

    word_frequencies = Counter()
    pos_f, posf, tag_f = Counter(), Counter(), Counter()
    for sentence in doc.sents:
        words = []
        pos_ = []
        pos = []
        tag_ = []
        for token in sentence:
            # Let's filter out punctuation
            if not token.is_punct:
                words.append(token.text)
                pos_.append(token.pos_)
                pos.append(token.pos)
                tag_.append(token.tag_)
        word_frequencies.update(words)
        pos_f.update(pos_)
        posf.update(pos)
        tag_f.update(tag_)

    print(word_frequencies)
    num_tokens = len(doc)
    num_words = sum(word_frequencies.values())
    num_types = len(word_frequencies.keys())

    print(num_tokens, num_words, num_types)

    #most_10 = doc.most_common(10)
    print(tag_f)
    #table = np.zeros((6,10))
    #for i, word in enumeratemost_10:



   # COOL