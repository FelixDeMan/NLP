# Implement linguistic analyses using spacy
# Run them on data/preprocessed/train/sentences.txt
import spacy
import numpy as np
from collections import Counter
from spacy import displacy


def firstpart(input):
    # Let's run the NLP pipeline on our test input
    doc = nlp(input)

    # these are hard-coded. i couldn't find a way to extract the number of sentences/words
    word_nr_insentence = np.zeros(701)
    lengword_nr = np.zeros(13242)

    i = 0   # Counter for sentences
    j = 0   # Counter for words

    # Defining Counters
    word_frequencies = Counter()
    pos_f, posf, tag_f = Counter(), Counter(), Counter()

    # Go over all sentences
    for sentence in doc.sents:
        words = []
        pos_ = []
        pos = []
        tag_ = []

        # Go over all tokens
        for token in sentence:
            # Let's filter out punctuation
            if not token.is_punct:
                token_word = nlp(token.text.lower())[0]

                words.append(token_word.text) # For counting word occurences
                pos_.append(token_word.pos_) # For counting universal pos tags
                pos.append(token_word.pos)   # universal pos under code nr
                tag_.append(token_word.tag_) # finegrained pos tags

                # for calculating average word and sentence length
                lengword_nr[j] = len(token_word.text.lower()) # for average word length
                j += 1  # counter, goes up for each word
                word_nr_insentence[i] += 1 # for average sentence length

        # counters, updating dictionaries
        i += 1 # counter, goes up for each sentence
        word_frequencies.update(words)
        pos_f.update(pos_)
        posf.update(pos)
        tag_f.update(tag_)

    print(word_frequencies) # occurences of words
    num_tokens = len(doc)
    num_words = sum(word_frequencies.values())
    num_types = len(word_frequencies.keys())

    print("Number of tokens", num_tokens, "number of words", num_words, "number of types", num_types)
    print("Checking:", i, j, "Average nr words per sentence", np.mean(word_nr_insentence), "Average word length", np.mean(lengword_nr))
    print("total number of tags", sum(tag_f.values()), "number of tag types", len(tag_f.keys()))

    print("fine-grained pos tags", tag_f)
    # print("coarse-grained pos tags", pos_f)

    # find the 10 most common
    allitems = list(tag_f.items())
    allvalues = list(tag_f.values())
    tenmostcommon = []

    for i in range(10):
        for items in allitems:
            if items[-1] == max(allvalues):
                tenmostcommon.append(items[0])
                allitems.remove(items)
                allvalues.remove(max(allvalues))

    return tenmostcommon


def frequencies(input):
    # Count the most frequent tags, and provide the most frequent words that occur.
    tencommons = firstpart(input)

    # Let's run the NLP pipeline on our test input
    doc = nlp(input)

    # Defining Counters
    word_1, word_2, word_3, word_4, word_5, word_6, word_7, word_8, word_9, word_10 = Counter(), Counter(), Counter(), Counter(), Counter(), Counter(), Counter(), Counter(), Counter(), Counter()

    # Go over all sentences
    for sentence in doc.sents:
        words1, words2, words3, words4, words5, words6, words7, words8, words9, words10 = [], [], [], [], [], [], [], [], [], []
        # Go over all tokens
        for token in sentence:
            # Let's filter out punctuation
            if not token.is_punct:
                token_word = nlp(token.text.lower())[0]

                if token_word.tag_ == tencommons[0]:
                    words1.append(token_word.text) # For counting word occurences
                elif token_word.tag_ == tencommons[1]:
                    words2.append(token_word.text)
                elif token_word.tag_ == tencommons[2]:
                    words3.append(token_word.text)
                elif token_word.tag_ == tencommons[3]:
                    words4.append(token_word.text)
                elif token_word.tag_ == tencommons[4]:
                    words5.append(token_word.text)
                elif token_word.tag_ == tencommons[5]:
                    words6.append(token_word.text)
                elif token_word.tag_ == tencommons[6]:
                    words7.append(token_word.text)
                elif token_word.tag_ == tencommons[7]:
                    words8.append(token_word.text)
                elif token_word.tag_ == tencommons[8]:
                    words9.append(token_word.text)
                elif token_word.tag_ == tencommons[9]:
                    words10.append(token_word.text)
        word_1.update(words1)
        word_2.update(words2)
        word_3.update(words3)
        word_4.update(words4)
        word_5.update(words5)
        word_6.update(words6)
        word_7.update(words7)
        word_8.update(words8)
        word_9.update(words9)
        word_10.update(words10)

    print("frequency of words for", tencommons[0], word_1)
    print("frequency of words for", tencommons[1], word_2)
    print("frequency of words for", tencommons[2], word_3)
    print("frequency of words for", tencommons[3], word_4)
    print("frequency of words for", tencommons[4], word_5)
    print("frequency of words for", tencommons[5], word_6)
    print("frequency of words for", tencommons[6], word_7)
    print("frequency of words for", tencommons[7], word_8)
    print("frequency of words for", tencommons[8], word_9)
    print("frequency of words for", tencommons[9], word_10)

    return

def ngrams(input, ngram_size, type):
    """
    input = textfile
    ngram_size is 2 for bigrams, 3 for trigrams, etc
    type = pos_ for universal pos-tag n-grams, text for general ngrams
    """
    # Let's run the NLP pipeline on our test input
    doc = nlp(input)

    # Defining Counters
    n_gram = Counter()

    # Go over all sentences
    for sentence in doc.sents:
        words = []
        epoch = []
        # Go over all tokens
        for token in sentence:
            # Let's filter out punctuation
            if not token.is_punct:
                token_word = nlp(token.text.lower())[0]

                if type == "text":
                    words.append(token_word.text) # For counting word occurences
                elif type == "pos_":
                    words.append(token_word.pos_)

        for i in range(len(sentence) - ngram_size):
            epoch.append(tuple(words[i:i+ngram_size]))
            # counters, updating dictionaries
        n_gram.update(epoch)

    print("total ngrams", len(n_gram.keys()), n_gram)

    return

def lemmatization(input):
    # Let's run the NLP pipeline on our test input
    doc = nlp(input)

    # Defining Counters
    lemma_freq = Counter()
    lemma_dict = {}

    # Go over all sentences
    for sentence in doc.sents:
        words = []
        # Go over all tokens
        for token in sentence:
            # Let's filter out punctuation
            if not token.is_punct:
                token_word = nlp(token.text.lower())[0]

                words.append(token_word.lemma_) # For counting word occurences

                if str(token_word.lemma_) not in lemma_dict:
                    lemma_dict[str(token_word.lemma_)] = [str(token_word.text)]
                else:
                    lemma_dict[str(token_word.lemma_)].append(str(token_word.text))

        lemma_freq.update(words)

    print("------------------------------------lemma frequencies", len(lemma_freq.keys()), lemma_freq)
    print("-----------------------------------lemma and their instances", lemma_dict)
    return

def NER(input):
    # Let's run the NLP pipeline on our test input
    doc = nlp(input)

    # Defining Counters
    NER_freq = Counter()
    NER_dict = {}
    i = 0

    # Go over all sentences
    for sentence in doc.sents:
        words = []
        # Go over all tokens
        for token in sentence.ents:
            if i < 6:
                print(token.text, token.label_)

            words.append(token.label_) # For counting word occurences
            if str(token.label_) not in NER_dict:
                NER_dict[str(token.label_)] = [str(token)]
            else:
                NER_dict[str(token.label_)].append(str(token))
        if i < 6:
            print(i)
        i += 1
        NER_freq.update(words)

    print("---Named entity recognition frequencies", len(NER_freq.keys()), "number of NERs", sum(NER_freq.values()), NER_freq)
    print("---NERs and their instances", NER_dict)
    return



if __name__ == '__main__':
    nlp = spacy.load('en_core_web_sm')

    filename = "data/preprocessed/train/sentences.txt"
    with open(filename, 'r', encoding='utf-8') as f:
        newline_break =""
        for readline in f:
            line_strip = (readline.rstrip() +" ")
            newline_break+=line_strip
    input = newline_break

    # firstpart(input)
    # frequencies(input)
    # ngrams(input, 2, "text")
    # lemmatization(input)
    NER(input)



