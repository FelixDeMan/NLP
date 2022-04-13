# Implement four baselines for the task.
# Majority baseline: always assigns the majority class of the training data
# Random baseline: randomly assigns one of the classes. Make sure to set a random seed and average the accuracy over 100 runs.
# Length baseline: determines the class based on a length threshold
# Frequency baseline: determines the class based on a frequency threshold
import random
from statistics import mean
import numpy as np
from collections import Counter
from model.data_loader import DataLoader

# Each baseline returns predictions for the test data. The length and frequency baselines determine a threshold using the development data.

def majority_baseline(train_sentences, train_labels, testinput, testlabels):

     # TODO: determine the majority class based on the training data
     # ...
    print((train_labels[0]))
    counterN = 0
    counterC = 0
    for labels in train_labels:
        counterN += labels.count("N")
        counterC += labels.count("C")
    if counterN > counterC:
        majority_class = "N"
    if counterC > counterN:
        majority_class = "C"

    predictions = []
    instances = []


    for instance in testinput:
        print(instance)
        tokens = instance.split(" ")
        instance_predictions = [majority_class for t in tokens]
        predictions.append(instance_predictions)
        instances.append(instance)
        print(instance_predictions)

    # TODO: calculate accuracy for the test input
    # ...
    labelcountN = 0
    labelcountC = 0

    for labels in testlabels:
        labelcountN += labels.count("N")
        labelcountC += labels.count("C")

    if majority_class == "C":
        accuracy = labelcountC / (labelcountN + labelcountC)

    if majority_class == "N":
        accuracy = labelcountN / (labelcountN + labelcountC)

    return accuracy, predictions


def random_baseline(train_sentences, train_labels, testinput, testlabels, B = 100):
    np.random.seed(2020)
    predictions = []
    instances = []
    correct = []
    accuracy = []
    for b in range(B):
        for i, instance in enumerate(testinput):
            print(instance)
            tokens = instance.split(" ")
            predictions.append(instance_predictions)

            for j, label in enumerate(testlabels[i].split(' ')):
                if label == instance_predictions[j]: correct.append(1)
                else: correct.append(0)
            instances.append(instance)
            print(instance_predictions)
        accuracy.append(sum(correct) / len(correct))



    print(accuracy)
    accuracy = mean(accuracy)
    print(accuracy)
    return accuracy, predictions



def freq_baseline(train_sentences, train_labels, testinput, testlabels, treshold = 3):
    predictions = []
    instances = []
    correct = []

    word_frequencies = Counter()

    for sentence in train_sentences:
        words = []
        for token in sentence:
            words.append(token)
        word_frequencies.update(words)

    print(word_frequencies)
    for i, instance in enumerate(testinput):
        print(instance)
        tokens = instance.split(" ")
        instance_predictions = ["N" if word_frequencies[t] > treshold else "C" for t in tokens]
        predictions.append(instance_predictions)
        instances.append(instance)
        print(instance_predictions)
        print(testlabels[i])

        for j, label in enumerate(testlabels[i].split(" ")):
            if label == instance_predictions[j]: correct.append(1)
            else: correct.append(0)

    accuracy = sum(correct) / len(correct)

    return accuracy, predictions

def length_baseline(train_sentences, train_labels, testinput, testlabels, treshold = 3):
    predictions = []
    instances = []
    correct = []
    for i, instance in enumerate(testinput):
        print(instance)
        tokens = instance.split(" ")
        instance_predictions = ["C" if len(t) > treshold else "N" for t in tokens]
        predictions.append(instance_predictions)
        instances.append(instance)
        print(instance_predictions)
        print(testlabels[i])

        for j, label in enumerate(testlabels[i].split(" ")):
            if label == instance_predictions[j]: correct.append(1)
            else: correct.append(0)

    accuracy = sum(correct) / len(correct)

    return accuracy, predictions


# TODO: output the predictions in a suitable way so that you can evaluate them
def write_output(outfile, predictions, labels, input):
    labels_list = []
    print("Labels = ", labels)
    for labs in labels:

        x = labs.split(" ")
        for label in x: label.strip()
        #print(x[-1])
        #if "\n" in x[-1]: x[-1].replace("\n", "")
        labels_list.append(x)
    print(labels_list)
    with open(outfile, "w") as f:
        for i, sentences in enumerate(input):
            tokens = sentences.split(" ")
            for j, token in enumerate(tokens):
                #print([token, labels[i][j], predictions[i][j]])
                f.write("\t".join([token.strip(), labels_list[i][j].strip(), predictions[i][j]]))
                f.write("\n")



if __name__ == '__main__':
    train_path = "data/preprocessed/train/"
    dev_path = "data/preprocessed/val/"
    test_path = "data/preprocessed/test/"

    # Note: this loads all instances into memory. If you work with bigger files in the future, use an iterator instead.

    with open(train_path + "sentences.txt") as sent_file:
        train_sentences = sent_file.readlines()

    with open(train_path + "labels.txt") as label_file:
        train_labels = label_file.readlines()


    with open(dev_path + "sentences.txt") as dev_file:
        dev_sentences = dev_file.readlines()

    with open(train_path + "labels.txt") as dev_label_file:
        dev_labels = dev_label_file.readlines()
    with open(test_path + "sentences.txt") as testfile:
        testinput = testfile.readlines()

    with open(test_path + "labels.txt") as test_label_file:
        testlabels = test_label_file.readlines()

    majority_accuracy, majority_predictions = length_baseline(train_sentences, train_labels, testinput, testlabels, treshold= 5)
    write_output("length_baseline.tsv", majority_predictions, testlabels, testinput)
    print(majority_accuracy)

