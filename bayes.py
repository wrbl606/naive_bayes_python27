# Example of Naive Bayes implemented from Scratch in Python
import csv
import random
import math
from tabulate import tabulate

"""
ilosc dzieci, stezenie glukozy, cisnienie krwi, grubosc faldu skornego tricepsowego,
poziom insuliny, masa ciala, funkcja rodowodowa cukrzycy
"""

def load_csv(filename):
    lines = csv.reader(open(filename, "rt"))
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    return dataset


def split_data_set(data_set, split_ratio):
    train_size = int(len(data_set) * split_ratio)
    train_set = []
    copy = list(data_set)
    while len(train_set) < train_size:
        index = random.randrange(len(copy))
        train_set.append(copy.pop(index))
    return [train_set, copy]


def separate_by_class(data_set):
    separated = {}
    for i in range(len(data_set)):
        vector = data_set[i]
        if vector[-1] not in separated:
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated


def mean(numbers):
    return sum(numbers) / float(len(numbers))


def standard_deviation(numbers):
    avg = mean(numbers)
    variance = sum([pow(x - avg, 2) for x in numbers]) / float(len(numbers) - 1)
    return math.sqrt(variance)


def summarize(data_set):
    summaries = [(mean(attribute), standard_deviation(attribute)) for attribute in zip(*data_set)]
    del summaries[-1]
    return summaries


def summarize_by_class(data_set):
    separated = separate_by_class(data_set)
    summaries = {}
    for classValue, instances in separated.iteritems():
        summaries[classValue] = summarize(instances)
    return summaries


def calculate_probability(x, mean, standard_deviation):
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(standard_deviation, 2))))
    return (1 / (math.sqrt(2 * math.pi) * standard_deviation)) * exponent


def calculate_class_probabilities(summaries, input_vector):
    probabilities = {}
    for classValue, classSummaries in summaries.iteritems():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = input_vector[i]
            probabilities[classValue] *= calculate_probability(x, mean, stdev)
    return probabilities


def predict(summaries, input_vector):
    probabilities = calculate_class_probabilities(summaries, input_vector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.iteritems():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel


def get_predictions(summaries, test_set):
    predictions = []
    for i in range(len(test_set)):
        result = predict(summaries, test_set[i])
        predictions.append(result)
        print("predicted: ", result, "should be:", test_set[i][-1])
    return predictions


def get_accuracy(test_set, predictions):
    correct = 0
    for i in range(len(test_set)):
        if test_set[i][-1] == predictions[i]:
            correct += 1
    return (correct / float(len(test_set))) * 100.0


def main():
    filename = "pima-indians-diabetes.data.csv"
    split_ratio = 0.5
    data_set = load_csv(filename)
    training_set, test_set = split_data_set(data_set, split_ratio)
    print('Split {0} rows into train={1} and test={2} rows').format(len(data_set), len(training_set), len(test_set))
    # prepare model
    summaries = summarize_by_class(training_set)
    print(tabulate(summaries, headers=["Klasa 0", "Klasa 1"]))
    # test model
    predictions = get_predictions(summaries, test_set)
    accuracy = get_accuracy(test_set, predictions)
    print('Accuracy: {0}%').format(accuracy)


main()
