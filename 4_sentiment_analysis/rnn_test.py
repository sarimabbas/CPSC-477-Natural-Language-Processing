import sys
import os
import numpy as np
import math
import time
import itertools
import shutil
import tensorflow as tf
import tree as tr
from utils import Vocab
import rnn

def compute_test_accuracy(model, weights_path):

    predictions, _ = model.predict(model.test_data, weights_path)
    labels = [t.root.label for t in model.test_data]
    test_acc = np.equal(predictions, labels).mean()

    return test_acc

def calculate_rnn_score(num_repetitions=1):
    config = rnn.Config()
    accuracies = []
    score = 0

    for i in range(num_repetitions):
        model = rnn.RNN_Model(config)
        try:
            accuracies.append(compute_test_accuracy(model, './weights/%s.temp'%model.config.model_name))
        except Exception as e:
            print(e)
            print("------")
            print("Score = {}/3".format(score))
            return

    print("Test accuracies over {} runs:".format(num_repetitions))
    print(accuracies)

    max_acc = max(accuracies)
    if max_acc >= 0.72:
        score = 3
    elif max_acc >= 0.65:
        score = 2
    else:
        score = 1

    print("------")
    print("Maximum accuracy = {}".format(max_acc))
    print("Score = {}/3".format(score))

def main():
    calculate_rnn_score(3)

if __name__ == "__main__":
    main()
