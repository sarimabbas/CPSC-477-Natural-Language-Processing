import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import itertools
import shutil
import tree as tr
from utils import Vocab
from backprop import RecursiveNeuralNet

def test1():
    hiddenDim = 3
    classes = 2

    vocab = Vocab()
    vocab.construct(["i", "love", "apple", "juice"])
    tree = tr.Tree("(1 (0 I) (1 (1 love) (1 (1 love) (0 (0 apple) (0 juice)))))")

    rnn = RecursiveNeuralNet(hiddenDim, classes, vocab)
    rnn.initialize_matrices(W=np.matrix([[1.0, 2.0, 0.0, -0.4, 0.2, -0.8],
                                        [-0.5, 1.0, 2.0, 0.0, -0.4, 0.2],
                                        [-0.6, 0.9, 1.1, 1.0, -2.0, 0.0]], dtype=np.float32),
                                        b = np.matrix([[-0.4], [0.5], [0.2]], dtype=np.float32),
                                        Ws = np.matrix([[0.0, 0.1, -0.2], [1.4, -0.7, 0.1]], dtype=np.float32),
                                        bs = np.matrix([[-0.1], [0.4]], dtype=np.float32),
                                        L = np.matrix([[0.4, -0.3, -0.1],
                                                        [0.1, 0.1, 0.2],
                                                        [0.04, -0.9, 1.2],
                                                        [0.2, 0.5, 0.6],
                                                        [0.2, 0.5, 0.6]], dtype=np.float32))

    result = rnn.forward_prop(tree.root)
    rnn.backward_prop(tree.root)

    return {'dW':rnn.dW, 'db':rnn.db, 'dWs':rnn.dWs, 'dbs':rnn.dbs}

def test2():
    hiddenDim = 3
    classes = 2
    
    vocab = Vocab()
    vocab.construct(["i", "am", "a", "an", "student"])
    tree = tr.Tree("(0 (0 I) (1 (1 am) (0 (0 a) (0 student))))")
    
    rnn = RecursiveNeuralNet(hiddenDim, classes, vocab)
    rnn.initialize_matrices(W=np.matrix([[1.0, -1.0, 0.9, -0.6, 0.2, 0.0],
                                         [-0.3, 1.2, 0.0, 0.4, -0.4, 0.0],
                                         [-0.8, 0.1, 1.1, 0.0, -2.0, 0.0]], dtype=np.float32),
                            b = np.matrix([[-0.4], [0.0], [0.0]], dtype=np.float32),
                            Ws = np.matrix([[0.0, 0.0, -0.5], [1.0, -0.9, 0.0]], dtype=np.float32),
                            bs = np.matrix([[-0.4], [0.0]], dtype=np.float32),
                            L = np.matrix([[0.4, -0.3, -0.1],
                                           [0.6, -0.3, 0.4],
                                           [0.04, -0.08, 1.25],
                                           [0.2, 0.5, 0.6],
                                           [0.1, 0.7, 0.6],
                                           [0.1, 0.3, 0.0]], dtype=np.float32))
        
    result = rnn.forward_prop(tree.root)
    rnn.backward_prop(tree.root)
                                           
    return {'dW':rnn.dW, 'db':rnn.db, 'dWs':rnn.dWs, 'dbs':rnn.dbs}

def test3():
    hiddenDim = 3
    classes = 2
    
    vocab = Vocab()
    vocab.construct(["i", "hate", "cat", "fur"])
    tree = tr.Tree("(0 (0 I) (1 (1 hate) (1 (1 cat) (0 (0 cat) (0 fur)))))")
    
    rnn = RecursiveNeuralNet(hiddenDim, classes, vocab)
    rnn.initialize_matrices(W=np.matrix([[1.0, -1.0, 0.5, -0.6, 0.2, -0.8],
                                         [-0.3, 1.2, 2.0, 0.4, -0.4, 0.2],
                                         [-0.8, 0.9, 1.1, 1.0, -2.0, 0.1]], dtype=np.float32),
                            b = np.matrix([[-0.4], [0.3], [0.4]], dtype=np.float32),
                            Ws = np.matrix([[0.2, 0.1, -0.5], [1.2, -0.9, 0.3]], dtype=np.float32),
                            bs = np.matrix([[-0.4], [0.5]], dtype=np.float32),
                            L = np.matrix([[0.4, -0.3, -0.1],
                                           [0.6, -0.3, 0.4],
                                           [0.04, -0.08, 1.25],
                                           [0.2, 0.7, 0.6],
                                           [0.2, 0.7, 0.6]], dtype=np.float32))
        
    result = rnn.forward_prop(tree.root)
    rnn.backward_prop(tree.root)
                                           
    return {'dW':rnn.dW, 'db':rnn.db, 'dWs':rnn.dWs, 'dbs':rnn.dbs}


def run_all_tests(test_list, to_check, prefix="tests/backprop"):

    num_correct = 0
    for i, test in enumerate(test_list):
        all_passed = True
        npzfile = np.load(prefix+str(i+1)+".npz")
        student_mats = test()
        for mat in to_check:
            if not np.allclose(student_mats[mat], npzfile[mat]):
                print("Error: Test {} Matrix {}".format(i+1, mat))
                print("Student solution:")
                print(student_mats[mat])
                print("Expected solution:")
                print(npzfile[mat])
                all_passed = False
        if all_passed:
            num_correct += 1
    print("--------")
    print("{}/{} tests passed".format(num_correct, len(test_list)))


def main():
    
    run_all_tests([test1, test2, test3], ['dW', 'db', 'dWs', 'dbs'], prefix="tests/backprop")

if __name__ == "__main__":
    main()
