from __future__ import division # had to be included because its always unclear whether we use py2 or 3!

import math
import nltk
import time

import collections
import pprint
pp = pprint.PrettyPrinter(indent=4)

# Constants to be used by you when you fill the functions
START_SYMBOL = '*'
STOP_SYMBOL = 'STOP'
MINUS_INFINITY_SENTENCE_LOG_PROB = -1000

# TODO: IMPLEMENT THIS FUNCTION
# Calculates unigram, bigram, and trigram probabilities given a training corpus
# training_corpus: is a list of the sentences. Each sentence is a string with tokens separated by spaces, ending in a newline character.
# This function outputs three python dictionaries, where the keys are tuples expressing the ngram and the value is the log probability of that ngram
def calc_probabilities(training_corpus):
    unigram_p = {}
    bigram_p = {}
    trigram_p = {}

    ##### UNIGRAMS
    # split each sentence into tokens
    unigram_tokenized_sentences = []
    for sentence in training_corpus:
        split_sentence = sentence.strip().split() # split sentences by whitespace
        split_sentence.append(STOP_SYMBOL) # add 'STOP' to the end
        unigram_tokenized_sentences.extend(split_sentence) # add the sentence to our list of sentences

    unigram_counter = collections.Counter(unigram_tokenized_sentences)
    unigram_sum = sum(unigram_counter.itervalues())

    unigram_p = {(unigram,) : math.log(unigram_counter[unigram] / unigram_sum, 2) for unigram in unigram_counter}
    for word in unigram_counter:
        unigram_p[(word,)] = math.log(float(unigram_counter[word])/ unigram_sum, 2)

    ##### BIGRAMS
    unigram_tokenized_sentences_with_start = []
    bigram_tokenized_sentences = []
    for sentence in training_corpus:
        split_sentence = sentence.strip().split()
        split_sentence.insert(0, START_SYMBOL)
        split_sentence.append(STOP_SYMBOL)

        # we want to create a list of all bigrams
        bigram_tokenized_sentences.extend(list(nltk.bigrams(split_sentence)))

        # but also a list of all unigrams, this time with START_SYMBOL
        unigram_tokenized_sentences_with_start.extend(split_sentence)

    # count up all the unigrams, this time with START_SYMBOL
    unigram_counter_with_start = collections.Counter(unigram_tokenized_sentences_with_start)

    # count up all the bigrams, and their sum
    bigram_counter = collections.Counter(bigram_tokenized_sentences)
    bigram_sum = sum(bigram_counter.itervalues())

    # compute all the bigram probabilities
    for bigram in bigram_counter:
        bigram_p[bigram] = math.log(bigram_counter[bigram] / unigram_counter_with_start[bigram[0]], 2)

    ##### TRIGRAMS
    # split each sentence into tokens
    bigram_tokenized_sentences_with_start = []
    trigram_tokenized_sentences = []
    for sentence in training_corpus:
        split_sentence = sentence.strip().split()
        split_sentence.insert(0, START_SYMBOL)
        split_sentence.insert(0, START_SYMBOL)
        split_sentence.append(STOP_SYMBOL)
        trigram_tokenized_sentences.extend(list(nltk.trigrams(split_sentence)))
        bigram_tokenized_sentences_with_start.extend(list(nltk.bigrams(split_sentence)))

    bigram_counter_with_start = collections.Counter(bigram_tokenized_sentences_with_start)
    trigram_counter = collections.Counter(trigram_tokenized_sentences)
    trigram_sum = sum(trigram_counter.itervalues())

    # compute all the trigram probabilities
    for trigram in trigram_counter:
        trigram_p[trigram] = math.log(trigram_counter[trigram] / bigram_counter_with_start[(trigram[0], trigram[1])], 2)

    return unigram_p, bigram_p, trigram_p

# Prints the output for q1
# Each input is a python dictionary where keys are a tuple expressing the ngram, and the value is the log probability of that ngram
def q1_output(unigrams, bigrams, trigrams, filename):
    # output probabilities
    outfile = open(filename, 'w')

    unigrams_keys = unigrams.keys()
    unigrams_keys.sort()
    for unigram in unigrams_keys:
        outfile.write('UNIGRAM ' + unigram[0] + ' ' + str(unigrams[unigram]) + '\n')

    bigrams_keys = bigrams.keys()
    bigrams_keys.sort()
    for bigram in bigrams_keys:
        outfile.write('BIGRAM ' + bigram[0] + ' ' + bigram[1]  + ' ' + str(bigrams[bigram]) + '\n')

    trigrams_keys = trigrams.keys()
    trigrams_keys.sort()
    for trigram in trigrams_keys:
        outfile.write('TRIGRAM ' + trigram[0] + ' ' + trigram[1] + ' ' + trigram[2] + ' ' + str(trigrams[trigram]) + '\n')

    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Calculates scores (log probabilities) for every sentence
# ngram_p: python dictionary of probabilities of uni-, bi- and trigrams.
# n: size of the ngram you want to use to compute probabilities
# corpus: list of sentences to score. Each sentence is a string with tokens separated by spaces, ending in a newline character.
# This function must return a python list of scores, where the first element is the score of the first sentence, etc.
def score(ngram_p, n, corpus):
    scores = []
    if(n == 1):
        for sentence in corpus: # iterate through the list of sentences
            split_sentence = sentence.strip().split() # strip and split by whitespace
            split_sentence.append(STOP_SYMBOL)
            sum_probability = 0
            for word in split_sentence: # for every word in the split sentence
                if (word,) in ngram_p: # if the unigram exists in dictionary
                    sum_probability += ngram_p[(word,)] # add its probability to the sum
                else:
                    sum_probability = MINUS_INFINITY_SENTENCE_LOG_PROB # the sentence does not occur
                    break
            scores.append(sum_probability) # store the sentence's probability in scores
    elif(n == 2):
        for sentence in corpus:
            split_sentence = sentence.strip().split()
            split_sentence.insert(0, START_SYMBOL)
            split_sentence.append(STOP_SYMBOL)
            bigrams = list(nltk.bigrams(split_sentence))
            sum_probability = 0
            for bigram in bigrams:
                if bigram in ngram_p:
                    sum_probability += ngram_p[bigram]
                else:
                    sum_probability = MINUS_INFINITY_SENTENCE_LOG_PROB
                    break
            scores.append(sum_probability)
    elif(n == 3):
        for sentence in corpus:
            split_sentence = sentence.strip().split()
            split_sentence.insert(0, START_SYMBOL)
            split_sentence.insert(0, START_SYMBOL)
            split_sentence.append(STOP_SYMBOL)
            trigrams = list(nltk.trigrams(split_sentence))
            sum_probability = 0
            for trigram in trigrams:
                if trigram in ngram_p:
                    sum_probability += ngram_p[trigram]
                else:
                    sum_probability = MINUS_INFINITY_SENTENCE_LOG_PROB
                    break
            scores.append(sum_probability)
    else:
        return False
    # return the list of probabilities, one for each sentence
    return scores

# Outputs a score to a file
# scores: list of scores
# filename: is the output file name
def score_output(scores, filename):
    outfile = open(filename, 'w')
    for score in scores:
        outfile.write(str(score) + '\n')
    outfile.close()

# TODO: IMPLEMENT THIS FUNCTION
# Calculates scores (log probabilities) for every sentence with a linearly interpolated model
# Each ngram argument is a python dictionary where the keys are tuples that express an ngram and the value is the log probability of that ngram
# Like score(), this function returns a python list of scores
def linearscore(unigrams, bigrams, trigrams, corpus):
    lambda1 = lambda2 = lambda3 = float(1/3)
    scores = []
    for sentence in corpus:
        # split into tokens
        split_sentence = sentence.strip().split()
        split_sentence.insert(0, START_SYMBOL)
        split_sentence.insert(0, START_SYMBOL)
        split_sentence.append(STOP_SYMBOL)

        # turn tokens into trigrams
        trigram_list = list(nltk.trigrams(split_sentence))

        # add up all the log probabilities
        sum_probability = 0
        for trigram in trigram_list:

            if trigram not in trigrams and bigram not in bigrams and unigram not in unigrams:
                sum_probability = MINUS_INFINITY_SENTENCE_LOG_PROB
            else:

                # find the trigram prob
                tri_log = 0
                if trigram in trigrams:
                    tri_log = trigrams[trigram]

                # find the bigram prob
                bi_log = 0
                bigram = (trigram[1], trigram[2])
                if bigram in bigrams:
                    bi_log = bigrams[bigram]

                # find the unigram prob
                uni_log = 0
                unigram = (trigram[2],)
                if unigram in unigrams:
                    uni_log = unigrams[unigram]

                # add the prob for the total sentence
                sum_probability += math.log((lambda1 * (2 ** uni_log)) + (lambda2 * (2 ** bi_log)) + (lambda3 * (2 ** tri_log)), 2)

        # append the probability for the sentence
        scores.append(sum_probability)

    # return the probabilities for each sentence
    return scores

DATA_PATH = '/home/classes/cs477/data/' # absolute path to use the shared data
OUTPUT_PATH = 'output/'

# DO NOT MODIFY THE MAIN FUNCTION
def main():
    # start timer
    time.clock()

    # get data
    infile = open(DATA_PATH + 'Brown_train.txt', 'r')
    corpus = infile.readlines() # SARIM: a [list] of "str"
    infile.close()

    # calculate ngram probabilities (question 1)
    unigrams, bigrams, trigrams = calc_probabilities(corpus)

    # question 1 output
    q1_output(unigrams, bigrams, trigrams, OUTPUT_PATH + 'A1.txt')

    # score sentences (question 2)
    uniscores = score(unigrams, 1, corpus)
    biscores = score(bigrams, 2, corpus)
    triscores = score(trigrams, 3, corpus)

    # question 2 output
    score_output(uniscores, OUTPUT_PATH + 'A2.uni.txt')
    score_output(biscores, OUTPUT_PATH + 'A2.bi.txt')
    score_output(triscores, OUTPUT_PATH + 'A2.tri.txt')

    # linear interpolation (question 3)
    linearscores = linearscore(unigrams, bigrams, trigrams, corpus)

    # question 3 output
    score_output(linearscores, OUTPUT_PATH + 'A3.txt')

    # open Sample1 and Sample2 (question 5)
    infile = open(DATA_PATH + 'Sample1.txt', 'r')
    sample1 = infile.readlines()
    infile.close()
    infile = open(DATA_PATH + 'Sample2.txt', 'r')
    sample2 = infile.readlines()
    infile.close()

    # score the samples
    sample1scores = linearscore(unigrams, bigrams, trigrams, sample1)
    sample2scores = linearscore(unigrams, bigrams, trigrams, sample2)

    # question 5 output
    score_output(sample1scores, OUTPUT_PATH + 'Sample1_scored.txt')
    score_output(sample2scores, OUTPUT_PATH + 'Sample2_scored.txt')

    # print total time to run Part A
    print("Part A time: " + str(time.clock()) + ' sec')

if __name__ == "__main__": main()
