from __future__ import division # had to be included because its always unclear whether we use py2 or 3!

import sys
import nltk
import math
import time

import collections
import pprint
pp = pprint.PrettyPrinter(indent=4)

START_SYMBOL = '*'
STOP_SYMBOL = 'STOP'
RARE_SYMBOL = '_RARE_'
RARE_WORD_MAX_FREQ = 5
LOG_PROB_OF_ZERO = -1000


# TODO: IMPLEMENT THIS FUNCTION
# Receives a list of tagged sentences and processes each sentence to generate a list of words and a list of tags.
# Each sentence is a string of space separated "WORD/TAG" tokens, with a newline character in the end.
# Remember to include start and stop symbols in yout returned lists, as defined by the constants START_SYMBOL and STOP_SYMBOL.
# brown_words (the list of words) should be a list where every element is a list of the tags of a particular sentence.
# brown_tags (the list of tags) should be a list where every element is a list of the tags of a particular sentence.
def split_wordtags(brown_train):
    brown_words = []
    brown_tags = []

    for sentence in brown_train:
        # prepare two separate lists
        words = []
        tags = []
        words.append(START_SYMBOL)
        words.append(START_SYMBOL)
        tags.append(START_SYMBOL)
        tags.append(START_SYMBOL)

        # split the sentence by whitespace
        split_sentence = sentence.strip().split()

        # consider each token
        for token in split_sentence:
            # split by "/"
            split_token = token.split("/")

            # the last item is the tag
            tag = split_token[-1]
            tags.append(tag)

            # remove the last item
            split_token.pop()

            # join and add the rest
            word = "/".join(split_token)
            words.append(word)

        # add end symbols
        words.append(STOP_SYMBOL)
        tags.append(STOP_SYMBOL)

        # append the two lists
        brown_words.append(words)
        brown_tags.append(tags)

    return brown_words, brown_tags

# TODO: IMPLEMENT THIS FUNCTION
# This function takes tags from the training data and calculates tag trigram probabilities.
# It returns a python dictionary where the keys are tuples that represent the tag trigram, and the values are the log probability of that trigram
def calc_trigrams(brown_tags):
    q_values = {}
    # have a list of bigrams
    bigram_tokenized_sentences = []
    # have a list of trigrams
    trigram_tokenized_sentences = []
    # iterate over the sentences
    for split_sentence in brown_tags:
        # create a list of trigrams and append to total list
        trigram_tokenized_sentences.extend(list(nltk.trigrams(split_sentence)))
        # create a list of bigrams and append to total list
        bigram_tokenized_sentences.extend(list(nltk.bigrams(split_sentence)))
    # count up the bigrams
    bigram_counter = collections.Counter(bigram_tokenized_sentences)
    # count up the trigrams
    trigram_counter = collections.Counter(trigram_tokenized_sentences)
    # sum up all the trigrams
    trigram_sum = sum(trigram_counter.itervalues())
    # compute the trigram probabilities
    for trigram in trigram_counter:
        tri_count = trigram_counter[trigram]
        bi_count = bigram_counter[(trigram[0], trigram[1])]
        q_values[trigram] = math.log(tri_count / bi_count, 2)
    # return the dictionary of log probabilities
    return q_values

#TODO: IMPLEMENT THIS FUNCTION
# This function takes tags from the training data and calculates the tag trigrams in reverse.  In other words, instead of looking at the probabilities that the third tag follows the first two, look at the probabilities of the first tag given the next two.
# Hint: This code should only differ slightly from calc_trigrams(brown_tags)
def calc_trigrams_reverse(brown_tags):
    q_values = {}
    # have a list of bigrams
    bigram_tokenized_sentences = []
    # have a list of trigrams
    trigram_tokenized_sentences = []
    # iterate over the sentences
    for split_sentence in brown_tags:
        # create a list of trigrams and append to total list
        trigram_tokenized_sentences.extend(list(nltk.trigrams(split_sentence[::-1])))
        # create a list of bigrams and append to total list
        bigram_tokenized_sentences.extend(list(nltk.bigrams(split_sentence[::-1])))
    # count up the bigrams
    bigram_counter = collections.Counter(bigram_tokenized_sentences)
    # count up the trigrams
    trigram_counter = collections.Counter(trigram_tokenized_sentences)
    # sum up all the trigrams
    trigram_sum = sum(trigram_counter.itervalues())
    # compute the trigram probabilities
    for trigram in trigram_counter:
        tri_count = trigram_counter[trigram]
        bi_count = bigram_counter[(trigram[0], trigram[1])]
        q_values[trigram] = math.log(tri_count / bi_count, 2)
    # return the dictionary of log probabilities
    return q_values

# This function takes output from calc_trigrams() and outputs it in the proper format
def q2_output(q_values, filename):
    outfile = open(filename, "w")
    trigrams = q_values.keys()
    trigrams.sort()
    for trigram in trigrams:
        output = " ".join(['TRIGRAM', trigram[0], trigram[1], trigram[2], str(q_values[trigram])])
        outfile.write(output + '\n')
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Takes the words from the training data and returns a set of all of the words that occur more than 5 times (use RARE_WORD_MAX_FREQ)
# brown_words is a python list where every element is a python list of the words of a particular sentence.
# Note: words that appear exactly 5 times should be considered rare!
def calc_known(brown_words):
    known_words = set([])

    # create a word dump
    word_list = []
    for sentence in brown_words:
        word_list.extend(sentence)

    # count the dump
    word_counter = collections.Counter(word_list)

    # get desired words
    for word in word_counter.items():
        if word[1] > RARE_WORD_MAX_FREQ:
            known_words.add(word[0])

    return known_words

# TODO: IMPLEMENT THIS FUNCTION
# Takes the words from the training data and a set of words that should not be replaced for '_RARE_'
# Returns the equivalent to brown_words but replacing the unknown words by '_RARE_' (use RARE_SYMBOL constant)
def replace_rare(brown_words, known_words):
    brown_words_rare = []
    for sentence in brown_words:
        new_sentence = []
        for token in sentence:
            if token in known_words:
                new_sentence.append(token)
            else:
                new_sentence.append(RARE_SYMBOL)
        brown_words_rare.append(new_sentence)

    return brown_words_rare

# This function takes the ouput from replace_rare and outputs it to a file
def q3_output(rare, filename):
    outfile = open(filename, 'w')
    for sentence in rare:
        outfile.write(' '.join(sentence[2:-1]) + '\n')
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Calculates emission probabilities and creates a set of all possible tags
# The first return value is a python dictionary where each key is a tuple in which the first element is a word
# and the second is a tag, and the value is the log probability of the emission of the word given the tag
# The second return value is a set of all possible tags for this data set
def calc_emission(brown_words_rare, brown_tags):
    e_values = {}

    word_tags = []
    tags = []
    for word_sentence, tag_sentence in zip(brown_words_rare, brown_tags):
        word_tags.extend(zip(word_sentence, tag_sentence))
        tags.extend(tag_sentence)

    # count up all the (word, tag) pairs
    word_tag_counter = collections.Counter(word_tags)

    # count up all the tags
    tag_counter = collections.Counter(tags)

    # the probability of an emission is the Count((word, tag)) / Count(, tag)
    for word_tag in word_tag_counter:
        e_values[word_tag] = math.log(word_tag_counter[word_tag] / tag_counter[word_tag[1]], 2)

    taglist = set([])
    # iterate over the tags
    for sentence in brown_tags:
        # add to the set only unique tags
        for token in sentence:
            if token not in taglist:
                taglist.add(token)

    return e_values, taglist

# This function takes the output from calc_emissions() and outputs it
def q4_output(e_values, filename):
    outfile = open(filename, "w")
    emissions = e_values.keys()
    emissions.sort()
    for item in emissions:
        output = " ".join([item[0], item[1], str(e_values[item])])
        outfile.write(output + '\n')
    outfile.close()

# TODO: IMPLEMENT THIS FUNCTION
# This function takes data to tag (brown_dev_words), a set of all possible tags (taglist), a set of all known words (known_words),
# trigram probabilities (q_values) and emission probabilities (e_values) and outputs a list where every element is a tagged sentence
# (in the WORD/TAG format, separated by spaces and with a newline in the end, just like our input tagged data)
# brown_dev_words is a python list where every element is a python list of the words of a particular sentence.
# taglist is a set of all possible tags
# known_words is a set of all known words
# q_values is from the return of calc_trigrams()
# e_values is from the return of calc_emissions()
# The return value is a list of tagged sentences in the format "WORD/TAG", separated by spaces. Each sentence is a string with a
# terminal newline, not a list of tokens. Remember also that the output should not contain the "_RARE_" symbol, but rather the
# original words of the sentence!
def viterbi(brown_dev_words, taglist, known_words, q_values, e_values):
    """
    inspired by Piazza post: video on YouTube by Arnaldo Figueira
    """
    # a list of tagged sentences
    tagged = []

    for split_sentence in brown_dev_words:
        ############ INITIALIZATION
        pi = collections.defaultdict(float) # pi[(k, u, v)]: max prob of tag sequence u, v at position k
        bp = {} # bp[(k, u, v)]: backpointers i.e. the argmax of pi[(k, u, v)]
        pi[(0, START_SYMBOL, START_SYMBOL)] = 1.0 # the base case
        split_sentence_refine = [word if word in known_words else RARE_SYMBOL for word in split_sentence]
        n = len(split_sentence_refine)
        ############ VITERBI
        for k in range(1, n + 1):
            index = k - 1
            ######### STORE WORDS
            v_word = split_sentence_refine[index]
            v_tags = [tag for tag in taglist if (v_word, tag) in e_values]
            u_tags = [START_SYMBOL]
            if index > 0:
                u_word = split_sentence_refine[index - 1]
                u_tags = [tag for tag in taglist if (u_word, tag) in e_values]
            w_tags = [START_SYMBOL]
            if index > 1:
                w_word = split_sentence_refine[index - 2]
                w_tags = [tag for tag in taglist if (w_word, tag) in e_values]
            ######### BEGIN ITERATION
            # iterate over the tags available
            for u_tag in u_tags:
                for v_tag in v_tags:
                    best_prob = float('-Inf')
                    best_tag = None
                    for w_tag in w_tags:
                        if (v_word, v_tag) in e_values:
                            total_prob = (
                                pi.get((k - 1, w_tag, u_tag), LOG_PROB_OF_ZERO) +
                                q_values.get((w_tag, u_tag, v_tag), LOG_PROB_OF_ZERO) +
                                e_values.get((v_word, v_tag)) # (THE, DET) == -4.23
                            )
                            if total_prob > best_prob:
                                best_prob = total_prob
                                best_tag = w_tag
                    pi[(k, u_tag, v_tag)] = best_prob
                    bp[(k, u_tag, v_tag)] = best_tag
        ######### DO THE SAME FOR STOP
        best_prob = float('-Inf')
        best_u_tag = None
        best_v_tag = None
        # collect tags
        u_word = split_sentence_refine[n - 2]
        u_tags = [tag for tag in taglist if (u_word, tag) in e_values]
        v_word = split_sentence_refine[n - 1]
        v_tags = [tag for tag in taglist if (v_word, tag) in e_values]
        # find best probs
        for u_tag in u_tags:
            for v_tag in v_tags:
                total_prob = (
                        pi.get((n, u_tag, v_tag), LOG_PROB_OF_ZERO) +
                        q_values.get((u_tag, v_tag, STOP_SYMBOL), LOG_PROB_OF_ZERO)
                    )
                if total_prob > best_prob:
                    best_prob = total_prob
                    best_u_tag = u_tag
                    best_v_tag = v_tag
        ######## RECOVER SENTENCE FROM BACKPOINTERS
        tagged_sentence = []
        tagged_sentence.append(best_v_tag)
        tagged_sentence.append(best_u_tag)
        # build up the tags
        for i, k in enumerate(range(n - 2, 0, -1)):
            tagged_sentence.append(bp[(k + 2, tagged_sentence[i + 1], tagged_sentence[i])])
        tagged_sentence.reverse()
        # build up the full sentence
        full_sentence = []
        for word_index in range(0, n):
            full_sentence.append(split_sentence[word_index] + '/' + tagged_sentence[word_index])
        full_sentence.append('\n')
        tagged.append(' '.join(full_sentence))

    return(tagged)

# This function takes the output of viterbi() and outputs it to file
def q5_output(tagged, filename):
    outfile = open(filename, 'w')
    for sentence in tagged:
        outfile.write(sentence)
    outfile.close()

# TODO: IMPLEMENT THIS FUNCTION
# This function uses nltk to create the taggers described in question 6
# brown_words and brown_tags is the data to be used in training
# brown_dev_words is the data that should be tagged
# The return value is a list of tagged sentences in the format "WORD/TAG", separated by spaces. Each sentence is a string with a
# terminal newline, not a list of tokens.
def nltk_tagger(brown_words, brown_tags, brown_dev_words):
    # Hint: use the following line to format data to what NLTK expects for training
    training = [ zip(brown_words[i],brown_tags[i]) for i in xrange(len(brown_words)) ]

    tagged = []

    default_tagger = nltk.DefaultTagger('NOUN')
    bigram_tagger = nltk.BigramTagger(training, backoff=default_tagger)
    trigram_tagger = nltk.TrigramTagger(training, backoff=bigram_tagger)

    # use the taggers to tag the sentence
    for split_sentence in brown_dev_words:
        tagged_sentence = []
        for word, tag in trigram_tagger.tag(split_sentence):
            tagged_sentence.append(word + '/' + tag)
        tagged_sentence.append('\n')
        tagged.append(' '.join(tagged_sentence))

    return tagged

# This function takes the output of nltk_tagger() and outputs it to file
def q6_output(tagged, filename):
    outfile = open(filename, 'w')
    for sentence in tagged:
        outfile.write(sentence)
    outfile.close()

DATA_PATH = '/home/classes/cs477/data/'
OUTPUT_PATH = 'output/'

def main():
    # start timer
    time.clock()

    # open Brown training data
    infile = open(DATA_PATH + "Brown_tagged_train.txt", "r")
    brown_train = infile.readlines()
    infile.close()

    # split words and tags, and add start and stop symbols (question 1)
    brown_words, brown_tags = split_wordtags(brown_train)

    # calculate tag trigram probabilities with an option to reverse (question 7)
    if len(sys.argv) > 1 and sys.argv[1] == "-reverse":
        q_values = calc_trigrams_reverse(brown_tags)
    else:
        q_values = calc_trigrams(brown_tags)

    # question 2 output
    q2_output(q_values, OUTPUT_PATH + 'B2.txt')

    # calculate list of words with count > 5 (question 3)
    known_words = calc_known(brown_words)

    # get a version of brown_words with rare words replace with '_RARE_' (question 3)
    brown_words_rare = replace_rare(brown_words, known_words)

    # question 3 output
    q3_output(brown_words_rare, OUTPUT_PATH + "B3.txt")

    # calculate emission probabilities (question 4)
    e_values, taglist = calc_emission(brown_words_rare, brown_tags)

    # question 4 output
    q4_output(e_values, OUTPUT_PATH + "B4.txt")

    # delete unneceessary data
    del brown_train
    del brown_words_rare

    # open Brown development data (question 5)
    infile = open(DATA_PATH + "Brown_dev.txt", "r")
    brown_dev = infile.readlines()
    infile.close()

    # format Brown development data here
    brown_dev_words = []
    for sentence in brown_dev:
        brown_dev_words.append(sentence.split(" ")[:-1])

    # do viterbi on brown_dev_words (question 5)
    viterbi_tagged = viterbi(brown_dev_words, taglist, known_words, q_values, e_values)

    # question 5 output
    q5_output(viterbi_tagged, OUTPUT_PATH + 'B5.txt')

    # do nltk tagging here
    nltk_tagged = nltk_tagger(brown_words, brown_tags, brown_dev_words)

    # question 6 output
    q6_output(nltk_tagged, OUTPUT_PATH + 'B6.txt')

    # print total time to run Part B
    print "Part B time: " + str(time.clock()) + ' sec'

if __name__ == "__main__": main()
