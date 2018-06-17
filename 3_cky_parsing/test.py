from utils import *
from nltk.ccg.chart import printCCGDerivation
from parser import CCGParser
from subprocess import call
import argparse, os, sys

def accept(sent, lexicon):
	""" True iff the grammar from lexicon.ccg can parse sent. """
	with open(lexicon + ".ccg") as fh:
		parser = CCGParser(Lexicon(fh.read()))
		return parser.accepts(sent)

def parse(sent, lexicon):
	""" Returns iterator of parses for sent w.r.t. grammar in lexicon.ccg. """
	with open(lexicon + ".ccg") as fh:
		parser = CCGParser(Lexicon(fh.read()))
		return parser.getParseTrees(sent)

if __name__ ==  "__main__":

	parser = argparse.ArgumentParser(description="Test script for Hwk2.")

	parser.add_argument("--accept", type=str, help="check whether a grammar accepts stdin")
	parser.add_argument("--parse", type=str, help="print parses for stdin according to a grammar")
	parser.add_argument("--grade", action='store_true', help="run the autograder")
	parser.add_argument("-c", action="store_true", help="tokenize by character instead of words")

	args = parser.parse_args()

	if args.accept:
		raw = sys.stdin.read().strip()
		sent = [c for c in raw] if args.c else raw.split()
		print accept(sent, args.accept)

	if args.parse:
		sent = sys.stdin.read().split()
		for i, p in enumerate(parse(sent, args.parse)):
			print "===== Parse Sequence #{} =====".format(i + 1)
			printCCGDerivation(p)
			print "\n\n"

	if args.grade:
		os.chdir("autograder")
		call("python autograder.py".split())
