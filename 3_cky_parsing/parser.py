import nltk

from collections import defaultdict
from nltk.ccg.chart import printCCGDerivation
from nltk.ccg.lexicon import Token

from utils import *

class CCGParser:

	""" Parse CCG according to the CKY algorithm. """

	DEFAULT_START = AtomicCategory("S")
	DEFAULT_RULE_SET = []

	def __init__(self, lexicon, rules=DEFAULT_RULE_SET):
		self.lexicon = lexicon
		self.rules = rules

	@staticmethod
	@rule(DEFAULT_RULE_SET, 2, "A")
	def application(cat1, cat2):
		"""
		Implements the combination rule for function application.
		If cat1 and cat2 can be left or right-applied to each other (assuming cat1 is left of cat2),
		return the resulting category. Otherwise, return None.
		Hints:
			* isinstance(cat, CombinedCategory) tells you whether a category is combined or atomic.
			* For a combined category, cat.left and cat.right give the left and right subcategories.
			* For a combined category, cat.direction is either "\\" or "/".
			* You can use == to check whether categories are the same
		"""
		# forward function application
		if (isinstance(cat1, CombinedCategory) 
			and cat1.direction == "/" 
			and cat1.right == cat2):
				return cat1.left
		# backward function application
		elif (isinstance(cat2, CombinedCategory) 
			and cat2.direction == "\\" 
			and cat2.right == cat1):
				return cat2.left
		else:
			return None

	@staticmethod
	@rule(DEFAULT_RULE_SET, 2, "C")
	def composition(cat1, cat2):
		"""
		Implements the combination rule for function composition.
		If cat1 and cat2 can be left or right-composed, return the resulting category.
		Otherwise, return None.
		"""
		if(isinstance(cat1, CombinedCategory) and isinstance(cat2, CombinedCategory)):
			if cat1.direction == "/" and cat1.right == cat2.left: 
				return CombinedCategory(cat1.left, cat2.direction, cat2.right)
			elif cat2.direction == "\\" and cat2.right == cat1.left:
				return CombinedCategory(cat2.left, cat1.direction, cat1.right)
			else:
				return None
		else:
			return None
	
	@staticmethod
	@rule(DEFAULT_RULE_SET, 2, "C2")
	def composition2(cat1, cat2):
		if(isinstance(cat1, CombinedCategory) and isinstance(cat2, CombinedCategory)):
			if cat1.direction == "/" and isinstance(cat2.left, CombinedCategory) and cat1.right == cat2.left.left:
				left_part = CombinedCategory(cat1.left, cat2.left.direction, cat2.left.right)
				return CombinedCategory(left_part, cat2.direction, cat2.right)
			if cat2.direction == "\\" and isinstance(cat1.left, CombinedCategory) and cat1.left.left == cat2.right:
				left_part = CombinedCategory(cat2.left, cat1.left.direction, cat1.left.right)
				return CombinedCategory(left_part, cat1.direction, cat1.right)
		return None

	@staticmethod
	@rule(DEFAULT_RULE_SET, 1, "T")
	def typeRaising(cat1, cat2):
		"""
		Implements the combination rule for type raising.
		If cat2 satisfies the type-raising constraints, type-raise cat1 (and vice-versa).
		Return value when successful should be the tuple (cat, dir):
			* cat is the resulting category of the type-raising
			* dir is "/" or "\\" and represents the direction of the raising
			* If no type-raising is possible, return None
		Hint: use cat.innermostFunction() to implement the conditional checks described in the
			specification.
		"""
		# forward type raising
		if(isinstance(cat1, AtomicCategory) 
			and isinstance(cat2, CombinedCategory)
			# innermost is guaranteed to be a CombinedCategory
			# we get the argument of innermost and check if it is atomic
			and isinstance(cat2.innermostFunction().right, AtomicCategory)
			and cat2.innermostFunction().direction == "\\"):
			finalCat = CombinedCategory(
				cat2.innermostFunction().left, 
				"/", 
				CombinedCategory(cat2.innermostFunction().left, "\\", cat1))
			finalDir = finalCat.direction
			return(finalCat, finalDir)
		# backward type raising
		elif(isinstance(cat2, AtomicCategory)
			and isinstance(cat1, CombinedCategory)
			and isinstance(cat1.innermostFunction().right, AtomicCategory)
			and cat1.innermostFunction().direction == "/"):
			finalCat = CombinedCategory(
				cat1.innermostFunction().left,
				"\\",
				CombinedCategory(cat1.innermostFunction().left, "/", cat2))
			finalDir = finalCat.direction
			return(finalCat, finalDir)
		else:
			return None

	class VocabException(Exception):
		pass
			
	def fillParseChart(self, tokens):
		"""
		Builds and fills in a CKY parse chart for the sentence represented by tokens.
		The argument tokens is a list of words in the sentence.
		Each entry in the chart should be a list of Constituents:
			* Use AtomicConstituent(cat, word) to construct initialize Constituents of words.
			* Use CombinedConstituent(cat, [leftPtr, rightPtr], rule) to construct Constituents
			  produced by rules. leftPtr and rightPtr are the Constituent objects that combined to
			  form the new Constituent, and rule should be the rule object itself.
			* For a unary rule (type-raising), you should do: CombinedConstituent(cat, [ptr], rule)
		Should return (chart, parses), where parses is the final (top right) entry in the chart. 
		Each tuple in parses corresponds to a parse for the sentence.
		Hint: initialize the diagonal of the chart by looking up each token in the lexicon and then
			use self.rules to fill in the rest of the chart. Rules in self.rules are sorted by
			increasing arity (unary or binary), and you can use rule.arity to check the arity of a
			rule.
		"""
		chart = defaultdict(list)
		# INITIALIZATION
		index_i = 0
		index_j = 1
		for token in tokens:
			lookup = self.lexicon.getCategories(token)
			if lookup is not None:
				for category in lookup:
					constituent = AtomicConstituent(category, token)
					chart[index_i, index_j].append(constituent)
			index_i += 1
			index_j += 1
		# FILL THE CHART
		# start from the base of each column and move upward
		for column in range(2, len(tokens) + 1):
			for balloon in range(column - 1, -1, - 1):
				# generate binarized splits of (balloon, column)
				# for example: (0, 2) == [(0, 1), (1, 2)]
				# but only for differences greater than 1 (so we avoid the original diagonal)
				if (column - balloon) > 1:
					for split_range in createRanges((balloon, column)):
						# check if the previous results can be combined
						for rule in self.rules:
							if rule.arity == 1:
								for left_constituent in chart[split_range[0]]:
									for right_constituent in chart[split_range[1]]:
										new_cat = rule(left_constituent.cat, right_constituent.cat)
										if new_cat is not None:
											if new_cat[1] == "/":
												new_constituent = CombinedConstituent(new_cat[0],
																[left_constituent], rule)
												chart[(split_range[0])].append(new_constituent)
											else:
												new_constituent = CombinedConstituent(new_cat[0],
																[right_constituent], rule)
												chart[(split_range[1])].append(new_constituent)
							if rule.arity == 2:
								for left_constituent in chart[split_range[0]]:
									for right_constituent in chart[split_range[1]]:
										combined_cat = rule(left_constituent.cat, right_constituent.cat)
										if combined_cat is not None:
											combined_constituent = CombinedConstituent(combined_cat,
																[left_constituent, right_constituent], rule)
											chart[(balloon, column)].append(combined_constituent)
		parses = chart[0, len(tokens)]
		return (chart, parses)

	@staticmethod
	def generateParseTree(cons, chart):
		"""
		Helper function that returns an NLTK Tree object representing a parse.
		"""
		token = Token(None, cons.cat, None)
		if isinstance(cons, AtomicConstituent):
			return nltk.tree.Tree(
				(token, u"Leaf"),
				[nltk.tree.Tree(token, [cons.word])]
			)
		else:
			if cons.rule == CCGParser.typeRaising:
				return nltk.tree.Tree(
					(token, cons.rule.name),
					[CCGParser.generateParseTree(cons.ptrs[0], chart)]
				)
			else:
				return nltk.tree.Tree(
					(token, cons.rule.name),
					[CCGParser.generateParseTree(cons.ptrs[0], chart),
                                            CCGParser.generateParseTree(cons.ptrs[1], chart)]
				)

	def getParseTrees(self, tokens):
		"""
		Reconstructs parse trees for the sentences by following backpointers.
		"""
		chart, parses = self.fillParseChart(tokens)
		for cons in parses:
			yield CCGParser.generateParseTree(cons, chart)

	def accepts(self, tokens, sentCat=DEFAULT_START):
		"""
		Return True iff the sentence represented by tokens is in this language (i.e. has at least
			one valid parse).
		"""
		_, parses = self.fillParseChart(tokens)
		for cons in parses:
			if cons.cat == sentCat:
				return True
		return False

def createRanges(tuple):
		# (0, 2) == [ [(0, 1), (1, 2)],  ]
		i = tuple[0]
		rangeList = []
		while(i < tuple[1] - 1):
			rangeList.append([(tuple[0], i + 1), (i + 1, tuple[1])])
			i += 1
		return rangeList
