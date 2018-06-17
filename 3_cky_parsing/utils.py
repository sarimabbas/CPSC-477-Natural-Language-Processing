import re
from collections import defaultdict

"""
Written for Yale CPSC 477 (Natural Language Processing), Spring 2018
@author Will Merrill and Jo-Jo Feng
"""

COMMENTS_RE = re.compile('''([^#]*)(?:#.*)?''')

class Lexicon:
	def __init__(self, lex_str):
		self.categories = {}  # maps from string to AtomicCategory object
		self.entries = defaultdict(list) # maps from token to Category object
		# helper function to parse cat_str and at it (and its contained tokens) to categories
		def parseCategory(cat_str):
			SYMBOLS = set(["(", ")", "/", "\\"])
			stack = [[]]
			index = 0
			while index < len(cat_str):
				char = cat_str[index]
				if char == "(":  # append empty list to stack
					stack.append([])
				elif char == ")":  # take single token from list on top of stack and push it onto next list
					stack[-2].append(stack.pop()[0])
				elif char == "/" or char == "\\":  # append to list on top of stack
					stack[-1].append(char)
				else:  # a character
					str_start = index  # keep track of beginning of token
					while index + 1 < len(cat_str) and cat_str[index + 1] not in SYMBOLS:
						index += 1
					primitive = cat_str[str_start:index + 1]
					if primitive in families:
						cat = families[primitive]
					elif primitive in self.categories:
						cat = self.categories[primitive]
					else:
						self.categories[primitive] = AtomicCategory(primitive)
						cat = self.categories[primitive]
					stack[-1].append(cat)
				# check length of top of stack, if 3, smash it into a CombinedCategory
				if len(stack[-1]) == 3:
						stack[-1] = [CombinedCategory(stack[-1][0], stack[-1][1], stack[-1][2])]
				index += 1
			return stack[0][0]  # returns final object
		families = {}
		for line in lex_str.splitlines():
			line = COMMENTS_RE.match(line).groups()[0].strip()
			if line == "":
				continue
			if line.startswith(':-'):
				# A line of primitive categories.
				# The first one is the target category
				# ie, :- S, N, NP, VP
				for prim in [prim.strip() for prim in line[2:].strip().split(',')]:
					self.categories[prim] = AtomicCategory(prim)
			else:
				# Family definition or word definitions
				left, sep, right = line.split()
				# Define new family
				if sep == '::':
					families[left] = parseCategory(right)
				else:  # define a new word
					self.entries[left].append(parseCategory(right))
		self.entries = dict(self.entries)
	def getAllCategories(self):
		return self.categories
	def getAllEntries(self):
		return self.entries
	def getCategories(self, token):
		return self.entries[token] if token in self.entries else None

class Category(object):
	""" Abstract class for syntactic categories. """
	def __len__(self):
		return len(repr(self))
	def __str__(self):
		return repr(self)

class AtomicCategory(Category):
	""" Class for atomic syntactic categories. """
	def __init__(self, name):
		self.name = name
	def __eq__(self, other):
		if not isinstance(other, AtomicCategory):
			return False
		return self.name == other.name
	def __repr__(self):
		return self.name
	def innermostFunction(self):
		return None

class CombinedCategory(Category):
	""" Syntactic category that is a combination of two other categories. """
	def __init__(self, left, direction, right):
		assert direction == "/" or direction == "\\"
		assert isinstance(left, Category) and isinstance(right, Category)
		self.left = left
		self.direction = direction
		self.right = right
	def __eq__(self, other):
		if not isinstance(other, CombinedCategory) or self.direction != other.direction:
			return False
		return self.left == other.left and self.right == other.right
	def __repr__(self):
		return "({} {} {})".format(self.left, self.direction, self.right)
	def innermostFunction(self):
		"""
		Retrives the left-most functional category.
		For example: (N\N)/(S/NP) => N\N.
		"""
		while isinstance(self.left, CombinedCategory):
			self = self.left
		return self

class rule(object):

	""" Function decorator for rules. """

	def __init__(self, ruleset, arity, name):
		self.ruleset = ruleset
		self.arity = arity
		self.name = name

	def __call__(self, f):
		f.arity = self.arity
		f.name = self.name
		self.ruleset.insert(0, f)
		return f

class Constituent(object):
	def __repr__(self):
		return repr(self.cat)

class CombinedConstituent(Constituent):
	def __init__(self, cat, ptrs, rule):
		assert isinstance(cat, Category)
		assert all(isinstance(ptr, Constituent) for ptr in ptrs)
		self.cat = cat
		self.ptrs = ptrs
		self.rule = rule

class AtomicConstituent(Constituent):
	def __init__(self, cat, word):
		assert isinstance(cat, Category)
		self.cat = cat
		self.word = word