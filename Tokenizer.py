
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import re
import json

class Tokenizer(object):

	def __init__(self):
		self.word_index = dict()

	def fit_on_texts(self, raw_texts):
		tokens = self.__tokenize_raw_texts(raw_texts)
		vocab = self.__generate_vocab(tokens)
		for i in range(len(vocab)):
			self.word_index[vocab[i]] = i + 1

	def transform(self, docs):
		tokens = self.__tokenize_raw_texts(docs)
		tokenized_docs = list()
		for doc in tokens:
			seq = list()
			for word in doc:
				try:
					seq.append(self.word_index[word])
				except KeyError:
					seq.append(0)
			tokenized_docs.append(seq)
		return tokenized_docs

	def save(self, file_path):
		with open(file_path, 'w') as file:
			json.dump(self.word_index, file)

	def load(self, file_path):
		with open(file_path, 'b') as file:
			self.word_index = json.load(file)

	def __tokenize_raw_texts(self, texts):
		tokenized_docs = list()
		for text in texts:
			text = text.strip().lower()
			tokens = text.split()
			tokens = [token for token in tokens if not re.match(r"[^a-zA-Z]+", token)]
			tokens = [token for token in tokens if token not in ENGLISH_STOP_WORDS]
			tokenized_docs.append(tokens)
		return tokenized_docs

	def __generate_vocab(self, tokenized_docs):
		return list(set(x for doc in tokenized_docs for x in doc))