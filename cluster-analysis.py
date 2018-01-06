# -*- coding: utf-8 -*-

import collections
import os.path
import re
import string

import gensim.models
import gensim.utils
import stop_words
import tinydb

_DB_NAME = 'db/201801011249.json'

_CANDIDATES = [
	'PietroGrasso',  # Pietro Grasso
	'matteorenzi',  # Matteo Renzi
	'luigidimaio',  # Luigi di Maio
	'berlusconi',  # Silvio Berlusconi
	'GiorgiaMeloni',  # Giorgia Meloni
	'matteosalvinimi',  # Matteo Salvini
	]

def LoadDataset():
	"""It loads the data from DB and returns a dict with candidates as keys."""
	db = tinydb.TinyDB(_DB_NAME)
	dataset = collections.defaultdict(dict)
	for candidate in _CANDIDATES:
		table = db.table(candidate)
		documents = table.all()
		dataset[candidate]['tweets'] = [document['text']
									    for document in documents]
	return dataset


def RemoveMentions(tweet):
	"""It removes mentions (@example), replacing them with ''."""
	return _MENTION_REGEX.sub('', tweet)


def RemoveLinks(tweet):
	"""It removes links (https://t.co/xxx), replacing them with ''."""
	return _LINK_REGEX.sub('', tweet)


def CleanTweet(tweet):
	"""It executes all clean-up functions and returns cleaned tweet."""
	tweet = RemoveMentions(tweet)
	tweet = RemoveLinks(tweet)
	clean = ''
	for char in tweet.lower():
		if (char in string.ascii_lowercase 
		    or char in string.digits or char == ' '):
			clean += char
		else:
			clean += ' '
	return clean


def ExtractCleanTweetsFromDataset(dataset):
	"""."""
	tweets = []
	for candidate in _CANDIDATES:
		for tweet in dataset[candidate]:
			tweets.append(CleanTweet(tweet))
	return tweets


def ExtractCandidatesFromDataset(dataset):
	"""."""
	candidates = []
	for candidate in _CANDIDATES:
		for _ in dataset[candidate]:
			candidates.append(candidate)
	return candidates


def TokenizeTweet(tweet):
	"""."""
	tokens = []
	for token in gensim.utils.tokenize(tweet):
		if token not in stop_words.get_stop_words('it'):
			tokens.append(token)
	return tokens


def ComputeAverageVector(vectors):
	"""."""
	# compute average
	return average_vector


def ComputeTweetVector(model, tokens):
	"""."""
	if tokens:
		vectors = []
		for token in tokens:
			vectors.append(model.wv[token])
		return ComputeAverageVector(vectors)
	else:
		return # zero vector

# main
dataset = LoadDataset()
if not os.path.isfile('word2vec.model'):
	tweets = ExtractCleanTweetsFromDataset(dataset)
	candidates = ExtractCandidatesFromDataset(dataset)
	list_of_tokens = [TokenizeTweet(tweet) for tweet in tweets]
	model = gensim.models.word2vec.Word2Vec(
		list_of_tokens,
		size=100,
		window=5,
		min_count=5,
		workers=4)
	model.save('word2vec.model')
else:
	model = gensim.models.word2vec.Word2Vec.load(fname)

vectors = [ComputeTweetVector(model, tokens) for tokens in list_of_tokens]
# Each element in the vector should be a different columns
dataframe = pd.DataFrame(vectors, index=candidates)
# apply PCA or other technique
# plot