#!/usr/bin/python3.5
# -*- coding: utf-8 -*-

import collections
import os.path
import re
import string

import ipdb  # erase me
import gensim.models
import gensim.utils
import numpy as np
import pandas as pd
import pygal
import sklearn.decomposition
import sklearn.feature_extraction.text
import sklearn.manifold
import stop_words
import tinydb

_DB_NAME = 'db/201801011249.json'
_EMBEDDINGS_VECTOR_SIZE = 200

_CANDIDATES = [
	'PietroGrasso',  # Pietro Grasso
	'matteorenzi',  # Matteo Renzi
	'luigidimaio',  # Luigi di Maio
	'berlusconi',  # Silvio Berlusconi
	'GiorgiaMeloni',  # Giorgia Meloni
	'matteosalvinimi',  # Matteo Salvini
	]

_LINK_REGEX = re.compile(r'(https?|www)[^ ]+')
_MENTION_REGEX = re.compile(r'@[^ ]+')

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
		for tweet in dataset[candidate]['tweets']:
			tweets.append(CleanTweet(tweet))
	return tweets


def ExtractCandidatesFromDataset(dataset):
	"""."""
	candidates = []
	for candidate in _CANDIDATES:
		for _ in dataset[candidate]['tweets']:
			candidates.append(candidate)
	return candidates


def TokenizeTweet(tweet):
	"""."""
	tokens = []
	for token in gensim.utils.tokenize(tweet):
		if token not in stop_words.get_stop_words('it'):
			tokens.append(token)
	return tokens


def GenerateMegaTweets(dataset):
	"""."""
	mega_tweets = []
	for candidate in _CANDIDATES:
		mega_tweet = ''
		for tweet in dataset[candidate]['tweets']:
			mega_tweet += ' '
			mega_tweet += CleanTweet(tweet)
		mega_tweets.append(mega_tweet)
	return mega_tweets


def ComputeTfidf(mega_tweets):
	"""."""
	vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(
		input='content',
		analyzer='word',
		ngram_range=(1, 1),
		stop_words=stop_words.get_stop_words('it'),
		max_df=7.9  # Ignore words that occur in 4 or more mega tweets
		)
	raw_tfidf = vectorizer.fit_transform(mega_tweets)
	tfidf = pd.SparseDataFrame(
		raw_tfidf,
		index=_CANDIDATES,
		columns=vectorizer.get_feature_names())
	return tfidf


def ComputeTokensImportance(tfidf):
	"""."""
	tokens_importance = {}
	for token in tfidf.columns:
		tokens_importance[token] = np.nanmax(
			tfidf.loc[:, token].values.to_dense())
	return tokens_importance


def ComputeTweetVector(model, tokens, tokens_importance):
	"""."""
	if tokens:
		vectors = []
		weights = []
		# For each token (most of the times a word) we want
		# to retrieve a vector representation, if available
		for token in tokens:
			try:
				vectors.append(model.wv[token])
			except KeyError:
				continue
			else:  # Only executed if no exception
				try:
					weights.append(tokens_importance[token])
				except KeyError:
					# If for any reason we don't have a tfidf value
					# for a token, we'll simply default to weight 0
					weights.append(0.0000001)	
		if vectors:
			# Compute the average vector for the tweet
			return np.average(vectors, axis=0, weights=weights)
		else:
			return np.zeros(_EMBEDDINGS_VECTOR_SIZE)
	else:
		# If for any reason we have no tokens (for example a tweet
		# made entirely of stop words), we simply return a vector of 0s
		return np.zeros(_EMBEDDINGS_VECTOR_SIZE)


def FilterZeroVectorsOut(dataframe):
	"""."""
	filtered_candidates = []
	filtered_vectors = []
	for candidate, vector in dataframe.iterrows():
		if not np.array_equal(vector.values, np.zeros(_EMBEDDINGS_VECTOR_SIZE)):
			filtered_candidates.append(candidate)
			filtered_vectors.append(vector)
	return pd.DataFrame(filtered_vectors, index=filtered_candidates)


def GenerateTweetsScatterPlot(pc_dataframe):
	"""."""
	xy_chart = pygal.XY(stroke=False)
	xy_chart.title = 'Correlation'
	for candidate in _CANDIDATES:
		subset = pc_dataframe.loc[candidate, :]
		values = list(subset.itertuples(index=False, name=None))
		xy_chart.add(candidate, values)
	xy_chart.render_to_file('images/tweets_plot.svg')


# main
dataset = LoadDataset()
tweets = ExtractCleanTweetsFromDataset(dataset)
candidates = ExtractCandidatesFromDataset(dataset)
list_of_tokens = [TokenizeTweet(tweet) for tweet in tweets]
if not os.path.isfile('word2vec.model'):
	model = gensim.models.word2vec.Word2Vec(
		list_of_tokens,
		size=_EMBEDDINGS_VECTOR_SIZE,
		window=5,
		min_count=5,
		workers=4)
	model.save('word2vec.model')
else:
	model = gensim.models.word2vec.Word2Vec.load('word2vec.model')

mega_tweets = GenerateMegaTweets(dataset)
tfidf = ComputeTfidf(mega_tweets)
tokens_importance = ComputeTokensImportance(tfidf)

vectors = [ComputeTweetVector(model, tokens, tokens_importance)
		   for tokens in list_of_tokens]
dataframe = pd.DataFrame(vectors, index=candidates)
filtered_dataframe = FilterZeroVectorsOut(dataframe)

# pca = sklearn.decomposition.PCA(n_components=2)
pca = sklearn.manifold.TSNE(n_components=2)
principal_components = pca.fit_transform(filtered_dataframe)
pc_dataframe = pd.DataFrame(
	principal_components,
	index=filtered_dataframe.index)

GenerateTweetsScatterPlot(pc_dataframe)

# remove zeroes OK
# change title
# change colours?
# add pc1, pc2 labels
# weight by by tfidf score 
# I would plot both PCA and t-SNE