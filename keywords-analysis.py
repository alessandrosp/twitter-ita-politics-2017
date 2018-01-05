# -*- coding: utf-8 -*-

import collections
import re
import string

import numpy as np
import pandas as pd
import sklearn.feature_extraction.text
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
		if char in string.ascii_lowercase or char == ' ':
			clean += char
		else:
			clean += ' '
	return clean


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


def ComputeCandidatesVocabulary(tfidf):
	"""."""
	vocabulary = collections.defaultdict(list)
	for candidate in _CANDIDATES:
		for idx, value in enumerate(tfidf.loc[candidate,:]):
			if not np.isnan(value):
				word = tfidf.columns[idx]
				if len(word) > 3 and word != 'https':
					vocabulary[candidate].append((word, value))
	return vocabulary


def ComputeCandidatesKeywords(vocabulary, num_words):
	"""."""
	sorted_vocabulary = collections.defaultdict(list)
	for candidate in _CANDIDATES:
		sorted_vocabulary[candidate] = sorted(
			vocabulary[candidate],
			key=lambda tup: tup[1],
			reverse=True)

	keywords = collections.defaultdict(list)
	for candidate in _CANDIDATES:
		for word, value in sorted_vocabulary[candidate][0:num_words]:
			keywords[candidate].append(word)

	return keywords


if __name__ == '__main__':
	dataset = LoadDataset()
	mega_tweets = GenerateMegaTweets(dataset)
	tfidf = ComputeTfidf(mega_tweets)
	vocabulary = ComputeCandidatesVocabulary(tfidf)
	keywords = ComputeCandidatesKeywords(vocabulary, num_words=25)
	output = pd.DataFrame(keywords, columns=_CANDIDATES)
	output.to_csv(path_or_buf='keywords.csv', index=False)
