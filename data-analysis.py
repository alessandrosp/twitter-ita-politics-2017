import collections
import re

import ipdb  # erase me
import numpy as np
import polyglot.text
import polyglot.downloader
# polyglot.downloader.downloader.download('sentiment2.it')
import pygal
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
_HASHTAG_REGEX = re.compile(r'#[^ ]+')

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
	"""Remove mentions (@example), replacing them with ''."""
	return _MENTION_REGEX.sub('', tweet)


def RemoveHashtags(tweet):
	"""Remove hashtags (#example), replacing them with ''."""
	return _HASHTAG_REGEX.sub('', tweet)


def RemoveLinks(tweet):
	"""Remove links (https://t.co/xxx), replacing them with ''."""
	return _LINK_REGEX.sub('', tweet)


def CleanTweet(tweet):
	"""It executes all clean-up functions and returns cleaned tweet."""
	tweet = RemoveMentions(tweet)
	tweet = RemoveHashtags(tweet)
	tweet = RemoveLinks(tweet)
	return tweet


def ComputeSentimentForTweet(tweet):
	"""It computes the sentimental score (-1 to +1) for the given tweet."""
	text = polyglot.text.Text(tweet.lower(), hint_language_code='it')
	scores = [word.polarity for word in text.words if word.polarity != 0]
	return np.mean(scores) if scores else 0.0


def AnnotateWithSentimentScores(dataset):
	"""It computes the sentiment scores for all the tweets."""
	for candidate in dataset.keys():
		sentiment_scores = []
		for tweet in dataset[candidate]['tweets']:
			sentiment_scores.append(ComputeSentimentForTweet(tweet))
		dataset[candidate]['sentiment_scores'] = sentiment_scores


def GenerateSentimentScoresPlot(dataset):
	"""."""
	perc_negatives = []
	perc_neutrals = []
	perc_positives = []
	for candidate in dataset.keys():
		total = len(dataset[candidate]['sentiment_scores'])
		negatives = [score
					 for score in dataset[candidate]['sentiment_scores']
					 if score < 0.0]
		neutrals = [score
					for score in dataset[candidate]['sentiment_scores']
					if score == 0.0]
		positives = [score
					 for score in dataset[candidate]['sentiment_scores']
					 if score > 0.0]
		perc_negatives.append(len(negatives) / total)
		perc_neutrals.append(len(neutrals) / total)
		perc_positives.append(len(positives) / total)
	plot = pygal.StackedBar()
	plot.title = "Sentiment Scores for Candidates' Tweets (in %)"
	plot.x_labels = dataset.keys()
	plot.add('Negative', perc_negatives)
	plot.add('Neutral', perc_neutrals)
	plot.add('Positive', perc_positives)
	plot.render_to_file('images/sentiment_plot.svg')

# main
dataset = LoadDataset()
AnnotateWithSentimentScores(dataset)
GenerateSentimentScoresPlot(dataset)

ipdb.set_trace()