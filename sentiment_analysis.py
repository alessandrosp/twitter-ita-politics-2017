#!/usr/bin/python3.5
# -*- coding: utf-8 -*-

"""It computes sentiment scores for each tweet and plot the results.

It retrieves the tweets from the .json specified in _DB_NAME. For each
token in each tweet, a polarity score is computed (either -1, +1 or 0). For
each tweet a sentiment score is computed by averaging the non-0 polarity
scores for that tweet (if all 0s, 0 is returned).

The sentiment scores are then grouped together per candidates and plotted.
"""

import collections

import numpy as np
import polyglot.text
# You only need to use the downloader once
# import polyglot.downloader
# polyglot.downloader.downloader.download('sentiment2.it')
import pygal  # pylint: disable=wrong-import-order
import tinydb  # pylint: disable=wrong-import-order

_DB_NAME = 'db/201801011249_cleaned.json'

_CANDIDATES = [
    'PietroGrasso',  # Pietro Grasso
    'matteorenzi',  # Matteo Renzi
    'luigidimaio',  # Luigi di Maio
    'berlusconi',  # Silvio Berlusconi
    'GiorgiaMeloni',  # Giorgia Meloni
    'matteosalvinimi',  # Matteo Salvini
    ]


def load_dataset():
    """It loads the data from DB and returns a dict with candidates as keys."""
    db = tinydb.TinyDB(_DB_NAME)  # pylint: disable=invalid-name
    dataset = collections.defaultdict(dict)
    for candidate in _CANDIDATES:
        table = db.table(candidate)
        documents = table.all()
        dataset[candidate]['tweets'] = [document['text']
                                        for document in documents]
    return dataset


def compute_sentiment_for_tweet(tweet):
    """It computes the sentiment score (-1 to +1) for the given tweet."""
    text = polyglot.text.Text(tweet.lower(), hint_language_code='it')
    scores = [word.polarity for word in text.words if word.polarity != 0]
    return np.mean(scores) if scores else 0.0


def annotate_with_sentiment_scores(dataset):
    """It computes the sentiment scores for all the candidates."""
    for candidate in dataset.keys():
        sentiment_scores = []
        for tweet in dataset[candidate]['tweets']:
            if tweet:
                sentiment_scores.append(compute_sentiment_for_tweet(tweet))
            else:
                sentiment_scores.append(0.0)
        dataset[candidate]['sentiment_scores'] = sentiment_scores


def generate_sentiment_scores_plot(dataset):
    """It generates the stacked bar plot from the sentiment scores."""
    perc_negatives = []
    perc_neutrals = []
    perc_positives = []
    for candidate in _CANDIDATES:
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
    plot.x_labels = _CANDIDATES
    plot.add('Negative', perc_negatives)
    plot.add('Neutral', perc_neutrals)
    plot.add('Positive', perc_positives)
    plot.render_to_file('images/sentiment_plot.svg')


def main():
    """Main function for the module."""
    dataset = load_dataset()
    annotate_with_sentiment_scores(dataset)
    generate_sentiment_scores_plot(dataset)


if __name__ == '__main__':
    main()
