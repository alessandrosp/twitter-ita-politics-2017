#!/usr/bin/python3.5
# -*- coding: utf-8 -*-

"""It computes the top keywords for each candidate.

Specifically, the script uses TF-IDF to compute the most important tokens
for each candidate (tokens that appear often in their tweets but not
so often in the tweets of the other candidates).
"""

import collections

import numpy as np
import pandas as pd
import sklearn.feature_extraction.text
import sklearn.metrics.pairwise
import stop_words
import tinydb

_DB_NAME = 'db/201801011249_cleaned.json'

_CANDIDATES = [
    'PietroGrasso',  # Pietro Grasso
    'matteorenzi',  # Matteo Renzi
    'luigidimaio',  # Luigi di Maio
    'berlusconi',  # Silvio Berlusconi
    'GiorgiaMeloni',  # Giorgia Meloni
    'matteosalvinimi',  # Matteo Salvini
    ]

_NUM_KEYWORDS = 25

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


def generate_mega_tweets(dataset):
    """It joins all the tweets for one candidate together into one string."""
    mega_tweets = []
    for candidate in _CANDIDATES:
        mega_tweet = ''
        for tweet in dataset[candidate]['tweets']:
            mega_tweet += ' '
            mega_tweet += tweet
        mega_tweets.append(mega_tweet)
    return mega_tweets


def compute_tfidf(mega_tweets):
    """It uses sklearn TfidfVectorizer to generates a tfidf matrix."""
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


def compute_candidates_vocabulary(tfidf):
    """It returns dict containing tuples of (token, tfidf-value)."""
    vocabulary = collections.defaultdict(list)
    for candidate in _CANDIDATES:
        for idx, value in enumerate(tfidf.loc[candidate, :]):
            if not np.isnan(value):
                word = tfidf.columns[idx]
                if len(word) > 3 and word != 'https':
                    vocabulary[candidate].append((word, value))
    return vocabulary


def compute_candidates_keywords(vocabulary, num_words):
    """It returns the num_words tokens with highest tfidf-score as a list."""
    sorted_vocabulary = collections.defaultdict(list)
    for candidate in _CANDIDATES:
        sorted_vocabulary[candidate] = sorted(
            vocabulary[candidate],
            key=lambda tup: tup[1],
            reverse=True)

    keywords = collections.defaultdict(list)
    for candidate in _CANDIDATES:
        for word, _ in sorted_vocabulary[candidate][0:num_words]:
            keywords[candidate].append(word)

    return keywords


def main():
    """Main function for the module."""
    dataset = load_dataset()
    mega_tweets = generate_mega_tweets(dataset)
    tfidf = compute_tfidf(mega_tweets)

    vocabulary = compute_candidates_vocabulary(tfidf)
    keywords = compute_candidates_keywords(vocabulary, num_words=_NUM_KEYWORDS)

    kw_output = pd.DataFrame(keywords, columns=_CANDIDATES)
    kw_output.to_csv(path_or_buf='keywords.csv', index=False)

if __name__ == '__main__':
    main()
