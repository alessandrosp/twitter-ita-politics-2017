#!/usr/bin/python3.5
# -*- coding: utf-8 -*-

"""It computes descriptive statistics for each candidate activity on Twitter.

The statistics are then outputted to CSV.
"""

import collections
import re

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


def add_num_tweets(dataset):
    """It adds the number of tweets per candidate to dataset."""
    for candidate in _CANDIDATES:
        dataset[candidate]['num_tweets'] = len(dataset[candidate]['tweets'])


def add_average_length(dataset):
    """It adds the average tweet length per candidate to dataset."""
    for candidate in _CANDIDATES:
        tweets = dataset[candidate]['tweets']
        total_length = sum([len(tweet) for tweet in tweets])
        num_tweets = len(dataset[candidate]['tweets'])
        dataset[candidate]['average_length'] = total_length / num_tweets


def add_longest_length(dataset):
    """It adds the longest tweet length per candidate to dataset."""
    for candidate in _CANDIDATES:
        tweets = dataset[candidate]['tweets']
        longest_length = max([len(tweet) for tweet in tweets])
        dataset[candidate]['longest_length'] = longest_length


def add_shortest_length(dataset):
    """It adds the shortest tweet length per candidate to dataset."""
    for candidate in _CANDIDATES:
        tweets = dataset[candidate]['tweets']
        longest_length = min([len(tweet) for tweet in tweets])
        dataset[candidate]['longest_length'] = longest_length


def add_ratio_tweets_with_links(dataset):
    """It adds the ratio of tweets with links over the total number."""
    for candidate in _CANDIDATES:
        tweets = dataset[candidate]['tweets']
        num_tweets = len(dataset[candidate]['tweets'])
        with_links = [1 if _LINK_REGEX.match(tweet) else 0
                      for tweet in tweets]
        dataset[candidate]['longest_length'] = sum(with_links) / num_tweets


def output_dataset(dataset):
    """."""
    # remove tweets
    # cast as pandas
    # tocsv
    del dataset


def main():
    """Main function for the module."""
    dataset = load_dataset()
    add_num_tweets(dataset)
    add_average_length(dataset)
    add_longest_length(dataset)
    add_shortest_length(dataset)
    add_ratio_tweets_with_links(dataset)

    output_dataset(dataset)

if __name__ == '__main__':
    main()
