#!/usr/bin/python3.5
# -*- coding: utf-8 -*-

"""It computes descriptive statistics for each candidate activity on Twitter.

The statistics are then outputted to CSV.
"""

import collections
import copy
import re

import pandas as pd
import tinydb

_DB_NAME = 'db/201801011249.json'

_CANDIDATES = [
    'PietroGrasso',  # Pietro Grasso
    'matteorenzi',  # Matteo Renzi
    'luigidimaio',  # Luigi di Maio
    'berlusconi',  # Silvio Berlusconi
    'GiorgiaMeloni',  # Giorgia Meloni
    'matteosalvinimi',  # Matteo Salvini
    ]

_LINK_REGEX = re.compile(r'(https?|www)[^ ]+')
_MENTION_REGEX = re.compile(r'@[^ ]+')
_HASHTAG_REGEX = re.compile(r'#[^ ]+')


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
        dataset[candidate]['shortest_length'] = longest_length


def add_number_of_links(dataset):
    """It adds the number of links per candidate to dataset."""
    for candidate in _CANDIDATES:
        count_links = 0
        tweets = dataset[candidate]['tweets']
        for tweet in tweets:
            count_links += len(_LINK_REGEX.findall(tweet))
        dataset[candidate]['num_links'] = count_links


def add_number_of_hashes(dataset):
    """It adds the number of hashes per candidate to dataset."""
    for candidate in _CANDIDATES:
        count_hashes = 0
        tweets = dataset[candidate]['tweets']
        for tweet in tweets:
            count_hashes += len(_HASHTAG_REGEX.findall(tweet))
        dataset[candidate]['num_hashes'] = count_hashes


def add_number_of_mentions(dataset):
    """It adds the number of mentions per candidate to dataset."""
    for candidate in _CANDIDATES:
        count_mentions = 0
        tweets = dataset[candidate]['tweets']
        for tweet in tweets:
            count_mentions += len(_MENTION_REGEX.findall(tweet))
        dataset[candidate]['num_mentions'] = count_mentions


def _remove_tweets(candidate_dict):
    """It removes the key 'tweets' from the candidate_dict."""
    new_dict = copy.deepcopy(candidate_dict)
    del new_dict['tweets']
    return new_dict


def output_dataset(dataset):
    """It strips the candidates' dicts of 'tweets' and then outputs to CSV."""
    output = collections.defaultdict(dict)
    for candidate in _CANDIDATES:
        output[candidate] = _remove_tweets(dataset[candidate])
    df_to_output = pd.DataFrame(output)
    df_to_output.to_csv('descriptive_stats.csv')


def main():
    """Main function for the module."""
    dataset = load_dataset()
    add_num_tweets(dataset)
    add_average_length(dataset)
    add_longest_length(dataset)
    add_shortest_length(dataset)
    add_number_of_links(dataset)
    add_number_of_hashes(dataset)
    add_number_of_mentions(dataset)
    output_dataset(dataset)

if __name__ == '__main__':
    main()
