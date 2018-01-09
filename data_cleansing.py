#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

"""It reads the data stored in db/ and performs some data cleasing.

Specifically, the script creates two additional version of whatever .json
file is used as input:

- A _cleaned version, where each tweet has been lowercased and stripped of
  links, mentions and any special character (anything but English characters,
  numbers and spaces). Note that hastags (without #) are mantained.
- A _lemmatized version, equal to the _cleaned version but hashtags have been
  completely removed and also each token has been replaced with its
  lemmatized version (e.g., 'cani' becomes 'cane').
"""

import collections
import re
import string

import pattern.it  # pylint: disable=no-name-in-module,import-error
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


def remove_mentions(tweet):
    """It removes mentions (@example), replacing them with ''."""
    return _MENTION_REGEX.sub('', tweet)


def remove_links(tweet):
    """It removes links (https://t.co/xxx), replacing them with ''."""
    return _LINK_REGEX.sub('', tweet)


def remove_hashtags(tweet):
    """It removes hashtags (#example), replacing them with ''."""
    return _HASHTAG_REGEX.sub('', tweet)


def remove_special_characters(tweet):
    """It removes anything but English characters, digits and spaces."""
    clean = ''
    for char in tweet.lower():
        if (char in string.ascii_lowercase or char in string.digits
                or char == ' '):
            # We only keep English default 26 letters, digits or spaces
            clean += char
        else:
            clean += ' '
    return clean


def lemmatize_tweet(tweet):
    """For each token in tweet, it replaces the token with lemmatized version.

    For example plural nouns are replaces with their singular versions, while
    verbs are replaced with their infinitive form (e.g. 'gatte' becomes
    'gatta', while 'vado' becomes 'andare').
    """
    tree = pattern.it.parsetree(  # pylint: disable=no-member
        tweet, lemmata=True)
    if tree:
        # Note that the first item of tree is always accessed (e.g., [0]). This
        # is because all punctuation marks are stripped away before this
        # function is invoked, which means that tree will always be made by
        # only one sentence
        return ' '.join(token.lemma for token in tree[0])
    return ''


def transform_tweets(tweets, transformations):
    """It applies all functions in transformations to all tweets."""
    new_tweets = []
    for tweet in tweets:
        for transformation in transformations:
            tweet = transformation(tweet)
        new_tweets.append(tweet)
    return new_tweets


def save_tweets(tweets, db_name, table_name):
    """It saves tweets on disk in table table_name of db db_name."""
    db = tinydb.TinyDB(db_name)  # pylint: disable=invalid-name
    table = db.table(table_name)
    for tweet in tweets:
        table.insert({'text': tweet})


def main():
    """Main function for the module."""
    dataset = load_dataset()
    cleaned_db_name = _DB_NAME.replace('.json', '_cleaned.json')
    lemmatized_db_name = _DB_NAME.replace('.json', '_lemmatized.json')
    for candidate in _CANDIDATES:
        tweets = dataset[candidate]['tweets']
        tweets_cleaned = transform_tweets(
            tweets,
            transformations=[
                remove_mentions, remove_links, remove_special_characters])
        tweets_lemmatized = transform_tweets(
            tweets,
            transformations=[
                remove_mentions, remove_links, remove_hashtags,
                remove_special_characters, lemmatize_tweet])
        save_tweets(tweets_cleaned, cleaned_db_name, candidate)
        save_tweets(tweets_lemmatized, lemmatized_db_name, candidate)


if __name__ == '__main__':
    main()
