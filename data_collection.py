#!/usr/bin/python3.5
# -*- coding: utf-8 -*-

"""It uses Twitter API to scrape tweets for each candidate.

It uses Twitter API via tweepy to scrape tweets for each candidate. Retweets,
both native and non, are not considered and immidiately discarded. No string
manipulation is done on the tweets, which are stored as they are.

Tweets are stored in a .json database using TinyDB.
"""

import datetime
import logging
import re

import tinydb
import tweepy  # pylint: disable=import-error

_CONSUMER_KEY = ''
_CONSUKER_SECRET = ''
_ACCESS_TOKEN = ''
_ACCESS_TOKEN_SECRET = ''

_MIN_DATE = datetime.datetime(2017, 1, 1, 0, 0, 0)  # inclusive
_MAX_DATE = datetime.datetime(2017, 12, 24, 23, 59, 59)  # inclusive
_MIN_TWEET_ID = 796192757836554240  # exclusive
_MAX_TWEET_ID = 946071696200650759  # inclusive
# test 823490915541143552
# prod 946071696200650759

_RETWEET_REGEX = re.compile(r'^(RT|rt)( @\w*)?[: ]')

_CANDIDATES = [
    'PietroGrasso',  # Pietro Grasso
    'matteorenzi',  # Matteo Renzi
    'luigidimaio',  # Luigi di Maio
    'berlusconi',  # Silvio Berlusconi
    'GiorgiaMeloni',  # Giorgia Meloni
    'matteosalvinimi',  # Matteo Salvini
    ]

def create_twitter_connection():
    """It uses authentication keys to connect to Twitter API."""
    auth = tweepy.OAuthHandler(_CONSUMER_KEY, _CONSUKER_SECRET)
    auth.set_access_token(_ACCESS_TOKEN, _ACCESS_TOKEN_SECRET)
    return tweepy.API(auth)


def get_tweets(connection, screen_name):
    """Get all the tweets for the user specified by screen_name."""
    tweets = []
    min_id = _MIN_TWEET_ID
    max_id = _MAX_TWEET_ID
    done = False

    while not done:
        # Note: we can only get 200 tweets per call
        result_set = connection.user_timeline(
            screen_name=screen_name,
            count=200,
            tweet_mode='extended',
            since_id=min_id,
            max_id=max_id)
        if result_set:
            max_id = min([tweet.id for tweet in result_set]) - 1
            tweets += [tweet.full_text
                       for tweet in result_set
                       if (_MIN_DATE <= tweet.created_at <= _MAX_DATE
                           and not _RETWEET_REGEX.match(tweet.full_text))]
        else:
            done = True

    return tweets


def save_tweets(tweets, db_name, table_name):
    """Save the scraped tweets into a TinyDB table."""
    db = tinydb.TinyDB(db_name)  # pylint: disable=invalid-name
    table = db.table(table_name)
    for tweet in tweets:
        table.insert({'text': tweet})


def main():
    """Main function for the module."""
    db_name = datetime.datetime.today().strftime('db/%Y%m%d%H%M.json')
    connection = create_twitter_connection()
    for candidate in _CANDIDATES:
        logging.info('Processing tweets for %s', candidate)
        tweets = get_tweets(connection, candidate)
        save_tweets(tweets, db_name, candidate)


if __name__ == '__main__':
    main()
