"""
bunch of checks for internal use only :)
"""

from GrammarCheckAPI import GrammarCheckAPI
from TextProcessor import TextProcessor as Processor
from gingerit import gingerit
import os
import datetime
import pandas as pd
import codecs
import random

api = GrammarCheckAPI()
processor = Processor()


def check_single_tweet_by_naive(tweet):
    normalized_tweet = processor.clean_text(tweet)
    api.set_threshold(0.9)
    print(api.is_correct_by_naive(normalized_tweet))


def check_single_tweet_by_ml(tweet):
    normalized_tweet = processor.clean_text(tweet)
    return api.is_correct_by_ml(normalized_tweet)


def check_multiple_tweets_by_naive(tweets):
    api.set_threshold(1.0)
    for tweet in tweets:
        normalized_tweet = processor.clean_text(tweet)
        is_correct = api.is_correct_by_naive(normalized_tweet)
        print("{}: {}".format(normalized_tweet, "CORRECT" if is_correct else "INCORRECT"))


def check_multiple_tweets_by_ml(tweets, output_dir=""):
    for tweet in tweets:
        normalized_tweet = processor.clean_text(tweet)
        is_correct = api.is_correct_by_ml(tweet)
        tweet_result = "{}: {}".format(normalized_tweet, "CORRECT" if is_correct else "INCORRECT")

        if output_dir != "":
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            file_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            file_path = os.path.join(output_dir, '.'.join((file_name, "txt")))
            with open(file_path, 'w') as f:
                print("{}\n".format(tweet_result), file=f)

        else:
            print(tweet_result)


def check_single_tweet_hybrid(tweet):
    is_correct, errors = api.check_single_tweet(tweet)
    tweet_result = "{}: {}; {}".format(tweet, "CORRECT" if is_correct else "INCORRECT",
                                       errors if not is_correct and len(errors) > 0 else "")
    print(tweet_result)
    return tweet_result


def check_multiple_tweets_hybrid(tweets, output_dir=""):
    tweet_results = []
    for tweet in tweets:
        is_correct, errors = api.check_single_tweet(tweet)
        tweet_result = "{}: {}; {}".format(tweet, "CORRECT" if is_correct else "INCORRECT",
                                                    errors if not is_correct and len(errors) > 0 else "")
        tweet_results.append(tweet_result)

    #tweet_results = api.check_multiple_tweets(tweets)

    if output_dir != "":
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        file_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        file_path = os.path.join(output_dir, '.'.join((file_name, "txt")))
        with open(file_path, "w") as f:
            for tweet_result in tweet_results:
                f.write(tweet_result + '\n')
    else:
        print(*tweet_results, sep="\n")


def run_manual_checks():
    tweet1 = "I am sooooo happy! This is how true happiness looks like 👍😜 #RandomTweets #blizzard2017 #blizzard #Happiness #funny #studentlife"
    tweet2 = "Poop Money android app is now available at tntdevelopment store. https://goo.gl/ddAn1E"
    tweet3 = "Relationship I was hoping for instead flew to #moscow #laughing #relationshipgoals #geek"
    tweet4 = "Stay in contact w/ your group w/ @mention #iPhone application"
    tweet5 = "This is how true happiness looks like 👍😜 #RandomTweets #blizzard2017 #blizzard"
    tweet6 = "This time it's Discovr for iPhone at #sxsw. Climbing the App Store charts!"
    #print(" =========== NAIVE CHECK ==============")
    #check_multiple_tweets_by_naive([tweet1, tweet2, tweet3, tweet4, tweet5])

    #print(" =========== ML CHECK ==============")
    #check_multiple_tweets_by_ml([tweet1, tweet2, tweet3, tweet4, tweet5])

    #print(" =========== HYBRID CHECK ==============")
    #check_multiple_tweets_hybrid([tweet1, tweet2, tweet3, tweet4, tweet5])

    check_single_tweet_hybrid(tweet6)


def run_bulk_check():
    codecs.register_error("strict", codecs.ignore_errors)
    df = pd.read_csv("./data/raw_tweets.csv", error_bad_lines=False, encoding='Windows-1254', names=['tweet'])
    tweets = df['tweet'].tolist()

    print(len(tweets))

    tweets = random.sample(tweets, 5)
    check_multiple_tweets_hybrid(tweets, "output")


if __name__ == '__main__':
    # ginger = gingerit.GingerIt()
    # print(ginger.parse("were how true happiness looks like 👍😜"))
    # exit()
    run_manual_checks()
    # run_bulk_check()
