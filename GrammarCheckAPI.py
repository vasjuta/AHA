'''
Entry point for API calls;
no business logic here, everything flows down the stream to Engine;
only deals with response object lists' serialization;
keeps the first development steps - threshold idea, custom data types,
naive classifier (spelling errors counter), etc.;
was simplified since then
'''

from GrammarCheckEngine import GrammarCheckEngine as Engine
import json


# utility function to serialize set() object for returning valid json over API calls
def serialize_sets(obj):
    if isinstance(obj, set):
        return list(obj)

    return obj


class GrammarCheckAPI:
    DEFAULT_THRESHOLD = 0.5
    CheckedTweet = tuple[str, bool]
    CheckedTweets = list[CheckedTweet]

    @staticmethod
    def is_single_tweet_correct(tweet: str) -> bool:
        print("Input: {}".format(tweet))
        is_correct, _ = Engine.check_single_tweet(tweet)
        return is_correct

    @staticmethod
    def check_single_tweet(tweet: str) -> tuple[bool, str]:
        print("Input: {}".format(tweet))
        is_correct, errors = Engine.check_single_tweet(tweet)
        # wrap errors list into json serializable object
        return is_correct, json.dumps(errors, default=serialize_sets)

    # todo: include tweet itself (id?) in response object to differentiate between tweets
    @staticmethod
    def check_multiple_tweets(tweets: list[str]) -> list[tuple[bool, str]]:
        print("Input: {}".format(tweets))
        engine = Engine()
        engine_results = engine.check_multiple_tweets(tweets)
        results = []
        for result in engine_results:
            # wrap errors list into json serializable object
            is_correct, errors = result
            errors = json.dumps(errors, default=serialize_sets)
            results.append((is_correct, errors))
        return results

    # obsolete
    def __init__(self):
        self.threshold = self.DEFAULT_THRESHOLD

    # obsolete
    def set_threshold(self, threshold: float) -> None:
        self.threshold = threshold

    # obsolete
    def is_correct_by_naive(self, tweet: str) -> bool:
        return self.get_correctness_score_by_naive(tweet) >= self.threshold

    # obsolete
    @staticmethod
    def is_correct_by_ml(tweet: str) -> bool:
        return Engine.is_correct_by_ml_classifier(tweet)


    # obsolete
    @staticmethod
    def check_bulk(tweets: list) -> CheckedTweets:
        result = []
        for tweet in tweets:
            result.append((tweet, Engine.get_correctness_score_by_naive_cls()))
        return result

    # obsolete
    @staticmethod
    def get_correctness_score_by_naive(tweet: str) -> float:
        # print("threshold is: {}".format(self.threshold))
        return Engine.get_correctness_score_by_naive_cls(tweet)
