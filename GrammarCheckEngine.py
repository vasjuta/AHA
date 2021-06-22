"""
The major module of the system;
gets an input, calls for TextProcessor to clean it up,
calls for MLClassifier to get a classification, and performs all the logics
to yield the result
"""

from NaiveClassifier import NaiveClassifier
from TextProcessor import TextProcessor
from MLClassifier import MLClassifier
from nltk import tokenize
import nltk
import json

nltk.download('punkt')


class GrammarCheckEngine:

    def __init__(self):
        pass

    # obsolete
    @staticmethod
    def get_correctness_score_by_naive_cls(tweet: str) -> float:
        return NaiveClassifier.simple_text_classifier(tweet)

    # obsolete
    @staticmethod
    def is_correct_by_ml_classifier(tweet: str) -> bool:
        cls = MLClassifier()
        return cls.evaluate_single_sentence(tweet)

    @staticmethod
    def check_single_tweet(tweet: str) -> tuple[bool, list[str]]:
        processor = TextProcessor()
        # normalize input
        clean_tweet = processor.clean_text(tweet)
        # get spelling errors
        spelling_errors = processor.get_spelling_errors(clean_tweet)

        # not a good idea - maybe will get back to it later
        # spelling_errors = processor.remove_social_media_acceptable(spelling_errors)

        cls = MLClassifier()
        # split into sentences
        sentences = nltk.tokenize.sent_tokenize(clean_tweet)
        classified_as_incorrect = False
        # check each sentence with classifier
        for sentence in sentences:
            is_sentence_correct = cls.evaluate_single_sentence(sentence)
            if not is_sentence_correct:
                classified_as_incorrect = True
                break

        # both spelling errors and classified as incorrect -> is definitely incorrect;
        # return along with spelling errors. caution! may be misleading: actual errors
        # may be different from the displayed ones! spellcheck identified smth as wrong
        # while classifier passed them as OK but failed on smth else
        if len(spelling_errors) > 0 and classified_as_incorrect:
            return False, spelling_errors

        # no spelling errors and classified as correct -> definitely correct;
        # return with empty list of errors for consistency
        if len(spelling_errors) == 0 and not classified_as_incorrect:
            return True, []

        # any other case we trust our classifier;
        # we don't have its errors (if there are any), so we return the spelling ones (if there are any)
        # not the sharpest logics here, may be revisited; upd - indeed revisited, it was confusing,
        # no errors are returned in any case
        return not classified_as_incorrect, []

    # check multiple tweets by calling the single check for each one;
    # may be revisited to allow parallel treatment
    def check_multiple_tweets(self, tweets: list[str]) -> list[tuple[bool, list]]:
        result = [self.check_single_tweet(tweet) for tweet in tweets]
        return result
