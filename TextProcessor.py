"""
Text manipulations module;
used by GrammarCheckEngine.py to relieve the latter from text processing;
has twitter-specific logics while normalizing (cleaning-up) text, both
on sentence and word levels;
uses both regular expressions and dictionaries (should be get rid off further);
performs spell check
"""
from nltk import TweetTokenizer as Tokenizer
from emoji import demojize, emoji_count
import re
import csv
from spellchecker import SpellChecker
import os


class TextProcessor:

    # initializes the preprocessor with nltk tweet tokenizer;
    # loads text replacements dictionary (regex patterns as keys, replacements as values) for possible use
    def __init__(self) -> None:
        self.tokenizer = Tokenizer()
        dictionary_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), "dicts/replacement_dict.csv")

        with open(dictionary_filename, 'r') as infile:
            reader = csv.reader(infile)
            next(reader)
            tmp_dict = dict((row[0], row[1]) for row in reader)
            # sort the keys in reverse order of length as we want the longer rules to be applied first
            sorted_keys_list = sorted(tmp_dict.keys(), key=len, reverse=True)
            sorted_dict = {item: tmp_dict[item] for item in sorted_keys_list}
            self.dict_regex = sorted_dict
            # Create a regular expression from all of the dictionary keys
            self.pat_regex = re.compile("\\b(" + ")\\b|\\b(".join(self.dict_regex.keys()) + ")\\b")

    # replaces text if matching pattern is found in the dictionary
    def regex_dict_replace(self, text: str) -> str:
        # for each pattern match, look up the corresponding value in the dictionary and replace it
        if self.pat_regex and self.dict_regex:
            return self.pat_regex.sub(lambda match: self.dict_regex[match.group(0)], text)
        else:
            return text

    # normalize text on a word/token level
    def normalize_token(self, token: str) -> str:
        token = token.lower()
        token = self.regex_dict_replace(token)

        # remove URLs
        if token.startswith("http") or token.startswith("www") or token.startswith("bit.ly"):
            return ""
        elif len(token) == 1:
            token = demojize(token)
            # remove emojis
            if token.startswith(":") and token.endswith(":"):
                return ""
            else:
                return token
        # some edge cases, should be refactored (moved to dictionary)
        else:
            if token == "’":
                return "'"
            elif token == "…":
                return "..."
            else:
                return token

    # normalize text on a sentence level
    def clean_text(self, sentence: str) -> str:
        norm_sentence = sentence.lower()
        # tokenize with nltk; in case it fails due to encoding - split by space
        try:
            tokens = self.tokenizer.tokenize(norm_sentence)
        except TypeError:
            tokens = sentence.split()
        # clean up on token level first
        norm_sentence = " ".join([self.normalize_token(token) for token in tokens])

        # clean up on sentence level
        # todo: move all the regex rules into config file for easier maintenance
        norm_sentence = re.sub(r"(\.{3,})", r"\.", norm_sentence) # ...... -> .
        norm_sentence = re.sub(r"(\brt\s+@[a-z]+\b)", r"", norm_sentence) # rt @mention -> ''
        # probably location hashtag -> preposition + Rome
        norm_sentence = re.sub(r"(at|in|from|off|to|toward|over|under)\s+(#[a-z]+)", r"\1 Rome", norm_sentence)
        # preposition + hashtag -> strip just the #
        norm_sentence = re.sub(r"(for|on|of)\s+#([a-z]+)", r"\1 \2", norm_sentence)
        # hashtag -> '' (definitely should be revisited)
        norm_sentence = re.sub(r"(#([a-z0-9]+)\b)", r"", norm_sentence)
        # @mention -> John
        norm_sentence = re.sub(r"(@[a-z0-9]+\b)", r"John", norm_sentence)

        # log the normalized sentence
        print("Normalized Input: {}".format(norm_sentence))
        return norm_sentence

    # return spelling errors by using SpellChecker package
    def get_spelling_errors(self, text: list[str]) -> list[str]:
        spell = SpellChecker()
        tokens = self.tokenizer.tokenize(text)
        misspelled = spell.unknown(tokens)
        return misspelled
