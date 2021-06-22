from spellchecker import SpellChecker


# obsolete
class NaiveClassifier:

    def __init__(self):
        pass

    @staticmethod
    def simple_text_classifier(text):
        spell = SpellChecker()
        words = text.split()
        # find those words that may be misspelled
        misspelled = spell.unknown(words)
        if len(misspelled) > 0:
            print(misspelled)
        score = 1 - len(misspelled) * 1.0 / len(words)
        print("score is: {}".format(score))
        return score
