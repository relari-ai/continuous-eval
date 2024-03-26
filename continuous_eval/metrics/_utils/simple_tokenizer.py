import re
import string
from copy import copy
from typing import List

from nltk import download as nltk_download
from nltk.corpus import stopwords
from nltk.tokenize.api import TokenizerI
from nltk.tokenize.destructive import MacIntyreContractions

# This is a workaround to avoid make sure that the stopwords are loaded before the tokenizer is used by a thread
try:
    stopwords.ensure_loaded()
except LookupError:
    nltk_download("punkt", quiet=True)
    nltk_download("stopwords", quiet=True)
    stopwords.ensure_loaded()


# Modified version of NLTKWordTokenizer
class SimpleTokenizer(TokenizerI):

    # List of contractions adapted from Robert MacIntyre's tokenizer.
    _contractions = MacIntyreContractions()
    CONTRACTIONS2 = list(map(re.compile, _contractions.CONTRACTIONS2))
    CONTRACTIONS3 = list(map(re.compile, _contractions.CONTRACTIONS3))

    IS_NUMBER = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")

    def tokenize(self, text: str, remove_stopwords=True) -> List[str]:
        text = copy(text.lower())

        # strip punctuation (including dollar symbol and commas in numbers)
        text = "".join([char for char in text if char not in string.punctuation])

        if remove_stopwords:
            stop_words = set(stopwords.words("english"))
            text = " ".join([word for word in text.split() if word not in stop_words])

        # add extra space to make things easier
        text = " " + text + " "

        for regexp in self.CONTRACTIONS2:
            text = regexp.sub(r" \1 \2 ", text)
        for regexp in self.CONTRACTIONS3:
            text = regexp.sub(r" \1 \2 ", text)

        return [t if (self.IS_NUMBER.match(t) is None) else float(t) for t in text.split()]
