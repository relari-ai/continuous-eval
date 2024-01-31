import re
from copy import copy
from typing import List

from nltk.tokenize.api import TokenizerI
from nltk.tokenize.destructive import MacIntyreContractions


class SmartTokenizer(TokenizerI):
    BASIC_PUNTCUATION = """!"#&'().:;?[]^_`{}~"""

    # Starting quotes.
    STARTING_QUOTES = [
        (re.compile("([«“‘„]|[`]+)", re.U), r" \1 "),
        (re.compile(r"^\""), r"``"),
        (re.compile(r"(``)"), r" \1 "),
        (re.compile(r"([ \(\[{<])(\"|\'{2})"), r"\1 `` "),
        (re.compile(r"(?i)(\')(?!re|ve|ll|m|t|s|d|n)(\w)\b", re.U), r"\1 \2"),
    ]

    # Ending quotes.
    ENDING_QUOTES = [
        (re.compile("([»”’])", re.U), r" \1 "),
        (re.compile(r"''"), " '' "),
        (re.compile(r'"'), " '' "),
        (re.compile(r"([^' ])('[sS]|'[mM]|'[dD]|') "), r"\1 \2 "),
        (re.compile(r"([^' ])('ll|'LL|'re|'RE|'ve|'VE|n't|N'T) "), r"\1 \2 "),
    ]

    # Punctuation.
    PUNCTUATION = [
        (re.compile(r'([^\.])(\.)([\]\)}>"\'' "»”’ " r"]*)\s*$", re.U), r"\1 \2 \3 "),
        (re.compile(r"([:,])([^\d])"), r" \1 \2"),
        (re.compile(r"([:,])$"), r" \1 "),
        (
            re.compile(r"\.{2,}", re.U),
            r" \g<0> ",
        ),  # See https://github.com/nltk/nltk/pull/2322
        (re.compile(r"[;@#%&]"), r" \g<0> "),
        (
            re.compile(r'([^\.])(\.)([\]\)}>"\']*)\s*$'),
            r"\1 \2\3 ",
        ),  # Handles the final period.
        (re.compile(r"[?!]"), r" \g<0> "),
        (re.compile(r"([^'])' "), r"\1 ' "),
        (
            re.compile(r"[*]", re.U),
            r" \g<0> ",
        ),  # See https://github.com/nltk/nltk/pull/2322
    ]

    DOUBLE_DASHES = (re.compile(r"--"), r" -- ")

    # List of contractions adapted from Robert MacIntyre's tokenizer.
    _contractions = MacIntyreContractions()
    CONTRACTIONS2 = list(map(re.compile, _contractions.CONTRACTIONS2))
    CONTRACTIONS3 = list(map(re.compile, _contractions.CONTRACTIONS3))

    def tokenize(self, text: str) -> List[str]:
        text = copy(text.lower())

        # strip basic punctuation
        text = ''.join([char for char in text if char not in self.BASIC_PUNTCUATION])

        for regexp, substitution in self.STARTING_QUOTES:
            text = regexp.sub(substitution, text)

        for regexp, substitution in self.PUNCTUATION:
            text = regexp.sub(substitution, text)

        # Handling dollar symbol attached to numbers
        text = re.sub(r'\$(\d+)', r'$\1', text)

        # Removing commas from numeric values
        text = re.sub(r'(\d+),(\d+)', r'\1\2', text)

        # Handles double dash.
        regexp, substitution = self.DOUBLE_DASHES
        text = regexp.sub(substitution, text)

        # add extra space to make things easier
        text = " " + text + " "

        for regexp, substitution in self.ENDING_QUOTES:
            text = regexp.sub(substitution, text)

        for regexp in self.CONTRACTIONS2:
            text = regexp.sub(r" \1 \2 ", text)
        for regexp in self.CONTRACTIONS3:
            text = regexp.sub(r" \1 \2 ", text)

        return text.split()
