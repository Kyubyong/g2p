# -*- coding: utf-8 -*-
# /usr/bin/python
"""
By kyubyong park(kbpark.linguist@gmail.com) and Jongseok Kim(https://github.com/ozmig77)
https://www.github.com/kyubyong/g2p
"""
from nltk import pos_tag
from nltk.corpus import cmudict
import nltk
from nltk.tokenize.casual import (
    URLS,
    EMOTICONS,
    _replace_html_entities,
    remove_handles,
    reduce_lengthening,
    EMOTICON_RE,
    HANG_RE,
    PHONE_REGEX,
)
from nltk.tokenize import TweetTokenizer

import numpy as np
import codecs
import re
import os
import regex
from typing import List
import unicodedata
from builtins import str as unicode
from .expand import normalize_numbers

try:
    nltk.data.find("taggers/averaged_perceptron_tagger.zip")
except LookupError:
    nltk.download("averaged_perceptron_tagger")
try:
    nltk.data.find("corpora/cmudict.zip")
except LookupError:
    nltk.download("cmudict")

dirname = os.path.dirname(__file__)


class CMUDictTokenizer(TweetTokenizer):
    _REGEXPS = (
        URLS,
        # ASCII Emoticons
        EMOTICONS,
        # HTML tags:
        r"""<[^>\s]+>""",
        # ASCII Arrows
        r"""[\-]+>|<[\-]+""",
        # Twitter username:
        r"""(?:@[\w_]+)""",
        # Twitter hashtags:
        r"""(?:\#+[\w_]+[\w\'_\-]*[\w_]+)""",
        # email addresses
        r"""[\w.+-]+@[\w-]+\.(?:[\w-]\.?)+[\w-]""",
        # Zero-Width-Joiner and Skin tone modifier emojis
        """.(?:
            [\U0001F3FB-\U0001F3FF]?(?:\u200d.[\U0001F3FB-\U0001F3FF]?)+
            |
            [\U0001F3FB-\U0001F3FF]
        )""",
        # Remaining word types:
        r"""
        (?:[a-zA-Z'](?:[a-zA-Z']|['\-_])+[a-zA-Z']) # Words with apostrophes or dashes.
        |
        (?:[+\-]?\d+[,/.:-]\d+[+\-]?)  # Numbers, including fractions, decimals.
        |
        (?:[\w_]+)                     # Words without apostrophes or dashes.
        |
        (?:\.(?:\s*\.){1,})            # Ellipsis dots.
        |
        (?:\S)                         # Everything else that isn't whitespace.
        """,
    )
    _REGEXPS_PHONE = (_REGEXPS[0], PHONE_REGEX, *_REGEXPS[1:])
    _WORD_RE = regex.compile(
        f"({'|'.join(_REGEXPS)})",
        regex.VERBOSE | regex.I | regex.UNICODE,
    )
    _PHONE_WORD_RE = regex.compile(
        f"({'|'.join(_REGEXPS_PHONE)})",
        regex.VERBOSE | regex.I | regex.UNICODE,
    )

    @property
    def WORD_RE(self):
        return self._WORD_RE

    @property
    def PHONE_WORD_RE(self):
        return self._PHONE_WORD_RE

    def tokenize(self, text: str) -> List[str]:
        """Tokenize the input text.

        :param text: str
        :rtype: list(str)
        :return: a tokenized list of strings; joining this list returns\
        the original string if `preserve_case=False`.
        """
        # Fix HTML character entities:
        text = _replace_html_entities(text)
        # Remove username handles
        if self.strip_handles:
            text = remove_handles(text)
        # Normalize word lengthening
        if self.reduce_len:
            text = reduce_lengthening(text)
        # Shorten problematic sequences of characters
        safe_text = HANG_RE.sub(r"\1\1\1", text)
        # Recognise phone numbers during tokenization
        if self.match_phone_numbers:
            words = self.PHONE_WORD_RE.findall(safe_text)
        else:
            words = self.WORD_RE.findall(safe_text)
        # Possibly alter the case, but avoid changing emoticons like :D into :d:
        if not self.preserve_case:
            words = list(
                map((lambda x: x if EMOTICON_RE.search(x) else x.lower()), words)
            )
        return words


tokenizer = CMUDictTokenizer()


def construct_homograph_dictionary():
    f = os.path.join(dirname, "homographs.en")
    homograph2features = dict()
    for line in codecs.open(f, "r", "utf8").read().splitlines():
        if line.startswith("#"):
            continue  # comment
        headword, pron1, pron2, pos1 = line.strip().split("|")
        homograph2features[headword.lower()] = (pron1.split(), pron2.split(), pos1)
    return homograph2features


# def segment(text):
#     '''
#     Splits text into `tokens`.
#     :param text: A string.
#     :return: A list of tokens (string).
#     '''
#     print(text)
#     text = re.sub('([.,?!]( |$))', r' \1', text)
#     print(text)
#     return text.split()


class G2p(object):
    def __init__(self):
        super().__init__()
        self.graphemes = ["<pad>", "<unk>", "</s>"] + list("abcdefghijklmnopqrstuvwxyz")
        self.phonemes = ["<pad>", "<unk>", "<s>", "</s>"] + [
            "AA0",
            "AA1",
            "AA2",
            "AE0",
            "AE1",
            "AE2",
            "AH0",
            "AH1",
            "AH2",
            "AO0",
            "AO1",
            "AO2",
            "AW0",
            "AW1",
            "AW2",
            "AY0",
            "AY1",
            "AY2",
            "B",
            "CH",
            "D",
            "DH",
            "EH0",
            "EH1",
            "EH2",
            "ER0",
            "ER1",
            "ER2",
            "EY0",
            "EY1",
            "EY2",
            "F",
            "G",
            "HH",
            "IH0",
            "IH1",
            "IH2",
            "IY0",
            "IY1",
            "IY2",
            "JH",
            "K",
            "L",
            "M",
            "N",
            "NG",
            "OW0",
            "OW1",
            "OW2",
            "OY0",
            "OY1",
            "OY2",
            "P",
            "R",
            "S",
            "SH",
            "T",
            "TH",
            "UH0",
            "UH1",
            "UH2",
            "UW",
            "UW0",
            "UW1",
            "UW2",
            "V",
            "W",
            "Y",
            "Z",
            "ZH",
        ]
        self.g2idx = {g: idx for idx, g in enumerate(self.graphemes)}
        self.idx2g = {idx: g for idx, g in enumerate(self.graphemes)}

        self.p2idx = {p: idx for idx, p in enumerate(self.phonemes)}
        self.idx2p = {idx: p for idx, p in enumerate(self.phonemes)}

        self.cmu = cmudict.dict()
        self.load_variables()
        self.homograph2features = construct_homograph_dictionary()

    def load_variables(self):
        self.variables = np.load(os.path.join(dirname, "checkpoint20.npz"))
        self.enc_emb = self.variables["enc_emb"]  # (29, 64). (len(graphemes), emb)
        self.enc_w_ih = self.variables["enc_w_ih"]  # (3*128, 64)
        self.enc_w_hh = self.variables["enc_w_hh"]  # (3*128, 128)
        self.enc_b_ih = self.variables["enc_b_ih"]  # (3*128,)
        self.enc_b_hh = self.variables["enc_b_hh"]  # (3*128,)

        self.dec_emb = self.variables["dec_emb"]  # (74, 64). (len(phonemes), emb)
        self.dec_w_ih = self.variables["dec_w_ih"]  # (3*128, 64)
        self.dec_w_hh = self.variables["dec_w_hh"]  # (3*128, 128)
        self.dec_b_ih = self.variables["dec_b_ih"]  # (3*128,)
        self.dec_b_hh = self.variables["dec_b_hh"]  # (3*128,)
        self.fc_w = self.variables["fc_w"]  # (74, 128)
        self.fc_b = self.variables["fc_b"]  # (74,)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def grucell(self, x, h, w_ih, w_hh, b_ih, b_hh):
        rzn_ih = np.matmul(x, w_ih.T) + b_ih
        rzn_hh = np.matmul(h, w_hh.T) + b_hh

        rz_ih, n_ih = (
            rzn_ih[:, : rzn_ih.shape[-1] * 2 // 3],
            rzn_ih[:, rzn_ih.shape[-1] * 2 // 3 :],
        )
        rz_hh, n_hh = (
            rzn_hh[:, : rzn_hh.shape[-1] * 2 // 3],
            rzn_hh[:, rzn_hh.shape[-1] * 2 // 3 :],
        )

        rz = self.sigmoid(rz_ih + rz_hh)
        r, z = np.split(rz, 2, -1)

        n = np.tanh(n_ih + r * n_hh)
        h = (1 - z) * n + z * h

        return h

    def gru(self, x, steps, w_ih, w_hh, b_ih, b_hh, h0=None):
        if h0 is None:
            h0 = np.zeros((x.shape[0], w_hh.shape[1]), np.float32)
        h = h0  # initial hidden state
        outputs = np.zeros((x.shape[0], steps, w_hh.shape[1]), np.float32)
        for t in range(steps):
            h = self.grucell(x[:, t, :], h, w_ih, w_hh, b_ih, b_hh)  # (b, h)
            outputs[:, t, ::] = h
        return outputs

    def encode(self, word):
        chars = list(word) + ["</s>"]
        x = [self.g2idx.get(char, self.g2idx["<unk>"]) for char in chars]
        x = np.take(self.enc_emb, np.expand_dims(x, 0), axis=0)

        return x

    def predict(self, word):
        # encoder
        enc = self.encode(word)
        enc = self.gru(
            enc,
            len(word) + 1,
            self.enc_w_ih,
            self.enc_w_hh,
            self.enc_b_ih,
            self.enc_b_hh,
            h0=np.zeros((1, self.enc_w_hh.shape[-1]), np.float32),
        )
        last_hidden = enc[:, -1, :]

        # decoder
        dec = np.take(self.dec_emb, [2], axis=0)  # 2: <s>
        h = last_hidden

        preds = []
        for i in range(20):
            h = self.grucell(
                dec, h, self.dec_w_ih, self.dec_w_hh, self.dec_b_ih, self.dec_b_hh
            )  # (b, h)
            logits = np.matmul(h, self.fc_w.T) + self.fc_b
            pred = logits.argmax()
            if pred == 3:
                break  # 3: </s>
            preds.append(pred)
            dec = np.take(self.dec_emb, [pred], axis=0)

        preds = [self.idx2p.get(idx, "<unk>") for idx in preds]
        return preds

    def __call__(self, text):
        # preprocessing
        text = unicode(text)
        text = normalize_numbers(text)
        text = "".join(
            char
            for char in unicodedata.normalize("NFD", text)
            if unicodedata.category(char) != "Mn"
        )  # Strip accents
        text = text.lower()
        text = re.sub("[^ a-z'.,?!\-]", "", text)
        text = text.replace("i.e.", "that is")
        text = text.replace("e.g.", "for example")

        # tokenization
        words = tokenizer.tokenize(text)
        tokens = pos_tag(words)  # tuples of (word, tag)
        prons = []
        for word, pos in tokens:
            if re.search("[a-z]", word) is None:
                pron = [word]

            elif word in self.homograph2features:  # Check homograph
                pron1, pron2, pos1 = self.homograph2features[word]
                if pos.startswith(pos1):
                    pron = pron1
                else:
                    pron = pron2
            elif word in self.cmu:  # lookup CMU dict
                pron = self.cmu[word][0]
            else:  # predict for oov
                pron = self.predict(word)

            prons.extend(pron)
            prons.extend([" "])

        return prons[:-1]

    def check_lookup(self, text):
        # preprocessing
        text = unicode(text)
        text = normalize_numbers(text)
        text = "".join(
            char
            for char in unicodedata.normalize("NFD", text)
            if unicodedata.category(char) != "Mn"
        )  # Strip accents
        text = text.lower()
        text = re.sub("[^ a-z'.,?!\-]", "", text)
        text = text.replace("i.e.", "that is")
        text = text.replace("e.g.", "for example")

        # tokenization
        words = tokenizer.tokenize(text)
        tokens = pos_tag(words)  # tuples of (word, tag)

        # steps
        lookup_result_dict = {}
        for word, pos in tokens:

            if re.search("[a-z]", word) is None:
                lookup_result = "non-alphanumeric"
                pron = [word]

            elif word in self.homograph2features:  # Check homograph
                lookup_result = "homograph"
                pron1, pron2, pos1 = self.homograph2features[word]
                if pos.startswith(pos1):
                    pron = pron1
                else:
                    pron = pron2
            elif word in self.cmu:  # lookup CMU dict
                lookup_result = "CMU"
                pron = self.cmu[word][0]

            else:  # predict for oov
                lookup_result = "RNN"
                pron = self.predict(word)

            if lookup_result in lookup_result_dict.keys():
                lookup_result_dict[lookup_result].append(word)
            else:
                lookup_result_dict[lookup_result] = [word]

        return lookup_result_dict


if __name__ == "__main__":
    texts = [
        "I have $250 in my pocket.",  # number -> spell-out
        "popular pets, e.g. cats and dogs",  # e.g. -> for example
        "I refuse to collect the refuse around here.",  # homograph
        "I'm an activationist.",
    ]  # newly coined word
    g2p = G2p()
    for text in texts:
        out = g2p(text)
        print(out)
