# g2p: A Simple Python Module for English Grapheme To Phoneme Conversion

This module is designed to convert English graphemes (spelling) to phonemes (pronunciation).
It is considered essential in several tasks such as speech synthesis.
Unlike many languages like Spanish or German where pronunciation of a word can be inferred from its spelling,
English words are often far from people's expectations.
Therefore, it will be the best idea to consult a dictionary if we want to know the pronunciation of some word.
However, there are at least two tentative issues in this approach.
First, you can't disambiguate the pronunciation of homographs, words which have multiple pronunciations. (See `a` below.)
Second, you can't check if the word is not in the dictionary. (See `b` below.)

* a. I refuse to collect the refuse around here. (rɪ|fju:z as verb vs. |refju:s as noun)
* b. I am an activationist. (activationist: newly coined word which means `n. A person who designs and implements programs of treatment or therapy that use recreation and activities to help people whose functional abilities are affected by illness or disability.`
from [WORD SPY](https://wordspy.com/index.php?word=activationist])

For the first homograph issue, fortunately many homographs can be disambiguated using their part-of-speech, if not all.
When it comes to the words not in the dictionary, however, we should make our best guess using our knowledge.
In this project, we employ a deep learning seq2seq framework based on TensorFlow.

## Algorithm

1. Spells out arabic numbers and some currency symbols. (e.g. $200 -> two hundred dollars) (This is borrowed from [Keith Ito's code](https://github.com/keithito/tacotron/blob/master/text/numbers.py))
2. Attempts to retrieve the correct pronunciation for homographs based on their POS)
3. Looks up [The CMU Pronouncing Dictionary](http://www.speech.cs.cmu.edu/cgi-bin/cmudict) for non-homographs.
4. For OOVs, we predict their pronunciations using our neural net model.

## Environment

* python 2.x or 3.x

## Dependencies

* numpy >= 1.13.1
* tensorflow >= 1.3.0
* nltk >= 3.2.4
* python -m nltk.downloader "averaged_perceptron_tagger" "cmudict"
* inflect >= 0.3.1
* Distance >= 0.1.3
* future >= 0.16.0

## Installation

    pip install g2p
OR

    python setup.py install


## Training (Note that pretrained model is already included)

    python train.py

## Usage

    from g2p import g2p

    text = "I refuse to collect the refuse around here."
    print(g2p(text))
    >>>[u'AY1', ' ', u'R', u'IH0', u'F', u'Y', u'UW1', u'Z', ' ', u'T', u'UW1', ' ', u'K', u'AH0', u'L', u'EH1', u'K', u'T', ' ', u'DH', u'AH0', ' ', u'R', u'EH1', u'F', u'Y', u'UW2', u'Z', ' ', u'ER0', u'AW1', u'N', u'D', ' ', u'HH', u'EH1', u'R']

    text = "I am an activationist."
    print(g2p(text))
    >>>[u'AY1', u'M', ' ', u'AE1', u'N', ' ', u'AE2', u'K', u'T', u'AH0', u'V', u'EY1', u'SH', u'AH0', u'N', u'IH0', u'S', u'T']

May, 2018.

Kyubyong Park & [Jongseok Kim](https://github.com/ozmig77)
