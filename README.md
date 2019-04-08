# [UPDATE] g2p_en: A Simple Python Module for English Grapheme To Phoneme Conversion

* [v.2.0] We removed TensorFlow from the dependencies. After all, it changes its APIs quite often, and we don't expect you to have a GPU. Instead, NumPy is used for inference.

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

* python 3.x

## Dependencies

* numpy >= 1.13.1
* nltk >= 3.2.4
* python -m nltk.downloader "averaged_perceptron_tagger" "cmudict"
* inflect >= 0.3.1
* Distance >= 0.1.3

## Installation

    pip install g2p_en
OR

    python setup.py install

nltk package will be automatically downloaded at your first run.

## Training (Note that pretrained model is already included)

    python train.py

## Usage

    from g2p_en import g2p

    text = "I refuse to collect the refuse around here."
    print(g2p(text))
    >>>['AY1', ' ', 'R', 'IH0', 'F', 'Y', 'UW1', 'Z', ' ', 'T', 'UW1', ' ', 'K', 'AH0', 'L', 'EH1', 'K', 'T', ' ', 'DH', 'AH0', ' ', 'R', 'EH1', 'F', 'Y', 'UW2', 'Z', ' ', 'ER0', 'AW1', 'N', 'D', ' ', 'HH', 'EH1', 'R', '.']

    text = "I am an activationist."
    print(g2p(text))
    >>>['AY1', 'M', ' ', 'AE1', 'N', ' ', 'AE2', 'K', 'T', 'AH0', 'V', 'EY1', 'SH', 'AH0', 'N', 'IH0', 'S', 'T']


May, 2018.

Kyubyong Park & [Jongseok Kim](https://github.com/ozmig77)
