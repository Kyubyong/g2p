# g2pE: A Simple Python Module for English Grapheme To Phoneme Conversion

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
2. Attempts to retrieve the correct pronunciation for heteronyms based on their POS)
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


## Usage

    from g2p_en import G2p
    
    texts = ["I have $250 in my pocket.", # number -> spell-out
             "popular pets, e.g. cats and dogs", # e.g. -> for example
             "I refuse to collect the refuse around here.", # homograph
             "I'm an activationist."] # newly coined word
    g2p = G2p()
    for text in texts:
        out = g2p(text)
        print(out)
    >>> ['AY1', ' ', 'HH', 'AE1', 'V', ' ', 'T', 'UW1', ' ', 'HH', 'AH1', 'N', 'D', 'R', 'AH0', 'D', ' ', 'F', 'IH1', 'F', 'T', 'IY0', ' ', 'D', 'AA1', 'L', 'ER0', 'Z', ' ', 'IH0', 'N', ' ', 'M', 'AY1', ' ', 'P', 'AA1', 'K', 'AH0', 'T', ' ', '.']
    >>> ['P', 'AA1', 'P', 'Y', 'AH0', 'L', 'ER0', ' ', 'P', 'EH1', 'T', 'S', ' ', ',', ' ', 'F', 'AO1', 'R', ' ', 'IH0', 'G', 'Z', 'AE1', 'M', 'P', 'AH0', 'L', ' ', 'K', 'AE1', 'T', 'S', ' ', 'AH0', 'N', 'D', ' ', 'D', 'AA1', 'G', 'Z']
    >>> ['AY1', ' ', 'R', 'IH0', 'F', 'Y', 'UW1', 'Z', ' ', 'T', 'UW1', ' ', 'K', 'AH0', 'L', 'EH1', 'K', 'T', ' ', 'DH', 'AH0', ' ', 'R', 'EH1', 'F', 'Y', 'UW2', 'Z', ' ', 'ER0', 'AW1', 'N', 'D', ' ', 'HH', 'IY1', 'R', ' ', '.']
    >>> ['AY1', ' ', 'AH0', 'M', ' ', 'AE1', 'N', ' ', 'AE2', 'K', 'T', 'IH0', 'V', 'EY1', 'SH', 'AH0', 'N', 'IH0', 'S', 'T', ' ', '.']

## References

If you use this code for research, please cite:

```
@misc{g2pE2019,
  author = {Park, Kyubyong & Kim, Jongseok},
  title = {g2pE},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Kyubyong/g2p}}
}
```

## Cited in
* [Learning pronunciation from a foreign language in speech synthesis networks](https://arxiv.org/abs/1811.09364)

May, 2018.

Kyubyong Park & [Jongseok Kim](https://github.com/ozmig77)
