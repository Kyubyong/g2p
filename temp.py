import urllib.request
import codecs
import re
from expand import normalize_numbers
import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import cmudict
import unicodedata
nltk.download('cmudict')
cmu = cmudict.dict()
#
# # step1. construct cmu lookup dictionary
# url = "http://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict-0.7b"
#
# cmu = dict()
# lines = urllib.request.urlopen(url).read().decode('utf-8', errors="replace").splitlines()
# for line in lines:
#     if line.startswith(";;;"): continue # comments
#     word, pron = line.split("  ")
#     cmu[word] = pron

# step2. construct homograph dictionary
f = 'homographs.en'
homograph2features = dict()
for line in codecs.open(f, 'r', 'utf8').read().splitlines():
    if line.startswith("#"): continue # comment
    # print(line.strip().split("|"))
    headword, pron1, pron2, pos1 = line.strip().split("|")
    homograph2features[headword] = (pron1, pron2, pos1)

# step3. Load trained model
def word2pron(word):
    pron = ""
    pref, suf = word, ""
    while len(pref) > 0:
        print(pref, suf)

        if pref in cmu:
            # print(pref)
            pron += cmu[pref] + " "
            pref, suf = suf, ""
        else:
            pref, suf = pref[:-1], pref[-1] + suf
            if pref == "":
                pron += suf[0] + " "
                pref, suf = suf[1:], ""

    pron = pron.strip()
    if pron == "":
        return word
    else:
        return pron

word = "resistentialism".upper()
out = word2pron(word)
print(out)
def token2pron(token):
    word, pos = token

    word = ''.join(char for char in unicodedata.normalize('NFD', word)
                   if unicodedata.category(char) != 'Mn')  # Strip accents
    word = word.lower()

    if word in homograph2features: # Check homograph
        pron1, pron2, pos1 = homograph2features[word]
        if pos.startswith(pos1):
            pron = pron1
        else:
            pron = pron2
    elif word in word2pron: # CMU dict
        pron = word2pron(word)
    else:
        pron = model.predict
    return pron

#
#
#
#
def g2p(text):
    text = normalize_numbers(text)
    tokens = word_tokenize(text)
    ret = []
    for token in tokens:
        pron = token2pron(token) # list of phonemes
        ret.extend(pron)
        ret.extend([" "])
    ret = ret[:-1]
    return ret
#
#
# AH0 F AO1 R D AH0 B AH0 L
#
# ["A", "I", ".", " ", "A"]
#
#
