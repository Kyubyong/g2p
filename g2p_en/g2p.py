# -*- coding: utf-8 -*-
# /usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/g2p
'''
from __future__ import print_function

import tensorflow as tf

from nltk import pos_tag
from nltk.corpus import cmudict
import nltk
try:
    nltk.data.find('taggers/averaged_perceptron_tagger.zip')
except LookupError:
    nltk.download('averaged_perceptron_tagger')
try:
    nltk.data.find('corpora/cmudict.zip')
except LookupError:
    nltk.download('cmudict')

from .train import Graph, hp, load_vocab
from .expand import normalize_numbers

import numpy as np
import codecs
import re
import os
import unicodedata
from builtins import str as unicode

dirname = os.path.dirname(__file__)

cmu = cmudict.dict()

# Load vocab
g2idx, idx2g, p2idx, idx2p = load_vocab()

# Load Graph
g = tf.Graph()
with g.as_default():
    with tf.device('/cpu:0'):
        graph = Graph(); print("Graph loaded for g2p")
        saver = tf.train.Saver()
config = tf.ConfigProto(
             device_count={'GPU' : 0},
             gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.0001)
         )

g_sess = None # global session
class Session: # make/remove global session
    def __enter__(self):
        global g_sess
        if g_sess != None:
            raise Exception('Session already exist in g2p')
        g_sess = tf.Session(graph=g, config=config)
        saver.restore(g_sess, tf.train.latest_checkpoint(os.path.join(dirname,hp.logdir)))

    def __exit__(self, exc_type, exc_val, exc_tb):
        global g_sess
        g_sess.close()
        g_sess = None


def predict(words, sess):
    '''
    Returns predicted pronunciation of `words` which do NOT exist in the dictionary.
    :param words: A list of words.
    :return: pron: A list of phonemes
    '''
    if len(words) > hp.batch_size:
        after = predict(words[hp.batch_size:], sess)
        words = words[:hp.batch_size]
    else:
        after = []
    x = np.zeros((len(words), hp.maxlen), np.int32)  # 0: <PAD>
    for i, w in enumerate(words):
        for j, g in enumerate((w + "E")[:hp.maxlen]):
            x[i][j] = g2idx.get(g, 2)  # 2:<UNK>

    ## Autoregressive inference
    preds = np.zeros((len(x), hp.maxlen), np.int32)
    for j in range(hp.maxlen):
        _preds = sess.run(graph.preds, {graph.x: x, graph.y: preds})
        preds[:, j] = _preds[:, j]

    # convert to string
    pron = []
    for i in range(len(preds)):
        p = [u"%s" % unicode(idx2p[idx]) for idx in preds[i]]  # Make p into unicode.
        if "<EOS>" in p:
            eos = p.index("<EOS>")
            p = p[:eos]
        pron.append(p)

    return pron + after

# Construct homograph dictionary
f = os.path.join(dirname,'homographs.en')
homograph2features = dict()
for line in codecs.open(f, 'r', 'utf8').read().splitlines():
    if line.startswith("#"): continue # comment
    headword, pron1, pron2, pos1 = line.strip().split("|")
    homograph2features[headword.lower()] = (pron1.split(), pron2.split(), pos1)

def token2pron(token):
    '''
    Returns pronunciation of word based on its pos.
    :param token: A tuple of (word, pos)
    :return: A list of phonemes. If word is not in the dictionary, [] is returned.
    '''
    word, pos = token

    if re.search("[a-z]", word) is None:
        pron = [word]

    elif word in homograph2features: # Check homograph
        pron1, pron2, pos1 = homograph2features[word]
        if pos.startswith(pos1):
            pron = pron1
        else:
            pron = pron2
    elif word in cmu: # CMU dict
        pron = cmu[word][0]
    else:
        return []

    return pron

def tokenize(text):
    '''
    Splits text into `tokens`.
    :param text: A string.
    :return: A list of tokens (string).
    '''
    text = re.sub('([.,?!]( |$))', r' \1', text)
    return text.split()

def g2p(text):
    '''
    Returns the pronunciation of text.
    :param text: A string. A sequence of words.
    :return: A list of phonemes.
    '''
    # normalization
    text = unicode(text)
    text = normalize_numbers(text)
    text = ''.join(char for char in unicodedata.normalize('NFD', text)
                   if unicodedata.category(char) != 'Mn')  # Strip accents
    text = text.lower()
    text = re.sub("[^ a-z'.,?!\-]", "", text)
    text = text.replace("i.e.", "that is")
    text = text.replace("e.g.", "for example")

    # tokenization
    words = tokenize(text)
    tokens = pos_tag(words) # tuples of (word, tag)

    # g2p
    oovs, u_loc = [], []
    ret = []
    for token in tokens:
        pron = token2pron(token) # list of phonemes
        if pron == []: # oov
            oovs.append(token[0])
            u_loc.append(len(ret))
        ret.extend(pron)
        ret.extend([" "])

    if len(oovs)>0:
        global g_sess
        if g_sess is not None: # check global session
            prons = predict(oovs, g_sess)
            for i in range(len(oovs)-1,-1,-1):
                    ret = ret[:u_loc[i]]+prons[i]+ret[u_loc[i]:]
        else: # If global session is not defined, make new one as local.
            with tf.Session(graph=g, config=config) as sess:
                saver.restore(sess, tf.train.latest_checkpoint(os.path.join(dirname, hp.logdir)))
                prons = predict(oovs, sess)
                for i in range(len(oovs)-1,-1,-1):
                    ret = ret[:u_loc[i]]+prons[i]+ret[u_loc[i]:]
    return ret[:-1]


if __name__ == '__main__':
    texts = ["I have $250 in my pocket.", # number -> spell-out
             "popular pets, e.g. cats and dogs", # e.g. -> for example
             "I refuse to collect the refuse around here.", # homograph
             "I'm an activationist."] # newly coined word
    for text in texts:
        out = g2p(text)
        print(out)

