# -*- coding: utf-8 -*-
# /usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/g2p
'''
from __future__ import print_function

import tensorflow as tf
from train import Graph, hp, load_vocab
import numpy as np
import codecs
import re
import os, sys
import unicodedata
from expand import normalize_numbers

from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import cmudict



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

g_sess = None
class Session:
    def __enter__(self):
        global g_sess
        if g_sess != None:
            raise Exception('Session already exist in g2p')
        g_sess = tf.Session(graph=g, config=config)
        saver.restore(g_sess, tf.train.latest_checkpoint(hp.logdir))

    def __exit__(self, exc_type, exc_val, exc_tb):
        global g_sess
        g_sess.close()
        g_sess = None

def predict(word, sess):
    '''
    Returns predicted pronunciation of `word` which does NOT exist in the dictionary.
    :param word: string.
    :return: pron: A list of phonemes
    '''
    if len(word)>5:
        pron = predict(word[5:],sess)
    else:
        pron = []

    x = np.zeros((len(word),hp.maxlen), np.int32) # 0: <PAD>
    for i,w in enumerate(word):
        for j,g in enumerate((w+"E")[:hp.maxlen]):
            x[i][j] = g2idx.get(g,2) # 2:<UNK>
    preds = np.zeros((len(word),hp.maxlen), np.int32)
    '''
    graphemes = word + "E"  # EOS
    graphemes += "P" * hp.maxlen  # Padding

    x = [g2idx.get(g, 2) for g in graphemes[:hp.maxlen]]  # 2: <UNK>
    x = np.array(x, np.int32)
    x = np.expand_dims(x, 0) # (1, maxlen)


    ## Autoregressive inference
    preds = np.zeros((1, hp.maxlen), np.int32)
    '''
    for j in range(hp.maxlen):
        _preds = sess.run(graph.preds, {graph.x: x, graph.y: preds})
        preds[:, j] = _preds[:, j]
    '''
    # convert to string
    pron = [idx2p[idx] for idx in preds[0]]
    if "<EOS>" in pron:
        eos = pron.index("<EOS>")
        pron = pron[:eos]
    '''
    # convert to string
    for i in range(len(word)):
        p = [idx2p[idx] for idx in preds[i]]
        if "<EOS>" in p:
            eos = p.index("<EOS>")
            p = p[:eos]
        pron.append(p)
    return pron

# Construct homograph dictionary
f = 'homographs.en'
homograph2features = dict()
for line in codecs.open(f, 'r', 'utf8').read().splitlines():
    if line.startswith("#"): continue # comment
    headword, pron1, pron2, pos1 = line.strip().split("|")
    homograph2features[headword.lower()] = (pron1.split(), pron2.split(), pos1)

def token2pron(token):
    '''
    Returns pronunciation of word based on its pos.
    :param token: A tuple of (word, pos)
    :return: A list of phonemes
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
        return [], True
        #pron = predict(word,sess)

    return pron, False

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

    # tokenization
    words = word_tokenize(text)
    tokens = pos_tag(words) # tuples of (word, tag)

    # g2p
    unseen = [] # Process unseen word at last 
    u_loc = []
    ret = []
    for token in tokens:
        pron, is_unseen = token2pron(token) # list of phonemes
        if is_unseen:
            #unseen.append((token[0], len(ret))) # add (word, location)
            unseen.append(token[0])
            u_loc.append(len(ret))
        ret.extend(pron)
        ret.extend([" "])
    if len(unseen)>0:
        global g_sess
        if g_sess != None: # Already defined
            prons = predict(unseen,g_sess)
            for i in range(len(unseen)-1,-1,-1):
                    ret = ret[:u_loc[i]]+prons[i]+ret[u_loc[i]:]
        else: # If not defined, assign new one.
            with tf.Session(graph=g, config=config) as sess:
                saver.restore(sess, tf.train.latest_checkpoint(hp.logdir))#; print("Restored!")
                #for u in reversed(unseen):
                #    pron = predict(u[0],sess)
                #    ret = ret[:u[1]]+pron+ret[u[1]:]
                prons = predict(unseen,sess)
                for i in range(len(unseen)-1,-1,-1):
                    ret = ret[:u_loc[i]]+prons[i]+ret[u_loc[i]:]
    return ret[:-1]


if __name__ == '__main__':
    text = u"I need your Résumé. She is my girl. He's my activationist activationist activationist."
    text = u"abb "*20
    out = g2p(text)
    print(out)



# GPU : 7.83 (#3)
# CPU : 3.32 (#3)
# CPU : using Graph. 2.91 (#3)
# CPU : run session at first 1.51 (#3)
# GPU : run session at first 5.61 (#3)

# CPU : not using batch, activationist#10, 5.73 (#3)
# CPU : using batch, activationist#10, 3.24 (#3)
