# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/g2p
'''
from __future__ import print_function

import tqdm
import distance

import tensorflow as tf
import numpy as np

from nltk.corpus import cmudict
cmu = cmudict.dict()

# Hyper parameters
class hp:
    batch_size = 128
    lr = 0.0001
    logdir = "logdir"
    maxlen = 20
    num_epochs = 15
    hidden_units = 128
    graphemes = ["P", "E", "U"] + list("abcdefghijklmnopqrstuvwxyz") # Padding, EOS, UNK
    phonemes = ["<PAD>", "<EOS>", 'AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2', 'AO0',
                'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2', 'B', 'CH', 'D', 'DH',
                'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0', 'EY1', 'EY2', 'F', 'G', 'HH',
                'IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'IY2', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW0', 'OW1',
                'OW2', 'OY0', 'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH0', 'UH1', 'UH2', 'UW',
                'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH']

def load_vocab():
    g2idx = {g: idx for idx, g in enumerate(hp.graphemes)}
    idx2g = {idx: g for idx, g in enumerate(hp.graphemes)}

    p2idx = {p: idx for idx, p in enumerate(hp.phonemes)}
    idx2p = {idx: p for idx, p in enumerate(hp.phonemes)}

    return g2idx, idx2g, p2idx, idx2p

g2idx, idx2g, p2idx, idx2p = load_vocab()

def load_data(mode="train"):
    # Vectorize
    xs, ys = [], []  # vectorized sentences
    for word, prons in cmu.items():
        graphemes = word + "E" # EOS
        if len(graphemes) > hp.maxlen: continue
        graphemes += "P" * hp.maxlen # Padding

        x = [g2idx.get(g, 2) for g in graphemes[:hp.maxlen]] # 2: <UNK>

        pron = prons[0]
        phonemes = list(pron) + ["<EOS>"]
        if len(phonemes) > hp.maxlen: continue
        phonemes += ["<PAD>"] * hp.maxlen
        y = [p2idx[p] for p in phonemes[:hp.maxlen]]

        xs.append(x)
        ys.append(y)

    # Convert to 2d-arrays
    X = np.array(xs, np.int32)
    Y = np.array(ys, np.int32)

    if mode=="train":
        X, Y = X[:-hp.batch_size], Y[:-hp.batch_size]
    else: # eval
        X, Y = X[-hp.batch_size:], Y[-hp.batch_size:]

    return X, Y

class Graph():
    '''Builds a model graph'''
    def __init__(self):
        self.x = tf.placeholder(tf.int32, shape=(None, hp.maxlen), name="grapheme")
        self.y = tf.placeholder(tf.int32, shape=(None, hp.maxlen), name="phoneme")

        # Sequence lengths
        self.seqlens = tf.reduce_sum(tf.sign(self.x), -1)

        # Embedding
        self.inputs = tf.one_hot(self.x, len(hp.graphemes))

        # Encoder: BiGRU
        cell_fw = tf.nn.rnn_cell.GRUCell(hp.hidden_units)
        cell_bw = tf.nn.rnn_cell.GRUCell(hp.hidden_units)
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, self.inputs, self.seqlens, dtype=tf.float32)
        memory = tf.concat(outputs, -1)

        # Decoder : Attentional GRU
        decoder_inputs = tf.concat((tf.zeros_like((self.y[:, :1])), self.y[:, :-1]), -1)
        decoder_inputs = tf.one_hot(decoder_inputs, len(hp.phonemes))
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(hp.hidden_units, memory, self.seqlens)
        cell = tf.nn.rnn_cell.GRUCell(hp.hidden_units)
        cell_with_attention = tf.contrib.seq2seq.AttentionWrapper(cell,
                                                                  attention_mechanism,
                                                                  hp.hidden_units,
                                                                  alignment_history=True)
        outputs, _ = tf.nn.dynamic_rnn(cell_with_attention, decoder_inputs, dtype=tf.float32)  # ( N, T', 16)
        logits = tf.layers.dense(outputs, len(hp.phonemes))
        self.preds = tf.to_int32(tf.argmax(logits, -1))

        ## Loss and training
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.y)
        self.mean_loss = tf.reduce_mean(loss)
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        optimizer = tf.train.AdamOptimizer(hp.lr)
        self.train_op = optimizer.minimize(self.mean_loss, global_step=self.global_step)

if __name__ == '__main__':
    # Data loading
    X_train, Y_train = load_data(mode="train")
    x_val, y_val = load_data(mode="val")

    # Graph loading
    g = Graph(); print("Training Graph loaded")

    # Session
    sv = tf.train.Supervisor(logdir=hp.logdir, save_model_secs=0)
    with sv.managed_session() as sess:
        for epoch in range(hp.num_epochs):
            # shuffle
            ids = np.arange(len(X_train))
            np.random.shuffle(ids)
            X_train, Y_train = X_train[ids], Y_train[ids]

            # batch train
            for i in tqdm.tqdm(range(0, len(X_train), hp.batch_size), total=len(X_train) // hp.batch_size):
                x_train = X_train[i: i + hp.batch_size]
                y_train = Y_train[i: i + hp.batch_size]
                _, loss = sess.run([g.train_op, g.mean_loss], {g.x: x_train, g.y: y_train})

            # Write checkpoint files at every epoch
            gs = sess.run(g.global_step)
            sv.saver.save(sess, hp.logdir + '/model_epoch_%02d_gs_%d' % (epoch, gs))

            # Evaluation
            ## Autoregressive inference
            preds = np.zeros((hp.batch_size, hp.maxlen), np.int32)
            for j in range(hp.maxlen):
                _preds = sess.run(g.preds, {g.x: x_val, g.y: preds})
                preds[:, j] = _preds[:, j]

            ## Parsing & Calculation
            total, errors = 0, 0
            bugs = 0
            for xx, yy, pred in zip(x_val, y_val, preds): # sample-wise
                inputs = "".join(idx2g[g] for g in xx).split("E")[0]
                expected = " ".join(idx2p[p] for p in yy).split("<EOS>")[0].strip()
                got = " ".join(idx2p[p] for p in pred).split("<EOS>")[0].strip()

                print("* Input   : {}".format(inputs))
                print("* Expected: {}".format(expected))
                print("* Got     : {}".format(got))

                error = distance.levenshtein(expected.split(), got.split())
                errors += error
                total += len(expected.split())

            cer = errors / float(total)
            print("epoch: %02d, training loss: %02f,  CER: %02f\n" % (epoch+1, loss, cer))