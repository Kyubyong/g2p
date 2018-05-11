import tensorflow as tf
from nltk.corpus import cmudict
import unicodedata
cmu = cmudict.dict()
import numpy as np
import codecs
import tqdm

# Hyper parameters
BATCH_SIZE = 1024
LEARNING_RATE = 0.0001
LOGDIR = "logdir"
MAXLEN = 20
NUM_EPOCHS = 10
HIDDEN_UNITS = 128
GRAPHEMES = ["<PAD>", "<EOS>", "<UNK>"] + list("abcdefghijklmnopqrstuvwxyz")
PHONEMES = ["<PAD>", "<EOS>", 'AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2', 'AO0',
            'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2', 'B', 'CH', 'D', 'DH',
            'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0', 'EY1', 'EY2', 'F', 'G', 'HH',
            'IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'IY2', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW0', 'OW1',
            'OW2', 'OY0', 'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH0', 'UH1', 'UH2', 'UW',
            'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH']

def load_vocab():
    g2idx = {g: idx for idx, g in enumerate(GRAPHEMES)}
    idx2g = {idx: g for idx, g in enumerate(GRAPHEMES)}

    p2idx = {p: idx for idx, p in enumerate(PHONEMES)}
    idx2p = {idx: p for idx, p in enumerate(PHONEMES)}

    return g2idx, idx2g, p2idx, idx2p

g2idx, idx2g, p2idx, idx2p = load_vocab()

def load_data(mode="train"):
    # Vectorize
    xs, ys = [], []  # vectorized sentences
    for word, prons in cmu.items():
        graphemes = list(word) + ["<EOS>"]
        if len(graphemes) > MAXLEN: continue
        graphemes += ["<PAD>"]*MAXLEN

        x = [g2idx.get(g, 2) for g in graphemes[:MAXLEN]] # 2: <UNK>

        for pron in prons:
            phonemes = list(pron) + ["<EOS>"]
            if len(phonemes) > MAXLEN: continue
            phonemes += ["<PAD>"] * MAXLEN
            y = [p2idx[p] for p in phonemes[:MAXLEN]]

            xs.append(x)
            ys.append(y)

    # Convert to 2d-arrays
    X = np.array(xs, np.int32)
    Y = np.array(ys, np.int32)

    if mode=="train":
        X, Y = X[:-BATCH_SIZE], Y[:-BATCH_SIZE]
    else: # eval
        X, Y = X[-BATCH_SIZE:], Y[-BATCH_SIZE:]

    return X, Y

class Graph():
    '''Builds a model graph'''
    def __init__(self):
        self.x = tf.placeholder(tf.int32, shape=(None, MAXLEN), name="grapheme")
        self.y = tf.placeholder(tf.int32, shape=(None, MAXLEN), name="phoneme")

        # Sequence lengths
        self.seqlens = tf.reduce_sum(tf.sign(self.x), -1)

        # Embedding
        self.inputs = tf.one_hot(self.x, len(GRAPHEMES))

        # Encoder
        cell_fw = tf.nn.rnn_cell.GRUCell(HIDDEN_UNITS)
        cell_bw = tf.nn.rnn_cell.GRUCell(HIDDEN_UNITS)
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, self.inputs, self.seqlens, dtype=tf.float32)
        memory = tf.concat(outputs, -1)

        # Decoder
        decoder_inputs = tf.concat((tf.zeros_like((self.y[:, :1])), self.y[:, :-1]), -1)
        decoder_inputs = tf.one_hot(decoder_inputs, len(PHONEMES))
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(HIDDEN_UNITS, memory)
        cell = tf.nn.rnn_cell.GRUCell(HIDDEN_UNITS)
        cell_with_attention = tf.contrib.seq2seq.AttentionWrapper(cell,
                                                                  attention_mechanism,
                                                                  HIDDEN_UNITS,
                                                                  alignment_history=True)
        outputs, _ = tf.nn.dynamic_rnn(cell_with_attention, decoder_inputs, dtype=tf.float32)  # ( N, T', 16)
        logits = tf.layers.dense(outputs, len(PHONEMES))
        self.preds = tf.to_int32(tf.argmax(logits, -1))

        ## Loss and training
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.y)
        self.mean_loss = tf.reduce_mean(loss)
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
        self.train_op = optimizer.minimize(self.mean_loss, global_step=self.global_step)

if __name__ == '__main__':
    # Data loading
    X_train, Y_train = load_data(mode="train")
    x_val, y_val = load_data(mode="val")

    # Graph loading
    g = Graph()
    print("Training Graph loaded")

    # Session
    sv = tf.train.Supervisor(logdir=LOGDIR, save_model_secs=0)
    with sv.managed_session() as sess:
        for epoch in range(NUM_EPOCHS):
            # shuffle
            ids = np.arange(len(X_train))
            np.random.shuffle(ids)
            X_train, Y_train = X_train[ids], Y_train[ids]

            # batch train
            for i in tqdm.tqdm(range(0, len(X_train), BATCH_SIZE), total=len(X_train) // BATCH_SIZE):
                x_train = X_train[i:i + BATCH_SIZE]
                y_train = Y_train[i:i + BATCH_SIZE]
                _, loss = sess.run([g.train_op, g.mean_loss], {g.x: x_train, g.y: y_train})

                print(loss)

            # Write checkpoint files at every epoch
            gs = sess.run(g.global_step)
            sv.saver.save(sess, LOGDIR + '/model_epoch_%02d_gs_%d' % (epoch, gs))


#             # Evaluation
#             preds = sess.run([g.preds, g.acc], {g.x: x_val, g.y: y_val})
#             fout.write(u"\nepoch = {}\n".format(epoch+1))
#             for xx, yy, pred in zip(x_val, y_val, preds): # sentence-wise
#                 inputs, expected, got = [], [], []
#                 for xxx, yyy, ppp in zip(xx, yy, pred):  # character-wise
#                     if xxx==0: break
#                     inputs.append(idx2hangul[xxx])
#                     expected.append(idx2hanja[yyy] if yyy!=1 else idx2hangul[xxx])
#                     got.append(idx2hanja[ppp] if ppp != 1 else idx2hangul[xxx])
#
#                 fout.write(u"* Input   : {}\n".format("".join(inputs)))
#                 fout.write(u"* Expected: {}\n".format("".join(expected)))
#                 fout.write(u"* Got     : {}\n".format("".join(got)))
# fout.write("\n")