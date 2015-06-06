"""
This tutorial introduces denoising auto-encoders (dA) using Theano.

Starter code from the deeplearning.net's denoising auto-encoder.

References :
   - P. Vincent, H. Larochelle, Y. Bengio, P.A. Manzagol: Extracting and
   Composing Robust Features with Denoising Autoencoders, ICML'08, 1096-1103,
   2008
   - Y. Bengio, P. Lamblin, D. Popovici, H. Larochelle: Greedy Layer-Wise
   Training of Deep Networks, Advances in Neural Information Processing
   Systems 19, 2007

Barak Oshri, Nishith Khandwala
"""
import csv
import os
import sys
import pickle
import time

import numpy
import numpy as np

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

class dA(object):
    def __init__(
        self,
        numpy_rng,
        input_size=None,
        theano_rng=None,
        wvec_dim=50,
        n_dict=None,
        input=None,
        n_hidden=50,
        H1=None,
        H2=None,
        C=None,
        Y=None,
        S=None,
        bhid=None,
        word_vectors=None,
        bvis=None
        batch_size = 5
    ):

        self.n_hidden = n_hidden
        self.word_vectors = word_vectors
        self.n_dict = n_dict
        self.f = T.nnet.sigmoid
        self.input_size = input_size
        self.batch_count = 0
        self.batch_size = batch_size

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        if not H1:
            initial_H1 = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (self.n_hidden)),
                    high=4 * numpy.sqrt(6. / (self.n_hidden)),
                    size=(self.n_hidden, self.n_hidden)
                ),
                dtype=theano.config.floatX
            )
            H1 = theano.shared(value=initial_H1, name='H1', borrow=True)

        if not H2:
            initial_H2 = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (self.n_hidden)),
                    high=4 * numpy.sqrt(6. / (self.n_hidden)),
                    size=(self.n_hidden, self.n_hidden)
                ),
                dtype=theano.config.floatX
            )
            H2 = theano.shared(value=initial_H2, name='H2', borrow=True)

        if not C:
            initial_C = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (self.n_hidden)),
                    high=4 * numpy.sqrt(6. / (self.n_hidden)),
                    size=(self.n_hidden, self.n_hidden)
                ),
                dtype=theano.config.floatX
            )
            C = theano.shared(value=initial_C, name='C', borrow=True)

        if not Y:
            initial_Y = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (self.n_hidden)),
                    high=4 * numpy.sqrt(6. / (self.n_hidden)),
                    size=(self.n_hidden, self.n_dict)
                ),
                dtype=theano.config.floatX
            )
            Y = theano.shared(value=initial_Y, name='Y', borrow=True)

        if not S:
            initial_S = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (self.n_hidden)),
                    high=4 * numpy.sqrt(6. / (self.n_hidden)),
                    size=(self.n_dict, self.n_hidden)
                ),
                dtype=theano.config.floatX
            )
            S = theano.shared(value=initial_S, name='S', borrow=True)

        self.H1 = H1
        self.H2 = H2
        self.C = C
        self.Y = Y
        self.S = S

        self.theano_rng = theano_rng

        if input is None:
            self.x = T.dmatrix(name='input')
        else:
            self.x = input

        self.params = [self.H1, self.S, self.Y, self.C, self.H2]
        self.updates = [0] * len(self.params)

    def get_params(self):
        return {'H1': self.H1, 'H2': self.H2, 'C': self.C, 'Y': self.Y, 'S': self.S}

    def get_hidden_values(self, sentence):
        self.input_count = 0
        h0 = theano.shared(value=np.zeros((self.n_hidden), dtype=theano.config.floatX))
        h, _ = theano.scan(fn = self._rnn_recurrence, sequences = sentence, outputs_info = h0)

        return h[-1]

    def _rnn_recurrence(self, x_t, h_tm1):
        h_t = self.f(T.dot(self.H1, h_tm1) + self.word_vectors[x_t])
        self.input_count += 1
        return h_t

    def get_reconstructed_input(self, c):
        h0 = self.f(T.dot(self.H1, c) + T.dot(self.C, c))
        a0 = T.dot(self.S, h0)
        s0 = T.reshape(T.nnet.softmax(a0), a0.shape)

        [h, s], _ = theano.scan(fn = self._decoder_recurrence, outputs_info = [h0, s0], non_sequences = c, n_steps = self.input_size)
        y = T.argmax(s, axis=1)

        return y, s

    def _decoder_recurrence(self, ht_1, yt_1, hidden):
        h_t = self.f(T.dot(self.H2, ht_1) + T.dot(self.Y, yt_1) + T.dot(self.C, hidden))
        a = T.dot(self.S, h_t)
        s_t = T.reshape(T.nnet.softmax(a), a.shape)

        return [h_t, s_t]

    def _decoder_recurrence_until(self, ht_1, yt_1, hidden):
        h_t = self.f(T.dot(self.H2, ht_1) + T.dot(self.Y, yt_1) + T.dot(self.C, hidden))
        a = T.dot(self.S, h_t)
        s_t = T.reshape(T.nnet.softmax(a), a.shape)

        return [h_t, s_t], theano.scan_module.until(T.eq(T.argmax(s_t), self.end_token))

    def get_new_reconstructed_input(self, c):
        h0 = self.f(T.dot(self.H1, c) + T.dot(self.C, c))
        a0 = T.dot(self.S, h0)
        s0 = T.reshape(T.nnet.softmax(a0), a0.shape)

        [h, s], _ = theano.scan(fn = self._decoder_recurrence_until, outputs_info = [h0, s0], non_sequences = c, n_steps = self.input_size)
        y = T.argmax(s, axis=1)

        return y

    def get_cost_updates(self, corruption_level, learning_rate):
        #tilde_x = self.get_corrupted_input(self.x, corruption_level)
        hidden = self.get_hidden_values(self.x)
        output, softmaxes = self.get_reconstructed_input(hidden)
        output_hidden = self.get_hidden_values(output)

        cost = T.sqrt(T.sum(T.sqr(output_hidden - hidden))) + T.sum(T.nnet.categorical_crossentropy(softmaxes, self.x))

        gparams = T.grad(cost, self.params)

        if self.batch_count == self.batch_size:
            updates = [
                (self.params[i], self.params[i] - learning_rate * (gparams[i] + self.updates[i]))
                for i in range(len(gparams))
            ]
            self.updates = [0] * len(self.params) 
            self.batch_count = 0
        else:
            for i in range(len(gparams)):
                self.updates[i] += gparams[i]
            self.batch_count += 1

        return (cost, updates)

    def get_cost_and_reconstructions(self, corruption_level):
        hidden = self.get_hidden_values(self.x)
        output, softmaxes = self.get_new_reconstructed_input(hidden)
        output_hidden = 



def build(src_filename, delimiter=',', header=True, quoting=csv.QUOTE_MINIMAL):    
    # Thanks to Prof. Chris Potts, Stanford University, for this function
    reader = csv.reader(file(src_filename), delimiter=delimiter, quoting=quoting)
    colnames = None
    if header:
        colnames = reader.next()
        colnames = colnames[1: ]
    mat = []    
    rownames = []
    for line in reader:        
        rownames.append(line[0])            
        mat.append(np.array(map(float, line[1: ])))
    return (np.array(mat), rownames, colnames)

def make_dA(params=False, data=data):
    glove_matrix, glove_vocab, _ = build('glove.6B.50d.txt', delimiter=' ', header=False, quoting=csv.QUOTE_NONE)
    glove_matrix = theano.shared(value=np.array(glove_matrix, dtype=theano.config.floatX), borrow=True)

    if params == False:
        da = dA(
            numpy_rng=rng,
            theano_rng=theano_rng,
            input=data,
            wvec_dim=50,
            n_hidden=50,
            word_vectors=glove_matrix,
            n_dict=len(glove_vocab),
            input_size=input_size
        )
    else:
        da = dA(
            numpy_rng=rng,
            theano_rng=theano_rng,
            input=data,
            wvec_dim=50,
            n_hidden=50,
            word_vectors=glove_matrix,
            n_dict=len(glove_vocab),
            input_size=input_size
            H1=params['H1']
            H2=params['H2']
            C=params['C']
            Y=params['Y']
            S=params['S']
        )

def train_dA(learning_rate=0.1, training_epochs=15,
            dataset='mnist.pkl.gz', output_folder='dA_plots', params_dict = False, print_every = 100
            data=None):

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder)

    X_train = map(lambda x: np.array(x), sentences[1:101])

    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))
    x = T.lvector('x')
    input_size = T.scalar(dtype='int64')

    dA = make_dA(params=params_dict, data=data)

    cost, updates = dA.get_cost_updates(
        corruption_level=0.,
        learning_rate=learning_rate
    )

    train_da = theano.function(
        [x],
        cost,
        updates=updates,
        givens={
            input_size: x.shape[0]
        }
    )

    start_time = time.clock()
    for epoch in xrange(training_epochs):
        cost_history = []
        for index in len(data):
            cost_history.append(train_da(X_train[index]))
            if index % print_every == 0:
                print 'Iteration %d, mean cost %d' % (index, numpy.mean(cost_history))

        print 'Training epoch %d, cost ' % epoch, numpy.mean(cost_history)

    training_time = (time.clock() - start_time)

    print 'Finished training %d epochs, took %d seconds' % (training_epochs, training_time)

    return cost_history, da.get_params(), dA, train_da

def test_dA(params_dict, print_count=10, data=None):
    dA = make_dA(parms=params_dict, data=data)

    train_da = theano.function(
        [x],
        cost,
        updates=updates,
        givens={
            input_size: x.shape[0]
        }
    )

if __name__ == '__main__':
    f = open("all_wvi.pkl", 'r')
    sentences = pickle.load(f)
    train_sentences = sentences[:100]
    test_sentences = sentences[100:200]

    cost_history, params, dA, train_da = train_dA(data=train_sentences, epochs=5)
    sample_sentences, cost, test_da = test_dA(params, print_count=20, data=test_sentences)
