import numpy as np
import theano
from theano import tensor as T

from attrdict import AttrDict

class Decoder:

    def _set_parameters(self, params=None):
        raise NotImplementedError()

    def get_reconstructed_input(self, c):
        raise NotImplementedError()


class RNN(Decoder):

    def __init__(self, params=None, numpy_rng=None, n_hidden=None, dict_size=None):
        self.n_hidden = n_hidden
        self.dict_size = dict_size
        if numpy_rng is None:
            self.numpy_rng = np.random.RandomState(None)
        else:
            self.numpy_rng = numpy_rng
        self.params = self._set_parameters(params)
        self.end_token = T.constant(0)

    def _set_parameters(self, params=None):
        if not params:
            initial_H1 = np.asarray(
                self.numpy_rng.uniform(
                    low=-4 * np.sqrt(6. / (self.n_hidden + self.n_hidden)),
                    high=4 * np.sqrt(6. / (self.n_hidden + self.n_hidden)),
                    size=(self.n_hidden, self.n_hidden)
                ),
                dtype=theano.config.floatX
            )

            initial_Y = np.asarray(
                self.numpy_rng.uniform(
                    low=-4 * np.sqrt(6. / (self.n_hidden + self.dict_size)),
                    high=4 * np.sqrt(6. / (self.n_hidden + self.dict_size)),
                    size=(self.n_hidden, self.dict_size)
                ),
                dtype=theano.config.floatX
            )

            initial_C = np.asarray(
                self.numpy_rng.uniform(
                    low=-4 * np.sqrt(6. / (self.n_hidden + self.n_hidden)),
                    high=4 * np.sqrt(6. / (self.n_hidden + self.n_hidden)),
                    size=(self.n_hidden, self.n_hidden)
                ),
                dtype=theano.config.floatX
            )

            initial_S = np.asarray(
                self.numpy_rng.uniform(
                    low=-4 * np.sqrt(6. / (self.n_hidden + self.dict_size)),
                    high=4 * np.sqrt(6. / (self.n_hidden + self.dict_size)),
                    size=(self.dict_size, self.n_hidden)
                ),
                dtype=theano.config.floatX
            )

            initial_B = np.asarray(
                self.numpy_rng.uniform(
                    low=-4 * np.sqrt(6. / (self.dict_size)),
                    high=4 * np.sqrt(6. / (self.dict_size)),
                    size=(self.dict_size)
                ),
                dtype=theano.config.floatX
            )

            params = {
                'H1' : theano.shared(value=initial_H1, name='H1', borrow=True),
                'Y' : theano.shared(value=initial_Y, name='Y', borrow=True),
                'C' : theano.shared(value=initial_C, name='C', borrow=True),
                'S' : theano.shared(value=initial_S, name='S', borrow=True),
                'B' : theano.shared(value=initial_B, name='B', borrow=True),
            }

        return AttrDict(params)

    def get_reconstructed_input(self, c):
        h0 = T.tanh(T.dot(self.params.H1, c) + T.dot(self.params.C, c))
        a0 = T.dot(self.params.S, h0) + self.params.B
        s0 = T.reshape(T.nnet.softmax(a0), a0.shape)

        [h, s], _ = theano.scan(fn = self._recurrence, outputs_info = [h0, s0], non_sequences = c, n_steps = 20)
        y = T.argmax(s, axis=1)

        return y

    def _recurrence(self, ht_1, yt_1, hidden):
        h_t = T.tanh(T.dot(self.params.H1, ht_1) + T.dot(self.params.Y, yt_1) + T.dot(self.params.C, hidden))
        a = T.dot(self.params.S, h_t) + self.params.B
        s_t = T.reshape(T.nnet.softmax(a), a.shape)

        return [h_t, s_t], theano.scan_module.until(T.eq(np.argmax(T.nnet.softmax(T.dot(self.params.S, T.tanh(T.dot(self.params.H1, ht_1) + T.dot(self.params.Y, yt_1) + T.dot(self.params.C, hidden))) + self.params.B)), self.end_token))