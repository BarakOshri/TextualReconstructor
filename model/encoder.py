import numpy as np
import theano
from theano import tensor as T

from attrdict import AttrDict

class Encoder:

    def _set_parameters(self, params=None):
        raise NotImplementedError()
    
    def get_hidden_state(self, sentence):
        raise NotImplementedError()


class RNN(Encoder):

    def __init__(self, word_vectors=None, n_hidden=None, params=None, numpy_rng=None):
        self.word_vectors = word_vectors
        self.n_hidden = n_hidden
        if numpy_rng is None:
            self.numpy_rng = np.random.RandomState(None)
        else:
            self.numpy_rng = numpy_rng
        self.params = self._set_parameters(params=params)
        self.h0 = theano.shared(value=np.zeros((self.n_hidden), dtype=theano.config.floatX))

    def _set_parameters(self, params=None):
        if params is None:
            initial_H2 = np.asarray(
                self.numpy_rng.uniform(
                    low=-4 * np.sqrt(6. / (self.n_hidden + self.n_hidden)),
                    high=4 * np.sqrt(6. / (self.n_hidden + self.n_hidden)),
                    size=(self.n_hidden, self.n_hidden)
                ),
                dtype=theano.config.floatX
            )

            params = {
                'H2' : theano.shared(value=initial_H2, name='H2', borrow=True)
            }

        return AttrDict(params)


    def get_hidden_state(self, sentence):
        h, _ = theano.scan(fn = self._recurrence, sequences = sentence, outputs_info = self.h0)
        hidden_state = h[-1]

        return hidden_state

    def _recurrence(self, x_t, h_tm1):
        #h_t = T.tanh(T.dot(self.params.H2, h_tm1) + self.word_vectors[T.cast(x_t, 'int64')])
        h_t = T.tanh(T.dot(self.params.H2, h_tm1) + self.word_vectors[x_t])

        return h_t