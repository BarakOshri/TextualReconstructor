import theano
from theano import tensor as T
import numpy as np
from attrdict import AttrDict
import sys
import pdb

#def corrupt(input):


class AutoEncoder:
	# For Word Vectors Only. Input is a two dimensional array, with input[i] a list of integers accessing  
	def __init__(self, encoder_params, decoder_params, wordVectors, end_token, batch_size, input=None, n_hidden=500, encoder='RNN', f='sigmoid'):

		self.n_hidden = n_hidden
		self.end_token = end_token
		self.batch_size = batch_size

		# Decoder Parameters
		self.params = AttrDict(decoder_params)
		self.decoder_params = [self.params.H1, self.params.Y, self.params.C, self.params.S, self.params.B]

		# Neuron Choice
		if f == 'sigmoid':
			self.f = T.nnet.sigmoid
		elif f == 'tanh':
			self.f = T.tanh
		elif f == 'gate':
			pass
		else:
			sys.exit('Invalid neuron')

		# Encoder Choice
		self.encoder = globals()[encoder](wordVectors, n_hidden, batch_size, params=encoder_params, hdim=n_hidden, f=self.f)
		self.get_hidden_values = self.encoder.get_hidden_values

		if input is None:
			self.x = T.dmatrix(name='input')
		else:
			self.x = input

	def get_reconstructed_input(self, hidden):
		[h, s], _ = theano.scan(fn = self._decoder_recurrence, outputs_info = [hidden, hidden], non_sequences = hidden, n_steps = 40)
		self.s = s
		self.y = np.argmax(s, axis=1)

		return s

	def _decoder_recurrence(self, ht_1, yt_1, hidden):
		h_t = self.f(T.dot(self.params.H1, ht_1) + T.dot(self.params.Y, yt_1) + T.dot(self.params.C, hidden))
		s_t = T.nnet.softmax(T.dot(self.params.S, h_t) + self.params.B)

		# pdb.set_trace()

		return [h_t, s_t], theano.scan_module.until(theano.tensor.eq(T.cast(np.argmax(s_t), 'int64'), self.end_token))

	def get_cost_updates(self, learning_rate):
		hidden_input = self.encoder.get_hidden_values(self.x)
		output = self.get_reconstructed_input(hidden_input)
		hidden_output = self.encoder.get_hidden_values(output)

		L = np.linalg.norm(hidden_input - hidden_output)

		encoder_decoder_params = self.decoder_params + self.encoder.encoder_params
		gparams = T.grad(L, encoder_decoder_params)

		updates = [(param, param - learning_rate * gparam) for param, gparam in zip(self.params, gparams)]

		return (L, updates)

	def get_decoder_params(self):
		return self.params.__dict__

	def get_encoder_params(self):
		return self.encoder.get_params()

class RNN:

	def __init__(self, wordVectors, n_hidden, batch_size, params=None, hdim=None, f=None):
		self.params = AttrDict(params)
		self.encoder_params = [self.params.H2]
		self.n_hidden = n_hidden
		self.batch_size = batch_size
		self.f = f
		self.h0 = theano.shared(np.zeros((self.n_hidden, batch_size)))

		self.wordVectors = wordVectors

	def _recurrence(self, x_t, h_tm1):
		h_t = self.f(T.dot(self.params.H2, h_tm1) + self.wordVectors[T.cast(x_t, 'int64')])

		return h_t

	#Input is a list of word vectors
	def get_hidden_values(self, input):
		h, _ = theano.scan(fn = self._recurrence, sequences = input.T, outputs_info = self.h0)
		self.current_hidden = h[-1]

		return self.current_hidden

	def get_params(self):
		return self.params.__dict__

# class DeepRNN:

# class CNN:

# 	def __init__(self, wordVectors, params=None, context=3, numFilters=5)

	