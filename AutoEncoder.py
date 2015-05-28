import theano
from theano import tensor as T
import numpy as np
from attrdict import AttrDict
import sys

def corrupt(input):

class AutoEncoder:
	# For Word Vectors Only. Input is a two dimensional array, with input[i] a list of integers accessing  
	def __init__(self, wordVectors, n_hidden=500, encoder='RNN', encoder_params=None, decoder_params=None):

		self.n_hidden = n_hidden

		# Decoder Parameters
		if decoder_params is None:
			self.params.H = np.ones((n_hidden, n_hidden))
			self.params.Y = np.ones((n_hidden, len(wordVectors)))
			self.params.C = np.ones((n_hidden, n_hidden))
			self.params.S = np.ones((len(wordVectors), n_hidden))
			self.params.b = np.ones(len(wordVectors))
		else:
			self.params = AttrDict(decoder_params)

		# Neuron Choice
		if neuron == 'sigmoid':
			self.f = T.nnet.sigmoid
		if neuron == 'tanh':
			self.f = T.tanh
		if neuron == 'gate':
		else:
			sys.exit('Invalid neuron')

		# Encoder Choice
		self.encoder = globals()[encoder](wordVectors, params=encoder_params, hdim=n_hidden, f=self.f)
		self.get_hidden_values = self.encoder.get_hidden_values

	def get_reconstructed_input(self, hidden, end_token):
		[h, s], _ - theano.scan(fn = lambda a, b, c: _decoder_recurrence(a, b, c, end_token), 
								outputs_info = [hidden, hidden], non_sequences = hidden)
		self.s = s
		self.y = np.argmax(s, axis=1)

		return s

	def _decoder_recurrence(ht_1, yt_1, hidden, end_token):
		h_t = f(T.dot(self.H, ht_1) + T.dot(self.Y, yt_1) + T.dot(self.C, hidden))
		s_t = T.nnet.softmax(T.dot(self.S, h_t) + self.b)

		return [h_t, s_t], theano.scan_module.until(np.argmax(s_t) == end_token)

	def get_cost_updates(self, corruption_level, learning_rate, input, output):
		hidden_input = self.encoder.current_hidden
		hidden_output = encoder.get_hidden_values(output)

		L = np.linalg.norm(hidden_input - hidden_output)

		encoder_decoder_params = self.params AND self.encoder.params
		gparams = T.grad(L, encoder_decoder_params)

		updates = [(param, param - learning_rate * gparam) for param, gparam in zip(self.params, gparams)]

		return (L, updates)

	def get_params(self):
		return self.params.__dict__

class RNN:

	def __init__(self, wordVectors, params=None, hdim=None, f=None):
		if params = None:
			self.params.H = np.ones((hdim, hdim))
		else:
			self.params = AttrDict(params)

		self.wordVectors = wordVectors

	def _recurrence(x_t, h_tm1):
		h_t = self.f(T.dot(self.H, h_tm1) + wordVectors[x_t])

		return h_t

	#Input is a list of word vectors
	def get_hidden_values(self, input, corrupt=False):
		h, _ = theano.scan(fn=_recurrence, sequences=corrupt(input), outputs_info=self.h0)
		self.current_hidden = h[-1]
		
		return self.hidden

class DeepRNN:

class CNN:
	