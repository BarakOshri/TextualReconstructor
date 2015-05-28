import theano
from theano import tensor as T
import numpy as np
from attrdict import AttrDict
import sys

def denoiser(input):

class AutoEncoder:
	# For Word Vectors Only
	def __init__(self, input=None, encoder='RNN', n_hidden=500, encoder_params=None, decoder_params=None, augment=None):

		# Generate input
		self.x = augment(input) if augment else None

		# Decoder Parameters
		if decoder_params is None:
			self.params.H = 
			self.params.Y = 
			self.params.C = 
			self.params.S = 
			self.params.b = 
		else:
			self.params = AttrDict(decoder_params)

		self.n_hidden = n_hidden

		# Neuron Choice
		if neuron == 'sigmoid':
			self.f = T.nnet.sigmoid
		if neuron == 'tanh':
			self.f = T.tanh
		if neuron == 'gate':
		else:
			sys.exit('Invalid neuron')

		# Encoder Choice
		self.encoder = globals()[encoder](params=encoder_params, hdim=n_hidden, f=self.f)
		self.get_hidden_values = self.encoder.get_hidden_values

	def get_reconstructed_input(self, hidden):
		[h, s], _ - theano.scan(fn = _decoder_recurrence, outputs_info = [hidden, hidden], non_sequences = hidden)
		self.y = s
		return s, theano.scan_module.until(STOPPING CONDITION)

	def _decoder_recurrence(ht_1, yt_1, hidden):
		h_t = f(T.dot(self.H, ht_1) + T.dot(self.Y, yt_1) + T.dot(self.C, hidden))
		s_t = T.nnet.softmax(T.dot(self.S, h_t) + self.b)
		return [h_t, s_t]

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

	def __init__(self, params=None, hdim=None, f=None):
		if params = None:
			self.params.H = 
		else:
			self.params = AttrDict(params)

	def _recurrence(x_t, h_tm1):
		h_t = self.f(T.dot(self.H, h_tm1) + x_t)
		return h_t

	#Input is a list of word vectors
	def get_hidden_values(self, input):
		h, _ = theano.scan(fn=_recurrence, sequences=input, outputs_info=self.h0)
		self.current_hidden = h[-1]
		return self.hidden

class DeepRNN:

class CNN:
	