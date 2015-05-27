import theano
import numpy as np
from theano import tensor as T
import sys

class AutoEncoder:

	def __init__(self, input=None, encoder='rnn', n_hidden=500, params=None):
		self.n_hidden = n_hidden

		if input is None:
			self.x = T.dmatrix(name='input')
		else:
			self.x = input

		if encoder == 'rnn':
			rnn = RNN(params, n_hidden)
			self.get_hidden_values = rnn.get_hidden_values
		elif encoder == 'deeprnn':
			deeprnn = DeepRNN(params, n_hidden)
			self.get_hidden_values = deeprnn.get_hidden_values
		elif encoder == 'conv':
			cnn = CNN(params, n_hidden)
			self.get_hidden_values = cnn.get_hidden_values
		else:
			sys.exit('Invalid encoder')

	def get_reconstructed_input(self, hidden):

	def get_cost_updates(self, corruption_level, learning_rate):

class RNN:

	def __init__(self, params=None, hdim=None, neuron='sigmoid'):
		if params = None:
			self.H = 
		else:
			self.H = params['w']

		if neuron == 'sigmoid':
			self.f = T.nnet.sigmoid
		else:
			sys.exit('Invalid neuron')

	def recurrence(x_t, h_tm1):
		h_t = self.f(T.dot(self.H, h_tm1) + x_t)
		return [h_t]

	#Input is a list of word vectors
	def get_hidden_values(self, input):


class DeepRNN:

class CNN:



