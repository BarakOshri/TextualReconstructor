import theano
from theano import tensor as T
import numpy as np
from attrdict import AttrDict
from theano.ifelse import ifelse
import sys
import pdb

#def corrupt(input):


class AutoEncoder:
	# For Word Vectors Only. Input is a two dimensional array, with input[i] a list of integers accessing  
	def __init__(self, encoder_params, decoder_params, wordVectors, end_token, batch_size, input=None, n_hidden=500, encoder='RNN', f='sigmoid'):

		self.n_hidden = n_hidden
		self.end_token = end_token
		self.batch_size = batch_size
		self.wordVectors = wordVectors

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

	def get_reconstructed_input(self, c):
		# Get first word
		h0 = self.f(T.dot(self.params.H1, c) + T.dot(self.params.C, c))
		a0 = T.dot(self.params.S, h0) + self.params.B
		s0 = T.reshape(T.nnet.softmax(a0), a0.shape)
		#print s0.type
		#print h0.type
		#h0 = self.f(T.dot(self.params.H1, c) + T.dot(self.params.Y, s0) + T.dot(self.params.C, c))

		[h, s], _ = theano.scan(fn = self._decoder_recurrence, outputs_info = [h0, s0], non_sequences = c, n_steps = 20)
		y = T.argmax(s, axis=1)

		return y

	def _decoder_recurrence(self, ht_1, yt_1, hidden):
		#print T.dot(self.params.H1, ht_1).type
		h_t = self.f(T.dot(self.params.H1, ht_1) + T.dot(self.params.Y, yt_1) + T.dot(self.params.C, hidden))
		a = T.dot(self.params.S, h_t) + self.params.B
		s_t = T.reshape(T.nnet.softmax(a), a.shape)
		#print s_t.type
		#print h_t.type

		return [h_t, s_t], theano.scan_module.until(T.eq(np.argmax(T.nnet.softmax(T.dot(self.params.S, self.f(T.dot(self.params.H1, ht_1) + T.dot(self.params.Y, yt_1) + T.dot(self.params.C, hidden))) + self.params.B)), self.end_token))

	# def get_single_example(self, index):
	# 	sentence = self.x[index]
	# 	hidden_input = self.encoder.get_hidden_values(sentence)
	# 	output = self.get_reconstructed_input(hidden_input)
	# 	hidden_output = self.encoder.get_hidden_values(output)

	# 	L = np.linalg.norm(hidden_input - hidden_output)

	# 	return L

	def get_cost_updates(self, learning_rate):
		hidden_input = self.encoder.get_hidden_values(self.x)
		output = self.get_reconstructed_input(hidden_input)
		hidden_output = self.encoder.get_hidden_values(output)

		L = T.sqrt(T.sum(T.sqr(hidden_input - hidden_output)))

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
		self.h0 = np.zeros((self.n_hidden))

		self.wordVectors = wordVectors

	def _recurrence(self, x_t, h_tm1):
		h_t = self.f(T.dot(self.params.H2, h_tm1) + self.wordVectors[T.cast(x_t, 'int64')])

		return h_t

	#Input is a list of word vectors
	def get_hidden_values(self, sentence):
		h, _ = theano.scan(fn = self._recurrence, sequences = sentence, outputs_info = self.h0)
		self.current_hidden = h[-1]

		return self.current_hidden

	def get_params(self):
		return self.params.__dict__

# class DeepRNN:

# class CNN:

# 	def __init__(self, wordVectors, params=None, context=3, numFilters=5)

	