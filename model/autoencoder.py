import numpy as np
import theano
from theano import tensor as T

import encoder
import decoder

class Autoencoder:

	def __init__(self, word_vectors, batch_size, input=None, n_hidden=500, encoder_params=None, decoder_params=None, encoder_type='RNN', decoder_type='RNN'):
		self.n_hidden = n_hidden
		self.batch_size = batch_size
		self.current_batch_index = 0
		self.gparams = None
		self.word_vectors = theano.shared(value=word_vectors, borrow=True)
		self.word_vectors_dimension = word_vectors.shape[1]
		self.dict_size = word_vectors.shape[0]
		self.x = input

		if encoder_type == 'RNN':
			self.encoder = encoder.RNN(params=encoder_params, word_vectors=self.word_vectors, n_hidden=self.word_vectors_dimension)
		else:
			raise ValueError('Encoder type unknown')

		if decoder_type == 'RNN':
			self.decoder = decoder.RNN(params=decoder_params, n_hidden=self.word_vectors_dimension, dict_size=self.dict_size)
		else:
			raise ValueError('Decoder type unknown')

	def get_cost_updates(self, learning_rate):
		input_hidden_state = self.encoder.get_hidden_state(self.x)
		reconstructed_input = self.decoder.get_reconstructed_input(input_hidden_state)
		reconstructed_input_hidden_state = self.encoder.get_hidden_state(reconstructed_input)

		L = T.sqrt(T.sum(T.sqr(input_hidden_state - reconstructed_input_hidden_state)))

		all_params = self.decoder.params.values() + self.encoder.params.values()
		gparams = T.grad(L, all_params)

		if self.current_index % self.batch_size == 0:
			updates = [(param, param - learning_rate * gparam) for param, gparam in zip(all_params, gparams)]
			self.current_batch_index = 0
		else:
			updates = None
			self.gparams += gparams
			self.current_batch_index += 1

		return (L, updates)