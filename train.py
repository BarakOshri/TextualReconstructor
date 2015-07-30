import sys
sys.path.insert(0, 'model')
sys.path.insert(0, 'data_processing')
sys.path.insert(0, 'data')

import theano
from theano import tensor as T
import numpy as np
import pickle

from autoencoder import *
from wiki_to_vectors import *

word_vector_file = "data/vectors.6B.200d.txt"
wiki_files_directory = "data/wiki/"
sentences_file = 'data/all_wvi.pkl'

print 'Building word vectors'
glove_mat, glove_vocab = build_word_vectors(word_vector_file) # Matrix of shared theano tensor vectors

print 'Building sentences'
if sentences_file is not None:
	sentences = pickle.load(open('data/all_wvi.pkl', 'r'))
else:
	sentences = get_wiki_sentences(word_vector_file, wiki_files_directory) # num_sentences x index into word in dictionary
sentences = map(np.array, sentences)
train_sentences = sentences[:int(0.8*len(sentences))]
test_sentences = sentences[int(0.8*len(sentences)):]

#Initializing variables
batch_size = 5
learning_rate = 0.05
n_hidden = 50
n_training_epochs = 3
end_token = T.constant(0)

index = T.lscalar() # Index to a mini-batch
x = T.ivector('x') # Labels into words in glove dictionary

print 'Building autoencoder'
ae = Autoencoder(glove_mat, batch_size, input=x, n_hidden=n_hidden, encoder_type='RNN', decoder_type='RNN')
cost, updates = ae.get_cost_updates(learning_rate=learning_rate)
train_autoencoder = theano.function([index], cost, updates=updates, givens={x: train_sentences[index]})

print 'Training autoencoder'
start_time = time.clock()
batch_costs = []
for epoch in xrange(n_training_epochs):
	batch_cost = []
	for i in xrange(len(sentences)):
		batch_cost.append(train_autoencoder(i))
		if i % batch_size == 0 and i != 0:
			batch_costs.append(np.mean(batch_cost))
			batch_cost = []

	print 'Training epoch %d, cost ' % epoch, numpy.mean(batch_costs)
	batch_costs = []

end_time = time.clock()
training_time = end_time - start_time
print "Training took %d minutes" % (training_time / 60.0)