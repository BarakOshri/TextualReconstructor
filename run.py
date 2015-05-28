import theano
from theano import tensor as T
import numpy as np


da = AutoEncoder(numpy_rng=rng, theano_rng=theano_rng, input=x, n_visible=28 * 28, n_hidden=500)

cost, updates = AutoEncoder.get_cost_updates(corruption_level, learning_rate)

train_da