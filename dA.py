"""
 This tutorial introduces denoising auto-encoders (dA) using Theano.

 Denoising autoencoders are the building blocks for SdA.
 They are based on auto-encoders as the ones used in Bengio et al. 2007.
 An autoencoder takes an input x and first maps it to a hidden representation
 y = f_{\theta}(x) = s(Wx+b), parameterized by \theta={W,b}. The resulting
 latent representation y is then mapped back to a "reconstructed" vector
 z \in [0,1]^d in input space z = g_{\theta'}(y) = s(W'y + b').  The weight
 matrix W' can optionally be constrained such that W' = W^T, in which case
 the autoencoder is said to have tied weights. The network is trained such
 that to minimize the reconstruction error (the error between x and z).

 For the denosing autoencoder, during training, first x is corrupted into
 \tilde{x}, where \tilde{x} is a partially destroyed version of x by means
 of a stochastic mapping. Afterwards y is computed as before (using
 \tilde{x}), y = s(W\tilde{x} + b) and z as s(W'y + b'). The reconstruction
 error is now measured between z and the uncorrupted input x, which is
 computed as the cross-entropy :
      - \sum_{k=1}^d[ x_k \log z_k + (1-x_k) \log( 1-z_k)]


 References :
   - P. Vincent, H. Larochelle, Y. Bengio, P.A. Manzagol: Extracting and
   Composing Robust Features with Denoising Autoencoders, ICML'08, 1096-1103,
   2008
   - Y. Bengio, P. Lamblin, D. Popovici, H. Larochelle: Greedy Layer-Wise
   Training of Deep Networks, Advances in Neural Information Processing
   Systems 19, 2007

"""

import os
import sys
import time

import numpy
import numpy as np

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

#from utils import tile_raster_images

# try:
#     import PIL.Image as Image
# except ImportError:
#     import Image


class dA(object):
    """Denoising Auto-Encoder class (dA)

    A denoising autoencoders tries to reconstruct the input from a corrupted
    version of it by projecting it first in a latent space and reprojecting
    it afterwards back in the input space. Please refer to Vincent et al.,2008
    for more details. If x is the input then equation (1) computes a partially
    destroyed version of x by means of a stochastic mapping q_D. Equation (2)
    computes the projection of the input into the latent space. Equation (3)
    computes the reconstruction of the input, while equation (4) computes the
    reconstruction error.

    .. math::

        \tilde{x} ~ q_D(\tilde{x}|x)                                     (1)

        y = s(W \tilde{x} + b)                                           (2)

        x = s(W' y  + b')                                                (3)

        L(x,z) = -sum_{k=1}^d [x_k \log z_k + (1-x_k) \log( 1-z_k)]      (4)

    """

    def __init__(
        self,
        numpy_rng,
        theano_rng=None,
        wvec_dim=50,
        n_dict=124120,
        input=None,
        n_hidden=500,
        H1=None,
        H2=None,
        C=None,
        Y=None,
        S=None,
        bhid=None,
        bvis=None
    ):
        """
        Initialize the dA class by specifying the number of visible units (the
        dimension d of the input ), the number of hidden units ( the dimension
        d' of the latent or hidden space ) and the corruption level. The
        constructor also receives symbolic variables for the input, weights and
        bias. Such a symbolic variables are useful when, for example the input
        is the result of some computations, or when weights are shared between
        the dA and an MLP layer. When dealing with SdAs this always happens,
        the dA on layer 2 gets as input the output of the dA on layer 1,
        and the weights of the dA are used in the second stage of training
        to construct an MLP.

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: number random generator used to generate weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                     generated based on a seed drawn from `rng`

        :type input: theano.tensor.TensorType
        :param input: a symbolic description of the input or None for
                      standalone dA

        :type n_visible: int
        :param n_visible: number of visible units

        :type n_hidden: int
        :param n_hidden:  number of hidden units

        :type W: theano.tensor.TensorType
        :param W: Theano variable pointing to a set of weights that should be
                  shared belong the dA and another architecture; if dA should
                  be standalone set this to None

        :type bhid: theano.tensor.TensorType
        :param bhid: Theano variable pointing to a set of biases values (for
                     hidden units) that should be shared belong dA and another
                     architecture; if dA should be standalone set this to None

        :type bvis: theano.tensor.TensorType
        :param bvis: Theano variable pointing to a set of biases values (for
                     visible units) that should be shared belong dA and another
                     architecture; if dA should be standalone set this to None


        """
        self.n_hidden = n_hidden
        self.end_token = 0
        self.word_vectors = theano.shared(value = np.zeros((10, 50), dtype=theano.config.floatX), borrow=True)
        self.n_dict = 10
        self.f = T.nnet.sigmoid

        # create a Theano random generator that gives symbolic random values
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        # note : W' was written as `W_prime` and b' as `b_prime`

        if not H1:
            # W is initialized with `initial_W` which is uniformely sampled
            # from -4*sqrt(6./(n_visible+n_hidden)) and
            # 4*sqrt(6./(n_hidden+n_visible))the output of uniform if
            # converted using asarray to dtype
            # theano.config.floatX so that the code is runable on GPU
            initial_H1 = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_hidden)),
                    high=4 * numpy.sqrt(6. / (n_hidden)),
                    size=(n_hidden, n_hidden)
                ),
                dtype=theano.config.floatX
            )
            H1 = theano.shared(value=initial_H1, name='H1', borrow=True)

        if not H2:
            # W is initialized with `initial_W` which is uniformely sampled
            # from -4*sqrt(6./(n_visible+n_hidden)) and
            # 4*sqrt(6./(n_hidden+n_visible))the output of uniform if
            # converted using asarray to dtype
            # theano.config.floatX so that the code is runable on GPU
            initial_H2 = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_hidden)),
                    high=4 * numpy.sqrt(6. / (n_hidden)),
                    size=(n_hidden, n_hidden)
                ),
                dtype=theano.config.floatX
            )
            H2 = theano.shared(value=initial_H2, name='H2', borrow=True)

        if not C:
            # W is initialized with `initial_W` which is uniformely sampled
            # from -4*sqrt(6./(n_visible+n_hidden)) and
            # 4*sqrt(6./(n_hidden+n_visible))the output of uniform if
            # converted using asarray to dtype
            # theano.config.floatX so that the code is runable on GPU
            initial_C = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_hidden)),
                    high=4 * numpy.sqrt(6. / (n_hidden)),
                    size=(n_hidden, n_hidden)
                ),
                dtype=theano.config.floatX
            )
            C = theano.shared(value=initial_C, name='C', borrow=True)

        if not Y:
            # W is initialized with `initial_W` which is uniformely sampled
            # from -4*sqrt(6./(n_visible+n_hidden)) and
            # 4*sqrt(6./(n_hidden+n_visible))the output of uniform if
            # converted using asarray to dtype
            # theano.config.floatX so that the code is runable on GPU
            initial_Y = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_hidden)),
                    high=4 * numpy.sqrt(6. / (n_hidden)),
                    size=(n_hidden, n_dict)
                ),
                dtype=theano.config.floatX
            )
            Y = theano.shared(value=initial_Y, name='Y', borrow=True)

        if not S:
            # W is initialized with `initial_W` which is uniformely sampled
            # from -4*sqrt(6./(n_visible+n_hidden)) and
            # 4*sqrt(6./(n_hidden+n_visible))the output of uniform if
            # converted using asarray to dtype
            # theano.config.floatX so that the code is runable on GPU
            initial_S = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_hidden)),
                    high=4 * numpy.sqrt(6. / (n_hidden)),
                    size=(n_dict, n_hidden)
                ),
                dtype=theano.config.floatX
            )
            S = theano.shared(value=initial_S, name='S', borrow=True)

        self.H1 = H1
        self.H2 = H2
        self.C = C
        self.Y = Y
        self.S = S

        self.theano_rng = theano_rng
        # if no input is given, generate a variable representing the input
        if input is None:
            # we use a matrix because we expect a minibatch of several
            # examples, each example being a row
            self.x = T.dmatrix(name='input')
        else:
            self.x = input

        self.params = [self.H1, self.S, self.Y, self.C, self.H2]

    #Input is a list of word vectors
    def get_hidden_values(self, sentence):
        h0 = theano.shared(value=np.zeros((self.n_hidden), dtype=theano.config.floatX))
        h, _ = theano.scan(fn = self._rnn_recurrence, sequences = sentence, outputs_info = h0)

        return h[-1]

    def _rnn_recurrence(self, x_t, h_tm1):
        h_t = self.f(T.dot(self.H1, h_tm1) + self.word_vectors[T.cast(x_t, 'int64')])
        return h_t

    def get_reconstructed_input(self, c):
        h0 = self.f(T.dot(self.H1, c) + T.dot(self.C, c))
        a0 = T.dot(self.S, h0)
        s0 = T.reshape(T.nnet.softmax(a0), a0.shape)

        [h, s], _ = theano.scan(fn = self._decoder_recurrence, outputs_info = [h0, s0], non_sequences = c, n_steps = 20)
        y = T.argmax(s, axis=1)

        return y

        #return y

    def _decoder_recurrence(self, ht_1, yt_1, hidden):
        h_t = self.f(T.dot(self.H2, ht_1) + T.dot(self.Y, yt_1) + T.dot(self.C, hidden))
        a = T.dot(self.S, h_t)
        s_t = T.reshape(T.nnet.softmax(a), a.shape)

        return [h_t, s_t], theano.scan_module.until(T.eq(T.argmax(T.nnet.softmax(T.dot(self.S, self.f(T.dot(self.H2, ht_1) + T.dot(self.Y, yt_1) + T.dot(self.C, hidden))))), self.end_token))

    def get_cost_updates(self, corruption_level, learning_rate):
        """ This function computes the cost and the updates for one trainng
        step of the dA """

        #tilde_x = self.get_corrupted_input(self.x, corruption_level)
        hidden = self.get_hidden_values(self.x)
        #hidden2 = self.get_hidden_values(self.x)
        output = self.get_reconstructed_input(hidden)
        output_hidden = self.get_hidden_values(output)
        # note : we sum over the size of a datapoint; if we are using
        #        minibatches, L will be a vector, with one entry per
        #        example in minibatch
        #cost = T.mean(hidden) - T.mean(output) + T.mean(output) - T.mean(output_hidden)
        #cost = T.mean(hidden) - T.mean(output)
        cost = T.sqrt(T.sum(T.sqr(output_hidden - hidden + output - output)))
        # note : L is now a vector, where each element is the
        #        cross-entropy cost of the reconstruction of the
        #        corresponding example of the minibatch. We need to
        #        compute the average of all these to get the cost of
        #        the minibatch
        #cost = T.mean(L)

        # compute the gradients of the cost of the `dA` with respect
        # to its parameters
        gparams = T.grad(cost, self.params)
        # generate the list of updates
        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)
        ]

        return (cost, updates)


def test_dA(learning_rate=0.1, training_epochs=15,
            dataset='mnist.pkl.gz',
            batch_size=20, output_folder='dA_plots'):

    """
    This demo is tested on MNIST

    :type learning_rate: float
    :param learning_rate: learning rate used for training the DeNosing
                          AutoEncoder

    :type training_epochs: int
    :param training_epochs: number of epochs used for training

    :type dataset: string
    :param dataset: path to the picked dataset

    """
    # List of 1-d word vector sentences
    train_set_x = [theano.shared(numpy.zeros((30), dtype = theano.config.floatX), borrow=True) for i in range(30, 40)]

    # compute number of minibatches for training, validation and testing
    #n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

    # start-snippet-2
    # allocate symbolic variables for the data
    # index to a [mini]batch
     # the data is presented as rasterized images
    # end-snippet-2

    # if not os.path.isdir(output_folder):
    #     os.makedirs(output_folder)
    # os.chdir(output_folder)

    ####################################
    # BUILDING THE MODEL NO CORRUPTION #
    ####################################

    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    x = T.ivector('x') 

    da = dA(
        numpy_rng=rng,
        theano_rng=theano_rng,
        input=x,
        wvec_dim = 50,
        n_hidden=500
    )

    cost, updates = da.get_cost_updates(
        corruption_level=0.,
        learning_rate=learning_rate
    )

    train_da = theano.function(
        [x],
        cost,
        updates=updates,
    )

    print """
  ______   ______   .___  ___. .______    __   __       _______  _______  
 /      | /  __  \  |   \/   | |   _  \  |  | |  |     |   ____||       \ 
|  ,----'|  |  |  | |  \  /  | |  |_)  | |  | |  |     |  |__   |  .--.  |
|  |     |  |  |  | |  |\/|  | |   ___/  |  | |  |     |   __|  |  |  |  |
|  `----.|  `--'  | |  |  |  | |  |      |  | |  `----.|  |____ |  '--'  |
 \______| \______/  |__|  |__| | _|      |__| |_______||_______||_______/
 """

    # start_time = time.clock()

    ############
    # TRAINING #
    ############

    # # go through training epochs
    # for epoch in xrange(training_epochs):
    #     # go through trainng set
    #     c = []
    #     for batch_index in xrange(n_train_batches):
    #         c.append(train_da(batch_index))

    #     print 'Training epoch %d, cost ' % epoch, numpy.mean(c)

    # end_time = time.clock()

    # training_time = (end_time - start_time)

    # print >> sys.stderr, ('The no corruption code for file ' +
    #                       os.path.split(__file__)[1] +
    #                       ' ran for %.2fm' % ((training_time) / 60.))
    # image = Image.fromarray(
    #     tile_raster_images(X=da.W.get_value(borrow=True).T,
    #                        img_shape=(28, 28), tile_shape=(10, 10),
    #                        tile_spacing=(1, 1)))
    # image.save('filters_corruption_0.png')

    # start-snippet-3
    #####################################
    # BUILDING THE MODEL CORRUPTION 30% #
    #####################################

    # rng = numpy.random.RandomState(123)
    # theano_rng = RandomStreams(rng.randint(2 ** 30))

    # da = dA(
    #     numpy_rng=rng,
    #     theano_rng=theano_rng,
    #     input=x,
    #     n_visible=28 * 28,
    #     n_hidden=500
    # )

    # cost, updates = da.get_cost_updates(
    #     corruption_level=0.3,
    #     learning_rate=learning_rate
    # )

    # train_da = theano.function(
    #     [index],
    #     cost,
    #     updates=updates,
    #     givens={
    #         x: train_set_x[index * batch_size: (index + 1) * batch_size]
    #     }
    # )

    # start_time = time.clock()

    # ############
    # # TRAINING #
    # ############

    # # go through training epochs
    # for epoch in xrange(training_epochs):
    #     # go through trainng set
    #     c = []
    #     for batch_index in xrange(n_train_batches):
    #         c.append(train_da(batch_index))

    #     print 'Training epoch %d, cost ' % epoch, numpy.mean(c)

    # end_time = time.clock()

    # training_time = (end_time - start_time)

    # print >> sys.stderr, ('The 30% corruption code for file ' +
    #                       os.path.split(__file__)[1] +
    #                       ' ran for %.2fm' % (training_time / 60.))
    # # end-snippet-3

    # # start-snippet-4
    # # image = Image.fromarray(tile_raster_images(
    # #     X=da.W.get_value(borrow=True).T,
    # #     img_shape=(28, 28), tile_shape=(10, 10),
    # #     tile_spacing=(1, 1)))
    # # image.save('filters_corruption_30.png')
    # # end-snippet-4

    # os.chdir('../')


if __name__ == '__main__':
    test_dA()
