"""
This software is an implementation of

Deep MRI brain extraction: A 3D convolutional neural network for skull stripping

You can download the paper at http://dx.doi.org/10.1016/j.neuroimage.2016.01.024

If you use this software for your projects please cite:

Kleesiek and Urban et al, Deep MRI brain extraction: A 3D convolutional neural network for skull stripping,
NeuroImage, Volume 129, April 2016, Pages 460-469.

The MIT License (MIT)

Copyright (c) 2016 Gregor Urban, Jens Kleesiek

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

import numpy as np
import numpy


import theano
import theano.tensor as T

import TransferFunctions as transfer


if 1:
    try:
        try:
            from conv3d2d_cudnn import conv3d
        except:
            from conv3d2d import conv3d
    except:
        from theano.tensor.nnet.conv3d2d import conv3d



from maxPool3D import my_max_pool_3d


def max_pool_along_channel_axis(sym_input, pool_factor):
    """ for 3D conv."""
    s = None
    for i in xrange(pool_factor):
        t = sym_input[:,:,i::pool_factor]
        if s is None:
            s = t
        else:
            s = T.maximum(s, t)
    return s
#    Ns, Ts, C, Hs, Ws = 1, 70, 1, 70, 70  -> 70^3
#    Nf, Tf, C, Hf, Wf = 32, 5 , 1, 5 , 5  -> 32 filters of shape 5^3
#    signals = numpy.arange(Ns*Ts*C*Hs*Ws).reshape(Ns, Ts, C, Hs, Ws).astype('float32')
#    filters = numpy.arange(Nf*Tf*C*Hf*Wf).reshape(Nf, Tf, C, Hf, Wf).astype('float32')
#
# in 3D
#        input:  (1, 70,  3, 70, 70)
#       filters: (32, 5 , 3,  5 , 5)
#    --> output: (1, 66, 32, 66, 66)

import time





def offset_map(output_stride):
    for x in np.log2(output_stride):
        assert np.float.is_integer(x), 'Stride must be power of 2; is: '+str(output_stride)
    if np.all(output_stride == 2):
        return  np.array([[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]])
    else:
        prev = offset_map(output_stride/2)
        current = []
        for i in xrange(2):
            for j in xrange(2):
                for k in xrange(2):
                    for p in prev:
                        new = p.copy()
                        new[0] += i*2
                        new[1] += j*2
                        new[2] += k*2
                        current.append(new)
        return np.array(current)



class ConvPoolLayer3D(object):
    """Pool Layer of a convolutional network
        you could change this easily into using different pooling in any of the directions..."""

    def __init__(self,  input, filter_shape, input_shape, poolsize=2, bDropoutEnabled_=False,
                 bUpsizingLayer=False, ActivationFunction = 'abs',
                 use_fragment_pooling = False, dense_output_from_fragments = False, output_stride = None,
                 b_use_FFT_convolution=False, input_layer=None, 
                 W=None, b=None, b_deconvolution=False, verbose = 1):
        """
        Allocate a ConvPoolLayer with shared variable internal parameters.


        bUpsizingLayer = True: =>  bordermode = full (zero padding) thus increasing the output image size (as opposed to shrinking it in 'valid' mode)

        :type input: ftensor5
        :param input: symbolic image tensor, of shape input_shape

        :type filter_shape: tuple or list of length 5
        :param filter_shape: (number of filters, filter X, num input feature maps, filter Y,filter Z)

        :type input_shape: tuple or list of length 5
        :param input_shape: (batch size, X, num input feature maps,  Y, Z)

        :type poolsize: integer (typically 1 or 2)
        :param poolsize: the downsampling (max-pooling) factor


        accessible via "this"/self pointer:

        input -> conv_out -> ... -> output

        """
        assert len(filter_shape)==5


        if b_deconvolution:
            raise NotImplementedError()

        assert input_shape[2] == filter_shape[2]
        self.input = input
        prod_pool = np.prod(poolsize)
        try:
            if prod_pool==poolsize:
               prod_pool = poolsize**3
               poolsize = (poolsize,)*3
        except:
            pass
        poolsize = np.asanyarray(poolsize)
        self.pooling_factor=poolsize
        self.number_of_filters = filter_shape[0]
        self.filter_shape=filter_shape

        self.input_shape = input_shape
        self.input_layer = input_layer
        self.output_stride = output_stride


        if prod_pool>1 and use_fragment_pooling:
            assert prod_pool==8,"currently only 2^3 pooling"

        # n inputs to each hidden unit
        fan_in = 1.0*numpy.prod(filter_shape[1:])

        fan_out = 1.0*(numpy.prod(filter_shape[0:2]) * numpy.prod(filter_shape[3:]))/prod_pool
        # initialize weights with random weights
        W_bound = numpy.sqrt(3. / (fan_in + fan_out))#W_bound = 0.035#/(np.sqrt(fan_in/1400))##was 0.02 which was fine. #numpy.sqrt(0.04 / (fan_in + fan_out)) #6.0 / numpy.prod(filter_shape[1:]) #

        if verbose:
            print "ConvPoolLayer3D"+("(FFT_based)" if b_use_FFT_convolution else "")+":"
            print "   input (image) =",input_shape
            print "   filter        =",filter_shape," @ std =",W_bound
            print "   poolsize",poolsize
        
        if W==None:
            self.W = theano.shared(
            numpy.asarray(numpy.random.normal(0, W_bound, filter_shape), dtype=theano.config.floatX)
            ,  borrow=True, name='W_conv')
        else:
            self.W = W

        if ActivationFunction in ['ReLU', 'relu']:

            b_values =  numpy.ones((filter_shape[0],), dtype=theano.config.floatX)#/filter_shape[1]/filter_shape[3]/filter_shape[4]

        elif ActivationFunction in ['sigmoid', 'sig']:
            b_values =  0.5*numpy.ones((filter_shape[0],), dtype=theano.config.floatX)
        else:
            b_values =  numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)

        if b==None:
            self.b =  theano.shared(value=b_values, borrow=True, name='b_conv')
        else:
            self.b = b




        if b_use_FFT_convolution:


            self.mode = theano.compile.get_default_mode().including('conv3d_fft', 'convgrad3d_fft', 'convtransp3d_fft')


            filters_flip = self.W[:,::-1,:,::-1,::-1]  # flip x, y, z
            conv_res = T.nnet.conv3D(
            	V=input.dimshuffle(0,3,4,1,2),  # (batch, row, column, time, in channel)
            	W=filters_flip.dimshuffle(0,3,4,1,2), # (out_channel, row, column, time, in channel)
            	b=self.b,
            	d=(1,1,1))
            self.conv_out = conv_res.dimshuffle(0,3,4,1,2)  # (batchsize, time, channels, height, width)

        else:
            self.mode = theano.compile.get_default_mode()
            self.conv_out = conv3d(signals=input, filters=self.W, border_mode = 'full' if bUpsizingLayer else 'valid',
                filters_shape=filter_shape, signals_shape = input_shape if input_shape[0]!=None else None
                )

        if np.any(poolsize>1):
            #print "   use_fragment_pooling =",use_fragment_pooling

            if use_fragment_pooling:
                assert np.all(poolsize==2), "Fragment Pooling (currently) allows only a Poolingfactor of 2! GIVEN: "+str(poolsize)
                pooled_out = self.fragmentpool(self.conv_out)
            else:

                pooled_out =  my_max_pool_3d(self.conv_out, pool_shape = (poolsize[0],poolsize[1],poolsize[2])) 
        else:
            pooled_out = self.conv_out


        if bDropoutEnabled_:
            print "   dropout: on"
            if b_use_FFT_convolution:
                print "   !!! WARNING: b was already added, this might mess things up!\n"*2
                raise NotImplementedError("BAD: FFT & Dropout")


            self.SGD_dropout_rate = theano.shared(np.asarray(np.float32(0.5), dtype=theano.config.floatX)) # lower = less dropped units
            rng = T.shared_randomstreams.RandomStreams(int(time.time()))

            self.dropout_gate = (np.float32(1.)/(np.float32(1.)-self.SGD_dropout_rate))* rng.binomial(pooled_out.shape,1,1.0-self.SGD_dropout_rate,dtype=theano.config.floatX)
            pooled_out =  pooled_out * self.dropout_gate 


        if b_use_FFT_convolution==0:
            lin_output = pooled_out + self.b.dimshuffle('x', 'x', 0, 'x', 'x') #the value will be added EVERYWHERE!, don't choose a too big b!
        else:
            lin_output = pooled_out


        # MFP Code #
        if dense_output_from_fragments and (input_shape[0]>1 or (use_fragment_pooling and np.any(poolsize>1))):
            output_shape = list(  theano.function([input], lin_output.shape, mode = self.mode)(numpy.zeros((1 if input_shape[0]==None else input_shape[0],)+input_shape[1:],dtype=numpy.float32)))
            if input_shape[0]==None:
                output_shape[0] = input_shape[0]
            output_shape=tuple(output_shape)
            print '   dense_output_from_fragments:: (lin_output) reshaped into dense volume...' #class_probabilities, output too...
            lin_output = self.combine_fragments_to_dense_bxcyz(lin_output, output_shape)  #(batch, x, channels, y, z)

        self.lin_output = lin_output
        
        func, self.ActivationFunction, dic = transfer.parse_transfer_function(ActivationFunction)
        pool_ratio = dic["cross_channel_pooling_groups"]
        if pool_ratio is not None:
            self.output = max_pool_along_channel_axis(lin_output, pool_ratio)
        else:
            self.output = func(lin_output)
        
        output_shape = list(  theano.function([input], self.output.shape, mode = self.mode)(numpy.zeros((1 if input_shape[0]==None else input_shape[0],)+input_shape[1:],dtype=numpy.float32)))

        if input_shape[0]==None:
            output_shape[0] = input_shape[0]
        output_shape=tuple(output_shape)
        if verbose:
            print "   output        =",output_shape, "Dropout",("enabled" if bDropoutEnabled_ else "disabled")
            print "   ActivationFunction =",self.ActivationFunction
        self.output_shape = output_shape



        #lin_output:
        # bxcyz
        #dimshuffle((2,0,1,3,4))
        # cbxyz
        #flatten(2).dimshuffle((1,0))
        # bxyz,c
        self.class_probabilities = T.nnet.softmax( lin_output.dimshuffle((2,0,1,3,4)).flatten(2).dimshuffle((1,0))  )#e.g. shape is (22**3, 5) for 5 classes ( i.e. have to set n.of filters = 5) and predicting  22 * 22 * 22 labels at once

        #class_probabilities_realshape:
        # (b*x*y*z,c) -> (b,x,y,z,c) -> (b,x,c,y,z)  #last by: (0,1,4,2,3)
        self.class_probabilities_realshape = self.class_probabilities.reshape((output_shape[0],output_shape[1],output_shape[3],output_shape[4], self.number_of_filters)).dimshuffle((0,1,4,2,3))  #lin_output.shape[:2]+lin_output.shape[3:5]+(output_shape[2],)

        self.class_prediction = T.argmax(self.class_probabilities_realshape,axis=2)

        # store parameters of this layer
        self.params = [self.W, self.b]

        return




    def fragmentpool(self, conv_out):
      p000 = my_max_pool_3d(conv_out[:,:-1,:,:-1,:-1], pool_shape=(2,2,2))
      p001 = my_max_pool_3d(conv_out[:,:-1,:,:-1, 1:], pool_shape=(2,2,2))
      p010 = my_max_pool_3d(conv_out[:,:-1,:, 1:,:-1], pool_shape=(2,2,2))
      p011 = my_max_pool_3d(conv_out[:,:-1,:, 1:, 1:], pool_shape=(2,2,2))
      p100 = my_max_pool_3d(conv_out[:, 1:,:,:-1,:-1], pool_shape=(2,2,2))
      p101 = my_max_pool_3d(conv_out[:, 1:,:,:-1, 1:], pool_shape=(2,2,2))
      p110 = my_max_pool_3d(conv_out[:, 1:,:, 1:,:-1], pool_shape=(2,2,2))
      p111 = my_max_pool_3d(conv_out[:, 1:,:, 1:, 1:], pool_shape=(2,2,2))
      result = T.concatenate((p000, p001, p010, p011, p100, p101, p110, p111), axis=0)
      return result



    def combine_fragments_to_dense_bxcyz(self, tensor, sh):
      """ expected shape: (batch, x, channels, y, z)"""
      ttensor = tensor # be same shape as result, no significant time cost

      output_stride = self.output_stride
      if isinstance(output_stride, list) or isinstance(output_stride, tuple):
          example_stride = np.prod(output_stride)#**3
      else:
          example_stride = output_stride**3
          output_stride  = np.asarray((output_stride,)*3)
      zero = np.array((0), dtype=theano.config.floatX)
      embedding = T.alloc( zero, 1, sh[1]*output_stride[0], sh[2], sh[3]*output_stride[1], sh[4]*output_stride[2]) # first arg. is fill-value (0 in this case) and not an element of the shape
      ix = offset_map(output_stride)
      print "      output_stride",output_stride
      print "      example_stride",example_stride
      for i,(n,m,k) in enumerate(ix):
          embedding = T.set_subtensor(embedding[:,n::output_stride[0],:,m::output_stride[1],k::output_stride[2]], ttensor[i::example_stride])
      return embedding



    def randomize_weights(self, scale_w = 3.0):

        fan_in = 1.0*numpy.prod(self.filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width * filter depth" / pooling size**3
        fan_out = 1.0*(numpy.prod(self.filter_shape[0:2]) * numpy.prod(self.filter_shape[3:])/np.mean(self.pooling_factor)**3)
        # initialize weights with random weights
        W_bound = numpy.sqrt(scale_w / (fan_in + fan_out))

        self.W.set_value(numpy.asarray(numpy.random.normal(0, W_bound, self.filter_shape), dtype=theano.config.floatX))
        self.b.set_value(numpy.asarray(numpy.zeros((self.filter_shape[0],), dtype=theano.config.floatX)))



    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given 'target distribution'.

        ! ONLY WORKS IF y IS A VECTOR OF INTEGERS !

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        return -T.mean(T.log(self.class_probabilities)[T.arange(y.shape[0]),y]) #shape of class_probabilities is e.g. (14*14,2) for 2 classes and 14**2 labels


    def negative_log_likelihood_true(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        ! y must be a float32-matrix of shape ('batchsize', num_classes) !

        """

        return -T.mean(T.sum(T.log(self.class_probabilities)*y,axis=1)) #shape of class_probabilities is e.g. (14*14,2) for 2 classes and 14**2 labels



    def negative_log_likelihood_ignore_zero(self, y):
        """--Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        -- zeros in <y> code for "not labeled", these examples will be ignored!
            side effect: class 0 in the NNet is basically useless.

        --Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """


        return -T.mean((theano.tensor.neq(y,0))*T.log(self.class_probabilities)[T.arange(y.shape[0]),y]) #shape of class_probabilities is e.g. (14*14,2) for 2 classes and 14**2 labels



    def negative_log_likelihood_modulated(self, y, modulation):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        <modulation> is an float32 vector, value=1 is default behaviour, 0==ignore

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        return -T.mean(modulation*T.log(self.class_probabilities)[T.arange(y.shape[0]),y])




    def negative_log_likelihood_modulated_margin(self, y, modulation=1, margin=0.7, penalty_multiplier = 0):
        print "negative_log_likelihood_modulated_margin:: Penalty down to ",100.*penalty_multiplier,"% if prediction is close to the target! Threshold is",margin
        penalty_multiplier = np.float32(penalty_multiplier)
        margin = np.float32(margin)
        selected = self.class_probabilities[T.arange(y.shape[0]),y]
        r = modulation*T.log(selected)
        return -T.mean(r*(selected<margin) + (0 if penalty_multiplier==0 else penalty_multiplier*r*(selected>=margin))  )



    def squared_distance(self, Target,b_flatten=False):
        """Target is the TARGET image (vectorized), -> shape(x) = (batchsize, imgsize**2)
            output: scalar float32
        """
        if b_flatten:
            return T.mean( (self.output.flatten(2) - Target)**2 )
        else:
            return T.mean( (self.output - Target)**2 )


    def squared_distance_w_margin(self, TARGET, margin=0.3):
        """  output: scalar float32
        """
        print "Conv3D::squared_distance_w_margin (binary predictions)."
        margin = np.float32(margin)
        out = self.output
        NULLz  = T.zeros_like(out)
        sqsq_err = TARGET * T.maximum(NULLz, 1 - out - margin)**2 + (1-TARGET) * T.maximum(NULLz, out - margin)**2
        return T.mean(sqsq_err)




    def cross_entropy(self, Target):
        """Target is the TARGET image (vectorized), -> shape(x) = (batchsize, imgsize**2)
            output: scalar float32
        """
        #print np.shape( theano.function([self.input], self.colorchannel_probabilities)(np.random.random( self.input_shape ).astype(np.float32)) )
        return -T.mean( T.log(self.class_probabilities)*Target + T.log(1-self.class_probabilities)*(1-Target) )# #.reshape(new_shape)[index[0]:index[2],index[1]:index[3]]


    def cross_entropy_array(self, Target):
        """Target is the TARGET image (vectorized), -> shape(x) = (batchsize, imgsize**2)
            the output is of length: <batchsize>, Use cross_entropy() to get a scalar output.
        """
        return -T.mean( T.log(self.class_probabilities)*Target + T.log(1-self.class_probabilities)*(1-Target) ,axis=1)



    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.class_prediction.ndim:
            raise TypeError('y should have the same shape as self.class_prediction',
                ('y', y.type, 'class_prediction', self.class_prediction.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.class_prediction, y))
        else:
            print "something went wrong"
            raise NotImplementedError()






