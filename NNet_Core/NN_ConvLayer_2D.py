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
from theano.tensor.nnet import conv #('batch', 'channel', x, y)
#from theano.sandbox.cuda.basic_ops import gpu_contiguous

import time

def max_pool_along_second_axis(sym_input, pool_factor):
    """ for MLP and 2D conv"""
    s = None
    for i in xrange(pool_factor):
        t = sym_input[:,i::pool_factor]
        if s is None:
            s = t
        else:
            s = T.maximum(s, t)
    return s
    
    
try:
    import pylearn2.linear.conv2d_c01b as convnet#('channel', x, y, 'batch')
except:
    pass


def get_reconstructed_input(self, hidden):
    """ Computes the reconstructed input given the values of the hidden layer """
    repeated_conv = conv.conv2d(input = hidden, filters = self.W_prime, border_mode='full')

    multiple_conv_out = [repeated_conv.flatten()] * np.prod(self.poolsize)

    stacked_conv_neibs = T.stack(*multiple_conv_out).T

    stretch_unpooling_out = theano.sandbox.neighbours.neibs2images(stacked_conv_neibs, self.pl, self.x.shape)

    rectified_linear_activation = lambda x: T.maximum(0.0, x)
    return rectified_linear_activation(stretch_unpooling_out + self.b_prime.dimshuffle('x', 0, 'x', 'x'))



def my_max_pool_2d(sym_input, pool_shape = (2,2)):
    """ this one is pure theano. Hence all gradient-related stuff is working! No dimshuffling"""

    s = None
    for i in xrange(pool_shape[1]):
        t = sym_input[:,:,:,i::pool_shape[1]]
        if s is None:
            s = t
        else:
            s = T.maximum(s, t)

    temp = s
    s = None
    for i in xrange(pool_shape[0]):
        t = temp[:,:,i::pool_shape[0],:]
        if s is None:
            s = t
        else:
            s = T.maximum(s, t)

    sym_ret = s

    return sym_ret


class ConvPoolLayer(object):
    """2D Conv/Pool Layer of a convolutional network """

    def __init__(self,  input, filter_shape, input_shape, poolsize, bDropoutEnabled_, bUpsizingLayer=False, 
                 bTheanoConv=True, convolution_stride = 1, b_ReverseConvolution = False,
                 ActivationFunction = 'tanh', input_layer =  None, 
                 W=None, b=None, use_fragment_pooling = 0, dense_output_from_fragments = 0, output_stride=None,
                 input_axes="theano", output_axes="theano"):
        """
        Allocate a ConvPoolLayer with shared variable internal parameters.

        bUpsizingLayer = True: =>  bordermode = full (zero padding) thus increasing the output image size (as opposed to shrinking it in 'valid' mode)

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape input_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, input_channels, filter height,filter width)

        :type input_shape: tuple or list of length 4
        :param input_shape: (batch size, num input feature maps, image height, image width)

        :type poolsize: integer (typically 1 or 2)
        :param poolsize: the downsampling (max-pooling) factor

        :bTheanoConv:
            input=('batch', 'channel', x, y)\
            W=(n_filters,'channel', wx, wy)\
        else:\
            input = ('channel', x, y, 'batch')\
            W=('channel', wx, wy, n_filters)\
            BUT CONSIDER: the input can be presented as bc01 and will be reshaped to c01b automatically if input_axes=="theano"\
        :input_axes: possible values: 'theano' and 'native'; only has an effect if bTheanoConv==False.
        :output_axes: possible values: 'theano' and 'native'; only has an effect if bTheanoConv==False.
        :b_ReverseConvolution: set-up a deconvolutional layer (this will invert the convolution step \
        and force <bTheanoConv>==False. You may use <convolution_stride> > 1 in this case to create a larger output\
        than the size of the input

        """
        assert output_axes in ["theano","native"]
        assert input_axes  in ["theano","native"]
        if b_ReverseConvolution:
            print "DeconvPoolLayer(2D)"
            bTheanoConv = 0
        else:
            print "ConvPoolLayer(2D)"
        print "   input (image) =",input_shape
            
        assert use_fragment_pooling == 0,"todo"
        assert dense_output_from_fragments==0,"todo"
        assert len(filter_shape)==4
        if b_ReverseConvolution==False:
            assert input_shape[1] == filter_shape[1]
        else:
            assert input_shape[1] == filter_shape[0]
        
        
        filter_shape_xy = filter_shape[2:]
        input_shape_xy = input_shape[2:]


        self.input = input
        #self.rng=rng
        self.pooling_factor=poolsize
        self.number_of_filters = filter_shape[0]
        self.filter_shape=filter_shape

        self.input_shape = input_shape
        self.input_layer = input_layer


        # filter is 3D of shape (depth==#filtersPreviousLayer,x,y)

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = 1.0*numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = 1.0*(filter_shape[0] * numpy.prod(filter_shape[2:])/poolsize)
        # initialize weights with random weights
        W_bound = numpy.sqrt(3. / (fan_in + fan_out)) #6.0 / numpy.prod(filter_shape[1:]) #


        print "   filter        =",filter_shape," @ std =",W_bound

        #        print "w_init std =",W_bound
        if bTheanoConv==False:
            print "WARNING: layout of W has changed (bTheanoConv==False)! Beware if ConvPoolLayer.W is directly accessed!"
        if W==None:
            self.W = theano.shared(#numpy.asarray(        self.rng.normal(loc=0, scale = W_bound, size=filter_shape), dtype=theano.config.floatX)
            numpy.asarray(numpy.random.normal(0, W_bound, filter_shape if bTheanoConv else (filter_shape[1],filter_shape[2],filter_shape[3],filter_shape[0]) ), dtype=theano.config.floatX)
            ,  borrow=True, name='W_conv')
        else:
            self.W = W
        #print 'mean=',np.mean(np.abs(self.W.get_value(borrow=True))) ,'median=',np.median(np.abs(self.W.get_value(borrow=True)))

        Activation_f = lambda x: x

        self.ActivationFunction = ActivationFunction

        cross_channel_pooling_groups = 1 # 1== nothing is done.

        if ActivationFunction=='tanh':
            Activation_f = T.tanh# shape: (batch_size, num_outputs)
        elif ActivationFunction in ['ReLU', 'relu']: #rectified linear unit
            self.ActivationFunction = "relu"
            Activation_f = lambda x: x*(x>0)# shape: (batch_size, num_outputs)
        elif ActivationFunction in ['sigmoid', 'sig']:
            self.ActivationFunction = "sigmoid"
            Activation_f = T.nnet.sigmoid
        elif ActivationFunction in ['abs']:
            self.ActivationFunction = "abs"
            Activation_f = T.abs_
        elif "maxout" in ActivationFunction:
            r=int(ActivationFunction.split(" ")[1])
            assert r>=2
            cross_channel_pooling_groups = r
        elif ActivationFunction in ['linear',"lin"]:
            self.ActivationFunction = "linear"
            Activation_f = lambda x:x
        else:
            raise NotImplementedError()






        #rng.uniform(low=-W_bound, high=W_bound, size=filter_shape), dtype=theano.config.floatX)     ,  borrow=True, name='W')
            #self.W = theano.shared(numpy.zeros(filter_shape, dtype=theano.config.floatX)  ,  borrow=True, name='W')

        num_output_channels = int(filter_shape[0]/cross_channel_pooling_groups)

        if ActivationFunction in ['sigmoid', 'sig']:
            b_values =  0.5*numpy.ones((num_output_channels,), dtype=theano.config.floatX) #/filter_shape[3]/filter_shape[2]
            #self.b =  theano.shared(value=b_values, borrow=True, name='b_conv')
        else:
            # one bias per output feature map
            if ActivationFunction in ['relu', 'ReLu']:
                b_values =  1e-1*numpy.ones((num_output_channels,), dtype=theano.config.floatX)
            else:
                b_values =  numpy.zeros((num_output_channels,), dtype=theano.config.floatX)
        if b==None:
            self.b =  theano.shared(value=b_values, borrow=True, name='b_conv')
        else:
            self.b = b

        # convolve input feature maps with filters
        if bTheanoConv:
            assert convolution_stride==1, "ERROR: set bTheanoConv=False to use a convolution_stride other than 1!"
            
            if bUpsizingLayer==0:
                sp_after_conv = (input_shape_xy[0]-filter_shape_xy[0]+1, input_shape_xy[1]-filter_shape_xy[1]+1)
            else:
                sp_after_conv = (input_shape_xy[0]+filter_shape_xy[0]-1, input_shape_xy[1]+filter_shape_xy[1]-1)
            
            is_error=0
            if poolsize==2:
                if sp_after_conv[0]%2 !=0 or sp_after_conv[1]%2 !=0:
                    is_error=1
                
            if is_error:
                print
                print "ERROR/WARNING: Please use a different filter shape OR different input shape"
                print "2D Input shape: ",input_shape_xy
                print "2D Filter shape:",filter_shape_xy
                print "2D Conv. output:",sp_after_conv,"is not divisible by 2 (pooling)!"
                
                raise StandardError("ERROR/WARNING: Please use a different filter shape OR different input shape")
            
            #shape of pooled_out , e.g.: (1,2,27,27) for 2 class-output
            self.conv_out = conv.conv2d(input=input, 
                                        filters=self.W, 
                                        border_mode = 'full' if bUpsizingLayer else 'valid',
                                        filter_shape=filter_shape, 
                                        image_shape = input_shape if input_shape[0]!=None else None)
                                        #, unroll_batch=0, unroll_kern=0, unroll_patch=1, openmp = 0
        else:
            # input = ( in_channels, x, y, batch_size)#
            #Channels must be <=3, or be even.
            #channels must be divisible by 4. Must be C contiguous. You can enforce this by calling theano.sandbox.cuda.basic_ops.gpu_contiguous on it.
            #filters: (in_channels, filter_x, filter_y, num_filters)
            #output: (output channels, output rows, output cols, batch size)
            print "WARNING: reshaping input & output, to match the cuda-convnet convention! Modify to gain speed."
            print "Use batchsize = n*128  (n=1,2,3,...)"
            assert self.number_of_filters%16==0,"Not supported by Alex' code"
            print "input will be automatically reshaped (as if TheanoConv was used)"
            cc_input_shape = None
            if b_ReverseConvolution:
                print "b_ReverseConvolution:: using strange 'input_shape' (-2 instead of -1)"
                print "input_shape",input_shape
                if input_axes=="theano":
                    cc_input_shape = (convolution_stride*input_shape[2] + self.filter_shape[2]-2, convolution_stride*input_shape[3] + self.filter_shape[3]-2)
                else:
                    cc_input_shape = (convolution_stride*input_shape[1] + self.filter_shape[2]-2, convolution_stride*input_shape[2] + self.filter_shape[3]-2)
                print "cc_input_shape ",cc_input_shape 
            tmp = convnet.Conv2D(self.W, 
                                 input_axes =('b', 'c', 0, 1) if (b_ReverseConvolution==False and input_axes=="theano" or b_ReverseConvolution==True and output_axes=="theano")  else ('c', 0, 1, 'b'), 
                                 output_axes=('b', 'c', 0, 1) if (b_ReverseConvolution==False and output_axes=="theano" or b_ReverseConvolution==True and input_axes=="theano") else ('c', 0, 1, 'b'), 
                                 kernel_stride=(convolution_stride,convolution_stride), 
                                 pad= int((filter_shape[2]-1)/2.) if bUpsizingLayer else 0,input_shape=cc_input_shape, batch_size=None)#input_shape[0]



                 
                 
            input_mod = input#.dimshuffle(1,2,3,0) #gpu_contiguous( input.dimshuffle(1,2,3,0))
            if b_ReverseConvolution:
                
                self.conv_out = tmp.lmul_T(input_mod)#.dimshuffle(3,0,1,2)
            else:
                self.conv_out = tmp.lmul(input_mod)#.dimshuffle(3,0,1,2)
                                           
                                           
#            conv_out = filter_acts.FilterActs( stride=convolution_stride, pad= int((filter_shape[2]-1)/2) if bUpsizingLayer else 0)(input.dimshuffle(1,2,3,0), self.W).dimshuffle(3,0,1,2)
#            conv_out = cc_temp.make_node


#            print "todo: add alex max-pooling here..."
        # downsample each feature map individually, using maxpooling


#        self.conv_output = Activation_f(self.conv_out  + self.b.dimshuffle('x', 0, 'x', 'x'))

        if poolsize !=1:
            assert poolsize==2
            pooled_out =   my_max_pool_2d(sym_input = self.conv_out, pool_shape = (poolsize,poolsize))#downsample.max_pool_2d(input=self.conv_out,ds=(poolsize,poolsize), ignore_border=True) #(first + downsample.max_pool_2d(input=conv_out[1:,:],ds=poolsize, ignore_border=True)+ downsample.max_pool_2d(input=conv_out[:,1:],ds=poolsize, ignore_border=True)+ downsample.max_pool_2d(input=conv_out[1:,1:],ds=poolsize, ignore_border=True)).reshape((4,)+first.shape()[1:])
        else:
            pooled_out = self.conv_out

        if cross_channel_pooling_groups>1:
            pooled_out = max_pool_along_second_axis(pooled_out,cross_channel_pooling_groups)

        #lazy hack
#        print input_shape (None, 4, 32, 32)
        if input_shape[0]==None:
            output_shape = list(theano.function([input], pooled_out.shape,mode='FAST_COMPILE' if bTheanoConv else 'FAST_RUN')(numpy.zeros((1,)+input_shape[1:],dtype=numpy.float32)))
            output_shape[0] = None
            output_shape=tuple(output_shape)
        else:
            output_shape = tuple(theano.function([input], pooled_out.shape,mode='FAST_COMPILE' if bTheanoConv else 'FAST_RUN')(numpy.zeros(input_shape,dtype=numpy.float32)))
        print "   output        =",output_shape

        self.output_shape = output_shape



        lin_output = pooled_out + self.b.dimshuffle('x', 0, 'x', 'x')


        self.output = Activation_f(lin_output)





#       self.output = T.tanh(()) #(1, 49, 247, 247)

#       input (image) = (None, 4, 32, 32)
#       filter        = (15, 4, 5, 5)
#       output        = (None, 15, 28, 28)

        if bDropoutEnabled_:
            print "Dropout enabled."
            self.SGD_dropout_rate = theano.shared(np.asscalar(np.ones(1,dtype=np.float32)*0.5))
            rng = T.shared_randomstreams.RandomStreams(int(time.time()))
            #(self.output_shape[2],self.output_shape[2])
            self.dropout_gate = 2*rng.binomial(self.output.shape,1,1.0-self.SGD_dropout_rate,dtype=theano.config.floatX) #theano.shared(value=numpy.ones((n_out),dtype=np.float32), name='percep_dropout_gate')
            self.output =  self.output * self.dropout_gate#.dimshuffle(('x', 'x', 0, 1))#T.switch(self.bDropoutEnabled, self.output * self.dropout_gate.dimshuffle(('x', 0)), self.output )



        # (bs,ch,x,y) --> (ch,bs,x,y), flatten this --> (ch, bs*x*y), swap labels --> (bs*x*y, ch)
        self.class_probabilities = T.nnet.softmax( lin_output.dimshuffle((1,0,2,3)).flatten(2).dimshuffle((1,0))  )#e.g. shape is (484, 2) for 2 classes ( i.e. have to set n.of filters = 2) and predicting  22x22 labels at once
        #undo: 
        #(x*y*bs, ch) --> (bs,x,y, ch)  --swap-->  (bs, ch, x, y)
        self.class_probabilities_realshape = self.class_probabilities.reshape(((self.output_shape[0] if self.output_shape[0] is not None else 1),)+tuple(self.output_shape[2:])+(self.output_shape[1],)).dimshuffle((0,3,1,2))


    #   input (image) = (10, 32, 36, 36)
    #   filter        = (3, 32, 5, 5)
    #   output        = (10, 3, 32, 32)
#        to interpret as 10 color images 32x32

#e_x     = exp(x - input.max(axis=1, keep_dims=True))
#softmax = e_x / e_x.sum(axis=1, keep_dims=True)

#        if output_shape[0]==None:
#            print "Please change the last reshape() of the next line!"
#            raise NotImplementedError()

        #colorchannel_probabilities are done separately for each color channel and (of course) separately for each image in the batch
        self.colorchannel_probabilities = T.nnet.softmax( (lin_output).flatten(3).dimshuffle((2,0,1)).flatten(2).dimshuffle((1,0))  ).reshape((-1, np.product(self.output_shape[1:]) )) # reshape((self.output_shape[0], np.product(self.output_shape[1:]) ))
        #pooled_out + self.b.dimshuffle('x', 0, 'x', 'x')

#flatten(x) collapses the last dimensions until x dimensions remain.

#        print np.shape(theano.function([self.input],(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x')).flatten(3))(np.random.random(self.input_shape).astype(np.float32)) )
#        raw_input("BLAA")


        # compute prediction as class whose "probability" is maximal in symbolic form
        self.class_prediction = T.argmax(self.class_probabilities, axis=1)
        #(x*y*bs) --> (x,y,bs) --> (bs,x,y)
#        self.class_prediction_realshape = T.argmax(self.class_probabilities, axis=1).reshape(tuple(self.output_shape[2:])+(self.output_shape[0],)).dimshuffle((2,0,1))
        #output has shape e.g. (1,2,57,57); only the last two may change, 2 classes are predicted

        #self.class_probabilities = output_softmax


        # store parameters of this layer
        self.params = [self.W, self.b]

    def randomize_weights(self, scale_w = 1.0, value_b = 0.01):
        self.W.set_value(numpy.asarray(numpy.random.normal(0, 0.02 * scale_w, self.filter_shape), dtype=theano.config.floatX))
        self.b.set_value(numpy.asarray(value_b * numpy.ones((self.filter_shape[0],), dtype=theano.config.floatX)))


    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain [0,1,2,... n-1]
        # T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        #print "at least, y must be provided in a flattened view (a list of class values)!"

        return -T.mean(T.log(self.class_probabilities)[T.arange(y.shape[0]),y]) #shape of class_probabilities is e.g. (14*14,2) for 2 classes and 14**2 labels



    def negative_log_likelihood_classwise_masking(self, y, mask_class_labeled, mask_class_not_present):
        """
        todo: test.
        :y: true classes (as integer value): (batchsize, x, y)
        :mask_class_labeled: matrix: (batchsize, num_classes) allowed values: 0 or 1; setting everything to 1 leads to the ordinary nll; all zeroes is an invalid state.
                    a zero for one class indicates that this class may be present but is not labeled as such.
        :mask_class_not_present: (batchsize, num_classes): similar to mask_class_labeled, but now a 1 indicates that a class is CERTAINLY NOT PRESENT in the batch.
        
        values of -1 in y count as "absolutely not labeled / ignore predictions"; this has PRIORITY over anything else (including mask_class_not_present).
        """
        y                      = y.dimshuffle(0, 'x', 1, 2)                        #(batchsize, 1, x, y)
        mask_class_labeled     = mask_class_labeled.dimshuffle(0, 1, 'x', 'x')     #(batchsize, num_classes,1 ,1)
        mask_class_not_present = mask_class_not_present.dimshuffle(0, 1, 'x', 'x') #(batchsize, num_classes,1 ,1)
        global_loss_mask = (y != -1) #apply to overall loss after everything is calculated; marks positions 
        
        
        pred = self.class_probabilities_realshape # (batchsize, num_classes, x, y)
        mod_y = T.where(y<0,0,y)
        
        #dirty hack: compute "standard" nll when most predictive weight is put on classes which are in fact labeled
        votes_not_for_unlabeled = T.where( T.sum(pred*mask_class_labeled,axis=1)>=0.5, 1, 0 ).dimshuffle(0,'x',1,2)

        # could also add '* mask_class_labeled' inside, but this should not change anything , provided there is no logical conflict between y and mask_class_labeled !
        nll = -T.mean((T.log(pred) * votes_not_for_unlabeled * global_loss_mask)[:,mod_y]) #standard loss part -> increase p(correct_prediction); thus disabled if the "correct" class is not known
        
        # penalize predictions: sign is a plus! (yes: '+')
        # remove <global_loss_mask> if <mask_class_not_present> should override 'unlabeled' areas.
        nll += T.mean(T.log(pred) * mask_class_not_present * global_loss_mask) 
        
        return nll
        
#        no_cls   = T.alloc(np.int16(255), 1,1,1,1)
#        no_cls_ix = T.eq(no_cls,y) # (bs,x,y) tensor, 1 where y==255
#        # true if y==255 AND for outputneurons of (generally) labelled classes,
#        # i.e. at positions all where those classes are NOT appearing in the        data
#        no_cls_ix = no_cls_ix.dimshuffle * mask_class_labeled.dimshuffle(0, 1, 'x', 'x')
#        no_cls_ix = no_cls_ix.nonzero() # selects the output neurons of        negatively labelled pixels
#        
#        ix       =       T.arange(self.class_probabilities.shape[1]).dimshuffle('x', 0, 'x','x')  # (1,4,1, 1 )
#        select   = T.eq(ix,y).nonzero() # selects the output neurons of        positively labelled pixels
#        
#        push_up  = -T.log(self.class_probabilities)[select]
#        push_dn  =  T.log(self.class_probabilities)[no_cls_ix] / mask_class_labeled.sum(axis=1).dimshuffle(0, 'x', 'x', 'x')
#        nll_inst = push_up + push_dn
#        nll      = T.mean(nll_inst)



    def negative_log_likelihood_ignore_zero(self, y):
        """--Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        -- zeros in <y> code for "not labeled", these examples will be ignored!
            side effect: class 0 in the NNet is basically useless.

        --Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """

        return -T.mean((theano.tensor.neq(y,0))*T.log(self.class_probabilities)[T.arange(y.shape[0]),y]) #shape of class_probabilities is e.g. (14*14,2) for 2 classes and 14**2 labels







    def squared_distance(self, Target):
        """Target is the TARGET image (vectorized), -> shape(x) = (batchsize*imgsize**2)
            output: scalar float32
        """
        # use flatten(2) if the shape should be (batchsize,imgsize**2)
        return T.mean( (self.output.flatten() - Target)**2 )
        


    def cross_entropy(self, Target):#, index, new_shape):
        """Target is the TARGET image (vectorized), -> shape(x) = (batchsize, imgsize**2)
            output: scalar float32
        """
        #print np.shape( theano.function([self.input], self.colorchannel_probabilities)(np.random.random( self.input_shape ).astype(np.float32)) )
        return -T.mean( T.log(self.colorchannel_probabilities)*Target + T.log(1-self.colorchannel_probabilities)*(1-Target) )# #.reshape(new_shape)[index[0]:index[2],index[1]:index[3]]


    def cross_entropy_array(self, Target):#, index, new_shape):
        """Target is the TARGET image (vectorized), -> shape(x) = (batchsize, imgsize**2)
            the output is of length: <batchsize>, Use cross_entropy() to get a scalar output.
        """
        return -T.mean( T.log(self.colorchannel_probabilities)*Target + T.log(1-self.colorchannel_probabilities)*(1-Target) ,axis=1)


#    def add_normalization(self):
#
#        #lcn = LeCunLCN(img_shape = self.output_shape, kernel_size=7, batch_size=self.input_shape[0], threshold=1e-4, channels=None)
#        print "add_normalization::",self.output_shape
#
#        print "TODO: try normalization BEFORE max-pooling"
#        self.output_no_lcn = self.output
#        self.output = pylearn2_preprocessing.lecun_lcn( self.output, img_shape = self.output_shape, kernel_shape=(5,5),  threshold=1e-4)#lcn(self.output)
#        return


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