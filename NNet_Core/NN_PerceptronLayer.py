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


import numpy
import numpy as np
import time
import theano
import theano.tensor as T


from theano.tensor.shared_randomstreams import RandomStreams


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

class PerceptronLayer(object):
    def __init__(self, input, n_in, n_out, batchsize, bDropoutEnabled_, ActivationFunction = 'tanh', 
                 InputNoise=None, W=None, input_layer=None, b_experimental_inhibition_groups=False, flatW=False):
        """
        Typical hidden layer of a MLP: units are fully-connected.
        Weight matrix W is of shape (n_in,n_out), the bias vector b is of shape (n_out,).

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int or tuple/list
        :param n_out: number of hidden units

        :ActivationFunction: relu,sigmoid,tanh

        :InputNoise: theano.shared, float32 range 0 to 1 (0 = no noise)
        """
        self.input_layer = input_layer
        self.ActivationFunction = ActivationFunction
        
        
        if np.all(np.prod(n_out)==n_out):
            self.output_shape = (batchsize, n_out)
        else:
            self.output_shape = n_out#(batchsize, n_out)
            n_out = np.prod(n_out[1:])

        if InputNoise!=None:
            self.InputNoise=InputNoise
            print "PerceptronLayer::"+str(PerceptronLayer)+"InputNoise =",InputNoise
            rng = numpy.random.RandomState(int(time.time()))
            theano_rng = RandomStreams(rng.randint(2 ** 30))
            self.input = theano_rng.binomial(size=input.shape, n=1, p=1 - self.InputNoise,dtype=theano.config.floatX) * input
        else:
            self.input = input
            self.InputNoise=None
        print "PerceptronLayer( #Inputs =",n_in,"#Outputs =",n_out,")"

        if W==None:

            W_values = numpy.asarray(
            	numpy.random.uniform(
            	-numpy.sqrt(6. / (n_in + n_out)),
            	numpy.sqrt(6. / (n_in + n_out)),
            	(n_in, n_out)),
            	 dtype=theano.config.floatX)

            if flatW:
                self.flatW = theano.shared(value=W_values.flatten(), name='W_perceptron_flat_'+str(n_in)+'.'+str(n_out), borrow=True)
                self.W = self.flatW.reshape(W_values.shape)
            else:
                self.W = theano.shared(value=W_values, name='W_perceptron'+str(n_in)+'.'+str(n_out), borrow=True)
        else:
            print "Directly using given W (",W,"), not training on it in this layer!"#as this should happen in the other layer where this W came from.
            self.W = W


        b_values = numpy.asarray(numpy.random.uniform(-1e-8,1e-8,(n_out,)), dtype=theano.config.floatX) #1e-2*numpy.ones((n_out,), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, name='b_perceptron'+str(n_in)+'.'+str(n_out), borrow=True)

        self.conv_output = None

        lin_output = T.dot(self.input,self.W)  # + self.b



        self.Activation_noise = None
        if bDropoutEnabled_:
            print "Dropout..."
            self.Activation_noise = theano.shared(np.float32(0.5))
            rng = T.shared_randomstreams.RandomStreams(int(time.time()))
            #(n_out,)
            self.dropout_gate = np.float32(2.)*rng.binomial(lin_output.shape,np.float32( 1.), np.float32(1.)-self.Activation_noise ,dtype=theano.config.floatX) #rng.binomial((n_out,),1,1-self.Activations_noise_min,dtype=theano.config.floatX)
            lin_output =  lin_output * self.dropout_gate#.dimshuffle(('x', 0))             #(  1 - T.maximum(0.8-self.output, T.zeros_like(self.output))*



        lin_output = lin_output + self.b #add b after dropout


        if ActivationFunction=='tanh': #range = [-1,1]
            self.output = T.tanh(lin_output)# shape: (batch_size, num_outputs)
        elif ActivationFunction=='relu' or ActivationFunction=='ReLU': #rectified linear unit ,range = [0,inf]
            self.ActivationFunction = 'relu'

            self.output = T.maximum(lin_output,T.zeros_like(lin_output)) # 137.524226165 iterations/sec

        elif ActivationFunction=='abs': #symmetrically rectified linear unit ,range = [0,inf]
            self.output = T.abs_(lin_output)
        elif ActivationFunction=='sigmoid': #range = [0,1]
            print "WARNING: sig() used! Consider using abs() or relu() instead" # (abs > relu > tanh > sigmoid)
            b_values = 0.5*numpy.ones( (n_out,), dtype=theano.config.floatX)
            self.b.set_value(b_values)
            self.output = T.nnet.sigmoid(lin_output)#1/(1 + T.exp(-lin_output))
        elif ActivationFunction=='linear':
            if b_experimental_inhibition_groups==0:
                print "Warning: linear activation function! I hope this is the output layer?"
            self.output = (lin_output)
        elif ActivationFunction.startswith("maxout"):
            r=int(ActivationFunction.split(" ")[1])
            assert r>=2
            n_out = n_out/r
            self.output = max_pool_along_second_axis(lin_output,r)
        else:
            raise NotImplementedError("options are: ActivationFunction={tanh, relu, sigmoid,abs}")

        self.lin_output=lin_output


        self.class_probabilities = T.nnet.softmax(lin_output)# shape: (batch_size, num_outputs), num_outputs being e.g. the number of classes

        # compute prediction as class whose probability is maximal (in symbolic form)
        self.class_prediction = T.argmax(self.class_probabilities, axis=1)# shape: (batch_size,)


        if len(self.output_shape)>2:
            self.output = self.output.reshape(self.output_shape)
            
        self.n_in = n_in

        if W==None:
            try:
                a = self.flatW
                self.params = [self.flatW, self.b]
            except:
                self.params = [self.W, self.b]
        else:
            self.params = [self.b]




    def random_sparse_initialization(self, num_nonzero = 15, scale = 1.):
        """ exactly <num_nonzero> incoming weights per neuron will have a value of <scale>, the others will have a tiny random value"""
        n_in  = self.n_in
        n_out = self.output_shape[1]       
        print "MLP::random_sparse_initialization::(num_nonzero =",num_nonzero,", scale =",scale,")"
        assert n_in > num_nonzero
        
        w = numpy.asarray(numpy.random.uniform(
                -numpy.sqrt(0.1 / (n_in + n_out)),
                numpy.sqrt(0.1 / (n_in + n_out)),
                (n_in, n_out)), dtype=theano.config.floatX)
        
        base = np.random.permutation(range(n_in))
        for i in range(n_out):
            pick = np.random.permutation(base)[:num_nonzero]
            w[:,i][pick] = scale
        self.W.set_value(w)




    def randomize_weights(self, scale_w = 1.0):
        n_in  = self.n_in
        n_out = self.output_shape[1]

        self.W.set_value(numpy.asarray(numpy.random.uniform(
                -numpy.sqrt(scale_w * 6. / (n_in + n_out)),
                numpy.sqrt(scale_w * 6. / (n_in + n_out)),
                (n_in, n_out)), dtype=theano.config.floatX))

        if self.ActivationFunction == 'relu':
            b = 1.
        elif self.ActivationFunction == 'sigmoid':
            b=0.5
        else:
            b=0

        self.b.set_value(b * numpy.ones((n_out,), dtype=theano.config.floatX))



    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|} \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
                \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
      
        return -T.mean(T.log(self.class_probabilities)[T.arange(y.shape[0]), y])


    def negative_log_likelihood_modulated_margin(self, y, modulation=1, margin=0.7, penalty_multiplier = 0):
        print "negative_log_likelihood_modulated_margin:: Penalty down to ",100.*penalty_multiplier,"% if prediction is close to the target! Threshold is",margin
        penalty_multiplier = np.float32(penalty_multiplier)
        margin = np.float32(margin)
        selected = self.class_probabilities[T.arange(y.shape[0]),y]
        r = modulation*T.log(selected)
        return -T.mean(r*(selected<margin) + (0 if penalty_multiplier==0 else penalty_multiplier*r*(selected>=margin))  )



    def negative_log_likelihood_array(self, y):
        """Return the negative log-likelihood of the prediction
        of this model under a given target distribution for each element of the batch individually.

        """
        return -T.log(self.class_probabilities)[T.arange(y.shape[0]), y]


    def negative_log_likelihood_weighted(self, y, weight_vect):
        """
        weight_vect must be a vector of float32 of length = number_of_classes.
        Values: 1.0 (default), w < 1.0 (less important), w > 1.0 (more important class)
        """

        return -T.mean( weight_vect.dimshuffle('x',0)*(T.log(self.class_probabilities  )[T.arange(y.shape[0]), y]))
        



    def squared_distance(self, Target, Mask = None):
        """Target is the TARGET image (vectorized), -> shape(x) = (batchsize, imgsize**2)
            output: scalar float32
            mask: vectorized, 1==hole, 0==no_hole (== DOES NOT TRAIN ON NON-HOLES)
        """
        if Mask==None:
            return T.mean( (self.output - Target)**2 )
        else:
            print "squared_distance::Masked"
            return T.mean( ((self.output - Target)*T.concatenate( (Mask,Mask,Mask),axis=1 )  )**2 ) #assuming RBG input


    def squared_distance_array(self, Target, Mask = None):
        """Target is the TARGET image (vectorized), -> shape(x) = (batchsize, imgsize**2)
            output: scalar float32
            mask: vectorized, 1==hole, 0==no_hole (== DOES NOT TRAIN ON NON-HOLES)
        """
        if Mask==None:
            return T.mean( (self.output - Target)**2 ,axis=1)
        else:
            return T.mean( ((self.output - Target)*T.concatenate( (Mask,Mask,Mask),axis=1 ))**2 ,axis=1)#assuming RBG input



    def __make_window(self):
        print "window is on 32x32, fixed sigma, assuming RGB."
        denom = 29.8
        x0= 16
        sig = 19
        fun = lambda z,x,y: (32/denom* np.exp(-(abs(x - x0))**3/(2*sig**3)))*(32/denom*np.exp(-(abs(y - x0))**3/(2*sig**3)))#, {x, 0, 32}, {y, 0, 32}
        return np.fromfunction(fun,(3,32,32))



    def cross_entropy(self, Target, Mask = None):#, index, new_shape):
        """Target is the TARGET image (vectorized), -> shape(x) = (batchsize, imgsize**2)
            output: scalar float32
        """
        if Mask==None:
            #XX = window#T.TensorConstant(T.TensorType(theano.config.floatX,[True,False])(),data=window)
            return -T.mean( (T.log(self.class_probabilities )*Target + T.log(1.0 - self.class_probabilities)*(1.0-Target)) )# #.reshape(new_shape)[index[0]:index[2],index[1]:index[3]]
        else:
            print "cross_entropy::Masked, no window"
            return -T.mean( (T.log(self.class_probabilities )*Target + T.log(1.0 - self.class_probabilities)*(1.0-Target))*T.concatenate( (Mask,Mask,Mask),axis=1 ) )# #.reshape(new_shape)[index[0]:index[2],index[1]:index[3]]#assuming RBG input



    def cross_entropy_array(self, Target, Mask = None):#, index, new_shape):
        """Target is the TARGET image (vectorized), -> shape(x) = (batchsize, imgsize**2)
            the output is of length: <batchsize>, Use cross_entropy() to get a scalar output.
        """
        if Mask==None:
            return -T.mean( (T.log(self.class_probabilities )*Target + T.log(1.0 - self.class_probabilities)*(1.0-Target)) ,axis=1)
        else:
            return -T.mean( (T.log(self.class_probabilities )*Target + T.log(1.0 - self.class_probabilities)*(1.0-Target) )*T.concatenate( (Mask,Mask,Mask),axis=1 ),axis=1)#assuming RBG input



    def errors(self, y):
        """ Return a float representing the rel. number of errors in the minibatch (0 to 1=all wrong)
            0-1 loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """
        # check if y has same dimension of class_prediction
        if y.ndim != self.class_prediction.ndim:
            raise TypeError('y should have the same shape as self.class_prediction',
                ('y', y.type, 'class_prediction', self.class_prediction.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.class_prediction, y), dtype='float32')
        else:
            raise NotImplementedError()



    def CompileAutoencoderTrainingFunction(self, cnn_symbolic_input_x, cnn_symbolic_SGD_LR , b_use_cross_entropy_err=True, mode="sgd"):
        """
            using no momentum
            cnn_symbolic_input_x = cnn.x
        """
        all_params = self.params
        xin = self.input_layer
        layerz = 1
        while xin!=None:
            all_params+=xin.params
            xin = xin.input_layer
            layerz += 1

        print "CompileAutoencoderTrainingFunction... ChainLength =",layerz

        TARGET = T.fmatrix('x_raw_input')

        if b_use_cross_entropy_err==False:
            print "Using squared error (not using cross_entropy): training on output (e.g. sigmoid!) directly instead of softmax"
            cost = self.squared_distance(TARGET)
        else:
            cost = self.cross_entropy(TARGET)

        # create a list of gradients for all model parameters
        self.output_layer_Gradients = T.grad(cost, all_params)
        assert len(all_params)==len(self.output_layer_Gradients)

        if mode!="sgd":
            RPROP_penalty=0.25
            RPORP_gain=0.25
            self.RPROP_LRs=[]
            self.last_grads=[]
            for para in all_params:
                self.RPROP_LRs.append(theano.shared(  1e-4*np.ones(para.get_value().shape,dtype=theano.config.floatX) , name=para.name+str('_RPORP') , borrow=0))

                self.last_grads.append(theano.shared( np.zeros(para.get_value().shape,dtype=theano.config.floatX) , name=para.name+str('_LG') , borrow=0))
        else:
            self.last_grads = self.RPROP_LRs = [0]*len(all_params)


        self.SGD_updates=[]
        for param_i, grad_i, last_grad_i, pLR_i in zip(all_params, self.output_layer_Gradients, self.last_grads, self.RPROP_LRs ):
            print "warning: not sgd"
            
            if mode=="sgd":
                self.SGD_updates.append((param_i, param_i  - cnn_symbolic_SGD_LR *  grad_i ))#last_grad_i )) # use if Global_use_unique_LR==1 (1-self.SGD_weight_decay)*param_i
            else:
                self.SGD_updates.append((pLR_i, T.minimum( T.maximum( pLR_i * ( 1 - np.float32(RPROP_penalty)* ((last_grad_i*grad_i) < -1e-9) + np.float32(RPORP_gain)* ((last_grad_i*grad_i) > 1e-11)   ) , 2e-7*T.ones_like(pLR_i) ),8e-3 * T.ones_like(pLR_i)) ))
                self.SGD_updates.append((param_i, param_i  - pLR_i * grad_i/(T.abs_(grad_i) + 1e-6)  ))

                self.SGD_updates.append((last_grad_i, grad_i ))

        self.train_model_regression   = theano.function([cnn_symbolic_input_x, TARGET],  cost, updates=self.SGD_updates)# first input: image with holes etc. second input: clean image

        self.show_reconstruction = theano.function([cnn_symbolic_input_x], self.output if b_use_cross_entropy_err==False else self.class_probabilities) #input: holed or normal....
        
        return self.train_model_regression, self.show_reconstruction


