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

"""

example (2D CNN):


    filter_sizes    = [5, 7, 9,  5,  1] #layer0, layer1...
    pooling_sizes   = [2, 2, 2,   1, 1]
    nof_filters     = [40,41,42,200, 2]


Initial image size is 168x168
batchsize = 1 (hence 1 in each output[0])

ConvPoolLayer::init().
   input (image) = (1, 1, 168, 168)
   filter        = (40, 1, 5, 5)
   output        = (1, 40, 82, 82)

ConvPoolLayer::init().
   input (image) = (1, 40, 82, 82)
   filter        = (41, 40, 7, 7)
   output        = (1, 41, 38, 38)

ConvPoolLayer::init().
   input (image) = (1, 41, 38, 38)
   filter        = (42, 41, 9, 9)
   output        = (1, 42, 15, 15)

ConvPoolLayer::init().
   input (image) = (1, 42, 15, 15)
   filter        = (200, 42, 5, 5)
   output        = (1, 200, 11, 11)

ConvPoolLayer::init().
   input (image) = (1, 200, 11, 11)
   filter        = (2, 200, 1, 1)
   output        = (1, 2, 11, 11)


"""



import cPickle
import time
import numpy
import numpy as np

import theano
import theano.tensor as T

import NN_Optimizers

from NN_ConvLayer_2D import ConvPoolLayer
from NN_ConvLayer_3D import ConvPoolLayer3D

from NN_PerceptronLayer import PerceptronLayer
from matplotlib import pyplot as plot

from os import makedirs as _makedirs
from os.path import exists as _exists


def extract_filename(string, remove_trailing_ftype=True, trailing_type_max_len=7):
    """ removes path (in front of the file name) and removes the file-type after the '.' (optional).
    returns: path & file_name"""
    A = string.replace("\\","/").split("/")
    path = ("/".join(A[:-1]))+"/"
    if len(path)==1:
        path=""
    B=A[-1]
    if remove_trailing_ftype:
        file_name = ".".join(B.split(".")[:-1])
        if len(file_name)==0 or len(B)-len(file_name)>(trailing_type_max_len+1):
            file_name = B
    else:
        file_name=B
    return path, file_name

def mkdir(path):
    if len(path)>1 and _exists(path)==0:
        _makedirs(path)


def show_multiple_figures_add(fig, n, i, image, title, isGray=True):
    """ add <i>th (of n, start: 0) image to figure <fig> as subplot (GRAY)"""

    x = int(np.sqrt(n)+0.9999)
    y = int(n/x+0.9999)
    if(x*y<n):
        if x<y:
            x+=1
        else:
            y+=1

    ax = fig.add_subplot(x, y, i) #ith subplot in grid x,y
    ax.set_title(title)
    if isGray:
        plot.gray()
    ax.imshow(image,interpolation='nearest')

    return 0





#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------
class MixedConvNN():
    def __init__(self,input_image_size, batchsize=None, ImageDepth = 1,
                 InputImageDimensions = None, bSupportVariableBatchsize=True,
                 bDropoutEnabled_ = False, bInputIsFlattened=False, verbose = 1, bWeightDecay = False):
        """ Otherwise:
            Assuming that <input_image_size> == Image_width == Image_height UNLESS it is a tuple
            <InputImageDimensions> my be

            <ImageDepth> is 1 by default, but change it to 3 if you use RGB images, 4 if you use RGB-D images etc.

            bDropoutEnabled_ must be set to True if it is to be used anywhere in the network!
            You can disable it at any time in the future (incurring a speed-performance loss as compared to disabling it right here)
        """
        if bSupportVariableBatchsize==True:
            batchsize = None
            self.batchsize = None
            #print "bSupportVariableBatchsize is in EXPERIMENTAL stage!"

        else:
            self.batchsize = batchsize
        if not isinstance(InputImageDimensions, list) and not isinstance(InputImageDimensions, tuple):
            if InputImageDimensions is None:
                print "assuming input dimension==1 (if wrong: specify <InputImageDimensions> or set <input_image_size> as a tuple)"
                InputImageDimensions=1
            input_image_size = (int(input_image_size),)*InputImageDimensions

        self.y = T.wvector('y_cnn_labels')   # the labels are presented as 1D vector (int16) (ivector is int32)
        self.rng = numpy.random.RandomState(int(time.time()))
        self.layers = [] #will contain all layers ([0] input layer ---> [-1] output layer)
        self.autoencoderChains=[]
        self.output_layers = [] # this will stay empty, UNLESS you use addOutputFunction ... these layers will NOT be included in self.layers!
        self.SGD_global_LR_output_layers_multiplicator = theano.shared(np.float32(1.0))
        self.TotalForwardPassCost = 0 # number of multiplications done
        self.verbose = verbose
        self.output = None
        #self.output_layers_params = []
        self.params=[] # after calling CompileOutputFunctions():
        self.bDropoutEnabled = bDropoutEnabled_

        # Reshape matrix of rasterized images of shape (batch_size,input_image_size*input_image_size)
        # to a 4D tensor, compatible with our ConvPoolLayer

        if ImageDepth==1 and InputImageDimensions!=3:
            if bInputIsFlattened or InputImageDimensions==1:
                self.x = T.fmatrix('x_cnn_input')   # the data is presented as rasterized images (np.float32)
            else:
                self.x = T.ftensor4('x_cnn_input')
        else:
            if InputImageDimensions==3:
                self.x = T.TensorType('float32',(False,)*5,name='x_cnn_input')('x_cnn_input')
            else:
                self.x = T.ftensor4('x_cnn_input')   #

        assert InputImageDimensions in [1,2,3],"MixedConvNN::InputImageDimensions  currently unsupported"

        if InputImageDimensions==2:
            if self.batchsize != None:
                self.layer0_input = self.x.reshape((batchsize, ImageDepth, input_image_size[0], input_image_size[1])) #1st entry is batch_size, but it is 1 for the all-pure-convolutional net
            else:
                self.layer0_input = self.x
            self.input_shape = (batchsize, ImageDepth, input_image_size[0], input_image_size[1]) # valid for FIRST LAYER only. each layer has one entry called like this
        elif InputImageDimensions==3:
            if self.batchsize != None:

                self.layer0_input = self.x.reshape((batchsize, input_image_size[0], ImageDepth, input_image_size[1], input_image_size[2])) #1st entry is batch_size, but it is 1 for the all-pure-convolutional net
            else:
                self.layer0_input = self.x
            self.input_shape = (batchsize, input_image_size[0], ImageDepth, input_image_size[1], input_image_size[2])
        else:
            if self.batchsize != None:
                self.layer0_input = self.x.reshape((batchsize, input_image_size[0]))
            else:
                self.layer0_input = self.x
            self.input_shape = (batchsize, input_image_size[0]) # valid for FIRST LAYER only. each layer has one entry called like this

        self.SGD_global_LR = theano.shared(np.float32(1e-3))
        self.SGD_momentum = theano.shared(np.float32(0.9))
        

        self.debug_functions=[]
        self.debug_functions_conv_output=[]
        self.debug_gradients_function=None
        self.debug_lgradients_function=None

        self.output_stride = 1 #for fragment-max-pooling (fast segmentation/sliding window)

        self.bWeightDecay = bWeightDecay
        self.CompileSGD   = NN_Optimizers.CompileSGD
        self.CompileRPROP = NN_Optimizers.CompileRPROP
        #self.compileCG    = NN_Optimizers.compileCG
        #self.CompileARP   = NN_Optimizers.CompileARP
        self.CompileADADELTA   = NN_Optimizers.CompileADADELTA



#        self.trainingStepCG_g = NN_Optimizers.trainingStepCG_g
#        self._lineSearch_g    = NN_Optimizers._lineSearch_g
#        self._lineSearch      = NN_Optimizers._lineSearch
#        self._trainingStepCG  = NN_Optimizers._trainingStepCG




    def randomize_weights(self,b_reset_momenta=False,b_ONLY_reset_momenta=False, scale_w = 1.0):
        """ reset weights to initial values (calls randomize_weights() on each layer)"""

        if b_ONLY_reset_momenta==False:
            for lay in self.layers + self.output_layers:
                try:
                    lay.randomize_weights(scale_w = scale_w)
                except:
                    print 'randomize_weights() failed in',lay

        if b_reset_momenta:
            for para,rp,lg in zip(self.params, self.RPROP_LRs, self.last_grads):
                sp = para.get_value().shape
                rp.set_value( 1e-3*np.ones(sp,dtype=theano.config.floatX) , borrow=0)
                lg.set_value(  np.zeros(sp,dtype=theano.config.floatX), borrow=0)
        return



    #adds one convolutional layer
    def addConvLayer(self, nof_filters, filter_size, pooling_factor = 2, pooling_size=None,
                     ndim=2, pooling_stride=None, b_forceNoDropout=False,
                     bAddInOutputLayer=False, bUpsizingLayer=False, ActivationFunction = 'abs',
                     bTheanoConv=True, convolution_stride = 1, b_ReverseConvolution=False, layer_input_shape=None,
                     layer_input=None, share_params_with_this_layer=None,
                     use_fragment_pooling = False, dense_output_from_fragments = False,
                     input_axes="theano", output_axes="theano"):
        """ Set b_forceNoDropout=True if this shall be the output layer
            <filter_size>: integer (filter size identical along each dimension) or list/tuple (custom filter dimensions)
            :input_axes:  possible values: 'theano' (-> bc01) and 'native' (-> c01b); only has an effect if bTheanoConv==False.
            :output_axes: possible values: 'theano' (-> bc01) and 'native' (-> c01b); only has an effect if bTheanoConv==False.
        """
        assert pooling_size is None and pooling_stride is None,"CNN::addConvLayer:: these parameters were unified into <pooling_factor>"

        if layer_input_shape==None:
            layer_input_shape = self.input_shape if self.layers==[] else self.layers[-1].output_shape
        elif len(self.layers)>0:
            if layer_input_shape != self.layers[-1].output_shape:
                assert np.prod(self.layers[-1].output_shape)==np.prod(layer_input_shape),"Error: cannot reshape <"+str(self.layers[-1].output_shape)+"> to match <layer_input_shape>=="+str(layer_input_shape)
                self.layers[-1].output_shape = layer_input_shape
                self.layers[-1].output = self.layers[-1].output.reshape(layer_input_shape)

        if layer_input==None:
            layer_input       = (self.layer0_input if len(self.layers)==0 else self.layers[-1].output)

        assert (layer_input==layer_input_shape==None or (layer_input!=None and layer_input_shape!=None)),"Provide both layer_input_shape and layer_input at the same time! (Or just leave them @ None)"


        assert len(layer_input_shape) in [4,5],"Please provide a valid <layer_input_shape> if you want to place a Convlayer on top of a Perceptron layer! 2D shape: (batchsize, channels, x, y), 3D shape: (batchsize, x, channels, y, z)"
        assert ndim in [2,3],"only 2d and 3d convolution supported!"


        if isinstance(filter_size, (int, long, float, complex)):
            filter_size = (filter_size,)*ndim

        self.output_stride = self.output_stride * np.asarray(pooling_factor)


        if self.batchsize != None:
        # in 2D:
        #   input (image) = (1, 41, 38, 38)
        #   filter        = (42, 41, 9, 9)
        #   output        = (1, 42, 15, 15)
        # in 3D
        #        input:  (1, 70,  3, 70, 70)
        #       filters: (32, 5 , 3,  5 , 5)
        #    --> output: (1, 66, 32, 66, 66)

            if ndim==2:
                n_pos = ((layer_input_shape[2]+1-filter_size[0])*(layer_input_shape[3]+1-filter_size[1])) #number positions
            if ndim==3:
                n_pos = ((layer_input_shape[1]+1-filter_size[0])*(layer_input_shape[3]+1-filter_size[1])*(layer_input_shape[4]+1-filter_size[2]))
            num_multiplications = np.product(filter_size) * n_pos * nof_filters * layer_input_shape[1 if ndim==2 else 2] * layer_input_shape[0]

            #print "Cost for passing to the next layer: 10^(",np.log(num_multiplications)/np.log(10),")    =",num_multiplications
            self.TotalForwardPassCost += num_multiplications

        if share_params_with_this_layer!=None:
            W = share_params_with_this_layer.W
            b = share_params_with_this_layer.b
        else:
            W=None
            b=None

        if ndim == 2:
            PLayer = ConvPoolLayer( input = layer_input,
                        input_shape = layer_input_shape, bUpsizingLayer = bUpsizingLayer,
                        filter_shape = (nof_filters, layer_input_shape[1], filter_size[0], filter_size[1]) if b_ReverseConvolution==False else (layer_input_shape[1], nof_filters, filter_size[0], filter_size[1]),
                        poolsize = pooling_factor, bDropoutEnabled_= (self.bDropoutEnabled and b_forceNoDropout==False),
                        ActivationFunction = ActivationFunction,
                        input_layer = self.layers[-1] if len(self.layers)>0 else None,
                        bTheanoConv=bTheanoConv, convolution_stride = convolution_stride,
                        b_ReverseConvolution=b_ReverseConvolution, W=W,b=b,
                        use_fragment_pooling = use_fragment_pooling, dense_output_from_fragments = dense_output_from_fragments,
                        output_stride = self.output_stride, input_axes=input_axes, output_axes=output_axes)

        if ndim== 3:
            PLayer = ConvPoolLayer3D( input = layer_input,
                        input_shape = layer_input_shape, bUpsizingLayer = bUpsizingLayer,
                        filter_shape = (nof_filters, filter_size[0], layer_input_shape[2], 
                                        filter_size[1], filter_size[2]), poolsize = pooling_factor, 
                                        bDropoutEnabled_= (self.bDropoutEnabled and b_forceNoDropout==False), 
                                        ActivationFunction = ActivationFunction, 
                                        input_layer = self.layers[-1] if len(self.layers)>0 else None, W=W,b=b, 
                                        use_fragment_pooling = use_fragment_pooling, 
                                        dense_output_from_fragments = dense_output_from_fragments, 
                                        output_stride = self.output_stride, verbose = self.verbose)



        if bAddInOutputLayer==False:
            self.layers.append(PLayer)
        else:
            self.output_layers.append(PLayer)
        return 0


    def add_layer(self, layer, **kwargs):
        """ Calls the init function of the new layer object with the following arguments:

        input: output of last layer in this NN/CNN
        input_shape: tuple
        **kwargs: obvious...


        The layer objust must provide:
        params (list)
        output (theano variable)
        output_shape (tuple or list)
        """
        layer_input_shape = self.input_shape if self.layers==[] else self.layers[-1].output_shape
        layer_input       = (self.layer0_input if len(self.layers)==0 else self.layers[-1].output)

        self.layers.append(layer(input = layer_input, input_shape = layer_input_shape, **kwargs))

    #adds one standard perceptron layer
    def addPerceptronLayer(self, n_outputs, bAddInOutputLayer=False, b_forceNoDropout=False, ActivationFunction = 'tanh', bInputNoise=False, b_experimental_inhibition_groups=False,flatW=False, W=None):
        """ Set b_forceNoDropout=True if this shall be the output layer.
            :n_outputs: may be either an integer or a tuple/list, in which case the output will be reshaped into this shape; should be: (batch, channels, x, y) or (batch, x, channels, y, z) """

        layer_input_shape = self.input_shape if self.layers==[] else self.layers[-1].output_shape
        layer_input       = (self.layer0_input if len(self.layers)==0 else self.layers[-1].output)


        if len(layer_input_shape)==4:
            layer_input = layer_input.flatten(2)
            nin = (layer_input_shape[0], np.product(layer_input_shape[1:]))
        elif len(layer_input_shape) == 2:
            nin = layer_input_shape
        else:
            print "ERROR! "*100

        #cost including batch size
        if self.batchsize != None:
            if np.all(np.prod(n_outputs)==n_outputs):
                num_multiplications =  n_outputs * np.product(nin)
            else:
                num_multiplications =  np.prod(n_outputs[1:]) * np.product(nin)
            if self.verbose:
                print "Cost for passing to the next layer: 10^(",np.log(num_multiplications)/np.log(10),")    =",num_multiplications
            self.TotalForwardPassCost += num_multiplications

        if bInputNoise:
            InputNoise=theano.shared(np.float32(0.2))
        else:
            InputNoise=None
        PLayer = PerceptronLayer(input = layer_input, n_in = nin[1], n_out = n_outputs, batchsize = nin[0], bDropoutEnabled_= (self.bDropoutEnabled and b_forceNoDropout==False) , ActivationFunction = ActivationFunction, InputNoise=InputNoise, input_layer = self.layers[-1] if len(self.layers)>0 else None, b_experimental_inhibition_groups=b_experimental_inhibition_groups, flatW=flatW, W=W)

        if bAddInOutputLayer==False:
            self.layers.append(PLayer)
        else:
            self.output_layers.append(PLayer)
        return 0




    def CompileDebugFunctions(self, gradients=False):
        """ output is 3D for 2D-CNN nets ! (#filters used by layer, x , y )
            the debug_functions return the network activations / output"""
        if len(self.debug_functions)!=0:
            return 1

        for i,lay in enumerate(self.layers):

            self.debug_functions.append( theano.function([self.x], lay.output) )
            try:
                self.debug_functions_conv_output.append( theano.function([self.x], lay.conv_output,on_unused_input='ignore') )
            except:
                pass

        if gradients:
            output_layer_Gradients = T.grad(self.output_layer_Loss, self.params, disconnected_inputs="warn")
            self.debug_gradients_function  = theano.function([self.x,self.y], [x for x in output_layer_Gradients], on_unused_input='warn')

        return 0





    def CompileOutputFunctions(self, b_isRegression=False, ndim_regression=2, b_ignore_pred_funct = False,
                               num_classes = None, batchsize=None, bUseModulatedNLL=False, 
                               margin_reweighted_error=False, b_regression_with_margin=False, 
                               ignore_points_labeled_as_zero=False, compile_test_model_function=False, 
                               override_training_loss_function=None, override_training_loss=None):
        """
        Composes loss and compiles theano output functions



        """
        if len(self.output_layers)!=0:
            print "WARNING! This function only applies to the LAST layer (and ignores elements of self.output_layers)"

        self.b_isRegression=b_isRegression

        if margin_reweighted_error:
            print "margin_reweighted_error = True: this is currently only supported for Conv_3D. Be aware of this, in case of failure."# but it is easy to add it to the other layers too

        if b_isRegression:
            print "CompileOutputFunctions::isRegression = True"
            self.y = T.TensorType('float32',(False,)*ndim_regression,name='y_cnn_regression_targets')('y_cnn_regression_targets')



        if self.batchsize != None:
            print "TotalForwardPassCost = 10^(",np.log(self.TotalForwardPassCost)/np.log(10),") =",self.TotalForwardPassCost
        layer_last = self.layers[-1]

        if override_training_loss_function is None and override_training_loss is None:
            if self.b_isRegression==False:
                self.output_layer_Loss = layer_last.negative_log_likelihood(self.y)
            elif b_regression_with_margin==False:
                self.output_layer_Loss = layer_last.squared_distance(self.y)
            else:
                self.output_layer_Loss = layer_last.squared_distance_w_margin(self.y, margin=0.3)

            if self.b_isRegression==False and margin_reweighted_error:
                self.output_layer_Loss = layer_last.negative_log_likelihood_modulated_margin(self.y) #std modulation = 1 (==None)
            if self.b_isRegression==False and ignore_points_labeled_as_zero:
                assert bUseModulatedNLL==0
                assert margin_reweighted_error==0
                print "CompileOutputFunctions::ignore_points_labeled_as_zero==True"
                print "NLL will not count points that have label 0. Needs an additional place-holder class 0, that will never actually be predicted."
                self.output_layer_Loss = layer_last.negative_log_likelihood_ignore_zero(self.y)
            if bUseModulatedNLL:
                print "CompileOutputFunctions::UseModulatedNLL = True"
                if b_isRegression:
                    for i in range(5):
                        print "NN::CompileOutputFunctions:: WARNING: b_isRegression and bUseModulatedNLL are incompatible! NLL=0, no training."
                    self.output_layer_Loss = 0
                    self.z = T.TensorType('float32',(False,)*ndim_regression,name='z_cnn_nll_modulation')('z_cnn_nll_modulation')#T.fvector('z_cnn_nll_modulation')
                else:
                    print "NN:: using modulated NLL. Modulation(np.float32_vector) should be 0 for to-be-ignored examples and (closer to) 1 otherwise."
                    self.z = T.fvector('z_cnn_nll_modulation')
                    if margin_reweighted_error:
                        self.output_layer_Loss = layer_last.negative_log_likelihood_modulated_margin(self.y, modulation = self.z)#margin=0.7, penalty_multiplier = 0.2
                    else:
                        self.output_layer_Loss = layer_last.negative_log_likelihood_modulated(self.y, self.z)

        elif override_training_loss_function is not None:
            print "using custom loss function..."
            self.output_layer_Loss = override_training_loss_function(self.y)

        elif override_training_loss is not None:
            print "using custom loss..."
            self.output_layer_Loss = override_training_loss

        self.bUseModulatedNLL = bUseModulatedNLL




        if compile_test_model_function:
            self.test_model = theano.function([self.x,self.y], layer_last.errors(self.y) if self.b_isRegression==False else layer_last.squared_distance(self.y))

        if b_ignore_pred_funct==False:

            self.class_probabilities = theano.function([self.x], layer_last.class_probabilities ) #use last.output_softmax for all of them.

        self.params=[]
        for lay in self.layers:
            if len(lay.params):
                self.params.append(lay.params[0])
            try:
                self.params.append(lay.params[1])
            except:
                print "Warning: layer.params[1] is empty."


        # in case one uses more than the params W and b
        for lay in self.layers:
            if len(lay.params)>2:
                self.params.append(lay.params[2:])


        return 0





    def enable_gradient_clipping(self, f_clip_at = 5e-3):
        self._GradientClipping = True
        self._GradientClipping_limit = f_clip_at





    def CompileTrainingFunctions(self, RPROP_penalty=0.35, RPORP_gain=0.2, SGD_LR_=5e-5,
                                 SGD_momentum_=0.9, b_Override_only_SGD=False, bOverride_OnlyGPROP=False,
                                 bOverride_OnlyRPORP=False, b_Override_only_RMSPROP=False, bWeightDecay=False,
                                 bHighActivationPenalty=False, b_layerwise_LR= False, b_external_top_error=False,
                                 b_use_clipped_gradients = False, f_clip_at = 5e-3):
        """ creates the functions for the last layer of <self.layers>
            trains all parameters included in <self.params>, i.e. ignoring the layer structure

            rmsprop and sgd share <last_grads>, so switching between them may behave a bit strangely

            """
        
        print "Called: CompileTrainingFunctions. You don't have to call this function, you may use .training_step() directly!"
        if len(self.params)==0:
            print "call CompileOutputFunctions() before calling CompileTrainingFunctions()!"
            return -1

        # create a list of gradients for all model parameters

        if b_external_top_error==False:
            if b_use_clipped_gradients==False:
                output_layer_Gradients = T.grad( self.output_layer_Loss, self.params, disconnected_inputs="warn")


            else:
                print "\nBE WARNED: Feature activated: use_clipped_gradients (f_clip_at =",f_clip_at,")"
                output_layer_Gradients_tmp = T.jacobian( self.layers[-1].negative_log_likelihood_array(self.y), self.params, disconnected_inputs="warn")
                #each element has shape: (batchsize, rest...)
                output_layer_Gradients = [T.mean(T.clip(x,-np.float32(np.abs(f_clip_at)),np.float32(np.abs(f_clip_at))),axis=0) for x in output_layer_Gradients_tmp]

        else:
            self.known_top_err = T.TensorType('float32',(False,)*5,name='known_top_err')('known_top_err')
            print "predictions are last_layer.output, which is (hopefully) sigmoid!"
            print "top error is specified externally: <self.known_top_err> (batchsize,x,n_classes,y,z)"
            output_layer_Gradients = theano.gradient.grad( T.sum(self.layers[-1].output*self.known_top_err) , self.params ,disconnected_inputs="warn")#.subgraph_grad()


        if b_Override_only_SGD==False:
            self.RPROP_LRs=[] # one for each parameter -> many
        self.last_grads=[]
        self.gprop_grad_variance=[]



        for i,para in enumerate(self.params):
            if para in self.params[:i]:
                print "Detected RNN or shared param @index =",i
                continue
            if b_Override_only_SGD==False:
#                print "warning: was 4e-5"
                self.RPROP_LRs.append(theano.shared(  1e-4*np.ones(para.get_value().shape,dtype=theano.config.floatX) , name=para.name+str('_RPORP') , borrow=0))
                self.gprop_grad_variance.append(theano.shared( 1e-2*np.ones(para.get_value().shape,dtype=theano.config.floatX) , name=para.name+str('_GPROP') , borrow=0))
#            print "WARNING change this if you want to use sgd/rmsprop"
            self.last_grads.append(theano.shared( np.zeros(para.get_value().shape,dtype=theano.config.floatX) , name=para.name+str('_LG') , borrow=0))
            #self.SGD_EigHessian_perturbed_grads.append(theano.shared(  zeros(para.get_value().shape,dtype=theano.config.floatX) , name=para.name+str('_pLG') , borrow=True))

        n = len(self.last_grads)
        for i,lay in enumerate(self.layers):
            low = (i*2)%n
            lay.last_grads = self.last_grads[low:low+2]




        SGD_updatesa=[]
        SGD_updatesb=[]

        if b_Override_only_SGD==False:
            RPROP_updates = []
        RMSPROP_updates = []



        self.SGD_global_LR.set_value(np.float32(SGD_LR_))
        if bWeightDecay:
            print "CNN::using Weight decay! Change via this.SGD_global_weightdecay.set_value()"
            self.SGD_global_weightdecay = theano.shared(np.asarray(0.0005).astype("float32"))
        self.SGD_momentum.set_value(np.float32(SGD_momentum_))


        if b_Override_only_SGD==False:
            assert len(self.params)==len(self.last_grads),"rnn/shared params not yet implemented in rprop/gprop"


#            print "Trading memory usage for more speed (SGD_updates_a), change it if it gets too big (removes momentum, too)."
            for param_i, grad_i, last_grad_i, pLR_i, gprop_var_i in zip(self.params, output_layer_Gradients, self.last_grads, self.RPROP_LRs, self.gprop_grad_variance):
                # capping RPROP-LR inside [1e-7,1e-2]
                print "RPROP: missing backtracking handling "
                RPROP_updates.append((pLR_i, T.minimum( T.maximum( pLR_i * ( 1 - np.float32(RPROP_penalty)* ((last_grad_i*grad_i) < -1e-9) + np.float32(RPORP_gain)* ((last_grad_i*grad_i) > 1e-11)   ) , 1e-7*T.ones_like(pLR_i) ),2e-3 * T.ones_like(pLR_i)) ))
                RPROP_updates.append((param_i, param_i  - pLR_i * grad_i/(T.abs_(grad_i) + 1e-6) - (0 if bWeightDecay==False else self.SGD_global_weightdecay*param_i) ))

                RPROP_updates.append((last_grad_i, grad_i ))#RPROP_updates.append((last_grad_i, (grad_i + 0.5*last_grad_i)/1.5)) #trailing exp-mean over last gradients: smoothing. check if useful...


        if b_layerwise_LR:
            print "Using layerwise LR multiplier. Speed penalty ~ 10%. Access it via this.SGD_local_LRs (default is 1. == no modification of the global LR)."
            self.SGD_local_LRs = [theano.shared(np.float32(1.)) for x in self.params] #one LR modifier per param group
        else:
            self.SGD_local_LRs = [1. for x in self.params]


        for param_i, grad_i, last_grad_i, local_lr_modifier in zip(self.params, output_layer_Gradients, self.last_grads, self.SGD_local_LRs):
            if len(self.params)>len(self.last_grads):
                grad_i = None
                print "grad_param::",param_i
                for i in range(len(self.params)):
                    if param_i == self.params[i]:
                        print ">>",i
                        grad_i = output_layer_Gradients[i] if grad_i==None else grad_i + output_layer_Gradients[i]

            SGD_updatesa.append((last_grad_i, grad_i             + last_grad_i * self.SGD_momentum))#use this if you want to use the gradient magnitude

        for i, param_i, grad_i, last_grad_i, local_lr_modifier in zip(range(len(self.params)), self.params, output_layer_Gradients, self.last_grads, self.SGD_local_LRs):
            if bWeightDecay and (i < len(self.params)-2): #no WeightDecay in last layer
                SGD_updatesb.append((param_i, param_i  - (self.SGD_global_LR * local_lr_modifier) * last_grad_i - self.SGD_global_LR *self.SGD_global_weightdecay*param_i   ))
            else:
                SGD_updatesb.append((param_i, param_i  - (self.SGD_global_LR * local_lr_modifier) * last_grad_i   ))

            RMSPROP_updates.append((last_grad_i, 0.95*last_grad_i + 0.05* (grad_i)**2  ))
            RMSPROP_updates.append((param_i, param_i - self.SGD_global_LR * grad_i/(  T.sqrt(last_grad_i+0.000001) ) ))
        print "RMSPROP: advice: a good LR is 2e-4  (value for <self.SGD_global_LR>)"



        if bHighActivationPenalty:
            self.HighActivationPenalty_coeff = theano.shared(np.float32(1e-4))
            print "Applying high-activation-penalty..."
            print "todo: test..."
            for lay in self.layers:
                type_ = lay.ActivationFunction
                ok=1

                if type_=="tanh":
                    grads = T.grad( T.mean((lay.output)**2), lay.params)
                elif type_=="sigmoid":
                    grads = T.grad( 2*T.mean((lay.output-0.5)**2), lay.params)
                elif type_=="relu":
                    print "relu...todo:test"
                    grads = T.grad( -T.mean((lay.output)**2), lay.params)
                else:
                    print "UNSUPPORTED ActivationFunction!"
                    ok=0

                if ok:

                    for param_i,grad_i in zip(lay.params,grads):

                        for i,u in enumerate(SGD_updatesb):
                            if u[0]==param_i:
                                SGD_updatesb[i] = (param_i,u[1] - (self.SGD_global_LR * self.HighActivationPenalty_coeff) * grad_i)
                                break
                        try:
                            for i,u in enumerate(RMSPROP_updates):
                                if u[0]==param_i:
                                    RMSPROP_updates[i] = (param_i,u[1] - (self.SGD_global_LR * self.HighActivationPenalty_coeff) * grad_i)
                                    break
                            for i,u in enumerate(RPROP_updates):
                                if u[0]==param_i:
                                    RPROP_updates[i] = (param_i,u[1] - (self.SGD_global_LR * self.HighActivationPenalty_coeff) * grad_i)
                                    break
                        except:
                            print "only sgd..."


        addthis = [self.z,] if self.bUseModulatedNLL else []

        if b_external_top_error:
            addthis = addthis + [self.known_top_err]

        if bOverride_OnlyRPORP or (b_Override_only_SGD==False and bOverride_OnlyGPROP==False and b_Override_only_RMSPROP==0):
            print "compiling RPROP..."
            self.train_model_RPROP = theano.function([self.x] + ([] if b_external_top_error else [self.y])+addthis, None if b_external_top_error else self.output_layer_Loss, updates=RPROP_updates,  on_unused_input='warn')

        if b_Override_only_SGD==False and bOverride_OnlyGPROP==False and bOverride_OnlyRPORP==False:
            print "compiling RMSPROP..."
            self.train_model_RMSPROP = theano.function([self.x] + ([] if b_external_top_error else [self.y])+addthis, None if b_external_top_error else self.output_layer_Loss, updates=RMSPROP_updates,  on_unused_input='warn')

        if bOverride_OnlyGPROP==0 and b_Override_only_RMSPROP==0 and bOverride_OnlyRPORP==False:
            print "compiling SGD..."
            # a only updates last_grads, it DOES NOT change any parameters
            #you could call it 10 times and would get the same nll every time... but if momentum is != 0 then this changes the search direction

            assert len(SGD_updatesa)==len(SGD_updatesb),str(len(SGD_updatesa))+" != "+str(len(SGD_updatesb))

            self.train_model_SGD_a     = theano.function([self.x] + ([] if b_external_top_error else [self.y])+addthis, None if b_external_top_error else self.output_layer_Loss, updates=SGD_updatesa,  on_unused_input='warn')#the output is the value you get BEFORE updates....
            
            try:
                self.train_model_SGD_a_ext = theano.function([self.x,self.y]+addthis, [self.output_layer_Loss, self.layers[-1].class_probabilities_realshape], updates=SGD_updatesa,  on_unused_input='warn')
            except:
                print "NNet.train_model_SGD_a_ext unavailable"
            # b ONLY changes the parameters
            self.train_model_SGD_b     = theano.function([], None, updates=SGD_updatesb)

        return 0




    def get_NLL(self,x,y):
        self.get_NLL = theano.function([self.x,self.y], self.output_layer_Loss, on_unused_input='warn')
        return self.get_NLL(x,y)









    def training_step(self, data, labels, mode=1, useSGD=None, modulation=None, b_extended_output = False):
        """
        execute one step of online (mini)-batch learning

        <mode> in [0,1,2,3,4,5,6]:
            0:
            use RPROP, which does neither uses the global learning rate nor the momentum-value.
            It is better than SGD if you do full-batch training and use NO dropout.
            Any source of noise leads to failure to converge (at all).

            1:
            use SGD. Good if data set is big and redundant.

            2:
            use RMSPROP. Uses learning rate of SGD but you should fix it to 5e-4. no momentum. works on minimatches.
            -> seems better than SGD

            3:
            GPROP (experimental)

            4:
            Conjugate Gradient

            5:
            ARP (experimental, unfinished)

            6:
            ADADELTA

        <useSGD> is a bool, only for backwards compatibility! (IGNORE IT!!!)

        modulation is only in effect, if it was added in this->CompileOutputFunctions()
        then it must have the shape&type of labels, and be 1 for IGNORED examples (0 is the "default" case)
        Only added it for sgd at the moment...

        returns: current NLL (or MSE if CompileTrainingFunctions() was changed to regression)
        """

        if type(mode)==type(""):
            modes = ["rprop","sgd","rmsprop","gprop","cg","arp","adadelta"]
            mode = [i for i in range(len(modes)) if modes[i]==mode][0]


        if useSGD!=None:
            mode=int(useSGD)

        if mode==1:
#            t0=time.clock()
            if hasattr(self,"train_model_SGD_a")== False or self.train_model_SGD_a==None:
                self.CompileSGD(self, bWeightDecay=self.bWeightDecay)
            trainf = self.train_model_SGD_a_ext if b_extended_output else self.train_model_SGD_a
            if self.bUseModulatedNLL:
                nll = trainf(data,labels,modulation)
            else:
                nll = trainf(data,labels)
            self.train_model_SGD_b()

        elif mode==0:
            if hasattr(self,"train_model_RPROP")== False or self.train_model_RPROP==None:
                self.CompileRPROP(self)
            nll=self.train_model_RPROP(data,labels)
        elif mode==3:
            nll=self.train_model_GPROP(data,labels)
        elif mode==2:
            nll=self.train_model_RMSPROP(data,labels)
        elif mode==4:
            if hasattr(self,"trainingStepCG_g")== False or self.trainingStepCG_g==None:
                self.compileCG()

            if modulation==None:
                para=(data, labels)
            else:
                para=(data, labels, modulation)
            nll, r1,r2,r3 = self.trainingStepCG_g(self, para, n_steps=10)#n_steps: n.of steps per batch
        elif mode==5:
            if hasattr(self,"train_model_ARP")== False or self.train_model_ARP==None:
                self.CompileARP(self)
            nll=self.train_model_ARP(data,labels)

        elif mode==6:
            if hasattr(self,"train_model_ADADELTA")== False or self.train_model_ADADELTA==None:
                self.CompileADADELTA(self)
            nll=self.train_model_ADADELTA(data,labels)

        else:
            print "INVALID mode!"
            nll=1e42

        return nll



    def set_SGD_LR(self,value=7e-4):
        self.SGD_global_LR.set_value(np.float32(value),borrow=False)

    def get_SGD_LR(self):
        return self.SGD_global_LR.get_value()

    def set_SGD_Momentum(self,value=0.9):
        self.SGD_momentum.set_value(np.float32(value),borrow=False)



    #---------------------------------------------------------------------------------------------------
    #---------------------------------------save / load-------------------------------------------------
    #---------------------------------------------------------------------------------------------------

    def save(self, myfile="CNN.save", layers_to_save = None, protocol=1):
        """ alias to self.SaveParameters"""
        return self.SaveParameters(myfile=myfile, layers_to_save = layers_to_save, protocol=protocol)

    def load(self,myfile="CNN.save", n_layers_to_load=-1):
        """ alias to self._LoadParametersAdaptive"""
        return self._LoadParametersAdaptive(myfile,n_layers_to_load=n_layers_to_load)

    def LoadParameters(self,myfile="CNN.save", n_layers_to_load=-1):
        return self.LoadParametersStrict(myfile,n_layers_to_load=n_layers_to_load)




    #function lacks error-handling
    def LoadParametersStrict(self,myfile="CNN.save", n_layers_to_load = -1):
        """load a parameter set which IS fully compatible to the current network configuration (FAILS otherwise)."""
        #print "Loading from",myfile
        f = open(myfile, 'rb')
        assert n_layers_to_load==-1,'not implemented'
        nn_params = []
        for layer in self.layers:
            nn_params += layer.params
        
        file_params = []
        while 1:
            try:
                p = cPickle.load(f)
                if isinstance(p,list):
                    file_params.extend(p)
                else:
                    file_params.append(p)
            except:
                break
        
        file_params = [x for x in file_params if np.prod(np.shape(x))>6]
        #print map(np.shape, file_params)
        #print [x.get_value().shape for x in nn_params]

        for nn_p, p in zip(nn_params, file_params):
            nn_p.set_value(p,borrow=False)

        f.close()
        return 0






    def SaveParameters(self, myfile="CNN.save", layers_to_save = None, protocol=1):
        """ default: layers_to_save = None will save all self.layers"""
        print "saving to",myfile
#        print "TODO: this save-code sucks! (NN_ConvNet, line 1714)"
        mkdir(extract_filename(myfile)[0])
        f = open(myfile, 'wb')

        if layers_to_save==None:
            layers_to_save = self.layers

        shape_nfo = []
        for layer in layers_to_save:#self.layers:
            if len(layer.params):
                shape_nfo.append(layer.params[0].get_value(borrow=True).shape)

        cPickle.dump(shape_nfo, f, protocol = protocol)

        for layer in layers_to_save:#self.layers:  #every layer.params is a list containing two references to theano-shared variables
            if len(layer.params)>0:
                cPickle.dump(layer.params[0].get_value(borrow=True), f, protocol = protocol)
            if len(layer.params)>1:
                cPickle.dump(layer.params[1].get_value(borrow=True), f, protocol = protocol)
        f.close()
        return 0


#---------------------------------------------------------------------------------------------
#-------------------------------------end class MixedConvNN-----------------------------------------------
#---------------------------------------------------------------------------------------------
















