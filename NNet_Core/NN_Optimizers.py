# -*- coding: utf-8 -*-
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


# Contains all optimizers like SGD/CG/RPROP/...

import numpy as np
import theano
import theano.tensor as T
#import time
#from matplotlib import pyplot as plot
#import scipy

#####################################################################################################################
###################################      OptimizerCore Base Object          #########################################
#####################################################################################################################

class OptimizerCore(object):
  def __init__(self, base_object=None, input_arguments = [], loss=None, params=None, use_shared_getter_functions = False):
    """
    Optimizer Base Object, initialialises generic variables needed for loss/gradient based optimization.
    Also provides functions for optimization fully contained in python/numpy (e.g. get_grads, set_params)

    Parameters
    ----------

    base_object: 
        a theano based class (instead of provinding input_arguments, loss, params manually), all other arguments are
        retrieved from this object if they are ``None``. If an argument is not ``None`` it will override the value from the model
        
    input_arguments: 
        list of all symbolic input variables (which are required to compute the loss & gradients)
   
    loss:  
        symbolic loss function: requires <input_arguments> for compilation
    params:    
        list of theano-parameters which will be the target of the optimization
        
    use_shared_getter_functions:
        if True: avoids multiple compilations of get_loss/get_grads, but this is a HORRIBLY BAD IDEA if the same
        model is optimized with different loss functions!

    """
    if base_object is None:
        assert len(input_arguments)>0 and loss is not None and params is not None, "Missing argument(s)!"
        
    else:
      self.base_object = base_object
      if input_arguments is None or len(input_arguments)==0:
          try:
             self.input_arguments = [base_object.x, base_object.y]
          except:
              assert hasattr(base_object,"input_arguments"),"ERROR::Optimizer.__init__:Please specify <input_arguments>"
          if hasattr(base_object,"input_arguments"):
              self.input_arguments = base_object.input_arguments
      else:
          self.input_arguments = input_arguments
      if params==None:
          params = base_object.params
      if loss==None:
          loss = base_object.output_layer_Loss

    assert len(params)>0,"no params, call compileOutputFunctions() before calling OptimizerCore.__init__()!"
    assert loss is not None

    self.loss = loss
    self.params = params
    self.gradients = T.grad(loss, params, disconnected_inputs="warn")
    
#    if hasattr(base_object, 'last_grads') and base_object.last_grads!=[] and  base_object.last_grads!=None:
#      self.last_grads = base_object.last_grads
#    else:
#    self.last_grads = []
#    for i, p in enumerate(self.params):
#        if p in self.params[:i]:
#            print "Detected shared param: param[%i]" %i
#        else:
#        self.last_grads.append(theano.shared(np.zeros(p.get_value().shape, dtype=theano.config.floatX),name=p.name+str('_LG'), borrow=0))


    if base_object is not None:
        # avoid multiple/redundant compilations
        #but this is a HORRIBLY BAD IDEA if the same model is optimized with different loss functions!
        if use_shared_getter_functions:
            if hasattr(base_object, 'get_loss'):
                print "OptimizerCore:: using shared get_loss() function"
                self.get_loss = base_object.get_loss
            else:
                base_object.get_loss = self.get_loss
                
            if hasattr(base_object, 'get_grads'):
                print "OptimizerCore:: using shared get_grads() function"
                self.get_grads = base_object.get_grads
            else:
                base_object.get_grads = self.get_grads
        
#            base_object.last_grads = self.last_grads # share last grads in model
#            base_object.get_loss    = self.get_loss



  def get_loss(self, *args):
    """
    get loss
    
    input: numpy-arrays: [data, labels, ...] 
    output: loss
    """
    print "compiling get_loss ..."
    get_loss = theano.function(self.input_arguments, self.loss)
    loss = get_loss(*args)
    self.get_loss = get_loss #overwrite
    return np.float32(loss)


  def get_grads(self,*args):
    """
    returns a list of gradients, compiles on first call
    
    input:
        numpy-arrays: [data, labels, ...] 
    """
    print "compiling get_grads ..."
    getGradients = theano.function(self.input_arguments, self.gradients, on_unused_input='warn')
    ret = getGradients(*args)
    self.get_grads = getGradients
    return ret


  def get_grads_as_vector(self,*args):
    """
    returns a single vector containing the gradients, compiles self.get_grads() on first call
    
    perfomance penalty!
    
    input: 
        numpy-arrays: [data, labels, ...] 
    """
    grads = self.get_grads(*args)
    self._vectorized_gradient = self._list_to_vector(grads, self._vectorized_gradient if hasattr(self, "_vectorized_gradient") else None)
    return self._vectorized_gradient


  def set_params(self, new_params):
    """
    obvious...
    
    input: 
        list of numpy-arrays: list of new parameter values
    """
    for new, p in zip(self.params, new_params):
        p.set_value(new)
    return 0


  def set_params_by_vector(self, vectorized_new_params):
    """
    obvious...
    
    perfomance penalty!
    
    input: 
        numpy-array: new parameter values as flat vector/array
    """
    if hasattr(self,"_list_numpy_params")==0:
        self._list_numpy_params = [p.get_value() for p in self.params]
    self._vector_to_list(vectorized_new_params, target_list = self._list_numpy_params)
    for new, p in zip(self.params, self._list_numpy_params):
        p.set_value(new)
    return 0
    

  def _vector_to_list(self, input_vector, target_list = None):
    """ internal use, modifies elements of <target_list> if provided, creates a new list if not (uses shapes of <self.params>)."""
    if target_list is None:
        target_list = [p.get_value() for p in self.params]
    i=0
    for p in target_list:
        j=np.prod(p.shape)
        p[...] = input_vector[i:i+j].reshape(p.shape)
        i+=j
    return target_list


  def _list_to_vector(self, input_list, target_vector = None):
    """ internal use, modifies <target_vector> if provided, creates a new vector if not."""
    if target_vector is None:
        params_total_size = np.sum([np.prod(p.shape) for p in input_list])
        target_vector = np.zeros(params_total_size,"float32")
    i=0
    for p in input_list:
        j=np.prod(np.shape(p))
        target_vector[i:i+j] = np.asarray(p).flatten()
        i+=j
    return target_vector













#####################################################################################################################
#######################################       Resilient backPROPagation      ########################################
#####################################################################################################################


def CompileRPROP(base_object=None, input_arguments=[], loss = None, params = None, 
                 RPROP_penalty=0.35, RPROP_gain=0.2, bWeightDecay=False, initial_update_size = 1e-4, 
                 modifies_base_object=True):
    """ default for top_error is <base_object.output_layer_Loss>
        default for    params is <base_object.params>
        total factor on sign-agreement is: 1+<RPROP_gain>.
        total factor on disagreement   is: 1-<RPROP_penalty>
        
        Change-notes:
        removed INPUT, TARGET,
        replaced with input_arguments
        
        Returns
        -------
        OptimizerCore object, train the model by calling <core.train_model_RPROP(...)>
        
        
        """
    print "setting up RPROP..."
    
    core = OptimizerCore(base_object=base_object, input_arguments=input_arguments, 
                         loss=loss, params=params, use_shared_getter_functions=False)
    
    RPROP_LRs=[]
    last_grads=[]
    RPROP_updates=[]
    if bWeightDecay:
        print "CNN::using Weight decay! Change value via this.global_weightdecay_param.set_value()"
        global_weightdecay_param = theano.shared(np.asarray(0.0005).astype("float32"))

    for i,para in enumerate(core.params):
        if para in core.params[:i]:
            print "Detected RNN or shared param @index =",i
        else:
            RPROP_LRs.append(theano.shared(  np.float32(initial_update_size)*np.ones(para.get_value().shape,dtype=theano.config.floatX) , name=para.name+str('_RPROP') , borrow=0))
            last_grads.append(theano.shared( np.zeros(para.get_value().shape,dtype=theano.config.floatX) , name=para.name+str('_LG') , borrow=0))
    print "RPROP: missing backtracking handling "
    for param_i, grad_i, last_grad_i, pLR_i in zip(core.params, core.gradients, last_grads, RPROP_LRs):
        # capping RPROP-LR inside [1e-7,1e-2]

        RPROP_updates.append((pLR_i, T.minimum( T.maximum( pLR_i * ( 1 - np.float32(RPROP_penalty)* ((last_grad_i*grad_i) < -1e-9) + np.float32(RPROP_gain)* ((last_grad_i*grad_i) > 1e-11)   ) , 1e-7*T.ones_like(pLR_i) ),5e-3 * T.ones_like(pLR_i)) ))
        RPROP_updates.append((param_i, param_i  - pLR_i * grad_i/(T.abs_(grad_i) + 1e-6) - (0 if bWeightDecay==False else global_weightdecay_param*param_i) )) #grad_i/(T.abs_(grad_i) + 1e-6)  #T.sgn(grad_i)
        RPROP_updates.append((last_grad_i, grad_i ))#RPROP_updates.append((last_grad_i, (grad_i + 0.5*last_grad_i)/1.5)) #trailing exp-mean over last gradients: smoothing. check if useful...


    def train_model_RPROP(*args):
        # this function will replace itself. It will still work (efficiently) if a pointer to it was saved and is used instead of the new function.
        if hasattr(core,"_train_model_RPROP_compiled"):
            return core._train_model_RPROP_compiled(*args)
        print "compiling RPROP..."
        fun = theano.function(core.input_arguments, core.loss, updates=RPROP_updates,  on_unused_input='warn')
        core.train_model_RPROP = fun
        core._train_model_RPROP_compiled = fun
        return fun(*args)

    core.train_model_RPROP = train_model_RPROP #theano.function(core.input_arguments, core.loss, updates=RPROP_updates,  on_unused_input='warn')

    if modifies_base_object and base_object is not None:
        print "RPROP...setting base_object's attributes & methods..."
        base_object.train_model_RPROP = core.train_model_RPROP
        if bWeightDecay:
            base_object.global_weightdecay_param = global_weightdecay_param
        base_object.last_grads = last_grads
        base_object.RPROP_LRs = RPROP_LRs

    return core






#####################################################################################################################
#######################################       Resilient backPROPagation      ########################################
#####################################################################################################################


def CompileADADELTA(base_object=None, INPUT=None, TARGET=None, top_error = None, params = None, 
                    UpdateLatency=0.95, epsilon = 0, scale_factor = 0.1, bWeightDecay=False, 
                    modifies_base_object=True):
    """ default for top_error is <base_object.output_layer_Loss>
        default for    params is <base_object.params>
        """
    print "compiling ADA-DELTA..."
    assert UpdateLatency>0. and UpdateLatency<1.
    
    if top_error==None:
        top_error = base_object.output_layer_Loss
    if params==None:
        params = base_object.params
    if INPUT==None:
        INPUT = base_object.x
    if TARGET==None:
        TARGET = base_object.y
    assert len(params)>0,"call CompileOutputFunctions() before calling CompileADADELTA()!"

    All_Gradients = T.grad( top_error, params, disconnected_inputs="warn")
    
    UpdateLatency = np.float32(UpdateLatency)
    epsilon = np.float32(epsilon)
    scale_factor = np.float32(scale_factor)
    
    
    ADADELTA_updates=[]
    trailing_sq_grads=[]
    trailing_sq_updates=[]
    
    if bWeightDecay:
        print "CNN::using Weight decay! Change value via this.global_weightdecay_param.set_value()"
        global_weightdecay_param = theano.shared(np.asarray(0.0005).astype("float32"))

    for i,para in enumerate(params):
        if para in params[:i]:
            print "Detected RNN or shared param @index =",i,"(skipping duplicate)"
        else:
#            ADADELTA_LRs.append(theano.shared(  np.float32(initial_update_size)*np.ones(para.get_value().shape,dtype=theano.config.floatX) , name=para.name+str('_ADADELTA') , borrow=0))
            trailing_sq_grads.append(theano.shared( np.float32(1e-2)*np.ones(para.get_value().shape,dtype=theano.config.floatX) , name=para.name+str('_sqG') , borrow=0))
            trailing_sq_updates.append(theano.shared( np.zeros(para.get_value().shape,dtype=theano.config.floatX) , name=para.name+str('_sqU') , borrow=0))

    for param_i, grad_i, sq_g, sq_u in zip(params, All_Gradients, trailing_sq_grads, trailing_sq_updates):
        # capping ADADELTA-LR inside [1e-7,1e-2]
        ADADELTA_updates.append((sq_g, UpdateLatency*sq_g + np.float32(1.-UpdateLatency)*grad_i**2 ))
        update = T.sqrt(sq_u + epsilon) / T.sqrt(sq_g + epsilon) * grad_i * scale_factor  #actually the neg. of the update
        ADADELTA_updates.append((sq_u, UpdateLatency*sq_u + np.float32(1.-UpdateLatency)*update**2 ))
        
        ADADELTA_updates.append((param_i, param_i  - update * scale_factor - (0 if bWeightDecay==False else global_weightdecay_param*param_i) )) #grad_i/(T.abs_(grad_i) + 1e-6)  #T.sgn(grad_i)
#        ADADELTA_updates.append((last_grad_i, grad_i ))

    train_model_ADADELTA = theano.function([INPUT,TARGET], top_error, updates=ADADELTA_updates,  on_unused_input='warn')

    if modifies_base_object and base_object is not None:
        print "ADADELTA...setting base_object's attributes & methods..."
        base_object.train_model_ADADELTA = train_model_ADADELTA
        if bWeightDecay:
            base_object.global_weightdecay_param = global_weightdecay_param


    print "compiling ADADELTA...DONE!"
    return train_model_ADADELTA






#####################################################################################################################
#######################################       Stochastic Gradient Descent     #######################################
#####################################################################################################################

def CompileSGD(base_object, top_loss = None, params = None, input_arguments = None, SGD_LR_=4e-5, 
               SGD_momentum_=0.9, bWeightDecay=False):
    """ default for top_loss is <base_object.output_layer_Loss>
        default for    params is <base_object.params>
        default for input_arguments is <[base_object.x, base_object.y]>
        
        Returns:
        ------------
            
            train_function(input,...,target,...)

        Added to base_object:
        -----------------------
            
            SGD_global_LR, SGD_momentum, global_weightdecay_param
        
        """
    print "NN:opt::compiling SGD..."
    if top_loss==None:
        top_loss = base_object.output_layer_Loss
    if params==None:
        try:
            params = base_object.params
        except:
            params = base_object._params
    if input_arguments==None:
        input_arguments = [base_object.x, base_object.y]
        if hasattr(base_object,"input_arguments"):
            input_arguments = base_object.input_arguments
            assert type(input_arguments)==type([])        
    assert len(params)>0,"call CompileOutputFunctions() before calling CompileRPROP()!"

    All_Gradients = T.grad( top_loss, params, disconnected_inputs="warn")
    if not hasattr(base_object,"SGD_global_LR"):
        base_object.SGD_global_LR = theano.shared(np.float32(SGD_LR_))
    if not hasattr(base_object,"SGD_momentum"):
        base_object.SGD_momentum = theano.shared(np.float32(SGD_momentum_))
#    base_object.SGD_global_LR.set_value(np.float32(SGD_LR_))
        
    use_GradientClipping = False
    clip_limits = [0 for i in range(len(All_Gradients))]
    if hasattr(base_object, "_GradientClipping"):
        if base_object._GradientClipping:
            print "CompileSGD:: GradientClipping ENABLED!"
            assert hasattr(base_object, "_GradientClipping_limit")
            clip_limits = [base_object._GradientClipping_limit] * len(All_Gradients)

            use_GradientClipping = True
        
        
        
        
    if bWeightDecay:
        print "CNN::using Weight decay! Change via this.SGD_global_weightdecay.set_value()"
        if not hasattr(base_object,"SGD_global_weightdecay"):
            base_object.SGD_global_weightdecay = theano.shared(np.asarray(0.0001).astype("float32"))
    base_object.SGD_momentum.set_value(np.float32(SGD_momentum_))
    base_object.last_grads=[]
    SGD_updatesa=[]
    SGD_updatesb=[]


    for i,para in enumerate(params):
        if para in params[:i]:
            print "Detected RNN or shared param @index =",i
        else:
            base_object.last_grads.append(theano.shared( np.zeros(para.get_value().shape,dtype=theano.config.floatX) , name=para.name+str('_LG') , borrow=0))


    for param_i, grad_i, velocity_i, clip_limit in zip(params, All_Gradients, base_object.last_grads, clip_limits):
        if use_GradientClipping:
        
            
            SGD_updatesa.append((velocity_i, - base_object.SGD_global_LR * T.sgn(grad_i)*T.minimum(T.abs_(grad_i), clip_limit*T.ones_like(grad_i))  + velocity_i * base_object.SGD_momentum)) #use this if you want to use the gradient magnitude
        else:
            SGD_updatesa.append((velocity_i, - base_object.SGD_global_LR * grad_i  + velocity_i * base_object.SGD_momentum)) #use this if you want to use the gradient magnitude

    for param_i,  velocity_i in zip(params,  base_object.last_grads):
        if bWeightDecay:
            SGD_updatesb.append((param_i, param_i  + base_object.SGD_momentum * velocity_i - base_object.SGD_global_LR *base_object.SGD_global_weightdecay*param_i   ))
        else:
            SGD_updatesb.append((param_i, param_i  + base_object.SGD_momentum * velocity_i   ))

    assert len(SGD_updatesa)==len(SGD_updatesb),str(len(SGD_updatesa))+" != "+str(len(SGD_updatesb))

    base_object.train_model_SGD_a     = theano.function(input_arguments, top_loss, updates=SGD_updatesa,  on_unused_input='warn')#the output is the value you get BEFORE updates....
    base_object.train_model_SGD_b     = theano.function([], None, updates=SGD_updatesb)# ONLY changes the parameters

    def local_f(*args):
        nll = base_object.train_model_SGD_a(*args)
        base_object.train_model_SGD_b()
        return nll
    base_object.train_model_SGD = local_f
    print "compiling SGD...DONE!"
    return base_object.train_model_SGD





def CompileSGD_legacy(base_object, top_error = None, params = None, input_arguments = None, SGD_LR_=4e-5, 
               SGD_momentum_=0.9, bWeightDecay=False):
    """ default for top_error is <base_object.output_layer_Loss>
        default for    params is <base_object.params>
        default for input_arguments is <[base_object.x, base_object.y]>
        """
    print "NN:opt::compiling SGD..."
    if top_error==None:
        top_error = base_object.output_layer_Loss
    if params==None:
        params = base_object.params
    if input_arguments==None:
        input_arguments = [base_object.x, base_object.y]
        if hasattr(base_object,"input_arguments"):
            input_arguments = base_object.input_arguments
            assert type(input_arguments)==type([])        
    assert len(params)>0,"call CompileOutputFunctions() before calling CompileRPROP()!"

    All_Gradients = T.grad( top_error, params, disconnected_inputs="warn")
    if not hasattr(base_object,"SGD_global_LR"):
        base_object.SGD_global_LR = theano.shared(np.float32(SGD_LR_))
    if not hasattr(base_object,"SGD_momentum"):
        base_object.SGD_momentum = theano.shared(np.float32(SGD_momentum_))
#    base_object.SGD_global_LR.set_value(np.float32(SGD_LR_))
    if bWeightDecay:
        print "CNN::using Weight decay! Change via this.SGD_global_weightdecay.set_value()"
        if not hasattr(base_object,"SGD_global_weightdecay"):
            base_object.SGD_global_weightdecay = theano.shared(np.asarray(0.0005).astype("float32"))
    base_object.SGD_momentum.set_value(np.float32(SGD_momentum_))
    base_object.last_grads=[]
    SGD_updatesa=[]
    SGD_updatesb=[]

    if bWeightDecay:
        print "CNN::using Weight decay! Change value via this.global_weightdecay_param.set_value()"
        base_object.global_weightdecay_param = theano.shared(np.asarray(0.0005).astype("float32"))

    for i,para in enumerate(params):
        if para in params[:i]:
            print "Detected RNN or shared param @index =",i
        else:
            base_object.last_grads.append(theano.shared( np.zeros(para.get_value().shape,dtype=theano.config.floatX) , name=para.name+str('_LG') , borrow=0))


    for param_i, grad_i, last_grad_i in zip(base_object.params, All_Gradients, base_object.last_grads):

        SGD_updatesa.append((last_grad_i, grad_i             + last_grad_i * base_object.SGD_momentum))#use this if you want to use the gradient magnitude

    for param_i, grad_i, last_grad_i in zip(base_object.params, All_Gradients, base_object.last_grads):
        if bWeightDecay:
            SGD_updatesb.append((param_i, param_i  - (base_object.SGD_global_LR ) * last_grad_i - base_object.SGD_global_LR *base_object.SGD_global_weightdecay*param_i   ))
        else:
            SGD_updatesb.append((param_i, param_i  - (base_object.SGD_global_LR ) * last_grad_i   ))

    assert len(SGD_updatesa)==len(SGD_updatesb),str(len(SGD_updatesa))+" != "+str(len(SGD_updatesb))
#    args = [base_object.x,base_object.y]
#    if hasattr(base_object,"z"):
#        args+=[base_object.z]
    base_object.train_model_SGD_a     = theano.function(input_arguments, top_error, updates=SGD_updatesa,  on_unused_input='warn')#the output is the value you get BEFORE updates....
    base_object.train_model_SGD_b     = theano.function([], None, updates=SGD_updatesb)# ONLY changes the parameters
#            base_object.train_model_SGD_a_ext = theano.function([base_object.x,base_object.y]+addthis, [top_error, self.layers[-1].class_probabilities_realshape], updates=SGD_updatesa,  on_unused_input='warn')

    def local_f(*args):
        nll = base_object.train_model_SGD_a(*args)
        base_object.train_model_SGD_b()
        return nll
    base_object.train_model_SGD = local_f
    print "compiling SGD...DONE!"
    return base_object.train_model_SGD





