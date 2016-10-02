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
import time
from matplotlib import pyplot as plot
import scipy

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


#  def __call__(self, *args):
#    """
#    Perform an update step
#    
#    input: numpy-arrays: [data, labels, ...]
#    
#    Returns: float32 loss
#    """
#    raise NotImplementedError()
#    loss = self.step(*args)
#    return np.float32(loss)








#####################################################################################################################
###################################      Adaptive Resilient backPropagation      ####################################
#####################################################################################################################


def CompileARP(base_object=None, INPUT=None, TARGET=None, top_error = None, params = None, 
               ARP_rel_penalty=0.02, ARP_rel_gain=0.04, ARP_abs_gain = 0.1, bWeightDecay=False, 
               initial_update_size = 3e-2, modifies_base_object=True):
    """ default for top_error is <base_object.output_layer_Loss>
        default for    params is <base_object.params>
        total factor on sign-agreement is: 1+<ARP_gain>.
        total factor on disagreement   is: 1-<ARP_penalty>"""
    print "compiling ARP..."
    
    
    
    if top_error==None:
        top_error = base_object.output_layer_Loss
    if params==None:
        params = base_object.params
    if INPUT==None:
        INPUT = base_object.x
    if TARGET==None:
        TARGET = base_object.y
    assert len(params)>0,"call CompileOutputFunctions() before calling CompileARP()!"

    All_Gradients = T.grad( top_error, params, disconnected_inputs="warn")
    ARP_LRs=[] # ARP_LRs will oscillate around 1.
    last_grads=[]
    ARP_updates=[]

    # Moving averages:
    ARP_sign_consistency = [] #values in 0...1 (for each parameter)
    ARP_block_sign_consistency =[] # values in 0..1 (one for each BLOCK of parameters, eg. one for one W matrix in one layer)
    #todo test if mov.avg. of abs. value is helpful

#    ARP_global_LR = theano.shared(np.float32(initial_update_size))
    SGD_global_LR = theano.shared(np.float32(initial_update_size))



    if bWeightDecay:
        print "CNN::using Weight decay! Change value via this.global_weightdecay_param.set_value()"
        global_weightdecay_param = theano.shared(np.asarray(0.0005).astype("float32"))

    for i,para in enumerate(params):
        if para in params[:i]:
            print "Detected RNN or shared param @index =",i
        else:

            ARP_LRs.append(theano.shared(  np.float32(initial_update_size)*np.ones(para.get_value().shape,dtype=theano.config.floatX) , name=para.name+str('_ARP_LR') , borrow=0))
            ARP_sign_consistency.append(theano.shared(  np.float32(0.5)*np.ones(para.get_value().shape,dtype=theano.config.floatX) , name=para.name+str('_ARP_SC') , borrow=0))
            ARP_block_sign_consistency.append(theano.shared( np.asarray( np.float32(0.4)) , name=para.name+str('_ARP_SCB') , borrow=0))
            last_grads.append(theano.shared( np.zeros(para.get_value().shape,dtype=theano.config.floatX) , name=para.name+str('_ARP_LG') , borrow=0))


    for param_i, grad_i, last_grad_i, pLR_i, SignCons_i, SignConsBlock_i in zip(params, All_Gradients, last_grads, ARP_LRs, ARP_sign_consistency, ARP_block_sign_consistency):
        ARP_updates.append((SignCons_i, 0.95*SignCons_i + 0.05 * (T.sgn(grad_i)==T.sgn(last_grad_i))))
        ARP_updates.append((SignConsBlock_i,  T.mean(SignCons_i) ))

        ARP_updates.append((pLR_i, T.minimum( T.maximum( pLR_i - pLR_i * np.float32(ARP_rel_penalty)* ((SignCons_i < SignConsBlock_i) ) + ( (SignCons_i > SignConsBlock_i) ) * (np.float32(ARP_rel_gain)*pLR_i + ARP_abs_gain)    , 1e-1*T.ones_like(pLR_i) ), 100. * T.ones_like(pLR_i)) ))
        #ARP_updates.append((pLR_i, T.minimum( T.maximum( pLR_i - pLR_i * np.float32(ARP_rel_penalty)* ((SignCons_i < SignConsBlock_i) + ((last_grad_i*grad_i) < 0)) + (((last_grad_i*grad_i) >= 0) + (SignCons_i > SignConsBlock_i) ) * (np.float32(ARP_rel_gain)*pLR_i + ARP_abs_gain)    , 5e-2*T.ones_like(pLR_i) ), 20. * T.ones_like(pLR_i)) ))

        #ARP_updates.append((SignConsBlock_i, 0.8*SignConsBlock_i + 0.2* T.mean(SignCons_i) ))


#        ARP_updates.append((pLR_i, T.minimum( T.maximum( pLR_i * ( 1 - np.float32(ARP_penalty)* ((last_grad_i*grad_i) < -1e-9) + np.float32(ARP_gain)* ((last_grad_i*grad_i) > 1e-11)   ) , 1e-7*T.ones_like(pLR_i) ),2e-3 * T.ones_like(pLR_i)) ))
        ARP_updates.append((param_i, param_i  - pLR_i * T.sgn(grad_i) * SGD_global_LR - (0 if bWeightDecay==False else global_weightdecay_param*param_i) )) #grad_i/(T.abs_(grad_i) + 1e-6)
        ARP_updates.append((last_grad_i, grad_i ))#ARP_updates.append((last_grad_i, (grad_i + 0.5*last_grad_i)/1.5)) #trailing exp-mean over last gradients: smoothing. check if useful...

    train_model_ARP = theano.function([INPUT,TARGET], top_error, updates=ARP_updates,  on_unused_input='warn')

    if modifies_base_object and base_object is not None:
        print "ARP...setting base_object's attributes & methods..."
        base_object.train_model_ARP = train_model_ARP
        if bWeightDecay:
            base_object.global_weightdecay_param = global_weightdecay_param
        base_object.last_grads = last_grads
        base_object.ARP_LRs = ARP_LRs
        base_object.SGD_global_LR = SGD_global_LR

    print "compiling ARP...DONE!"
    return train_model_ARP














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


def CompileRPROP_extended(base_object=None, input_arguments=[], loss = None, params = None, 
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
    print
    print "This function is not completed. Todo: implement RPROP with backtracking and other features to reduce oscillations. Most likely a pure numpy/python optimizer."
    raise NotImplementedError()
    
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


def CompileSGD(base_object, top_error = None, params = None, input_arguments = None, SGD_LR_=4e-5, SGD_momentum_=0.9, bWeightDecay=False):
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





#
#
######################################################################################################################
########################################       Stochastic 2nd Order Gradient Descent (Newton)    #######################################
######################################################################################################################
#
#
#def CompileS2GD(base_object, top_error = None, params = None ):
#    """ default for top_error is <base_object.output_layer_Loss>
#        default for    params is <base_object.params>
#        """
#    print "compiling Stochastic 2nd Order Gradient Descent (Newton)..."
#    raise "TODO: import slinalg or nlinalg!"
#    if top_error==None:
#        top_error = base_object.output_layer_Loss
#    if params==None:
#        params = base_object.params
#    assert len(params)>0,"call CompileOutputFunctions() before calling CompileS2GD()!"
#
#    All_Gradients = T.grad( top_error, params, disconnected_inputs="warn")
##    base_object.SGD_global_LR = theano.shared(np.float32(SGD_LR_))
##    base_object.SGD_momentum = theano.shared(np.float32(SGD_momentum_))
##    base_object.SGD_global_LR.set_value(np.float32(SGD_LR_))
#
#    base_object.last_grads=[]
#    SGD_updatesa=[]
#    SGD_updatesb=[]
#
#
##    for i,para in enumerate(params):
##        if para in params[:i]:
##            print "Detected RNN or shared param @index =",i
##        else:
##            base_object.last_grads.append(theano.shared( np.zeros(para.get_value().shape,dtype=theano.config.floatX) , name=para.name+str('_LG') , borrow=0))
#
#
##    for param_i, grad_i, last_grad_i in zip(base_object.params, All_Gradients, base_object.last_grads):
###            if len(base_object.params)>len(base_object.last_grads):
###                grad_i = None
###                print "grad_param::",param_i
###                for i in range(len(base_object.params)):
###                    if param_i == base_object.params[i]:
###                        print ">>",i
###                        grad_i = All_Gradients[i] if grad_i==None else grad_i + All_Gradients[i]
##
##        SGD_updatesa.append((last_grad_i, grad_i             + last_grad_i * base_object.SGD_momentum))#use this if you want to use the gradient magnitude
#
#    for param_i, grad_i in zip(base_object.params, All_Gradients):
#        H = T.hessian(top_error, param_i, disconnected_inputs='raise')
#        invV = Solve(H, grad_i)
#        SGD_updatesa.append((param_i, param_i  - (base_object.SGD_global_LR ) * invV   ))
#
#    assert len(SGD_updatesa)==len(SGD_updatesb),str(len(SGD_updatesa))+" != "+str(len(SGD_updatesb))
#    base_object.train_model_SGD_a     = theano.function([base_object.x,base_object.y], top_error, updates=SGD_updatesa,  on_unused_input='warn')#the output is the value you get BEFORE updates....
##    base_object.train_model_SGD_b     = theano.function([], None, updates=SGD_updatesb)# ONLY changes the parameters
##            base_object.train_model_SGD_a_ext = theano.function([base_object.x,base_object.y]+addthis, [top_error, self.layers[-1].class_probabilities_realshape], updates=SGD_updatesa,  on_unused_input='warn')
#
##    def local_f(x,y):
##        nll = base_object.train_model_SGD_a(x,y)
###        base_object.train_model_SGD_b()
##        return nll
#    base_object.train_model_SGD = base_object.train_model_SGD_a#local_f
#    print "compiling SGD...DONE!"
#    return base_object.train_model_SGD









#####################################################################################################################
#######################################       Return:  Gradients, Hessian(s)     #######################################
#####################################################################################################################


def GetGradients(base_object, top_error = None, params = None, return_loss_as_first_output=False ):
    """ list of gradients.
        if <base_object.input_arguments> exists, it is used as input to obtain the gradients!
        """
    print "compiling Getter-function (GetGradients)..."
    if top_error==None:
        top_error = base_object.output_layer_Loss
    if params==None:
        params = base_object.params
    assert len(params)>0,"call CompileOutputFunctions() before calling GetGradients()!"

    grads = T.grad( top_error, params, disconnected_inputs="warn")

    if return_loss_as_first_output:
        ret = [top_error]+grads
    else:
        ret = grads
        
    args = [base_object.x, base_object.y]
    if hasattr(base_object,"input_arguments"):
        args = base_object.input_arguments
    getter     = theano.function(args, ret,  on_unused_input='warn')
    return getter



def GetGradientsAndHessian(base_object, top_error = None, params = None ):
    """ list of gradients + list of hessians
        """
    print "compiling Getter-function (GetGradientsAndHessian)..."
    if top_error==None:
        top_error = base_object.output_layer_Loss
    if params==None:
        params = base_object.params
    assert len(params)>0,"call CompileOutputFunctions() before calling GetGradientsAndHessian()!"

    grads = T.grad( top_error, params, disconnected_inputs="warn")
    H = [T.hessian(top_error, param_i, disconnected_inputs='raise') for param_i in params]

    getter     = theano.function([base_object.x,base_object.y], grads + H, on_unused_input='warn')
    return getter


def GetLoss(base_object):
    """ returns theano_function of <base_object.output_layer_Loss> """
    get_NLL = theano.function([base_object.x,base_object.y], base_object.output_layer_Loss, on_unused_input='warn')
    return get_NLL














#####################################################################################################################
############## Test gradient magnitude in different layers and relative update magnitude ############################
############## Useful to figure out whether the initialization is good (or not)          ############################
#####################################################################################################################
def plot_histogram(data, normalize_height=0, n_bins=None):
#    print "data",data.shape
    data = data.flatten()
#    print "data.flatten",data.shape
    mi,ma=np.min(data),np.max(data)
    if n_bins is None:
        n_bins=int(len(data)/8.)+1
#    print "N=",len(data)
#    print "min=",mi,"max=",ma
    hist, bin_edges = np.histogram(data,bins=n_bins,range=(mi,ma))
    if normalize_height:
        hist = hist*(1./np.max(hist))
#    print len(hist),len(bin_edges)
    plot.plot(bin_edges[:-1],hist)
    

class GradientAnalyzer():
    """ Test gradient magnitude in different layers and relative update magnitude
        Useful to figure out whether the initialization is good (or not)"""
    def __init__(self, base_object):

        self.GetGradients = GetGradients(base_object, top_error = None, params = None, return_loss_as_first_output=False )
        self.base_object = base_object
        
    def show_histograms(self, learning_rate, *network_inputs):
        """ <network_inputs> typically is the tuple (data,labels)"""
        grads = self.GetGradients(*network_inputs)[::2]
        plot.figure()
        plot.title("Histogram: Gradients (per layer)")

        for gg in grads:
            gg=np.asanyarray(gg)
            plot_histogram(gg,normalize_height=1)
        plot.legend( ["layer_"+str(i) for i in range(len(grads))] )
#        plot.show()
        
        params = [self.base_object.params[i].get_value() for i in range(len(self.base_object.params))][::2]

        plot.figure()
        plot.title("Histogram: relative update magnitude (per layer)")

        for gg,p in zip(grads,params):
            up_mag = (np.asanyarray(gg)*learning_rate)/(1e-6+np.abs(p))
            me=np.mean(up_mag)
            st=np.std(up_mag)
            new_um = np.clip(up_mag,me-3*st,me+3*st)
            print "Outliers removed:",np.sum(new_um!=up_mag)
            plot_histogram(new_um,normalize_height=1)
        plot.legend( ["layer_"+str(i) for i in range(len(grads))] )
        plot.show()



#####################################################################################################################
#######################################       Newtons Method for Optimization (slow, full-batch method)     #######################################
#####################################################################################################################


class Optimizer_SecondOrder():
    """ a.k.a. Newtons method, applied to 1st derivative => uses the full Hessian and 'inverts' it (solving LSE, quite slow)"""
    def __init__(self, base_object):
        self.base_object = base_object
        self.__grad_getter = GetGradientsAndHessian(base_object)
        self.__Parameter_values = [p.get_value() for p in base_object.params]

    def optimization_step(self, x, y, LR=0.1, b_use_cached_parameters=True):
        """ stores a local copy of current parameters, will be overwritten/updated if b_use_cached_parameters==False (speed penalty)."""
        grads = self.__grad_getter(x,y)
        hessians = grads[len(grads)/2:]
        if b_use_cached_parameters==0:
            self.__Parameter_values = [p.get_value() for p in self.base_object.params]
        assert len(hessians)==len(self.__Parameter_values)==len(self.base_object.params)
        for g,h,p,symb_p in zip(grads, hessians, self.__Parameter_values, self.base_object.params):
            invV = np.linalg.solve(h,g)#Solve(H, grad_i)
            p -= LR  * invV
            symb_p.set_value(p)





#####################################################################################################################
#######################################       L-BFGS (fast, full-batch method)     #######################################
#####################################################################################################################


class Optimizer_LBFGS():
    """ L-BFGS (fast, full-batch method)"""
    def __init__(self, base_object, debug=False, theano_symb_params=None, top_error=None , custom_loss_and_gradients_getter_fnct=None):
        """ <theano_symb_params> has precedence over <base_object.params>
            <custom_loss_and_gradients_getter_fnct> has precedence over <this_file.GetGradients(base_object, return_loss_as_first_output=True)>
            if you specify both then the parameter <base_object> will be ignored."""
        if theano_symb_params is None:
            self.__theano_symb_params = base_object.params # list of symbolic theano-variables (target of optimization)
        else:
            self.__theano_symb_params = theano_symb_params
        if custom_loss_and_gradients_getter_fnct is None:
            self.__top_error = top_error
            self.__cnn = base_object
            #the following will be delayed until optimization() is executed
#            self.__loss_and_grads_getter = GetGradients(base_object, return_loss_as_first_output=True,top_error=top_error)# must return: list/tuple. first entry: loss (float32 scalar), second entry: list of numpy.tensors (same shape as parameters!)
        else:
            self.__loss_and_grads_getter = custom_loss_and_gradients_getter_fnct
        self.__Parameter_values = [p.get_value() for p in self.__theano_symb_params]
        self.__params_total_size = np.sum([np.prod(p.shape) for p in self.__Parameter_values]) #length of vectorized parameters
        self.__Parameter_values_flat = np.zeros(self.__params_total_size,"float32")
        self.__Gradient_values_flat = np.zeros(self.__params_total_size,"float32")
        self.debug=debug

    def __loss_and_grads_getter(self,*args):
        self.__loss_and_grads_getter = GetGradients(self.__cnn, return_loss_as_first_output=True,top_error=self.__top_error)# must return: list/tuple. first entry: loss (float32 scalar), second entry: list of numpy.tensors (same shape as parameters!)
        return self.__loss_and_grads_getter(*args)

    def __vect2list(self, p_vect):
        """ internal use, modifies <self.__Parameter_values>"""
        i=0
        for p in self.__Parameter_values:
            j=np.prod(p.shape)
            p[...] = p_vect[i:i+j].reshape(p.shape)
            i+=j
        return self.__Parameter_values

    def __list2vect(self, p_list, b_params=True):
        """ internal use, modifies <self.__Parameter_values_flat> if b_params else <self.__Gradient_values_flat>"""
        if b_params:
            new = self.__Parameter_values_flat
        else:
            new = self.__Gradient_values_flat
        i=0

        for p in p_list:
            j=np.prod(np.shape(p))
            new[i:i+j] = p.flatten() if b_params else np.asarray(p).flatten()
            i+=j
        return new

    def __get_loss_and_grads(self, new_param_vect, *args):
        """ internal use; updates <self.__theano_symb_params>"""
        if np.any(np.isnan(new_param_vect)):
            print "NaN detected...aborting..."
            new_param_vect[...]=0
            return (0,new_param_vect)
        pl = self.__vect2list(new_param_vect)
        for i,npp,p in zip(range(len(pl)),pl,self.__theano_symb_params):
            p.set_value(npp,borrow=False)
        ret = self.__loss_and_grads_getter(*args)
        loss,grads = ret[0], ret[1:]
        if len(grads)==1:
            grads=grads[0]
        vec = np.asarray(self.__list2vect(grads,b_params=0),dtype="float64")
        if self.debug:
            print "__get_grads:: x_min,max =",np.min(args[0]),np.max(args[0]),"y_min,max =",np.min(args[1]),np.max(args[1])
            print "__get_grads (av(g) =",np.mean(vec),", av(abs(g)) =",np.mean(np.abs(vec)),")"
        return (loss,vec)


    def optimize(self, argument_list, max_evals=40, max_iters=4, b_use_cached_parameters=0):
        """ return: final loss"""
        #References (cite one)
        #    R. H. Byrd, P. Lu and J. Nocedal. A Limited Memory Algorithm for Bound Constrained Optimization, (1995), SIAM Journal on Scientific and Statistical Computing, 16, 5, pp. 1190-1208.
        #    C. Zhu, R. H. Byrd and J. Nocedal. L-BFGS-B: Algorithm 778: L-BFGS-B, FORTRAN routines for large scale bound constrained optimization (1997), ACM Transactions on Mathematical Software, 23, 4, pp. 550 - 560.
        #    J.L. Morales and J. Nocedal. L-BFGS-B: Remark on Algorithm 778: L-BFGS-B, FORTRAN routines for large scale bound constrained optimization (2011), ACM Transactions on Mathematical Software, 38, 1.
        if b_use_cached_parameters==0:
            self.__Parameter_values = [p.get_value() for p in self.__theano_symb_params]
        assert len(self.__Parameter_values)==len(self.__theano_symb_params)
        self.__list2vect(self.__Parameter_values, b_params=1) #sets self.__Parameter_values_flat
        
        if self.debug:
            print "\nlbfgs->optimize::"
            print "__optimize:: x_min,max =",np.min(argument_list[0]),np.max(argument_list[0]),"y_min,max =",np.min(argument_list[1]),np.max(argument_list[1])
        new_params, loss, dic = scipy.optimize.fmin_l_bfgs_b(func=self.__get_loss_and_grads, x0=self.__Parameter_values_flat,
                                     fprime=None, args=argument_list, approx_grad=0,
                                    bounds=None, m=10, factr=1e2, pgtol=1e-09,
                                    iprint=-1, maxfun=max_evals, maxiter=max_iters, disp=None, callback=None)
        # todo: check if last param_setting applied by <__get_loss_and_grads> by fmin_l_bfgs_b is the same as <new_params>
        pl = self.__vect2list(new_params)
        for i,npp,p in zip(range(len(pl)),pl,self.__theano_symb_params):
            p.set_value(npp,borrow=False)
        aborted = dic["warnflag"] != 1
        if aborted:
            print "L-BFGS::ABORTED! (check labels / model ...)"
        return float(loss)

    def __call__(self, argument_list, max_evals=40, max_iters=4, b_use_cached_parameters=0):
        return self.optimize(argument_list=argument_list, max_evals=max_evals, max_iters=max_iters, b_use_cached_parameters=b_use_cached_parameters)


#####################################################################################################################
###########################################       Conjugate Gradient      ###########################################
#####################################################################################################################


def compileCG(base_object):
    print "Compiling Conjugate Gradient op."
    if len(base_object.params)==0:
      print "Call CompileOutputFunctions before calling CompileCG, CNN.params is empty!"

    base_object.last_grads    = []
    base_object.direc         = []
    CG_updates_direc   = []
    CG_updates_grads   = []
    CG_updates         = [] # params

    addthis = [base_object.z,] if base_object.bUseModulatedNLL else []

    # Initialise shared variables for the training algos
    for para in base_object.params:
      para_shape = para.get_value().shape
      base_object.last_grads.append(theano.shared(np.zeros(para_shape,dtype=theano.config.floatX),name=para.name+'_LG',      borrow=False))
      base_object.direc.append(theano.shared(np.zeros(para_shape,dtype=theano.config.floatX),name=para.name+'_CG_direc',borrow=False))

    # create a list of gradients for all model parameters
    output_layer_gradients = T.grad(base_object.output_layer_Loss, base_object.params, disconnected_inputs="warn")

    ### Kickstart of CG, initialise first direction to current gradient ###
    for grad_i, last_grad_i, direc_i in zip(output_layer_gradients, base_object.last_grads, base_object.direc):
      CG_updates_grads.append((last_grad_i, grad_i))
      CG_updates_direc.append((direc_i,    -grad_i))

    # update direc & last-grad
    updates = CG_updates_grads + CG_updates_direc
    base_object.CG_kickstart   = theano.function( [base_object.x,base_object.y]+addthis, None, updates=updates, on_unused_input='warn')


    ### Regular CG-step ###
    CG_updates_direc = [] # clear update-list
    CG_updates_grads = [] # clear update-list

    # Compute Polak-Ribiere coefficient b for updating search direction
    denom = theano.shared(np.float32(0))
    num   = theano.shared(np.float32(0))

    for grad_i, last_grad_i in zip(output_layer_gradients, base_object.last_grads):
      num   = num + T.sum(-grad_i * (-grad_i+last_grad_i))
      denom = denom + T.sum(last_grad_i * last_grad_i)
    coeff = num / denom
    coeff = T.max(T.stack([coeff, theano.shared(np.float32(0))])) # select, use maximum instead!

    # Search-direction and last-grad update
    for grad_i, last_grad_i, direc_i in zip(output_layer_gradients, base_object.last_grads, base_object.direc):
      CG_updates_grads.append((last_grad_i,  grad_i))
      CG_updates_direc.append((direc_i, -grad_i + direc_i * coeff))

    updates = CG_updates_grads + CG_updates_direc
    base_object.CG_step   = theano.function([base_object.x,base_object.y]+addthis, coeff, updates=updates, on_unused_input='warn')


    # Weights update (Line search), no input needed, as only params are changed
    delta = T.fscalar('delta') # used to parametrise the ray along we search (=0 at current params)
    base_object.t = theano.shared(np.float32(0)) ######################################################################
    for param_i, search_direc_i in zip(base_object.params, base_object.direc):
      CG_updates.append((param_i, param_i + search_direc_i * delta))

    CG_updates.append((base_object.t, base_object.t+delta))
    base_object.CG_update_params = theano.function([delta], base_object.t+delta, updates=CG_updates, on_unused_input='warn')


    # Linear-Approximation (from shared last_(grad|direc))
    linear_approx   = theano.shared(np.float32(0))
    for grad_i, last_direc_i in zip(base_object.last_grads, base_object.direc):
      linear_approx   = linear_approx + T.sum(grad_i * last_direc_i)

    base_object.CG_linear_approx  = theano.function([], linear_approx, updates=None, on_unused_input='warn')
    base_object.get_NLL = theano.function([base_object.x,base_object.y]+addthis, base_object.output_layer_Loss, on_unused_input='warn')

    #stuff for my line search (Greg)

    base_object.__CG_LineS_avg_stepsize = 0.005
#        base_object.__CG_LineS_rel_steps = 0.5

    return 0




def _trainingStepCG(base_object, *args):
  base_object.CG_kickstart(*args)
  timeline = []
  nll, nll_instance, t, count = base_object._lineSearch(*args)
  timeline.append([nll, t, np.NaN, count])
  base_object.t.set_value(np.float32(0)) # DBG: reset internal update-magnitude
  n_steps = base_object.CG_params['n_steps']
  for i in xrange(n_steps-1):
      if base_object.CG_params['only_descent']:
          base_object.CG_kickstart(*args)
          coeff = np.NaN
      else: # use actual CG
          coeff = base_object._CG_step(*args)

  nll, nll_instance, t, count = base_object._lineSearch(*args)
  base_object.t.set_value(np.float32(0))
  timeline.append([nll, t, coeff, count])
  base_object.CG_timeline.extend(timeline)
  return nll, nll_instance


def _lineSearch(base_object, *args):
    """ Needed for CG """
    nll_0, _ = base_object.get_NLL(*args)
    nll_0 = np.float32(nll_0)
    linear_approx = base_object.CG_linear_approx()
    counter = 0
    if linear_approx > 0: # if algorithm gets stuck, reset
      nll, nll_instance = base_object.get_NLL(*args)
      return nll_0, nll_instance, 0, counter
    max_step = base_object.CG_params['max_step']
    min_step = base_object.CG_params['min_step']
    beta     = base_object.CG_params['beta']

    max_count = int(np.log(min_step / (max_step)) / np.log(beta)) # limit iterations to lower bound
    points = [] ### For Plotting
    t = max_step
    # The next search poits DEcrement the search ray by the decaying negative factor delta
    # Thus parameters needn't to be reset before updating, we directly change
    # the parameters by the desired amount
    base_object.CG_update_params(np.float32(max_step)) # the first search point: max_step along the current search direction
    nll, nll_instance = base_object.get_NLL(*args)
    nll = np.float32(nll)
    counter += 1
    delta = max_step * (beta - 1.0)
    last_nll = 1000000
    for i in xrange(max_count):
      chord = nll_0 + t * linear_approx * base_object.CG_params['alpha']
      points.append([t, nll, chord])
      if (nll <= chord) or (nll > last_nll): # The second condition is a deviation from regular BT-LineSearch
        break
      base_object.CG_update_params(np.float32(delta))
      last_nll = nll
      nll, nll_instance = base_object.get_NLL(*args)
      nll = np.float32(nll)
      delta = delta * beta
      t     =     t * beta
      counter += 1

    if base_object.CG_params['show']:
      points.append([0, nll_0, nll_0])
      points = np.array(points)
      plot.figure()
      plot.plot(points[:,0], points[:,1:])
      plot.scatter(points[:,0], points[:,1])
      plot.legend(('fu', 'chord'))
      plot.draw()
      plot.pause(0.0001)
    return nll, nll_instance, base_object.t.eval(), counter



def trainingStepCG_g(base_object, params, n_steps, alpha=0.3, beta=0.5, max_step=3e-1, min_step=8e-7, sgd_only=False, show=False, always_print_nll=True):
    t0=time.clock() # Start timing

    base_object.CG_kickstart(*params)

    timeline = np.zeros((n_steps, 3))
    nll, t = base_object._lineSearch_g(params)#base_object.__lineSearch(params, alpha, beta, max_step, min_step, show)
    timeline[0] = [nll, t, np.NaN]
    base_object.t.set_value(np.float32(0)) # DBG: reset internal update-magnitude
    for i in xrange(n_steps-1):
      if sgd_only:
        base_object.CG_kickstart(*params)
        coeff = np.NaN
      else: # use actual CG
        coeff = base_object.CG_step(*params)
      nll, t = base_object._lineSearch_g(params)#base_object.__lineSearch(params, alpha, beta, max_step, min_step, show)
      if always_print_nll:
          print i,nll
      base_object.t.set_value(np.float32(0))
      timeline[i+1] = [nll, t, coeff]
    runtime = (time.clock() - t0)/n_steps + 1e-7 # End timing
    return nll, t, timeline, runtime



def _lineSearch_g(base_object, params, max_evaluations=8, show=0):
    """ Needed for CG,
    author: Greg.

    minimizes nll

    calls to CG_update_params() are ADDITIVE, calls to make_step() are NOT (i.e. absolute)."""

    xs=[0]
    ys=[]

    ys.append(base_object.get_NLL(*params)) #at current setting


    x = [np.float32(0)]


    def make_step(stepsize, x=x):
        stepsize = np.float32(stepsize)
        base_object.CG_update_params(np.float32(stepsize - x[0]))
        x[0] = stepsize
        xs.append(stepsize)
        ys.append(base_object.get_NLL(*params))

    make_step(base_object.__CG_LineS_avg_stepsize*0.5)
    make_step(base_object.__CG_LineS_avg_stepsize) #at expected best



    for i in xrange(max_evaluations-3):
        if len(ys)>3:

            abc = np.polyfit(xs[:-1], ys[:-1], deg=2)  #order: a,b,c  (as in)  ax^2+bx+c ,  a == abc[0]
        else:
            abc = np.polyfit(xs, ys, deg=2)
#            print "a,b,c =",abc
        if abc[0]!=0:
            spot = - abc[1]/2./(abc[0])
            if spot>5*max(xs):
                spot = 5*max(xs)
            if spot<0.2*min(xs):
                spot = 0.2*min(xs)

        else:
            spot = xs[np.argmin(ys)] + 10*(xs[np.argmin(ys)]-xs[np.argmax(ys)])# follow best direction with large step


        #xs always sorted
        if xs[-1]<xs[-2]:
            indx = np.asarray(sorted(range(len(xs)), key=xs.__getitem__))
            xs=list(np.asarray(xs)[indx])
            ys=list(np.asarray(ys)[indx])


        if np.all(ys[0]<=np.asarray(ys)): # everything is worse than doing nothing
            spot = 0.8*xs[0]+0.2*xs[1]
        else:
            #avoid checking very close points:
            mi,ma = np.argmin(abs(spot-np.asarray(xs))), np.argmax(abs(spot-np.asarray(xs))) #closest and farthest point
            close = abs(spot-xs[mi]) / abs(spot-xs[ma]) < 0.03

            if close:
                mm = np.argmin(ys)
                if mm==0:#lower end
#                        print "0, "
                    spot = 3*xs[0] - 2*xs[1]  # == ys[0] - 2*(ys[1] - ys[0])
                    if spot<0:
                        spot = 0.5*(xs[0]+xs[1])
                elif mm==len(ys)-1:#upper end
                    spot =  3*xs[-1] - 2*xs[-2] # == ys[-1] + 2*(ys[-1] - ys[-2])
                else:
                    if ys[mm+1] < ys[mm-1]:
                        spot = 0.5*(xs[mm] + xs[mm+1])
                    else:
                        spot = 0.5*(xs[mm] + xs[mm-1])

                mi,ma = np.argmin(abs(spot-np.asarray(xs))), np.argmax(abs(spot-np.asarray(xs))) #closest and farthest point
                close = abs(spot-xs[mi]) / abs(spot-xs[ma]) < 0.03
                if close:#still close
                    break

        make_step(spot)

    if np.argmin(ys)!=len(ys)-1:#last step was not the best step
        make_step(xs[np.argmin(ys)])



    base_object.__CG_LineS_avg_stepsize = 0.95*base_object.__CG_LineS_avg_stepsize + 0.05* xs[np.argmin(ys)] #update



    if show:

      plot.figure()

      plot.scatter(xs,ys)
      plot.legend(('stepsize', 'nll'))

      plot.show()

    return ys[-1], 0








