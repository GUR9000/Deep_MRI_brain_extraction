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

import theano.tensor as T



def parse_transfer_function(string_identifier, slope_parameter = None):
    """ This function returns the appropriate activation function, as selected by the string argument.
    
    string_identifier: 
        possible values are tanh, ReLU/relu, sigmoid/sig, abs, maxout <number>, linear/lin
    
    RETURNS: 
        transfer_function(python/theano function), string_identifier (normalized), dict (for special cases)
            
    """
    cross_channel_pooling_groups=None
    
    
    if string_identifier=='tanh':
        Activation_f = T.tanh
    elif string_identifier in ['ReLU', 'relu']: #rectified linear unit
        string_identifier = "relu"
        Activation_f = lambda x: x*(x>0)
    elif string_identifier in ['sigmoid', 'sig']:
        string_identifier = "sigmoid"
        Activation_f = T.nnet.sigmoid
    elif string_identifier in ['abs', 'Abs', 'absolute']:
        string_identifier='abs'
        Activation_f = T.abs_
    elif string_identifier in ['plu','PLu','PLU','piecewise']: #piece-wise linear function
        string_identifier = "PLU"
        print "parse_transfer_function::Remember to optimize the 'slope_parameter'"
        assert slope_parameter is not None,"...and better pass it to this function, as well! (type: Theano.Tensor, shape: same as activation, unif. random values [-1,1] should be fine)"
        Activation_f = lambda x: T.maximum(0,x) + T.minimum(0,x) * slope_parameter
    elif "maxout" in string_identifier:
        r=int(string_identifier.split(" ")[1])
        assert r>=2
        cross_channel_pooling_groups = r
    elif string_identifier in ['linear',"lin"]:
        string_identifier = "linear"
        Activation_f = lambda x:x
    else:
        raise NotImplementedError()
    return Activation_f, string_identifier, {"cross_channel_pooling_groups":cross_channel_pooling_groups}






