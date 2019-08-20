
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

import theano
import numpy as np



def Gregs_Regularizer(cnn, num_classes):
    print("Experimental Gregs_Regularizer:: enabled")
#    assert bExperimental_Semanic_Hashing_enforcer==0, "dont use both!"
    assert num_classes is not None, "You need to provide a value for <num_classes> when using 'Gregs_Regularizer' in CompileOutputFunctions()"
    reg = 0
    for lay in cnn.layers[:-1]:
        reg += lay.Gregs_Regularizer(cnn.y, num_classes) #currently only exists for MLPs
    return reg






def add_balanced_regularizers_to_NLL(cnn, regularizers=[], relative_weightings=[]):
    """
    pass all regularizers as elements of a list, each of them as a symbolic theano variable.
    Those will be added to the cnn/nn's NLL with relative magnitudes as specified in <relative_weightings>.
    You may _almost_ forget to worry about them from now on (but remember to call cnn.Gregs_Regularizer_balance() every x weight-update steps [x may be in the order of 10 or 100 ])
    
    """
    assert isinstance(regularizers,[list,tuple])
    assert len(regularizers)==len(relative_weightings), "Nope, try again."

    print("Experimental Gregs_Regularizer_balance:: enabled   [You need to call CNN.Gregs_Regularizer_balance(_data_, _labels_, ...) initially and then also from time to time, to update the internal factor(s)!]")
    cnn.Gregs_Regularizer_balance__current_factors = [theano.shared(np.float32(1e-5)) for x in relative_weightings]
    cnn.Gregs_Regularizer_balance__relative_weightings   = relative_weightings
    
    cnn._Gregs_Regularizer_balancer_loss_getter = theano.function([cnn.x, cnn.y],[cnn.output_layer_Loss]+regularizers)
    print("todo: make Gregs_Regularizer_balancer() smarter: add a call-counter and determine how often it actually needs to update the values! (save computational time, especially if it is unnecessarily called after every update step)")
    print("todo: make this a stand-alone function, e.g. 'add_regularizer_balanced()'")
    def Gregs_Regularizer_balancer(*args):
        ret = cnn._Gregs_Regularizer_balancer_loss_getter(*args)
        original_nll_, reg_losses = ret[0], ret[1:]
#        print "Gregs_Regularizer_balancer():: NLL:",original_nll_,"Regularizer is:",reg_losses
        for theano_fact, rel_wei in zip(cnn.Gregs_Regularizer_balance__current_factors, cnn.Gregs_Regularizer_balance__relative_weightings):
            newv = rel_wei * (abs(original_nll_)/(abs(reg_losses)+1e-11))
            newv = 0.8*cnn.Gregs_Regularizer_balance__current_factors.get_value() + 0.2 * newv
            cnn.Gregs_Regularizer_balance__current_factors.set_value(np.float32(newv))
#                    print "new value for <cnn.Gregs_Regularizer_balance__current_factors> =",newv
    cnn.Gregs_Regularizer_balance = Gregs_Regularizer_balancer
    cnn.output_layer_Loss += cnn.Gregs_Regularizer_balance__current_factors * reg


