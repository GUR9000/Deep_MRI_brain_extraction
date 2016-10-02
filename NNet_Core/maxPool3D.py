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

version @23.4.14
mod @ 26.9.14
"""


import numpy as np
import theano
from theano import tensor as T


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




def my_max_pool_3d(sym_input, pool_shape = (2,2,2)):
    """ this one is pure theano. Hence all gradient-related stuff is working! No dimshuffling"""

    s = None
    if pool_shape[2]>1:
        for i in xrange(pool_shape[2]):
            t = sym_input[:,:,:,:,i::pool_shape[2]]
            if s is None:
                s = t
            else:
                s = T.maximum(s, t)
    else:
        s = sym_input

    if pool_shape[0]>1:
        temp = s
        s = None
        for i in xrange(pool_shape[0]):
            t = temp[:,i::pool_shape[0],:,:,:]
            if s is None:
                s = t
            else:
                s = T.maximum(s, t)

    if pool_shape[1]>1:
        temp = s
        s = None
        for i in xrange(pool_shape[1]):
            t = temp[:,:,:,i::pool_shape[0],:]
            if s is None:
                s = t
            else:
                s = T.maximum(s, t)
    sym_ret = s

    return sym_ret


if __name__=="__main__":
    import time
    inp = np.random.rand(1,64,32,64,64).astype('float32')


    sym_input = T.TensorType(dtype=theano.config.floatX, broadcastable=[False]*5)()

    sym_ret = my_max_pool_3d(sym_input)#my_max_pool_3d_stupid(sym_input)
    f_maxp_3d = theano.function([sym_input],sym_ret)
    print "ok"
    print

    for i in range(5):
        inp = np.random.rand(1,16+32*i,32,16+32*i,16+32*i).astype('float32')
        print 16+32*i,"^ 3"
        print inp.shape
        t0 = time.time()
        print f_maxp_3d(inp).shape
        print time.time()-t0,"s"
    exit()

