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
import theano.tensor as T
import numpy as np


class Analyzer(object):
    def __init__(self, cnn):
        self._cnn = cnn
        self._ranonce = False
        self._ranonce2 = False
        ####################
        
    def _runonce(self):
        if self._ranonce:
            return
        print self,'compiling...'
        self._output_function = theano.function([self._cnn.layers[0].input], [lay.output for lay in self._cnn.layers])
        self._ranonce=True
        ####################
    
    def _runonce2(self):
        if self._ranonce2:
            return
        print self,'compiling...'
        output_layer_Gradients = T.grad(self._cnn.output_layer_Loss, self._cnn.params, disconnected_inputs="warn")
        self._output_function2 = theano.function([self._cnn.x, self._cnn.y], [x for x in output_layer_Gradients], on_unused_input='warn')            
            
#         = theano.function([self._cnn.layers[0].input, self._cnn.y], [lay.output for lay in self._cnn.layers])
        self._ranonce2=True
        ####################
        
    def analyze_forward_pass(self, *input):
        """ input should be a list of all inputs. ((DO NOT INCLUDE labels/targets!))"""
        self._runonce()
        outputs = self._output_function(*input)
        print
        print 'Analyzing internal outputs of network',self._cnn,' (I am',self,') ... '
        for lay,out in zip(self._cnn.layers, outputs):
            mi,ma = np.min(out), np.max(out)
            mea,med = np.mean(out),np.median(out)
            std = np.std(out)
            print '{:^100}: {:^30}, min/max = [{:9.5f}, {:9.5f}], mean/median = ({:9.5f}, {:9.5f}), std = {:9.5f}'.format(lay,out.shape,mi,ma,mea,med,std)
        print
        return outputs
        ####################


    def analyze_gradients(self, *input):
        """ input should be a list of all inputs and labels/targets"""
        self._runonce2()
        outputs = self._output_function2(*input)
        print
        print 'Analyzing internal gradients of network',self._cnn,' (I am',self,') ... '
        i = 0
        j = 0
        for lay in self._cnn.layers:

            try:
                
                j = len(lay.params)
                
            except:
                j = 0
            if j:
                for out in outputs[i:i+j]:
                    mi,ma = np.min(out), np.max(out)
                    mea,med = np.mean(out),np.median(out)
                    std = np.std(out)
                    print '{:^100}: {:^30}, min/max = [{:9.5f}, {:9.5f}], mean/median = ({:9.5f}, {:9.5f}), std = {:9.5f}'.format(lay,out.shape,mi,ma,mea,med,std)
            else:
                print '{:^100}: no parameters'.format(lay)
            i+=j
        print
        return outputs
        ####################









