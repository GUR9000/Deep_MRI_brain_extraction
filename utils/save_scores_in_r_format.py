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
Format:
    experiment_name, Datensatz, ID, Methode, Dice, Jac, Sen, Spec, 
    3a, IBSR, 0001, CNN, 0.95, 0.87, ... 
"""

def make_R_readable_string(dic, experiment_name, dataset_name):
    """funct
    
    Inputs:
    -------
    
    dic: dictionary with fields (KEY) and content (VALUE)"""

    ret = ", ".join(["Experiment, Datensatz"]+dic.keys())+"\n"
    n = len(dic.values()[0])
    for x in dic.values():
        if n != len(x):
            print "ERROR"
            print "make_R_readable_string:: length mismatch!"
            print dic
            break
    
    for elems in zip(*dic.values()):
        ret += experiment_name+", "+dataset_name+", "
        ret += ", ".join([str(x) for x in elems])+"\n"
        
    
    return ret
    

if __name__=="__main__":
    
    import numpy as np
    dic=dict()
    dic["ID"] = range(10)
    dic["Dice"] = np.random.random(10)
    dic["Jac"] = np.random.random(10)
    
    print make_R_readable_string(dic,"Test01","IBSR")


