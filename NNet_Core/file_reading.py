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

import os
import h5py
import numpy as np

def get_path(fname):
    return '/'.join(fname.replace('\\','/').split('/')[:-1])

def get_filename(fname, remove_trailing_ftype= False):
    fn = fname.replace('\\','/').split('/')[-1]
    if remove_trailing_ftype:
        return '.'.join(fn.split('.')[:-1])
    return fn


def mkdir(fname):
    path = get_path(fname)
    if path!="" and os.path.exists(path)==False:
        os.makedirs(path)


try:
    import nibabel as nib
    try:
        from medpy.io import load as load_mha
    except:
        pass


    def load_nifti(fname):
        """
        load a single .nii file

        Returns:
        --------

            tuple: (data, affine, header) """

        nifti_obj = nib.load(fname)
        nifti_affine = nifti_obj.get_affine()
        nifti_header = nifti_obj.get_header()
        nifti_data = nifti_obj.get_data()#.astype(dtype)
        return nifti_data, nifti_affine, nifti_header

    def save_nifti(fname, data, affine=None, header=None):
        """ file-name should end on 'nii'"""
#        nin = nib.AnalyzeImage(data, np.eye(4))
        mkdir(fname)
        nin = nib.Nifti1Image(data, affine, header)
        nin.to_filename(fname)
except:
    print("nibabel package not found. Functions load_nifti(), save_nifti() will be unavailable")
    pass




def load_h5(fname, key=None):
    """load h5 file"""
    try:
        hfile = h5py.File(fname,'r')
    except:
        assert 0, "\nload_h5()::ERROR: Cannot open <<"+str(fname)+">>\n"
    if key==None:
        try:
            key = hfile.keys()[0]
        except:
            assert 0, "\nload_h5()::ERROR: File is not h5 / is empty  <<"+str(fname)+">>\n"
    xx = hfile[key]
    if isinstance(xx, h5py.Group):
        xx=dict(xx)
        xx = xx[xx.keys()[0] if key==None else key]
    dat = np.asarray(xx, dtype = xx.dtype)
    hfile.close()
    return dat





def save_h5(fname, data, compress = 1, fast_compression=1):#save as h5
    """save h5 file. set_name='data'"""

    mkdir(fname)
    h5f = h5py.File(fname,mode="w")

    if compress:
        h5set = h5f.create_dataset( "data", data.shape,dtype=data.dtype, compression=("gzip" if fast_compression!=True else "lzf")) #fast & bad: "lzf"
    else:
        h5set = h5f.create_dataset( "data", data.shape,dtype=data.dtype)
    h5set[...] = data
    h5f.close()
    return 0



def save_text(fname,string):
    mkdir(fname)
    f=open(fname,'w')
    f.write(string)
    f.close()



def load_file(filename):
    try:
        d = load_h5(filename)
        assert d is not None
        return d
    except:
        pass

    try:
        d, nifti_affine, nifti_header = load_nifti(filename)
        return d
    except:
        pass

    try:
        d = nib.load(filename)
        d=d.get_data()
        return d
    except:
        pass
    assert 0, 'Could not load file <'+str(filename)+'>'







