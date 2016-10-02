
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
from os import listdir as _listdir
from os.path import isfile as _isfile,join as  _join



def list_files(dir_paths, endswith=None, contains=None, startswith=None, contains_not=None):
    """ endswith may be a sting like '.jpg' """
    files=[]
    if type(dir_paths)!=type([]):
        dir_paths=[dir_paths]
    for path in dir_paths:#'/home/nkrasows/phd/data/graham/Neurons/4dBinNeuronVolume/h5/',
        try:
            gg= [ (_join(path,f) if path!="." else f) for f in _listdir(path) if _isfile(_join(path,f)) and (startswith == None or f.startswith(startswith)) and (endswith == None or f.endswith(endswith)) and (contains == None or contains in f)  and (contains_not == None or (not (contains_not in f))) ]
            files+=gg
        except:
            print("path",path,"invalid")
    files.sort()
    return files





def filter_list(string_list,endswith=None, contains=None, startswith=None, contains_not=None):
    return [ f for f in string_list if (startswith == None or f.startswith(startswith)) and (endswith == None or f.endswith(endswith)) and (contains == None or contains in f)  and (contains_not == None or (not (contains_not in f))) ]


def list_directories(dir_paths, endswith=None, contains=None, startswith=None, contains_not=None):
    """ endswith may be a sting like '.jpg' """
    files=[]
    N_OK=0
    if type(dir_paths)!=type([]):
        dir_paths=[dir_paths]
    for path in dir_paths:
        try:
            gg= [ (_join(path,f) if path!="." else f) for f in _listdir(path) if _isfile(_join(path,f))==False and (startswith == None or f.startswith(startswith)) and (endswith == None or f.endswith(endswith)) and (contains == None or contains in f)  and (contains_not == None or (not (contains_not in f))) ]
            files+=gg
            N_OK+=1
        except:
            print("path <",path,"> is invalid")
    if N_OK==0:
        print('list_directories():: All paths were invalid!')
        raise ValueError()
    files.sort()
    return files




def LPBA40_data(location):
    all_ = list_directories(location)
    list_training_data_=[]
    list_training_labels_=[]
    for x in all_:
        candid = list_files(x,endswith=".nii.gz",startswith="S")
        dat = filter_list(candid, endswith="mri.nii.gz")
        lab = filter_list(candid, endswith="mask.nii.gz")
        assert len(dat)==1
        dat=dat[0]
        assert len(lab)==1
        lab=lab[0]
        list_training_data_.append(dat)
        list_training_labels_.append(lab)
    assert len(list_training_labels_) == 40
    assert len(list_training_data_) == len(list_training_labels_)
    return list_training_data_, list_training_labels_




def IBSR_data(location):

    all_ = list_directories(location)

    list_training_data_=[]
    list_training_labels_=[]
    for x in all_:
        candid = list_files(x,endswith=".nii.gz",startswith="IB")
        dat = filter_list(candid, endswith="ana.nii.gz")
        lab = filter_list(candid, endswith="mask.nii.gz")
        assert len(dat)==1
        dat=dat[0]
        assert len(lab)==1
        lab=lab[0]
        list_training_data_.append(dat)
        list_training_labels_.append(lab)
    assert len(list_training_labels_) == 18
    assert len(list_training_data_) == len(list_training_labels_)
    return list_training_data_, list_training_labels_



def ID_check(list_training_data_, list_training_labels_):
    IDs_data = ['.'.join(x.replace('\\','/').split('/')[-1].split('_')[:2]) for x in list_training_data_]
    IDs_labels = ['.'.join(x.replace('\\','/').split('/')[-1].split('_')[:2]) for x in list_training_labels_]
    for a,b in zip(IDs_data, IDs_labels):
        assert a==b, 'training data/labels are shuffled and do not match!: '+a+' <>  '+b



def OASIS_data(location, labels_location = None):
    """
    labels_location: if None, then <location> will be used
    """
    if labels_location is None:
        labels_location = location

    list_training_labels_ = list_files(labels_location,endswith='hardmask.nii.gz')
    dirs = list_directories(location+"/disc1") + list_directories(location+"/disc2")
    list_training_data_=[]
    for d in dirs:
        dat = list_files(d+"/PROCESSED/MPRAGE/T88_111",endswith="t88_gfc.hdr")
        assert len(dat)==1
        dat=dat[0]
        list_training_data_.append(dat)
    assert len(list_training_data_)  ==77, len(list_training_data_)
    assert len(list_training_labels_)==77, len(list_training_labels_)
    assert len(list_training_data_) == len(list_training_labels_)
    ID_check(list_training_data_, list_training_labels_)
    return list_training_data_, list_training_labels_


def Tumor_data_JensCustomCreated():
    list_training_data_   = list_files("/home/share/brain_mask/tumor_data_h5",endswith = 't1ce.nii.gz')
    list_training_labels_ = list_files("/home/share/brain_mask/tumor_data_h5",endswith = '_human_mask.nii.gz')
    assert len(list_training_data_) == len(list_training_labels_)
    ID_check(list_training_data_, list_training_labels_)
    return list_training_data_, list_training_labels_



def get_CrossVal_part(list_training_data, list_training_labels, CV_index, CV_total_folds = 2):
    """ Splits data into training data/labels and test set.

        Inputs:
            CV_index: int from 0 to <CV_total_folds> - 1 (selects current split)

            CV_total_folds: Total number of CV folds, i.e. this must remain constant while CV_index changes from 0 up to <CV_total_folds> - 1

        returns:
            a dictionary with keys: 'list_test_data', 'list_training_data', 'list_training_labels'
            """
    N = len(list_training_data)
    cross_val_n_per_test= int(N*1./CV_total_folds)

    offset = CV_index*cross_val_n_per_test # must be calculated BEFORE the next if check !!!!!!!!!

    if CV_index == CV_total_folds - 1:
        cross_val_n_per_test = N - CV_index*cross_val_n_per_test # all remaining examples

    list_test_data       = list_training_data[offset : offset + cross_val_n_per_test]
    list_training_data   = list_training_data[:offset]  + list_training_data[offset+cross_val_n_per_test:]
    list_training_labels = list_training_labels[:offset]  + list_training_labels[offset+cross_val_n_per_test:]
    assert len(list_training_data)==len(list_training_labels)
    return {'list_test_data':list_test_data,'list_training_data':list_training_data, 'list_training_labels':list_training_labels}


