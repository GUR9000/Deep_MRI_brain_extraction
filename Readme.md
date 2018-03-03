---

    """
    This software is an implementation of the paper

    "Deep MRI brain extraction: A 3D convolutional neural network for skull stripping"

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

    Readme version 1.0 - October 2016

    """

## I. Introduction

Brain extraction from magnetic resonance imaging (MRI) is crucial for many neuroimaging workflows. However, the anatomical variability and the difference in intensity distributions of MRI scans make the development of a one-fits-all solution very challenging. This work is based on a 3D convolutional deep learning architecture that deals with arbitrary MRI modalities (T1, T2, FLAIR,DWI, ...), contrast-enhanced scans and pathologically altered tissues, when trained appropriately, i.e. tuned to the specific needs and imaging criteria present at your institution.

The presented code is a modified version of the work used for the above mentioned publication. There have been some alterations that might affect performance and speed. 


## II. Prerequesites & Installation

We strongly suggest using a GPU if speed is of necessity, as speedups of ~ 40x and more over CPU mode are typical. Nevertheless, the code will run on both CPU or GPU without modification. If you chose to use a GPU, make sure to use a NVIDIA model (these are the only cards supporting CUDA).

#### Prerequisites

** Python 2:
We recommend the use of Anaconda (https://www.continuum.io/downloads), especially for Windows users.

** Theano -- 

This is a GPU-toolkit which our code uses to build and train convolutional neural networks.
It is straight forward to get working with GPUs on Linux, but slightly harder on Windows

E.g. see: http://deeplearning.net/software/theano/install.html

A quick summary to install Theano for Windows users:
1) Install Anaconda (Python 2.7+ but not 3.x, x64)
2) [in console]  conda update conda
3) [in console]  conda update --all
4) [in console]  pip install Theano
5) [in console]  conda install mingw libpython

for GPU on Windows:
1) install MS Visual Studio (sadly the express version currently does not work)
2) install CUDA
3) add to Path: 

C:\Program Files (x86)\Microsoft Visual Studio 11.0\VC\bin      [or equivalent]
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\bin     [or equivalent]
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\libnvvp [or equivalent]

It is still tricky to get it to work, as Theano doesn't properly support Windows

** Linux/Windows: [optional] install NVIDIA cuDNN -- better convolution implementation for GPU users, large speed-gain for training and testing (dowload is free but requires registration)


** nibabel -- allows python code to load nifti files:
[in console]  pip install nibabel

** h5py -- file format 
1) [in console] conda install h5py or pip install h5py


#### Configuring Theano

In order to configue theano to automatically use the GPU, create a file named .theanorc (.theanorc.txt for Windows users) in your user's home directory and add the following three lines to it -- this is a minimalistic setup, many more options are available:

[global]
floatX=float32
device=gpu0

You can replace "gpu0" with e.g. "cpu", or select another gpu using "gpu1" (provided you have two GPUs), etc.

#### Where can I obtain the data used in the publication

https://www.nitrc.org/projects/ibsr
http://www.oasis-brains.org/app/action/BundleAction/bundle/OAS1_CROSS
http://loni.usc.edu/atlases/Atlas_Detail.php?atlas_id=12

If you use this data please cite the corresponding publications and comply with the indicated copyright regulations stated.

Please understand, that the brain tumor data set used in our publication cannot be made publicly available.


## III. Data preprocessing

Data pre-processing is not required, as long as the data does not contain artifacts/NaN entries etc. but it will most likely improve the results. The provided code automatically standardizes the data (zero mean, unit variance per volume), thus only nonlinear data pre-processing operations have any effect.

If you use data from different scanners that produce data with varying orientation it might be necessary to transform all your data to a common orientation. For instance, this might be achieved using fslreorient2std [http://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FSL]. 



## IV. Examples


#### Brain mask prediction using an already trained CNN
```
python deep3Dpredict.py --help
usage: deep3Dpredict.py [-h] -i I [I ...] [-o O] [-n N] [-c C] [-f F]
                        [-prob PROB] [-gridsize GRIDSIZE]

Main module to apply an already trained 3D-CNN to segment data

optional arguments:
  -h, --help          show this help message and exit
  -i I [I ...]        Any number and combination of paths to files or folders
                      that will be used as input-data for training the CNN
  -o O                output path for the predicted brain masks
  -n N                name of the trained/saved CNN model (can be either a
                      folder or .save file)
  -c C                Filter connected components: removes all connected
                      components but the largest two (i.e. background and
                      brain) [default=True]
  -f F                File saving format for predictions. Options are "h5",
                      "nifti", "numpy" [default=nifti]
  -prob PROB          save probability map as well
  -gridsize GRIDSIZE  size of CNN output grid (optimal: largest possible
                      divisor of the data-volume axes that still fits into GPU
                      memory). This setting heavily affects prediction times:
                      larger values are better. Values that are too large will
                      cause a failure due to too little GPU-memory.


python deep3Dpredict.py -n OASIS_ISBR_LPBA40__trained_CNN.save -i /home/share/brain_mask/__NEW__/ibsr_data/02/IBSR_02_ana.nii.gz -gridsize 16
 
```
#### Train a new CNN (with your data)
```
python deep3Dtrain.py --help
usage: deep3Dtrain.py [-h] -data DATA [DATA ...] -labels LABELS [LABELS ...]
                      [-lr LR] [-name NAME] [-convert_labels CONVERTLABELS]

Main module to train a 3D-CNN for segmentation

optional arguments:
  -h, --help            show this help message and exit
  -data DATA [DATA ...]
                        Any number and combination of paths to files or
                        folders that will be used as input-data for training
                        the CNN
  -labels LABELS [LABELS ...]
                        Any number and combination of paths to files or
                        folders that will be used as target for training the
                        CNN (values must be 0/1)
  -lr LR               initial learning rate (step size) for training the CNN (default: 10^(-5))
  -name NAME           name of the model (affects filenames) -- specify the
                        same name when using deep3Dtest.py
  -convert_labels CONVERTLABELS
                        if labels are not binary: this will convert values >1
                        to 1
  -data_clip_range [LOWER UPPER]
                        [optional] specify two values



For each data file you have to supply an associated label file. The file names should indicate their relationship such that alphabetical ordering results in a correct matching of the corresponding data and label files,
e.g. [vol1_data.nii.gz, vol2_data.nii.gz, ...] <-> [vol1_label.nii.gz, vol2_label.nii.gz, ...]

python deep3Dtrain.py -data data/ -labels labels/
```

## VI. FAQ

* How is it possible that the results reported in your paper differ from the results I obtain?

This can be due to several reasons. The presented method is not deterministic, i.e. there are a lot of random steps like initialization of the weights and order of the data cubes presented during training. Therefor, every trained CNN is somewhat unique. Further, the hardware and software libraries used also have an impact on speed and performance. With this software code we also included a pre-trained net (OASIS_ISBR_LPBA40__trained_CNN.save) that exhibits very good results on these three data sets.
The speed of training and prediction depends on several factors: the computer hardware (and GPU model), the used convolution backend (e.g. cuDNN and the specific version of it vs. the slower Theano convolution) and the "gridsize"  parameter (higher values are more efficient for prediction but require more GPU memory, the default value of 16 is a conservative choice)


* I have a problem running the code / found a bug, who should I contact?

send an email to: gur9000@outlook.com
