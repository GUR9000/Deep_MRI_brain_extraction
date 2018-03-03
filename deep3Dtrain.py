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

import argparse
import utils.data_and_CV_handler as data_and_CV_handler
import numpy as np


def train(list_training_data, list_training_labels, list_test_data, 
         save_name, learning_rate, load_previous_save=False, auto_threshold_labels=False,
         data_clip_range = None, network_size_factor = 1):
    """This is the runner for the brain mask prediction project.
        It will either train the CNN or load the trained network and predict the test set.

    Input:
    --------


        list_training_data:

            lists of strings that specify the locations of training data files (each file must contain one 3D or 4D-volume of type float32; the 4th dimension containins the channels)

        list_training_labels:

            lists of strings that specify the locations of training labels files (each file must contain one 3D or 4D-volume of type int; the 4th dimension containins the channels)

        list_test_data:

            lists of strings that specify the locations of test data files (each file must contain one 3D or 4D-volume of type float32; the 4th dimension containins the channels)

        save_name:

            str, name of the folder (and files) for saving/loading the trained network parameters
        learning_rate:
            float, initial learning rate used for training


    """
    import utils.Segmentation_trainer as Segmentation_trainer
    
    assert len(list_training_data) == len(list_training_labels), "The total number of data files and label files differs: "+str(len(list_training_data))+' vs '+str(len(list_training_labels))
    autosave_frequency_minutes=60 # saves a copy of the CNN's parameters on the disk every X minutes
    autosave_n_files=3


    # if anything weird happens during training (e.g. loss increases by more than a factor of two), try a lower value
    Initial_learning_rate = learning_rate

    # probably no need to change these values:
    momentum = 0.9
    patch_depth = 1
    use_fragment_pooling = 0
    init_scale_factor = 3.

    #  ~~~~~  REGULARIZATION  ~~~~~
    b_use_data_augmentation = 0
    gradient_clipping       = 0
    bWeightDecay            = 0
    bDropoutEnabled         = 0
    
    
    n_labels_pred_per_dim = 5 # was 4

    # number of classes in the data set - e.g. 2 means binary classification.
    n_classes=2

    # This is where the CNN is specified:
    #
    # the first value in each of the lists corresponds to the first 3D-convolutional layer, the second value(s) to the second layer, etc.
    nnet_args={}
    # this defines the filter shapes in the 3D-conv. layers - e.g. a value of 5 will result in
    # a set of filters of shape 5 x 5 x 5 (i.e. each filter has 125 parameters) in the corresponding layer
    nnet_args["filter_sizes"]    = [4, 5, 5, 5, 5, 5, 5, 1]
    # this indicates where max-pooling is used ( a value of 1 means no pooling)
    nnet_args["pooling_factors"] = [2, 1, 1,  1, 1, 1, 1, 1]
    # this specifies the number of different filters in each layer:
    nnet_args["nof_filters"]     = [16, 24, 28, 34, 42, 50, 50,   n_classes]
    
    nnet_args["nof_filters"] = [int(np.ceil(network_size_factor * x)) for x in nnet_args["nof_filters"][:-1]] + [nnet_args["nof_filters"][-1]]
        

    num_patches_per_batch = 4  # a better setting is e.g. 4 if you have enough GPU memory or use the CPU, otherwise try 2
    
    input_to_cnn_depth = patch_depth #use 2 if you enable the pseudo-recursion



    override_data_set_filenames = {"train_data":list_training_data,
                                   "test_data":list_test_data,
                                   "train_labels":list_training_labels}

    n_labels_per_batch = n_labels_pred_per_dim**(3)



    cnn, patchCreator = Segmentation_trainer.Build3D(nnet_args, n_labels_per_batch=n_labels_per_batch, notrain= False,
                                     bDropoutEnabled = bDropoutEnabled,
                                     patch_depth = patch_depth,
                                     input_to_cnn_depth=input_to_cnn_depth ,
                                     override_data_set_filenames=override_data_set_filenames,
                                     num_patches_per_batch=num_patches_per_batch,
                                     actfunc = "relu",
                                     data_init_preserve_channel_scaling=0, 
                                     data_clip_range = data_clip_range,
                                     use_fragment_pooling = use_fragment_pooling,
                                     auto_threshold_labels = auto_threshold_labels,
                                     gradient_clipping = gradient_clipping,
                                     bWeightDecay = bWeightDecay)

    cnn.randomize_weights(scale_w = init_scale_factor)


    if load_previous_save:
        try:
            cnn.LoadParameters(save_name+"/end_"+str(save_name)+".save")
            print 'load_previous_save:: found and loaded saved parameters'
        except:
            print 'load_previous_save:: found no saved parameters or CNN is incompatible'
    Segmentation_trainer.train_net(cnn, patchCreator, num_patches_per_batch=num_patches_per_batch,
                                   LR_start=Initial_learning_rate, momentum=momentum,
                                   save_name=save_name, b_no_test_set=True, b_use_ext_error=0,
                                   autosave_frequency_minutes=autosave_frequency_minutes,
                                   autosave_n_files=autosave_n_files,
                                   b_use_data_augmentation = b_use_data_augmentation)





def findall(paths):
    rlist=[]
    for x in paths:
        rlist += data_and_CV_handler.list_files(x, contains_not='.hdr') if data_and_CV_handler.os.path.isdir(x) else [x]
    return rlist


def tolist(x):
    return x if isinstance(x,list) else [x]


def main():
    parser = argparse.ArgumentParser(description='Main module to train a 3D-CNN for segmentation')
    
    parser.add_argument('-data', required=True, type=str, nargs='+', help='Any number and combination of paths to files or folders that will be used as input-data for training the CNN')
    parser.add_argument('-labels', required=True, type=str, nargs='+', help='Any number and combination of paths to files or folders that will be used as target for training the CNN (values must be 0/1)')
    
    parser.add_argument('-lr', type=float, default=1e-5,  help='initial learning rate (step size) for training the CNN')
    parser.add_argument('-name', default='deep3Dtrain_model_1', type=str,  help='name of the model (affects filenames) -- specify the same name when using deep3Dtest.py')
    parser.add_argument('-convert_labels', default=1, type=int, nargs=1, help='if labels are not binary: this will convert values >1 to 1')
    
    parser.add_argument('-data_clip_range', default=None, type=float, nargs=2, help='[Mostly for single-channel data] Clip all pixel-values outside of the given range (important if values of volumes have very different ranges!)')
    
    parser.add_argument('-CNN_width_scale', default=1., type=float, help='Scale factor for the layer widths of the CNN; values larger than 1 will increase the total network size beyond the default size, but be careful to not exceed your GPU memory.')
    
    
    args = parser.parse_args()
    
    #print args
    data = findall(tolist(args.data))
    labels = findall(tolist(args.labels))
    load_previous_save = False


    train(list_training_data=data,    
         list_training_labels=labels,    
         list_test_data=[],            
         save_name=tolist(args.name)[0],
         learning_rate=args.lr,
         load_previous_save = load_previous_save,
         auto_threshold_labels=tolist(args.convert_labels)[0],
         data_clip_range = args.data_clip_range,
         network_size_factor = float(args.CNN_width_scale))








if __name__ == '__main__':
    main()
    









