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
import time
import utils.data_and_CV_handler as data_and_CV_handler


def predict(list_training_data, list_training_labels, list_test_data, 
         save_name, apply_cc_filtering, output_path, load_previous_save=False, 
         auto_threshold_labels=False, output_filetype = 'h5', save_prob_map = False, n_labels_pred_per_dim = 16):
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


    """
    import utils.Segmentation_trainer as Segmentation_trainer
    import utils.Segmentation_predictor as Segmentation_predictor
    assert len(list_training_data) == len(list_training_labels)
    
    save_name = filter_saves(save_name)
    
    patch_depth = 1
    use_fragment_pooling = 0

    # this number should be as large as possible to increase the speed-efficiency when making predictions
    # the only limit is the RAM of the GPU which will manifest as memory allocation errors
    
    n_labels_pred_per_dim =  n_labels_pred_per_dim #32


    # number of classes in the data set -  2 means binary classification.
    n_classes=2

    # CNN specification:
    nnet_args={}
    nnet_args["filter_sizes"]    = [4, 5, 5, 5, 5, 5, 5, 1]
    nnet_args["pooling_factors"] = [2, 1, 1,  1, 1, 1, 1, 1] # this indicates where max-pooling is used ( a value of 1 means no pooling)
#    nnet_args["nof_filters"]     = [1, 1, 1, 1, 1, 1, 1,   n_classes] # number of different filters in each layer:
    nnet_args["nof_filters"]     = [16, 24, 28, 34, 42, 50, 50,   n_classes] # number of different filters in each layer:


    bDropoutEnabled = 0
    num_patches_per_batch =  1 
    input_to_cnn_depth = patch_depth

    override_data_set_filenames = {"train_data":list_training_data,
                                   "test_data":list_test_data,
                                   "train_labels":list_training_labels
                                   }

    n_labels_per_batch = n_labels_pred_per_dim**(3)



    cnn, patchCreator = Segmentation_trainer.Build3D(nnet_args, n_labels_per_batch=n_labels_per_batch, notrain= True,
                                     bDropoutEnabled = bDropoutEnabled,
                                     patch_depth = patch_depth,
                                     input_to_cnn_depth=input_to_cnn_depth ,
                                     override_data_set_filenames=override_data_set_filenames,
                                     num_patches_per_batch=num_patches_per_batch,
                                     actfunc = "relu",
                                     data_init_preserve_channel_scaling=0, 
                                     use_fragment_pooling = use_fragment_pooling,
                                     auto_threshold_labels = auto_threshold_labels)

#    print 'WARNING: load loading save\n\n'*20
    cnn.LoadParameters(save_name)
    t0 = time.clock()
    
    
    if len(output_path) and output_path.replace('\\','/')[-1] != '/':
        output_path += '/'
    Segmentation_predictor.predict_all(cnn, patchCreator, apply_cc_filtering = apply_cc_filtering, 
                                       save_as = output_path, output_filetype = output_filetype,
                                       save_prob_map = save_prob_map)
    t1 = time.clock()
    print "Predicted all in",t1-t0,"seconds"




def filter_saves(path_or_file):
    candidates = findall(path_or_file)
    matches = []
    for c in candidates:
        if '.save' in c:
            matches.append(c)
            if 'end_' in c:
                return c
    if len(matches)==0:
        raise ValueError('The provided save file/directory does not contain any saved model (file ending in .save)')
    return matches[-1]


def findall(paths):
    """
    locate and return all files in the paths (list of directory/file names)
    """
    rlist=[]
    for x in paths:
        rlist += data_and_CV_handler.list_files(x) if data_and_CV_handler.os.path.isdir(x) else [x]
    return rlist


def tolist(x):
    return x if isinstance(x,list) else [x]


def main():
#    default_data = 'U:/DATA/IBSR/data'
    
    parser = argparse.ArgumentParser(description='Main module to apply an already trained 3D-CNN to segment data')
#    parser.add_argument('-i', default=default_data, type=str, nargs='+', help='Any number and combination of paths to files or folders that will be used as input-data for training the CNN')
    parser.add_argument('-i', type=str, nargs='+', help='Any number and combination of paths to files or folders that will be used as input-data for training the CNN', required=True)

    parser.add_argument('-o', default='predictions/', type=str, help='output path for the predicted brain masks')
    parser.add_argument('-n', default='OASIS_ISBR_LPBA40__trained_CNN.save', type=str,  help='name of the trained/saved CNN model (can be either a folder or .save file)')
    parser.add_argument('-c', default=True, type=bool,  help='Filter connected components: removes all connected components but the largest two (i.e. background and brain) [default=True]')
    parser.add_argument('-f', default='nifti', type=str,  help='File saving format for predictions. Options are "h5", "nifti", "numpy" [default=nifti]')
    parser.add_argument('-prob', default=1, type=bool,  help='save probability map as well')
    parser.add_argument('-gridsize', default=16, type=int,  help='size of CNN output grid (optimal: largest possible divisor of the data-volume axes that still fits into GPU memory). This setting heavily affects prediction times: larger values are better. Values that are too large will cause a failure due to too little GPU-memory.')
    
    
    args = parser.parse_args()
    
    data = findall(tolist(args.i))
    
    assert len(data)>0, 'Could not find the data. Please either pass all paths to the individual files or place them in a single folder and pass the path to this folder as "-i" argument'
    assert args.f in ['nifti', 'h5', 'numpy']
    
    predict(list_training_data=[],    
         list_training_labels=[],    
         list_test_data=data,            
         save_name=tolist(args.n)[0],
         apply_cc_filtering = bool(args.c),
         output_path = str(args.o),
         output_filetype = args.f,
         save_prob_map = args.prob,
         n_labels_pred_per_dim = args.gridsize)



if __name__ == '__main__':
    
    main()









