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

import numpy as np

import time


from   NNet_Core.NN_ConvNet import MixedConvNN
import NNet_Core.utilities as utilities
import helper_seg as helper



def data_augmentation_greyvalue(data, max_shift = 0.05, max_scale = 1.3, min_scale = 0.85, b_use_lesser_augmentation=0):
    """shift is applied in the range [-max_shift, max_shift]"""
    if b_use_lesser_augmentation:
         max_shift = 0.02
         max_scale = 1.1
         min_scale = 0.91
    sh = (0.5-np.random.random())*max_shift*2.
    scale = (max_scale - min_scale)*np.random.random() + min_scale
    return (sh+data*scale).astype("float32")


def train_net(cnn, patchCreator, LR_start , num_patches_per_batch = 1,
              momentum = 0.9, save_name="auto", b_no_test_set=False,
              b_use_ext_error=0,
              autosave_frequency_minutes=60, autosave_n_files=10,
              b_use_data_augmentation = False, b_use_lesser_augmentation = False):

    LR_end = LR_start / 10.   # this is ignored
    training_time_minutes = 0 # this is ignored
    LR_sched    = utilities.LR_scheduler(cnn, max_training_time_minutes=60*30, max_training_steps=None, LR_start=LR_start,
                                         automated_LR_scaling_enabled=True, automated_LR_scaling_magnitude=0.5,
                                         automated_LR_scaling_wait_steps=5000,
                                         automated_LR_scaling_max_LR_reduction_factor_before_termination=1000,
                                         automated_LR_scaling_minimum_n_steps_for_subsequent_reduction=4000,
                                         automated_kill_after_n_unchanged_steps=15000, automated_kill_if_bad_enabled=False,
                                         automated_kill_if_bad__time_of_decision_minutes=10,
                                         automated_kill_if_bad__killscore=-0.9)

    TimeControl = utilities.AutosaveControl(cnn, training_time_minutes, LR_start, LR_end,
                                            save_name= save_name,
                                            save_path = save_name,
                                            autosave_n_files=autosave_n_files,
                                            autosave_frequency_minutes=autosave_frequency_minutes)

    TimeControl.tick(0, force_save = 0)
    GLogger = utilities.Logger(save_file=save_name+"/LOG_"+save_name+'.txt')
    done_looping = False

    cnn.set_SGD_LR(LR_start)
    cnn.set_SGD_Momentum(momentum)

    batchsize=1

    NLL_history = []
    NLL_history_size = 500 #only the most recent 500 values are kept


    if num_patches_per_batch!=1:
        assert int(num_patches_per_batch)==num_patches_per_batch
        assert num_patches_per_batch>=2
        assert num_patches_per_batch<10000
        dlm = patchCreator.makeTrainingPatch(batchsize=batchsize)
        data , labels_proto = dlm[0], dlm[1]
        data = np.zeros((num_patches_per_batch,)+data.shape[1:],"float32")
        labels_proto = np.zeros((num_patches_per_batch,)+labels_proto.shape[1:], labels_proto.dtype)


    print "data_augmentation =",b_use_data_augmentation
    print


    nit=0# iteration number
#    trailing_mean_NLL=0.7


    training_mode = 1


    print "Trigger KeyboardInterrupt (Ctrl+C) to end training prematurely"
    while done_looping==False:
        try:
            while done_looping==False:

                if num_patches_per_batch==1:
                    dlm = patchCreator.makeTrainingPatch(batchsize=batchsize)

                    data , labels = dlm[0], dlm[1]#.flatten()

                    if b_use_data_augmentation:
                        data = data_augmentation_greyvalue(data, b_use_lesser_augmentation=b_use_lesser_augmentation)

                    if (labels.ndim==5 and labels.shape[-1]>=2):
                        labels=labels.reshape(-1,labels.shape[-1])
                    elif labels.shape[-1]>2:
                        labels=labels.flatten()
                    else:
                        labels = np.transpose( labels,(0,1,4,2,3))

                else:
                    for i in range(num_patches_per_batch):
                        dlm = patchCreator.makeTrainingPatch(batchsize=batchsize)
                        data_ , labels_ = dlm[0], dlm[1]#.flatten()
                        if b_use_data_augmentation:
                            data = data_augmentation_greyvalue(data, b_use_lesser_augmentation=b_use_lesser_augmentation)

                        data[i,...]=data_[0,...]
                        labels_proto[i,...]=labels_[0,...]


                    if labels_proto.ndim==5:
                        labels = labels_proto.reshape(-1,labels_proto.shape[-1])
                    else:
                        labels = labels_proto.flatten()



                nll = cnn.training_step(data, labels, mode=training_mode)
                
#                ret = cnn.debug_gradients_function(data,labels)
#                print 'len(ret)',len(ret)
#                for gr in ret:
#                    print np.min(gr), np.mean(np.abs(gr)), np.median(np.abs(gr)), np.max(gr)
#                print
                
                NLL_history.append(nll)
#                trailing_mean_NLL = 0.995* trailing_mean_NLL + 0.005* xnll

                nit+=1;

                if nit%10 == 0:
                    NLL_history = NLL_history[-NLL_history_size:]
                    xnll = np.mean(NLL_history)
                    TimeControl.tick(nit,
                                    additional_save_string = "" if autosave_frequency_minutes>1 else xnll,
                                    update_LR = False) #creates auto-saves too...
                   
                    done_looping = LR_sched.tick(nit, current_score = -xnll)

                    GLogger.log_and_print(["Iteration =",nit,"avg. NLL =", xnll])



        except KeyboardInterrupt:
            print "Training terminated via ctrl+C"
            break

#    cnn.SaveParameters("end_"+str(save_name)+".save")
    cnn.SaveParameters(save_name+"/end_"+str(save_name)+".save")

    GLogger.close()
    return 0



def Build3D(nnet_args, n_labels_per_batch = 300,  patch_depth = 1, actfunc='relu',
            bDropoutEnabled = False, notrain=0, input_to_cnn_depth = 1,
            override_data_set_filenames=None, num_patches_per_batch = 1,
            data_init_preserve_channel_scaling = 1, 
            data_clip_range = None,
            use_fragment_pooling = 0, auto_threshold_labels=False,
            gradient_clipping = True, bWeightDecay = True):
    """build net, load samples (patchCreator)"""


    _type = actfunc


    filter_sizes = nnet_args["filter_sizes"]
    pooling_factors = nnet_args["pooling_factors"]
    nof_filters = nnet_args["nof_filters"]

    try:
        if type("")==type(_type):
            assert 0
        _type[1]
    except:
        _type = (_type,)*len(nof_filters)



    patchCreator = helper.PatchCreator(filter_sizes, pooling_factors, n_labels_per_batch=n_labels_per_batch,
                                       override_data_set_filenames=override_data_set_filenames,
                                       data_init_preserve_channel_scaling=data_init_preserve_channel_scaling,
                                       data_clip_range = data_clip_range,
                                       use_max_fragment_pooling = use_fragment_pooling, 
                                       auto_threshold_labels=auto_threshold_labels,
                                       pad_last_dimension = not notrain)

    print 'Building CNN...'
    cnn = MixedConvNN( patchCreator.CNET_Input_Size, ImageDepth = input_to_cnn_depth, InputImageDimensions = 3,
                      bDropoutEnabled_ = bDropoutEnabled, bSupportVariableBatchsize=0, 
                      batchsize=num_patches_per_batch, verbose = 0, bWeightDecay = bWeightDecay)


    for i,nf,fs,pf in zip(range(len(nof_filters)),nof_filters, filter_sizes, pooling_factors):

        is_last_layer = (i==len(nof_filters)-1)
        cnn.addConvLayer( nf, fs, pooling_factor = pf, ActivationFunction=_type[i], ndim=3,
                         b_forceNoDropout = is_last_layer, bTheanoConv=1,
                         use_fragment_pooling = use_fragment_pooling,
                         dense_output_from_fragments = is_last_layer and use_fragment_pooling)

    cnn.use_fragment_pooling = use_fragment_pooling

    if gradient_clipping:
        cnn.enable_gradient_clipping(0.02)

    print "Compiling Output Functions"
    cnn.CompileOutputFunctions(b_isRegression = False,
                               b_ignore_pred_funct= not notrain, bUseModulatedNLL=0,
                               b_regression_with_margin=0, margin_reweighted_error=0,
                               override_training_loss_function=None)

    
    cnn.CompileDebugFunctions(gradients=0)
    print "done: Build3D()"
    return cnn, patchCreator




if __name__ == '__main__':
    print "please execute main_train.py instead!"


