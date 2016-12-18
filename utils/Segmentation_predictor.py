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
(Gregor Urban)

5.11.2013 (first functional rewrite)
31.3.2014 (fork from seg_main.py)
30.3.2016 (small changes for code-publishing)

"""


import time
import numpy as np
import NNet_Core.utilities as utilities
import file_reading
import helper_seg as helper
import scipy.ndimage as ndimage



def predict_all(cnn, patchCreator, apply_cc_filtering, set_selection = 'all', 
                save_as = None, output_filetype = 'h5', save_prob_map = False):
    assert set_selection in ["test","train","all"]
    assert output_filetype in ['nifti', 'h5', 'numpy']
            
    if len(save_as) and save_as[-1] != '/':
        save_as += '/'
            
    for input_to_cnn_depth in [4]:
        if hasattr(cnn,"output")== False or cnn.output==None:
            import theano
            print "compiling output function"
            cnn.output   = theano.function([cnn.x], cnn.layers[-1].class_probabilities_realshape )

        
        timings=[]
        run_Net_on_multiple(patchCreator, input_to_cnn_depth=input_to_cnn_depth, cnn = cnn, 
                            str_data_selection= set_selection,  
                            save_file_prefix = save_as, 
                            apply_cc_filtering = apply_cc_filtering,
                            output_filetype = output_filetype, save_prob_map = save_prob_map)

        

    print



def runNetOnSlice(cnn, patchCreator, DATA, n_classes=0, channel2_data=None):
    """ input: data block, out: data block with (dense) class probabilities of size cnn.number_of_labeled_points_per_dim * product(pool_sizes).
    There is an offset relative to the input data."""
    networkoutput_fnct = cnn.output if cnn.output else cnn.class_probabilities

    CNET_stride    = patchCreator.CNET_stride if cnn.use_fragment_pooling==0 else 1
    ImgInputSize = patchCreator.CNET_Input_Size

    pred_size = patchCreator.number_of_labeled_points_per_dim * (1 if cnn.use_fragment_pooling==0 else patchCreator.CNET_stride)#CNET_stride*imgshape[-1]

    if n_classes<=0:
        n_classes = cnn.layers[-1].number_of_filters

    if not isinstance(CNET_stride*pred_size, list) and not isinstance(CNET_stride*pred_size, tuple):
        CNET_stride = np.asarray((CNET_stride,)*3)
        pred_size   = np.asarray((pred_size,)*3)
    if not isinstance(ImgInputSize, list) and not isinstance(ImgInputSize, tuple):
        ImgInputSize = np.asarray((ImgInputSize,)*3)

    pred = np.zeros( (n_classes, )+tuple(CNET_stride*pred_size),dtype=np.float32) # output


    if DATA.ndim==3 or DATA.ndim==4:
        if DATA.ndim==3:
            DATA = DATA.reshape(1,np.shape(DATA)[0],1,np.shape(DATA)[1],np.shape(DATA)[2])
        else:
            if DATA.shape[3]==min(DATA.shape):
                DATA = np.transpose(DATA,(0,3,1,2))
                DATA = DATA.reshape((1,)+DATA.shape)
            else:
                print ("runNetOnSlice:WARNING: assuming channel size is:"+str(DATA.shape[3])+" of data:"+str(DATA.shape)+"\n")*5
                DATA = np.transpose(DATA,(0,3,1,2))
                DATA = DATA.reshape((1,)+DATA.shape)

        if channel2_data!=None:
            proxy_input = np.zeros((DATA.shape[0],ImgInputSize[0],2 if channel2_data.ndim==3 else (channel2_data.shape[1]+1),ImgInputSize[1],ImgInputSize[2]),dtype="float32")#this will hold the original data + a second channel

        for x in range(CNET_stride[0]):
            for y in range(CNET_stride[1]):
                for z in range(CNET_stride[2]):
                    if channel2_data==None:
                        rr = networkoutput_fnct(DATA[:, x:x+ImgInputSize[0], :, y:y+ImgInputSize[1], z:z+ImgInputSize[2]])#[:,0].reshape((pred_size,pred_size)) #shape is e.g. (196,2) and membrane prob. are in [:,0]
                    else:
                        proxy_input[:,:,0,:,:] = DATA[:, x:x+ImgInputSize[0], 0, y:y+ImgInputSize[1], z:z+ImgInputSize[2]]
                        if channel2_data.ndim==3:
                            proxy_input[0,:,1,:,:] = channel2_data[x:x+ImgInputSize[0],  y:y+ImgInputSize[1], z:z+ImgInputSize[2]]
                        else:
                            proxy_input[0,:,1:,:,:] = channel2_data[x:x+ImgInputSize[0], :, y:y+ImgInputSize[1], z:z+ImgInputSize[2]]
                        rr = networkoutput_fnct(proxy_input)


                    for c in range(n_classes):
                        if len(rr.shape)==5:
                            pred[c, x::CNET_stride[0], y::CNET_stride[1], z::CNET_stride[2]] = rr[0,:,c,:,:].reshape((pred_size[0], pred_size[1], pred_size[2]))
                        else:
                            pred[c, x::CNET_stride[0], y::CNET_stride[1], z::CNET_stride[2]] = rr[:,c].reshape((pred_size[0], pred_size[1], pred_size[2]))
    else:#2D
        for x in range(CNET_stride[0]):
            for y in range(CNET_stride[1]):
                rr = networkoutput_fnct(DATA[:,:,x:x+ImgInputSize[0],y:y+ImgInputSize[1]])
                for c in range(n_classes):
                    pred[c, x::CNET_stride[0], y::CNET_stride[1]] = rr[:,c].reshape((pred_size[0],pred_size[1]))
    return pred








def run_Net_on_Block(cnn, DATA, patchCreator, bool_predicts_on_softmax=None,
                     added_data="auto_detect", second_input_data=None, rescale_predictions_to_max_range = True):
    """ :DATA: is the original input data. This function will pad it!
        :second_input_data: is assumed to apply to the LABELS/PREDICTIONS and not to the input of the NNet"""

    if hasattr(cnn,"output")== False or cnn.output==None:
        import theano
        assert bool_predicts_on_softmax!=None,"must specify <bool_predicts_on_softmax>"
        if bool_predicts_on_softmax:
            print "compiling output function (class_probabilities_realshape)"
            print "WARNING: if this is a MALIS net, then this is WRONG!!!\n"*3
            print
            cnn.output   = theano.function([cnn.x], cnn.layers[-1].class_probabilities_realshape )
        else:
            print "compiling output function (direc neuron output)"
            cnn.output   = theano.function([cnn.x], cnn.layers[-1].output )

    CNET_stride    = patchCreator.CNET_stride if cnn.use_fragment_pooling==0 else 1
    input_s = cnn.input_shape[-1] + CNET_stride - 1 # input size for runNetOnSlice()

    n_classes = cnn.layers[-1].number_of_filters
    offset_l = patchCreator.CNET_labels_offset[0]
    offset_r = offset_l + input_s

    if added_data=="auto_detect":
        try:
            added_data = patchCreator._input_adder.data_add
            s = added_data.shape
        except:
            added_data = None

    if min(DATA.shape)!=DATA.shape[-1]:
        #print "Warning: channel dimension is seemingly not last dimension! Transposing axes... (",DATA.shape,")"
        DATA = helper.make_channel_axis_last_axis(DATA)

    target_labels_per_dim = DATA.shape[:3]
    if patchCreator.padded_once:
        target_labels_per_dim -= offset_l + offset_r

    if added_data!=None:
        #assuming channels are last dim.
        print "fusing:",DATA.shape,"+",s
        if DATA.ndim==added_data.ndim:
            newd = np.zeros( DATA.shape[:-1]+(s[-1]+DATA.shape[-1],),DATA.dtype)
            newd[...,:DATA.shape[-1]]=DATA
            newd[...,DATA.shape[-1]:]=added_data
        else:
            newd = np.zeros( DATA.shape+(s[-1]+1,),DATA.dtype)
            newd[...,0]=DATA
            newd[...,1:]=added_data
        DATA = newd
    
    
    print "Predicting data of shape:",DATA.shape

    if patchCreator.padded_once==False:
        DATA = helper.greyvalue_data_padding(DATA, offset_l, offset_r)

    assert second_input_data is None
    if second_input_data is not None:
        second_input_data_2 = np.zeros(np.asarray(second_input_data.shape)+offset_l+offset_r,second_input_data.dtype)
        s=second_input_data.shape
        assert len(s)==3
        second_input_data_2[offset_l:offset_l+s[0],offset_l:offset_l+s[1],offset_l:offset_l+s[2]]=second_input_data
        second_input_data=second_input_data_2
        print "second_input_data (new) ~",second_input_data.shape

    ret_size_per_runonslice = cnn.layers[-1].output_shape[-1]*CNET_stride

    n_runs_p_dim = [int(round(target_labels_per_dim[i] / ret_size_per_runonslice)) for i in [0,1,2]]#n_iterations# glue this many fully-predicted neighbouring blocks together

    for i in [0,1,2]:
        if (ret_size_per_runonslice*n_runs_p_dim[i] < target_labels_per_dim[i]):
            n_runs_p_dim[i] = n_runs_p_dim[i] + 1

    ret_3d_cube = np.zeros( (n_classes,)+tuple(   DATA.shape[:3]   ) , dtype="float32")

    for i in range(n_runs_p_dim[0]):
        print "COMPLETION =", 100.*i/n_runs_p_dim[0],"%"
        for j in range(n_runs_p_dim[1]):
            for k in range(n_runs_p_dim[2]):
                offset = (ret_size_per_runonslice*i,ret_size_per_runonslice*(j),ret_size_per_runonslice*k)
                if DATA.ndim==4:
                    daa = DATA[offset[0]:input_s+offset[0],offset[1]:input_s+offset[1],offset[2]:input_s+offset[2],:]
                else:
                    daa = DATA[offset[0]:input_s+offset[0],offset[1]:input_s+offset[1],offset[2]:input_s+offset[2]]
                ret = runNetOnSlice(cnn, patchCreator, daa, channel2_data = None) 
                assert ret.ndim==4 #standard case
                ret_3d_cube[:,offset[0]:ret_size_per_runonslice+offset[0], offset[1]:ret_size_per_runonslice+offset[1], offset[2]:ret_size_per_runonslice+offset[2]] = ret#[0,...]
    sav = ret_3d_cube[:,:target_labels_per_dim[0],:target_labels_per_dim[1],:target_labels_per_dim[2]]
    
    sav = sav[1] # pick class 1
    if rescale_predictions_to_max_range:
        sav = (sav-sav.min())/(sav.max()+1e-7) # rescale the predicted probabilities (assuming that there is *something* positive in the data, otherwise this is quite bad...)
    
    return sav










def remove_small_conneceted_components(raw):
    """
    All but the two largest connected components will be removed
    """
    data = raw.copy()
    # binarize image
    data[data>0.5] = 1
    cc, num_components = ndimage.label(np.uint8(data))
    cc=cc.astype("uint16")
    vals = np.bincount(cc.ravel())
    sizes = list(vals)
    try:
        second_largest = sorted(sizes)[::-1][1]       
    except:
        return raw.copy()
    data[...] = 0
    for i in range(0,len(vals)):
        # 0 is background
        if sizes[i]>=second_largest:
            data[cc==i] = raw[cc==i]
    return data




def run_Net_on_multiple(patchCreator, input_to_cnn_depth=1, cnn = None, 
                        str_data_selection="all", save_file_prefix="", 
                        apply_cc_filtering = False, output_filetype = 'h5', save_prob_map = False):
    """ run runNetOnSlice() on neighbouring blocks of data.
        if opt_cnn is not none, it should point to a CNN / Net that will be used.
        if patchCreator contains a list of 3D data blocks (patchCreator.second_input_data) then it will be used as second input to cnn.output()
    """
    assert str_data_selection in ["all", "train", "test"]
    MIN = 0 if str_data_selection in ["all", "train"] else patchCreator.training_set_size
    MAX = patchCreator.training_set_size if str_data_selection =="train" else len(patchCreator.data)
    
    second_input_data = None
    
    DATA = patchCreator.data
    timings=[]
#    if hasattr(patchCreator,"second_input_data"):
#        second_input_data = patchCreator.second_input_data[opt_list_index]
    
    for opt_list_index in range(MIN, MAX):
        
        print "-"*30
        print "@",opt_list_index+1,"of max.",len(patchCreator.data)
        postfix = "" if opt_list_index==None else "_" + utilities.extract_filename(patchCreator.file_names[opt_list_index])[1] if isinstance(patchCreator.file_names[0], str) else str(patchCreator.file_names[opt_list_index]) if not isinstance(patchCreator.file_names[opt_list_index], tuple) else utilities.extract_filename(patchCreator.file_names[opt_list_index][0])[1]
        if opt_list_index is not None:
            is_training = "_train" if (opt_list_index < patchCreator.training_set_size) else "_test"
        else:
            is_training=""
        this_save_name = save_file_prefix+"prediction"+postfix+"_"+is_training
        
        t0 = time.clock()
        sav = run_Net_on_Block(cnn, DATA[opt_list_index], patchCreator, bool_predicts_on_softmax=1,
                               second_input_data = second_input_data) #this one does all the work
        
        t1 = time.clock()
        timings.append(t1-t0)
        if apply_cc_filtering:
            sav = remove_small_conneceted_components(sav)
            sav = 1 - remove_small_conneceted_components(1 - sav)
        
        save_pred(sav, this_save_name, output_filetype, save_prob_map)
    
    print 'timings (len',len(timings),')',np.mean(timings),'+-',np.std(timings)
    return None




def save_pred(prediction, this_save_name, output_filetype, save_prob_map):
    sav = prediction
    if save_prob_map:
        if output_filetype == 'h5':
            file_reading.save_h5(this_save_name+'.h5',sav)
        elif output_filetype == 'nifti':
            file_reading.save_nifti(this_save_name+'.nii.gz',sav)
        elif output_filetype == 'numpy':
            file_reading.mkdir(this_save_name+'.npy')
            np.save(this_save_name, sav)
        else:
            raise NotImplementedError(output_filetype)
    
    sav = (sav>0.5).astype('int8')
    
    if output_filetype == 'h5':
        file_reading.save_h5(this_save_name+'_mask.h5',sav)
    elif output_filetype == 'nifti':
        file_reading.save_nifti(this_save_name+'_mask.nii.gz',sav)
        
        
    elif output_filetype == 'numpy':
        file_reading.mkdir(this_save_name+'_mask.npy')
        np.save(this_save_name+'_mask.npy', sav)
    else:
        raise NotImplementedError(output_filetype)

    print "File saved as:",this_save_name
    
    return 0








#def run_Net_on_multiple__recursive(patchCreator, input_to_cnn_depth=1, cnn = None, opt_list_index=None,
#                        opt_list_predict_all_data=True, str_data_selection="all",
#                        opt_random_ID=None, DATA=None, save_file_prefix="", timings=[], 
#                        apply_cc_filtering = False, output_filetype = 'h5', save_prob_map = False):
#    """ run runNetOnSlice() on neighbouring blocks of data.
#        if opt_cnn is not none, it should point to a CNN / Net that will be used.
#        if patchCreator contains a list of 3D data blocks (patchCreator.second_input_data) then it will be used as second input to cnn.output()
#    """
#    
#    second_input_data = None
#    if DATA==None:
#        if type(patchCreator.data)!=type([]) and (patchCreator.data.ndim)>=3:
#            DATA = patchCreator.data
#        elif type(patchCreator.data)==type([]):
#            if opt_list_index==None:
#                assert str_data_selection in ["all", "train", "test"]
#                opt_list_index=0 if str_data_selection in ["all", "train"] else patchCreator.training_set_size
#            if ((opt_list_index>=len(patchCreator.data)) and (str_data_selection in  ["all", "test"])) or ((opt_list_index>=patchCreator.training_set_size) and (str_data_selection =="train")):
#                print "Done. Final index is",opt_list_index
#                return 0
#            DATA = patchCreator.data[opt_list_index]
#            if hasattr(patchCreator,"second_input_data"):
#                second_input_data = patchCreator.second_input_data[opt_list_index]
#        else:
#            print "\nError: Unsupported input data; shape is",patchCreator.data.shape
#    else:
#        if DATA.ndim==5:
#            assert DATA.shape[0]==1
#            DATA=DATA[0]
#    print "-"*30
#    print "@",opt_list_index+1,"of max.",len(patchCreator.data)
#    if opt_random_ID==None:
#        opt_random_ID = str(np.random.randint(1e7,1e8-1))
#
#    postfix = "" if opt_list_index==None else "_" + utilities.extract_filename(patchCreator.file_names[opt_list_index])[1] if type(patchCreator.file_names[0])==type("blubb") else str(patchCreator.file_names[opt_list_index]) if type(patchCreator.file_names[opt_list_index])!=type((1,)) else utilities.extract_filename(patchCreator.file_names[opt_list_index][0])[1]
#    if opt_list_index is not None:
#        is_training = "_train" if (opt_list_index < patchCreator.training_set_size) else "_test"
#    else:
#        is_training=""
#    this_save_name = save_file_prefix+"prediction"+postfix+"_"+is_training
#    
#
#    t0 = time.clock()
#    sav = run_Net_on_Block(cnn, DATA, patchCreator, bool_predicts_on_softmax=1,
#                           second_input_data = second_input_data) #this one does all the work
#    
#    
#    t1 = time.clock()
#    timings.append(t1-t0)
#    print 'timings (len',len(timings),')',np.mean(timings),'+-',np.std(timings)
#    if apply_cc_filtering:
#        sav = remove_small_conneceted_components(sav)
#        sav = 1 - remove_small_conneceted_components(1 - sav)
#                
#    if save_prob_map:
#        if output_filetype == 'h5':
#            file_reading.save_h5(this_save_name+'.h5',sav)
#        elif output_filetype == 'nifti':
#            file_reading.save_nifti(this_save_name+'.nii.gz',sav)
#        elif output_filetype == 'numpy':
#            file_reading.mkdir(this_save_name+'.npy')
#            np.save(this_save_name, sav)
#        else:
#            raise NotImplementedError(output_filetype)
#    
#    sav = (sav>0.5).astype('int8')
#    
#    if output_filetype == 'h5':
#        file_reading.save_h5(this_save_name+'_mask.h5',sav)
#    elif output_filetype == 'nifti':
#        file_reading.save_nifti(this_save_name+'_mask.nii.gz',sav)
#        
#        
#    elif output_filetype == 'numpy':
#        file_reading.mkdir(this_save_name+'_mask.npy')
#        np.save(this_save_name+'_mask.npy', sav)
#    else:
#        raise NotImplementedError(output_filetype)
#
#    print "File saved as:",this_save_name
#
#    del DATA
#    del sav
#
#    if opt_list_predict_all_data==True and opt_list_index!=None:
#        return run_Net_on_multiple(patchCreator, input_to_cnn_depth=input_to_cnn_depth, cnn = cnn, opt_list_index=opt_list_index+1, 
#                                   str_data_selection=str_data_selection, opt_list_predict_all_data=opt_list_predict_all_data, 
#                                   opt_random_ID=opt_random_ID, save_file_prefix=save_file_prefix, output_filetype=output_filetype,
#                                   save_prob_map=save_prob_map, apply_cc_filtering = apply_cc_filtering)
#    return None






if __name__ == '__main__':
    print "please execute main_train.py instead!"


