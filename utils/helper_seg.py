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


import os,sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'NNet_Core'))


import numpy as np
import random
import itertools as it
import file_reading



def outputsize_after_convpool(img,filt,pool):
    #get output size after applying len(filt) many filters of shape filt
    #input: img = size of full image, filt = list of filters (first to last)
    if len(filt)==1:
        return int(1.0/pool[0]*(img-filt[0]+1))
    return outputsize_after_convpool(int(1.0/pool[0]*(img-filt[0]+1)),filt[1:],pool[1:])



def recField(filt,pool,img=1):
    """get receptive field of last neuron, when applying filter and then max-pooling for each layer"""
    #recursion, starting with receptive field of last neuron relative to itself (which is 1)
    if len(filt)==1:
        return (pool[0]*img+filt[0]-1)
    return recField(filt[:-1],pool[:-1],(pool[-1]*img+filt[-1]-1))



def PredictionsOffset(filter_size,pooling_factor):
    """ offset from left/top of image to where the first label is located (i.e. the center of the receptive field of the prediciton for this point"""
    return int((recField(filter_size,pooling_factor)-1.)/2.0)


def PredictionStride(pooling_factor):
    """
    in fact this will be the distance between adjacent labels that are predicted (in one pass of the network)
    the network thus needs PredictionStride()**2 passes to classify one complete 2D-image (except for the borders if you don't mirror them)
    """
    return np.product(pooling_factor)



def PredictMaximumInputSize(INPUT_img_size, filter_sizes, pooling_factors):
    """
    e.g. input size is 512 but labels will be predicted only on 510 image!
    image size can be reduced in steps of <PredictionStride(pooling_factors) === np.product(pooling_factor)>
    """
    workon = INPUT_img_size - recField(filter_sizes,pooling_factors)
    stride = PredictionStride(pooling_factors)

    #print "input image size must be", (INPUT_img_size - (workon % stride))
    return int(INPUT_img_size - (workon % stride))




def make_channel_axis_last_axis(DATA):
    assert DATA.ndim==4
    nn = np.argmin(DATA.shape)
    return np.transpose(DATA ,  tuple([x for x in range(4) if x != nn] + [nn]) )


def make_channel_axis_second_axis(DATA):
    assert DATA.ndim==4
    nn = np.argmin(DATA.shape)
    other = [x for x in range(4) if x != nn]
    return np.transpose(DATA ,  tuple(other[:1] + [nn] + other[1:]) )



def greyvalue_data_padding(DATA, offset_l, offset_r):
    assert DATA.ndim==4
    foldback = False
    if np.argmin(DATA.shape)!=3:#,'channel axis must be the last one!'
        foldback=1
        DATA = make_channel_axis_last_axis(DATA)
    avg_value = 1./6.*(np.mean(DATA[0])+np.mean(DATA[:,0])+np.mean(DATA[:,:,0])+np.mean(DATA[-1])+np.mean(DATA[:,-1])+np.mean(DATA[:,:,-1]))
    sp = DATA.shape
    axis=[0,1,2]
    
    dat = avg_value * np.ones( (sp[0]+offset_l+offset_r if 0 in axis else sp[0], sp[1]+offset_l+offset_r if 1 in axis else sp[1], sp[2]+offset_l+offset_r if 2 in axis else sp[2]) + tuple(sp[3:]), dtype="float32")
    dat[offset_l*(0 in axis):offset_l*(0 in axis)+sp[0], offset_l*(1 in axis):offset_l*(1 in axis)+sp[1], offset_l*(2 in axis):offset_l*(2 in axis)+sp[2]] = DATA.copy()
    
    if foldback:
        dat = make_channel_axis_second_axis(dat)
    return dat



def pad_data(x, n_padding, mode):
    ''' padding will be added to the last axis on both front and end (i.e. size increases by 2 * n_padding
    
    mode:
        
        constant or mean'''
    pad = [(0, 0) for i in range(x.ndim-1)]+[(n_padding, n_padding)]
    return np.pad(x, pad, mode=mode)
    
    
    


class PatchCreator():
    """
    <INPUT_img_size> must be the output of PredictMaximumInputSize() !

    use <training_image_reduction_factor> to reduce the size of training images (= mini-batches)

    The last <number_of_images_test_set> images are test data

    """
    def __init__(self, filter_size, pooling_factor,  
                 n_labels_per_batch=10, 
                 override_data_set_filenames=None, 
                 data_init_preserve_channel_scaling=0, 
                 data_clip_range = None,
                 use_max_fragment_pooling = False,
                 auto_threshold_labels = False,
                 pad_last_dimension = False,
                 padding_margin = 10):
        """ filter_size and pooling_factor are lists (if multilayer)
            
            
            pad_last_dimension:
                
                True/False; necessary when training data's last channel is smaller than the CNN input window. Will add <padding_margin> more pixels in total than the required minimum.
        """


        """
        (30, 3, 3)
        min
        [[  7 142 125] #low, high, difference #x
         [ 12 181 150]#y
         [  0 132 123]]#z
         max
        [[ 58 189 152] #-> 153 pixels
         [ 60 224 174] #> 175 pixels if you use x-z-slices
         [ 86 219 145]]
         mean
        [[  19.03333333  158.06666667  139.03333333]
         [  36.03333333  200.5         164.46666667]
         [  30.73333333  163.7         132.96666667]]

        """
        self.ndim =3
        b_shuffle_data = True

        self.training_set_size = None


        assert not (type(override_data_set_filenames)!=type([]) and type(override_data_set_filenames)!=type({1:0}))
        
        
        self.CNET_real_imagesize = 256 # only valid for this set  np.shape(self.data)[1]

        best = 1
        #find best matching input size (such that n_labels_per_batch is reached)
        for i in range(200):
            input_size = PredictMaximumInputSize(self.CNET_real_imagesize *  0.005*i, filter_size, pooling_factor)
            n_lab_p_dim = outputsize_after_convpool(input_size, filter_size[:-1],pooling_factor[:-1])
            if n_labels_per_batch <= n_lab_p_dim**self.ndim:
                best = i * 0.005
                break

        self.CNET_Input_Size = PredictMaximumInputSize(self.CNET_real_imagesize * best, filter_size, pooling_factor)
#        print "PatchCreator::CNET_Input_Size = ",self.CNET_Input_Size#," down from (real size):",self.CNET_real_imagesize
#        print 'Receptive field =',recField(filter_size[:-1], pooling_factor[:-1]),'^ 2'

        offs = PredictionsOffset(filter_size,pooling_factor)
#        print "PredictionsOffset() =",offs

        self.CNET_labels_offset = np.asarray((offs,)*self.ndim)
        self.CNET_stride = PredictionStride(pooling_factor)
        self.number_of_labeled_points_per_dim = outputsize_after_convpool(self.CNET_Input_Size, filter_size[:-1],pooling_factor[:-1])

        #need additional margin in order to make predicitons for the whole image (i.e. need (<self.CNET_stride>-1) many 1-pixel displacements)
        if self.CNET_real_imagesize - self.CNET_Input_Size < self.CNET_stride-1:
            self.CNET_Input_Size -= self.CNET_stride

        if use_max_fragment_pooling:
            # due to implementation details: increase input size if pooling is used!
            # the following is the same as (stride>=2 + 2*(stride>=4) + 4*(stride>=8) + 8*(stride>=16) +...)
            self.CNET_Input_Size = self.CNET_Input_Size + PredictionStride(pooling_factor)-1
        self.padded_once=False
        self.use_max_fragment_pooling = use_max_fragment_pooling

        if type(override_data_set_filenames) is dict:
            if "data" in override_data_set_filenames.keys():
                nfiles = zip(override_data_set_filenames["data"],override_data_set_filenames["labels"])
                assert len(override_data_set_filenames["data"]) == len(override_data_set_filenames["labels"]),"seems broken! Fix the dict contents."
                if b_shuffle_data:
                    random.seed(46473)#fixed seed: otherwise saves are INVALID/FRAUD (->const test set)
                    random.shuffle(nfiles)
                    random.seed()
            else:
                assert len(override_data_set_filenames["train_data"]) == len(override_data_set_filenames["train_labels"]),"seems broken! Fix the dict contents."
                nfiles =  zip(override_data_set_filenames["train_data"],override_data_set_filenames["train_labels"])
                self.training_set_size = len(nfiles)
                tmp = override_data_set_filenames["test_data"]
                nfiles += zip(tmp,[None]*len(tmp))
                
        

        self.data   = []
        self.labels = []
        self.mask   = []
        if type(nfiles[0])==type(""):
            self.file_names = nfiles
        else:
            self.file_names = [x[0] for x in nfiles]
        print "loading..."
        n = len(nfiles)

        self.num_channels = None
        self.num_classes  = 6 #[0,1,2,3,4,5]
        
        

        for i,f in zip(range(len(nfiles)),nfiles):
            
#            print i,f
            addtnl_info_str=''
            
            if type(f) is str:

                d = file_reading.load_file(f)
                d = d[0,...]
                l = None 

                
            else:
                assert type(f[0]) is str
                d = file_reading.load_file(f[0])
                d = np.squeeze(d)
                
                if d.ndim==3:
                    d=d.reshape(d.shape+(1,))# add single channel dimension
                
                if data_clip_range is None:
                    if data_init_preserve_channel_scaling:
                        d = (d-0.5)/3.5 
                    else:
    
                        d2 = np.transpose(d,axes=[3,0,1,2])
                        d2 = np.reshape(d2,(d2.shape[0],-1))
                        std_ = np.std(d2,axis=1)
                        mean_ = np.mean(d2,axis=1)
                        d = (d-mean_)/(4.*std_)
                else:
                    assert len(data_clip_range)==2
                    #warp large values to min
                    d = np.where(d > data_clip_range[1] + abs(data_clip_range[1]-data_clip_range[0])*0.1, data_clip_range[0], d)
                    #clip to range
                    d = np.clip(d, data_clip_range[0], data_clip_range[1])
                    
                    addtnl_info_str+='clip({},{})'.format(data_clip_range[0], data_clip_range[1])
                    if 0:
                        overflow = np.where(d==data_clip_range[1], 1, 0)
                        d = np.where(d==data_clip_range[1], data_clip_range[0], d)
                        d -= d.min()
                        d /= d.max()
                        d = np.concatenate([d, overflow], axis=-1)
                    else:
                        d -= d.min()
                        d /= d.max()
                    d *= 0.1
                
                if f[1] is not None:
                    l = file_reading.load_file(f[1])
                    l=np.squeeze(l)
                    uniq = np.unique(l)
                else:
                    l = np.zeros((1,1,1),"uint16")
                    uniq = [0,1] #small hack...
            
            if len(uniq)==2 and uniq[1]!=1:
                l[l==uniq[1]]=1
                l[l==uniq[0]]=0
                uniq=[0,1]
            if len(uniq) !=2:
                if auto_threshold_labels:
                    assert uniq[0]==0
                    l = (l>0).astype('int16')
                else:
                    assert len(uniq)==2, 'Labels must be binary, but found '+str(len(uniq))+' unique values in the labels!'
            
            
            if d.shape[:3]!=l.shape[:3] and l.shape[:3]!=(1,1,1):
                print "DATA SHAPE MISMATCH! transposing labels..."
                l=np.transpose(l,axes=[0,2,1])
            assert d.shape[:3]==l.shape[:3] or l.shape[:3]==(1,1,1)
            
            if self.num_channels is None:
                self.num_channels = d.shape[3]
            assert d.shape[3]==self.num_channels
            if self.num_channels==5:
                print "warning: removing channel 2 (starting at 0)"
                d = np.concatenate( (d[...,:2],d[...,3:]),axis=3) #x,y,z,channels
            
            
            d = np.transpose(d,(0,3,1,2))
            
            

            if l is not None:
                if l.dtype in [np.dtype('int'),np.dtype('int32'),np.dtype('int16'),np.dtype('uint32'),np.dtype('uint16')]:
                    l[l==5]=0
#                    print "WARNING: merging class 0 and 5!"
                    l = l.astype("int16")
#                else:
                
                
                
            print 'Loaded...',100.*(i+1)/n,"%",d.shape,addtnl_info_str, f  #, self.labels[-1].shape  #,np.unique(self.labels[-1]) #(143, 4, 175, 127) (143, 175, 127) LG0004_inclLabels.cp
            
            if pad_last_dimension and (d.shape[-1] < self.CNET_Input_Size + padding_margin):
                add_this = int((padding_margin + self.CNET_Input_Size - d.shape[-1])/2.)
                d = pad_data(d, add_this, mode='constant')
                #            if l.shape[-1] < self.CNET_Input_Size:
                l = pad_data(l, add_this, mode='constant')
                print '>> padded to:', d.shape
            self.data.append(d)# format: (x,channels,y,z)
            self.labels.append(l)
        
        

        self.CNET_data_NumImagesInData = len(self.data)#number of different images
        


        self.number_of_images_test_set = int(self.CNET_data_NumImagesInData - self.training_set_size)

        print "Total n. of examples:",self.CNET_data_NumImagesInData,"images/volumes"
        print 'Training on',self.training_set_size,'images/volumes'
        print 'Testing on ',self.number_of_images_test_set,'images/volumes'

        self._getTestImage_current_file=self.training_set_size # <self.training_set_size> is the first non-training file







    def greyvalue_pad_data(self, cnn):
        print self,':: greyvalue_pad_data()'
        assert self.padded_once==False
        self.padded_once=True
        CNET_stride    = self.CNET_stride if self.use_max_fragment_pooling==0 else 1
        input_s = cnn.input_shape[-1] + CNET_stride - 1
        input_s = cnn.input_shape[-1] + CNET_stride - 1 # input size for runNetOnSlice()

        offset_l = self.CNET_labels_offset[0]
        offset_r = offset_l + input_s
        print '\nold shapes',np.unique([d.shape for d in self.data])        
        self.data = [greyvalue_data_padding(dat, offset_l, offset_r) for dat in self.data]


        self.labels = [np.asarray(np.pad(lab, pad_width=[(offset_l,offset_r),(offset_l,offset_r),(offset_l,offset_r)],mode='constant'),dtype='int16') for lab in self.labels if lab.shape[0] != 1] + [lab for lab in self.labels if lab.shape[0] == 1]
        print '\nnew shapes',np.unique([d.shape for d in self.data])

        
        for d,l in zip(self.data,self.labels):
            if l.shape[0] != 1:
                assert d.shape[0]==l.shape[0]
                assert d.shape[2:]==l.shape[1:]
        


    def __get_cubes(self, i_min, i_max, num):
        """ picks <num> many cubes from [i_min,i_max)  (max is excluded) <num> many pictures."""
                
        i_ = np.random.randint(i_min, i_max, size=num)# 0, self.training_set_size,size=num)

        dat = np.zeros( (num, self.CNET_Input_Size, self.num_channels, self.CNET_Input_Size, self.CNET_Input_Size), dtype="float32")
        labshape = (num,)+(self.number_of_labeled_points_per_dim,)*self.ndim


        if self.labels[0].ndim==4:
            labshape += (self.labels[0].shape[3],)
        lab = np.zeros( labshape, dtype="int16")
        
        
        
        for n,i in zip(range(num),i_):
            sp = self.data[i].shape
            sp = [ sp[x + (1 if x>0 else 0)] for x in range(3)] #ignore channel axis
            
            off = [np.random.randint(0,sp[x]-self.CNET_Input_Size) for x in range(3)]
            dat[n,...] = self.data[i][off[0]:off[0]+self.CNET_Input_Size, :, off[1]:off[1]+self.CNET_Input_Size, off[2]:off[2]+self.CNET_Input_Size]
            loff = tuple(off) + self.CNET_labels_offset
            lab[n,...] = self.labels[i][loff[0]:loff[0]+self.number_of_labeled_points_per_dim*self.CNET_stride:self.CNET_stride, loff[1]:loff[1]+self.number_of_labeled_points_per_dim*self.CNET_stride:self.CNET_stride, loff[2]:loff[2]+self.number_of_labeled_points_per_dim*self.CNET_stride:self.CNET_stride]

        return dat, lab #np.asarray(out_lab,dtype=np.int16).flatten()
        #__get_cubes:: (1, 28, 5, 28, 28), (1, 10, 10, 10)



    def __get_random_string(self):
        r1 = np.random.randint(0,5)
        r2 = np.random.randint(0,5)
        r3 = np.random.randint(0,5)
        r4 = random.choice(list(it.permutations(range(3),3))) #   r4==(0,1,2) is IDENTITY
        return (r1,r2,r3,r4)



    def __transform_data(self, dat, transform, transformable=[1,3,4]):
        """ function is the inverse of itself!
        values in <transformable> are only considered if working in 5dim (currently)"""
        assert dat.ndim in [4,5],"__transform_data::TODO"
        ret = dat.copy()

        if dat.ndim==4: #__get_cubes::  (1, 10, 10, 10)

            (r1,r2,r3,r4)=transform
            if r1==1:
                ret = ret[:,::-1,:,:]
            if r2==1:
                ret = ret[:,:,::-1,:]
            if r3==1:
                ret = ret[:,:,:,::-1]

            ret = np.transpose(ret,(0,)+tuple(np.asarray(r4)+1) )


        elif dat.ndim==5: #__get_cubes:: (1, 28, 5, 28, 28)

            (r1,r2,r3,r4)=transform
            if r1==1:
                idx = [slice(None)] * (transformable[0]) + [slice(None,None,-1)] + [Ellipsis]
                ret = ret[idx]
            if r2==1:
                idx = [slice(None)] * (transformable[1]) + [slice(None,None,-1)] + [Ellipsis]
                ret = ret[idx]                
            if r3==1:
                idx = [slice(None)] * (transformable[2]) + [slice(None,None,-1)] + [Ellipsis]
                ret = ret[idx]
            
            transp = range(dat.ndim)
            pick_count=0
            for i in range(dat.ndim):

                if i in transformable:
                    transp[i] = transformable[r4[pick_count]]
                    pick_count+=1

            ret = np.transpose(ret, transp)
        return ret


    def makeTrainingPatch(self, batchsize):
        """ """
        da,la = self.__get_cubes(i_min=0,i_max=self.training_set_size, num=batchsize)
        tr = self.__get_random_string()
        da = self.__transform_data(da,tr,transformable=[1,3,4])
        la = self.__transform_data(la,tr,transformable=[1,2,3])
        return da,la




if __name__ == '__main__':
    print "please execute main_train.py instead!"

