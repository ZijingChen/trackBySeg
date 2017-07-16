# combine fcn probability here
import caffe
import pdb
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random
import scipy.io as sio
import scipy
import cv2
import math
import sys

class VOCSegDataLayer(caffe.Layer):

    def setup(self, bottom, top):

        # config
        params = eval(self.param_str)
        self.dataset_dir = params['dataset_dir']
        self.train_inds_txt = params['train_inds_txt']# txt path
        self.split = params['split']
        self.random = params.get('randomize', True)
        self.seed = params.get('seed', None)
        self.imsz = np.array(params['imsz'])#[640,480] # in order [width, height]
        self.mean = np.array(params['mean'],dtype=np.float32)

        #self.lbsz = map(divide_8,self.imsz)  # => self.lbsz = [640/8, 480/8]
        self.lbsz = self.imsz
        
        #pdb.set_trace();
        # two tops: data and label
        if len(top) != 2:
            raise Exception("Need to define TWO tops: data, label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        ## load indices for images and labels
        self.img_path = []
        self.label_path = []
        f_ = open(self.train_inds_txt, 'r')
        for line in f_:
            self.img_path.append(line.strip().split()[0])
            self.label_path.append(line.strip().split()[1])
        f_.close()
        
        self.train_num = len(self.img_path) # total training numbers
        #pdb.set_trace()

        self.idx = 0 # % the line number corresponds to img_path and label_path
#        pdb.set_trace()
        # make eval deterministic
        if 'train' not in self.split:
            self.random = False

        # randomization: seed and pick
        if self.random:
            random.seed(self.seed)
#            self.idx = random.randint(0, len(self.indices)-1)
            self.idx = 0#random.randint(0+self.acc,self.train_num-1)
#        pdb.set_trace()
#        self.get_enhance_data_params()


    def reshape(self, bottom, top):
        # load image + label image pair
        #pdb.set_trace()
        self.data = self.load_image(self.idx);#self.indices[self.idx])
        self.label = self.load_label(self.idx);#self.indices[self.idx])

        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(1, *self.data.shape)
        top[1].reshape(1, *self.label.shape)


    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.data
        top[1].data[...] = self.label

#        pdb.set_trace()
        ## pick next training index as input
#        self.get_enhance_data_params()
        sample_flag = 1
        if self.random:
            while sample_flag:
                self.idx = random.randint(0,self.train_num-1)
                # check if two imgs belong to the same video clip
                prev_path = self.img_path[self.idx-1]
                cur_path = self.img_path[self.idx]
                #pdb.set_trace()
                prev_clip_name = prev_path.strip().split('/')[3]
                cur_clip_name = cur_path.strip().split('/')[3]
                if prev_clip_name==cur_clip_name: # belong to same clip, no more samping
                    sample_flag = 0
            #print 'Next idx= '+str(self.idx)
            #print cur_clip_name
            #sys.stdout.flush() 

        else:
            self.idx += 1
            if self.idx == len(self.indices):
                self.idx = 0


    def backward(self, top, propagate_down, bottom):
        pass


    def load_image(self, idx):

        ind2 = self.idx 
        im2 = Image.open(self.dataset_dir+self.img_path[ind2])
        #print('Now img:')
        #print(self.dataset_dir+self.img_path[ind2])
        in_tmp2 = np.array(im2, dtype=np.uint8)
        in_tmp2 = scipy.misc.imresize(in_tmp2, self.imsz, interp='nearest', mode=None)
        in_tmp2 = in_tmp2[:,:,::-1]

        in_tmp2_float = in_tmp2.astype(np.float32)
        in_ = in_tmp2_float.transpose((2,0,1)) # C*H*W

        return in_


    def load_label(self, idx):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """

        ind2 = self.idx 
        label_im2 = cv2.imread(self.dataset_dir+self.label_path[ind2],0) #0/255
#        pdb.set_trace()

        label2 = np.array(label_im2, dtype=np.uint8)
        label2 = label2/255.0
        label_in = scipy.misc.imresize(label2, self.imsz, interp='nearest', mode=None)
        label = label_in[np.newaxis, ...]
#        label = diff_label[np.newaxis, ...]
        return label




class PunishLossLayer(caffe.Layer):
    """
    Punish loss when backpropagation
    """

    def setup(self, bottom, top):
        
        # config
        params = eval(self.param_str)
        self.phase = params['phase']

        # two tops: data and label
        if len(top) != 1:
            raise Exception("Need to define a top.")
        # data layers have no bottoms
        if self.phase=='TRAIN':
            if len(bottom) != 2:
                raise Exception("Need to define two bottoms, including a label.")
        else:
            if len(bottom) != 1:
                raise Exception("Need to define one bottoms, not include any label.")



    def reshape(self, bottom, top):

        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(*bottom[0].data.shape)


    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = bottom[0].data[...]


    def backward(self, top, propagate_down, bottom):
        weight = 10
#        pdb.set_trace()
        label = bottom[1].data[...]
        label = np.squeeze(label)
        [r0,c0] = np.where(label==0)
        [r2,c2] = np.where(label==2)
        for i in range(len(bottom)):
            if propagate_down[i]:
                diff_data = top[i].diff[...]
                diff_data[:,:,r0,c0] = diff_data[:,:,r0,c0] *weight
                diff_data[:,:,r2,c2] = diff_data[:,:,r2,c2] *weight
                bottom[i].diff[...] = diff_data



class SBDDSegDataLayer(caffe.Layer):
    """
    Load (input image, label image) pairs from the SBDD extended labeling
    of PASCAL VOC for semantic segmentation
    one-at-a-time while reshaping the net to preserve dimensions.

    Use this to feed data to a fully convolutional network.
    """

    def setup(self, bottom, top):
        """
        Setup data layer according to parameters:

        - sbdd_dir: path to SBDD `dataset` dir
        - split: train / seg11valid
        - mean: tuple of mean values to subtract
        - randomize: load in random order (default: True)
        - seed: seed for randomization (default: None / current time)

        for SBDD semantic segmentation.

        N.B.segv11alid is the set of segval11 that does not intersect with SBDD.
        Find it here: https://gist.github.com/shelhamer/edb330760338892d511e.

        example

        params = dict(sbdd_dir="/path/to/SBDD/dataset",
            mean=(104.00698793, 116.66876762, 122.67891434),
            split="valid")
        """
        # config
        params = eval(self.param_str)
        self.sbdd_dir = params['sbdd_dir']
        self.split = params['split']
        self.mean = np.array(params['mean'])
        self.random = params.get('randomize', True)
        self.seed = params.get('seed', None)

        # two tops: data and label
        if len(top) != 2:
            raise Exception("Need to define two tops: data and label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # load indices for images and labels
        split_f  = '{}/{}.txt'.format(self.sbdd_dir,
                self.split)
        self.indices = open(split_f, 'r').read().splitlines()
        self.idx = 0

        # make eval deterministic
        if 'train' not in self.split:
            self.random = False

        # randomization: seed and pick
        if self.random:
            random.seed(self.seed)
            self.idx = random.randint(0, len(self.indices)-1)


    def reshape(self, bottom, top):
        # load image + label image pair
        self.data = self.load_image(self.indices[self.idx])
        self.label = self.load_label(self.indices[self.idx])
        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(1, *self.data.shape)
        top[1].reshape(1, *self.label.shape)


    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.data
        top[1].data[...] = self.label

        # pick next input
        if self.random:
            self.idx = random.randint(0, len(self.indices)-1)
        else:
            self.idx += 1
            if self.idx == len(self.indices):
                self.idx = 0


    def backward(self, top, propagate_down, bottom):
        pass


    def load_image(self, idx):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        im = Image.open('{}/img/{}.jpg'.format(self.sbdd_dir, idx))
        in_ = np.array(im, dtype=np.float32)
        in_ = in_[:,:,::-1]
        in_ -= self.mean
        in_ = in_.transpose((2,0,1))
        return in_


    def load_label(self, idx):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        import scipy.io
        mat = scipy.io.loadmat('{}/cls/{}.mat'.format(self.sbdd_dir, idx))
        label = mat['GTcls'][0]['Segmentation'][0].astype(np.uint8)
        label = label[np.newaxis, ...]
        return label
