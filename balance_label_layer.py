import caffe

import scipy.io as sio
import numpy as np
import cv2
import random
import glob
import pdb

class LabelBalanceLayer(caffe.Layer):
    def setup(self, bottom, top):
        #params = eval(self.param_str)
        self.confident_thesh = 0.5
        self.balance_ratio = 0.5
        #self.sample_num = params.get('sample_num', 1024) 

    def reshape(self, bottom, top):
       
        self.data_shape = bottom[0].data.shape
        self.batch_num = self.data_shape[0]
        self.channel_num = self.data_shape[1]
        self.label_count = np.zeros( (self.batch_num, self.channel_num), np.float32)
        self.label_ratio = np.zeros( (self.batch_num, self.channel_num), np.float32)
        
        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(*bottom[0].data.shape)
        #top[1].reshape(*bottom[1].data.shape)

    def forward(self, bottom, top):
        # bottom[0]: feat_before_prob
        # bottom[1]: prob
        # bottom[2]: label (nxcxhxw)
        # top[0]: feat_before_prob
        self.prob = bottom[1].data
        self.label = bottom[2].data
        
        for ni in range(self.data_shape[0]):
            for ci in range(self.channel_num):
                self.label_count[ni, ci] = np.sum(self.label[ni,0,:,:] == ci)
            balance = 1. / float(self.channel_num)
            total_num = float(self.data_shape[2] * self.data_shape[3])
            for ci in range(self.channel_num):
                if self. label_count[ni,ci] < 1:
                    self.label_ratio[ni, ci] = balance
                else:
                    self.label_ratio[ni, ci] = balance / (float(self.label_count[ni,ci]) / total_num)
        # assign output
        top[0].data[...] = np.copy(bottom[0].data)

    def backward(self, top, propagate_down, bottom):
        if propagate_down[0]:
            bottom[0].diff[...] = np.copy(top[0].diff)
            
            for ni in range(self.data_shape[0]):
                for ci in range(self.data_shape[1]):
                    #mask = ( (self.label[ni,0,:,:] == ci) & (self.prob[ni,ci,:,:] > self.confident_thesh) )
                    mask = (self.label[ni,0,:,:] == ci)
                    bottom[0].diff[ni,:,mask] = bottom[0].diff[ni,:, mask] * self.label_ratio[ni,ci]
