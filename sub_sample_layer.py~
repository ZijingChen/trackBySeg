import caffe

import scipy.io as sio
import numpy as np
import cv2
import random
import glob
import pdb
import traceback

class SubSampleLayer(caffe.Layer):
    def setup(self, bottom, top):
        #params = eval(self.param_str)
        self.width_ratio = 0.2 #size of subsample
        self.height_ratio = 0.2
        self.candibox_ratio = 0.5 # range to generate subsample
        self.sample_num = 20 # 10 pairs of (+-) pair; 5 pairs of (+.+); 5 pairs of (- -)
        self.out_pixel = 3 # the data into next layer is 256*3*3
        #self.sample_num = params.get('sample_num', 1024) 



    def reshape(self, bottom, top):
        self.data_shape = bottom[0].data.shape
        self.batch_num = self.data_shape[0]*self.sample_num
        self.channel_num = self.data_shape[1]
        self.img_width = self.data_shape[2]
        self.img_height = self.data_shape[3]

        self.label_shape = bottom[1].data.shape
        self.batch_label_num = self.label_shape[0]*self.sample_num

        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(self.batch_num,self.channel_num*2,self.out_pixel,self.out_pixel)
        top[1].reshape(self.batch_label_num,1)
        #top[1].reshape(*bottom[1].data.shape)

    def forward(self, bottom, top):
        # bottom[0]: img features after conv3 but before cropping sample patches
        # bottom[1]: 0 1label after conv3
        # top[0]: cropped sample patches (in pair, concated in axis1)
        # top[1]: 0/1 labels corresponding to top[0],0 means same class; 1 means different class
        forward_img_size = [self.out_pixel,self.out_pixel]#3*3
        # step1: get subsample positions
        features = bottom[0].data
        labels = bottom[1].data
        sample_pos, sample_label = self.get_subsample(features,labels,self.img_width,self.img_height) # sample_pos and sample_label are numpy matrix

        # step2: get subfeature maps from sample_pos, that's same as color images
        resized_features = self.get_features_from_pos(features,sample_pos)


        # step4: assign output, concate matrix on axis 1
        top[0].data[...] = resized_features
        top[1].data[...] = sample_label[...,np.newaxis]

    def backward(self, top, propagate_down, bottom):
        pass

    def get_subsample(features,label,img_width,img_height):
    # sample_pos = [xmin ymin xmax ymax, xmin1 ymin1 xmax1 ymax1; xmin_,ymin_,xmax_,ymax_,xmin1_,ymin1_,xmax1_,ymax1_]
    # sample_label = [0;1]
        num_pos_sample = self.sample_num
        num_neg_sample = self.sample_num
        sample_pos = np.zeros(self.sample_num,8)
        sample_label = np.zeros(self.sample_num,1)
        pos_cnt = 0
        neg_cnt = 0

        # get sample range: around bounding boxes
        labelnp = np.array(label, dtype=np.uint8)
        labelnp = np.squeeze(labelnp)
        if labelnp.max()==255:
            labelnp = labelnp/255
        labelnp = np.uint8(labelnp)
        tmp_loc = np.where(labelnp>0)
        rows = tmp_loc[0]
        cols = tmp_loc[1]
        rowmin = rows.min()
        rowmax = rows.max()
        colmin = cols.min()
        colmax = cols.max()
        maskwidth = colmax-colmin+1
        maskheight = rowmax-rowmin+1
        sample_xmin = max(1, round(colmin-self.candibox_ratio*maskwidth)  )
        sample_ymin = max(1, round(rowmin-self.candibox_ratio*maskheight)  )
        sample_xmax = min(img_width-1, round(colmax+self.candibox_ratio*maskwidth)  )
        sample_ymax = min(img_height-1, round(rowmax+self.candibox_ratio*maskheight)  )
        sample_width = round(maskwidth*self.width_ratio)
        sample_height = round(maskwidth*self.height_ratio)
        # generate subsamples
        pos_mat = np.zeros((num_pos_sample,4))
        neg_mat = np.zeros((num_neg_sample,4))
        while(pos_cnt<num_pos_sample or neg_cnt<pos_neg_sample):
            xmin = random.randint(sample_xmin, sample_xmax - sample_width)# TODO
            ymin = random.randint(sample_ymin, sample_ymax - sample_height)#TODO:what if minus<0
            xmax = xmin + sample_width - 1
            ymax = ymin + sample_height -1
            # judge the label of this new subsample
            filtered_mask = labelnp[xmin:xmax+1;ymin:ymax+1]
            acc_value = sum(sum(filtered_mask))
            if acc_value>=(filtered_mask.shape[0]*filtered_mask.shape[1]*0.5):
                if pos_cnt<num_pos_sample:
                    # get positive subsample
                    pos_mat[pos_cnt,:] = [xmin, ymin, xmax, ymax]
                    pos_cnt = pos_cnt+1
            else:
                if neg_cnt<num_neg_sample:
                    # get negative subsample
                    neg_mat = [neg_cnt,:] = [xmin, ymin, xmax, ymax]
                    neg_cnt = neg_cnt + 1

        # arrange subsamples into sample_position which include positions in pair
        diff_cnt_thred = round(self.sample_num*0.5)# the num of diff patch pairs
        mat_pt = 0#-> point to each row
        # part1:[pos neg] or [neg pos] pair
        pos_mat_for_part1 = pos_mat[0:diff_cnt_thred]
        neg_mat_for_part1 = neg_mat[0:diff_cnt_thred]
        tmp_data_diffa = np.concatenate((pos_mat_for_part1[0:round(diff_cnt_thred*0.5),:],
neg_mat_for_part1[0:round(diff_cnt_thred*0.5),:]),axis=1) #5*8
        tmp_data_diffb = np.concatenate((neg_mat_for_part1[round(diff_cnt_thred*0.5):,:],
pos_mat_for_part1[round(diff_cnt_thred*0.5):,:]),axis=1)#5*8
        data_part1 = np.concatenate((tmp_data_diffa,tmp_data_diffb),axis=0)
        label_part1 = np.ones((diff_cnt_thred,1)) # label=1
 
        # part2: [pos pos] pair or [neg neg] pair
        pos_mat_for_part2 = pos_mat[diff_cnt_thred:,:]
        neg_mat_for_part2 = neg_mat[diff_cnt_thred:,:]
        nrow_part2_pos = pos_mat_for_part2.shape[0]
        nrow_part2_neg = neg_mat_for_part2.shape[0]
        if (nrow_part2_pos % 2 or nrow_part2_neg % 2) == 1:
             raise Exception('The rows of pos_mat(or neg_mat) reserved to generate part2 point should be even number')
        npairpos = nrow_part2_pos*0.5
        npairneg = nrow_part2_neg*0.5
   
        tmp_pos_data_samea = pos_mat_for_part2[:npairpos,:]
        tmp_pos_data_sameb = pos_mat_for_part2[npairpos:,:]
        tmp_pos_data = np.concatenate((tmp_pos_data_samea,tmp_pos_data_sameb),axis=1)

        tmp_neg_data_samea = neg_mat_for_part2[:npairneg,:]
        tmp_neg_data_sameb = neg_mat_for_part2[npairneg:,:]
        tmp_neg_data = np.concatenate((tmp_neg_data_samea,tmp_neg_data_sameb),axis=1)
        data_part2 = np.concatenate((tmp_pos_data,tmp_neg_data),axis=0)
        label_part2 = np.zeros((npairpos+npairneg,1)) # label=0

        # put part1 and part2 together
        # pay attention to the label!
        sample_pos = np.concatenate((data_part1,data_part2),axis = 0)
        sample_label = np.concatenate((label_part1,label_part2), axis=0)

        return sample_pos, sample_label

    def get_features_from_pos(feature_all,sample_pos,forward_img_size):
        # feature_all = 1:128:H*W(H and W are image size)
        # sample_pos = 20*8*h*w(h and w are subsample size)
        # forward_img_size =[3,3]
        num_channel_in = feature_all.shape[1]
        num_channel_out = 2*num_channel_in
        num_batch_out = sample_pos.shape[0]
        num_row_out = forward_img_size[0]
        num_col_out = forward_img_size[1]
        features = np.zeros(num_batch_out,num_channel_out,num_row_out,num_col_out)

        noutpixel = self.out_pixel
        for ind in range(0,num_batch_out):
            loc_vec = sample_pos(ind,:)
            xmin1 = loc_vec[0,0]
            ymin1 = loc_vec[0,1]
            xmax1 = loc_vec[0,2]
            ymax1 = loc_vec[0,3]
            xmin2 = loc_vec[0,4]
            ymin2 = loc_vec[0,5]
            xmax2 = loc_vec[0,6]
            ymax2 = loc_vec[0,7]

            # resize feature pair for each channel
            # actually it is down sample
            yinds1 = self.subsample_locs(ymin1,ymax1,noutpixel)
            xinds1 = self.subsample_locs(xmin1,xmax1,noutpixel)
            yinds2 = self.subsample_locs(ymin2,ymax2,noutpixel)
            xinds2 = self.subsample_locs(xmin2,xmax2,noutpixel)
            feature_all = feature_all.suqeeze()
            tmp_feature1 = feature_all[:,yinds1,xinds1] #128*3*3
            tmp_feature2 = feature_all[:,yind2,xinds2] #128*3*3
            tmp_feature = np.concatenate((tmp_feature1,tmp_feature2),axis=0) # concate on channels,256*3*3

        return features

    def subsample_locs(ymin1,ymax1,noutpixel):
        if (ymax1-ymin1+1)>=noutpixel:
            yinds1 = range(ymin1,ymax1+1,noutpixel)
        else:
            wrap_ratio = float(ymax1-ymin1 + 1) / float(noutpixel)
            yinds = range(0,outpixel)
            yinds = round(inds * wrap_ratio + ymin1)
            yinds = np.minimum(np.maximum(inds,ymin1),ymax1)
        return yinds1




