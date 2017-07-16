import caffe

import scipy.io as sio
import numpy as np
import cv2
import random
import glob
import pdb
import traceback
from math import isnan
import sys
import scipy.io as scio 


class SubSampleLayer(caffe.Layer):
    def setup(self, bottom, top):
        #params = eval(self.param_str)
        self.width_ratio = 0.2 #size of subsample
        self.height_ratio = 0.2
        self.wh_ratio_notarget = 0.1 # when no target on image. the ratio r.r.d img width(after conv3),5*8
        self.candibox_ratio = 0.5 # range to generate subsample
        self.sample_num = 20 # 10 pairs of (+-) pair; 5 pairs of (+.+); 5 pairs of (- -)
        self.out_pixel = 3 # the data into next layer is 256*3*3
        #self.sample_num = params.get('sample_num', 1024) 

        np.random.seed(7)




    def reshape(self, bottom, top):

        self.data_shape = bottom[0].data.shape
        self.batch_num = self.data_shape[0]*self.sample_num
        self.channel_num = self.data_shape[1]
        self.img_height = self.data_shape[2]
        self.img_width = self.data_shape[3]
        if self.data_shape[0]>1:
            raise ValueError('The batch num of bottom blob should be one!')

        self.label_shape = bottom[1].data.shape
        self.batch_label_num = self.label_shape[0]*self.sample_num

        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(self.batch_num,self.channel_num*2,self.out_pixel,self.out_pixel)
        #top[1].reshape(self.batch_label_num,1,1,1)
        top[1].reshape(self.batch_label_num,1)
        #top[1].reshape(*bottom[1].data.shape)

    def forward(self, bottom, top):
        # bottom[0]: img features after conv3 but before cropping sample patches
        # bottom[1]: 0 1label after conv3
        # top[0]: cropped sample patches (in pair, concated in axis1)
        # top[1]: 0/1 labels corresponding to top[0],0 means same class; 1 means different class
        forward_img_size = [self.out_pixel,self.out_pixel]#3*3; order ->[num_row_out,num_col_out]
        # step1: get subsample positions
        features = bottom[0].data
        labels = bottom[1].data

        sample_p, sample_l = self.get_subsample(features, labels, self.img_width, self.img_height) # sample_p and sample_l are numpy matrix

        # step2: get subfeature maps from sample_p, that's same as color images
        resized_features = self.get_features_from_position(features,sample_p,forward_img_size )# 20*256*3*3

        # step4: assign output, concate matrix on axis 1
        #label_out = sample_l[...,np.newaxis,np.newaxis]# TODO: CHECK SHAPE OF SAMPLE_L
        label_out = sample_l
        top[0].data[...] = resized_features
        top[1].data[...] = label_out

    def backward(self, top, propagate_down, bottom):
        pass

    def get_subsample(self,features, label, img_width, img_height):
        # sample_p = [xmin ymin xmax ymax, xmin1 ymin1 xmax1 ymax1; xmin_,ymin_,xmax_,ymax_,xmin1_,ymin1_,xmax1_,ymax1_]
        # sample_l = [0;1]
        num_pos_sample = self.sample_num
        num_neg_sample = self.sample_num
		# define positions and labels of generated patch pair boxes
        sample_p = np.zeros((self.sample_num, 8),dtype=np.uint32)
        sample_l = np.zeros((self.sample_num, 1),dtype=np.uint32)
        pos_cnt = 0
        neg_cnt = 0

        # get sample range: around bounding boxes
        labelnp = np.array(label, dtype=np.uint8).squeeze() 

        labelnp[labelnp>0] = 1
        rows, cols = np.nonzero(labelnp)
        if np.count_nonzero(rows)>20: # large target in this image
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
            sample_height = round(maskheight*self.height_ratio)
            #pdb.set_trace()
            # generate subsamples
            pos_mat = np.zeros((num_pos_sample,4),dtype=np.uint32)
            neg_mat = np.zeros((num_neg_sample,4),dtype=np.uint32)
     
            while(pos_cnt<num_pos_sample or neg_cnt<num_neg_sample):
                if sample_xmin<(sample_xmax-sample_width): # get xmin and generate xmax according to width
                	xmin = int(np.random.randint(sample_xmin, sample_xmax - sample_width))
                	xmax = int(xmin + sample_width - 1)
                else:# randomly generate xmin and xmax in location range(no more considering width)
				    xcandis = np.random.choice((sample_xmin,sample_xmax),2,replace=False)
				    xcandis = np.sort(xcandis)
				    xmin = int(xcandis[0])
				    xmax = int(xcandis[1])
				
                if sample_ymin<sample_ymax-sample_height:
                	ymin = int(np.random.randint(sample_ymin, sample_ymax - sample_height))
                	ymax = int(ymin + sample_height -1)
                else:
				    ycandis = np.random.choice((sample_ymin,sample_ymax),2,replace=False)
				    ycandis = np.sort(ycandis)
				    ymin=int(ycandis[0])
				    ymax=int(ycandis[1])



                # judge the label of this new subsample
                filtered_mask = labelnp[ymin:ymax+1,xmin:xmax+1]
                if isnan(filtered_mask.mean()  ):
                    print('Nan in Mean!!!!!')
                    sys.stdout.flush()    
                    pdb.set_trace()         

                if filtered_mask.mean() >= .5:
                    if pos_cnt<num_pos_sample:
                        # get positive subsample
                        pos_mat[pos_cnt,:] = [xmin, ymin, xmax, ymax]
                        pos_cnt = pos_cnt+1
                else:
                    if neg_cnt<num_neg_sample:
                        # get negative subsample
                        neg_mat[neg_cnt,:] = [xmin, ymin, xmax, ymax]
                        neg_cnt = neg_cnt + 1

            # arrange subsamples into sample_position which include positions in pair
            diff_cnt_thred = int(round(self.sample_num*0.5))# the num of diff patch pairs

            # part1:[pos neg] or [neg pos] pair
            pos_mat_for_part1 = pos_mat[0:diff_cnt_thred]
            neg_mat_for_part1 = neg_mat[0:diff_cnt_thred]
            tmp_data_diffa = np.concatenate((pos_mat_for_part1[0:round(diff_cnt_thred*0.5),:],
    neg_mat_for_part1[0:round(diff_cnt_thred*0.5),:]),axis=1) #5*8
            tmp_data_diffb = np.concatenate((neg_mat_for_part1[round(diff_cnt_thred*0.5):,:],
    pos_mat_for_part1[round(diff_cnt_thred*0.5):,:]),axis=1)#5*8
            data_part1 = np.concatenate((tmp_data_diffa,tmp_data_diffb),axis=0)
            label_part1 = np.ones((diff_cnt_thred,1),dtype = np.int32) # label=1
     
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
            label_part2 = np.zeros((npairpos+npairneg,1),dtype=np.int32) # label=0

            # put part1 and part2 together
            # pay attention to the label!
            sample_p = np.concatenate((data_part1,data_part2),axis = 0)
            sample_l = np.concatenate((label_part1,label_part2), axis=0)
        
        else: # no target or very tiny target in this image
            # randomly sample 40 patches in whole image area and compare the IOU with ground truth
            # when dividing 40 patches into patch pairs,
            # the minority part has higher priority to form diff pair
            # FOR EXAMPLE: pos patch=2;neg patch = 38;
            # Then first two pairs would be [pos neg] with label 1
            # The remains would be [neg neg] with label 0

            sample_xmin = 1
            sample_ymin = 1
            sample_xmax = img_width-1
            sample_ymax = img_height-1
            sample_width = round(sample_xmax*self.wh_ratio_notarget )
            sample_height = round(sample_ymax*self.wh_ratio_notarget )
            #pdb.set_trace()
            # generate subsamples
            # they are large enough to store all posible locations
            pos_mat = np.zeros((self.sample_num*2,4),dtype=np.uint32)
            neg_mat = np.zeros((self.sample_num*2,4),dtype=np.uint32)
            pos_cnt = 0
            neg_cnt = 0
            for idx_whole in range(0,self.sample_num*2):
                # generate patches
                xmin = int(np.random.randint(sample_xmin, sample_xmax - sample_width))
                xmax = int(xmin + sample_width - 1)
                ymin = int(np.random.randint(sample_ymin, sample_ymax - sample_height))
                ymax = int(ymin + sample_height -1)

                # judge the label of this new subsample
                filtered_mask = labelnp[ymin:ymax+1,xmin:xmax+1]
                if isnan(filtered_mask.mean()  ):
                    print('Nan in Mean!!!!!')
                    sys.stdout.flush()    
                    pdb.set_trace()         

                if filtered_mask.mean() >= .5:
                    pos_mat[pos_cnt,:] = [xmin, ymin, xmax, ymax]
                    pos_cnt = pos_cnt+1
                else:
                    # get negative subsample
                    neg_mat[neg_cnt,:] = [xmin, ymin, xmax, ymax]
                    neg_cnt = neg_cnt + 1

            pos_mat[~np.all(pos_mat==0,axis=1)] # double check
            neg_mat[~np.all(neg_mat==0,axis=1)]
            if pos_cnt==0: # all patches are negative
                sample_p = np.concatenate((neg_mat[0:self.sample_num,:],neg_mat[self.sample_num:self.sample_num*2,:]),axis = 1)
                sample_l = np.zeros((self.sample_num,1),dtype=np.int32) # label=0
            elif neg_cnt>pos_cnt:
                sample_p[0:pos_cnt,:] = np.concatenate(pos_mat[0:pos_cnt],neg_mat[0:pos_cnt,:],axis=1)# double check
                label_diff = np.ones((pos_cnt,1),dtype = np.int32)
                rest_num = neg_cnt - pos_cnt
                if rest_num%2 == 1:
                    raise Exception('rest_num should be even number')
                middle_ind = pos_cnt+ 0.5*rest_num
                sample_p[pos_cnt:self.sample_num,:] = np.concatenate(neg_mat[pos_cnt:middle_ind,:],neg_mat[middle_ind:pos_cnt,:],axis=1)# double check
                label_same = np.zeros((rest_num,1),dtype = np.int32)
                pdb.set_trace()
                sample_l = np.concatenate((label_diff,label_same),axis=0)
            else: #pos_cnt>neg_cnt
                sample_p[0:neg_cnt,:] = np.concatenate(pos_mat[0:neg_cnt],neg_mat[0:neg_cnt,:],axis=1)# double check
                label_diff = np.ones((neg_cnt,1),dtype = np.int32)
                rest_num = pos_cnt - neg_cnt
                if rest_num%2 == 1:
                    raise Exception('rest_num should be even number')
                middle_ind = neg_cnt+ 0.5*rest_num
                sample_p[neg_cnt:self.sample_num,:] = np.concatenate(pos_mat[neg_cnt:middle_ind,:],neg_mat[middle_ind:neg_cnt,:],axis=1)# double check
                label_same = np.zeros((rest_num,1),dtype = np.int32)
                pdb.set_trace()
                sample_l = np.concatenate((label_diff,label_same),axis=0)

        #dataNew = './matlab/location_box_oneimg_train.mat' 
        #scio.savemat(dataNew, {'position_out':sample_p})
        #labelNew = './matlab/label_box_oneimg_train.mat' 
        #scio.savemat(labelNew, {'pred_label':sample_l})
        return sample_p, sample_l

    def get_features_from_position(self,feature_all,sample_p,forward_img_size):
        # feature_all = 1:128:H*W(H and W are image size)
        # sample_p = 20*8*h*w(h and w are subsample size)
        # forward_img_size =[3,3]
        num_channel_in = feature_all.shape[1]
        num_channel_out = 2*num_channel_in
        num_batch_out = sample_p.shape[0]
        num_row_out = forward_img_size[0]
        num_col_out = forward_img_size[1]

        features = np.zeros((num_batch_out,num_channel_out,num_row_out,num_col_out))

        noutpixel = self.out_pixel
        for ind in range(0,num_batch_out):
            loc_vec = sample_p[ind,:]
            xmin1 = loc_vec[0] 
            ymin1 = loc_vec[1]
            xmax1 = loc_vec[2]
            ymax1 = loc_vec[3]
            xmin2 = loc_vec[4]
            ymin2 = loc_vec[5]
            xmax2 = loc_vec[6]
            ymax2 = loc_vec[7]

            # resize feature pair for each channel
            # actually it is down sample     
            yinds1 = (np.linspace(ymin1, ymax1, noutpixel)+0.5).astype(np.uint32)
            xinds1 = (np.linspace(xmin1, xmax1, noutpixel)+0.5).astype(np.uint32)
            yinds2 = (np.linspace(ymin2, ymax2, noutpixel)+0.5).astype(np.uint32)
            xinds2 = (np.linspace(xmin2, xmax2, noutpixel)+0.5).astype(np.uint32)

            tmp_feature1_rows = feature_all[:,:,yinds1,:] #20*128*3*3
            tmp_feature1 = tmp_feature1_rows[:,:,:,xinds1]
            tmp_feature2_rows = feature_all[:,:,yinds2,:] #20*128*3*3s
            tmp_feature2 = tmp_feature1_rows[:,:,:,xinds2]

            tmp_feature = np.concatenate((tmp_feature1,tmp_feature2),axis=1) # concate on channels,20*256*3*3
            features[ind,:,:,:]=tmp_feature

        return features

class SubSampleLayerTest(caffe.Layer):
    def setup(self, bottom, top):
        #params = eval(self.param_str)
        self.width_ratio = 0.2 #size of subsample
        self.height_ratio = 0.2
        self.wh_ratio_notarget = 0.1 # when no target on image. the ratio r.r.d img width(after conv3),5*8
        self.candibox_ratio = 0.5 # range to generate subsample
        self.sample_num = 20 # 10 pairs of (+-) pair; 5 pairs of (+.+); 5 pairs of (- -)
        self.out_pixel = 3 # the data into next layer is 256*3*3
        #self.sample_num = params.get('sample_num', 1024) 
        np.random.seed(7)


    def reshape(self, bottom, top):

        self.data_shape = bottom[0].data.shape
        self.batch_num = self.data_shape[0]*self.sample_num
        self.channel_num = self.data_shape[1]
        self.img_height = self.data_shape[2]
        self.img_width = self.data_shape[3]
        if self.data_shape[0]>1:
            raise ValueError('The batch num of bottom blob should be one!')

        self.label_shape = bottom[1].data.shape
        self.batch_label_num = self.label_shape[0]*self.sample_num

        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(self.batch_num,self.channel_num*2,self.out_pixel,self.out_pixel)
        #top[1].reshape(self.batch_label_num,1,1,1)
        top[1].reshape(self.batch_label_num,1)
        #top[1].reshape(*bottom[1].data.shape)

    def forward(self, bottom, top):
        # bottom[0]: img features after conv3 but before cropping sample patches
        # bottom[1]: 0 1label after conv3
        # top[0]: cropped sample patches (in pair, concated in axis1)
        forward_img_size = [self.out_pixel,self.out_pixel]#3*3; order ->[num_row_out,num_col_out]
        # step1: get subsample positions
        features = bottom[0].data
        labels = bottom[1].data

        sample_p, sample_l = self.get_subsample_by_positive_seed(features, labels, self.img_width, self.img_height) # set positive_seed in this function

        # step2: get subfeature maps from sample_p, that's same as color images
        resized_features = self.get_features_from_position(features,sample_p,forward_img_size )# 20*256*3*3

        # step4: assign output, concate matrix on axis 1
        label_out = sample_l
        top[0].data[...] = resized_features
        top[1].data[...] = label_out

    def backward(self, top, propagate_down, bottom):
        pass

    def get_subsample_by_positive_seed(self,features, label, img_width, img_height):
        # position_out = [xmin ymin xmax ymax, xmin1 ymin1 xmax1 ymax1; xmin_,ymin_,xmax_,ymax_,xmin1_,ymin1_,xmax1_,ymax1_]
        # label_out = [0;1]
        num_sample = self.sample_num # the first point is the seed bounding box which is positive here
        label_out = np.zeros((num_sample,1),dtype=np.int32)
        position_out = np.zeros((num_sample,8),dtype = np.uint32)
        # generate seed bounding box location
        # get sample range: around bounding boxes
        labelnp = np.array(label, dtype=np.uint8).squeeze() 
        labelnp[labelnp>0] = 1
        rows, cols = np.nonzero(labelnp)    
        #pdb.set_trace()
        # get a positive seed bounding box
        if np.count_nonzero(rows)>20: # must meet this condition
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
            sample_height = round(maskheight*self.height_ratio)
            #pdb.set_trace()
            # generate subsamples
            pos_mat = np.zeros((1,4),dtype=np.uint32)
            pair_mat = np.zeros((num_sample,4),dtype=np.uint32)
            pos_cnt = 0
            pair_cnt = 0
            pos_pair_cnt = 0

            while(pos_cnt<1 or pair_cnt < num_sample):
                if sample_xmin<(sample_xmax-sample_width): # get xmin and generate xmax according to width
                	xmin = int(np.random.randint(sample_xmin, sample_xmax - sample_width))
                	xmax = int(xmin + sample_width - 1)
                else:# randomly generate xmin and xmax in location range(no more considering width)
				    xcandis = np.random.choice((sample_xmin,sample_xmax),2,replace=False)
				    xcandis = np.sort(xcandis)
				    xmin = int(xcandis[0])
				    xmax = int(xcandis[1])
				
                if sample_ymin<sample_ymax-sample_height:
                	ymin = int(np.random.randint(sample_ymin, sample_ymax - sample_height))
                	ymax = int(ymin + sample_height -1)
                else:
				    ycandis = np.random.choice((sample_ymin,sample_ymax),2,replace=False)
				    ycandis = np.sort(ycandis)
				    ymin=int(ycandis[0])
				    ymax=int(ycandis[1])



                # judge the label of this new subsample
                filtered_mask = labelnp[ymin:ymax+1,xmin:xmax+1]
                if isnan(filtered_mask.mean()  ):
                    print('Nan in Mean!!!!!')
                    sys.stdout.flush()    
                    pdb.set_trace()         

                if filtered_mask.mean() >= .8 and pos_cnt<1:
                    # get 'good' positive subsample
                    #pdb.set_trace()
                    pos_mat[pos_cnt,:] = [xmin, ymin, xmax, ymax]
                    pos_cnt = pos_cnt+1 
                    continue
                elif filtered_mask.mean()>0.6 and pos_pair_cnt <10: # get a positive pair
                    #pdb.set_trace()
                    pair_mat[pair_cnt,:] = [xmin, ymin, xmax, ymax]
                    label_out[pair_cnt,:] = 0 # cause the reference is positivelabel_out 
                    pair_cnt = pair_cnt + 1
                    pos_pair_cnt = pos_pair_cnt + 1
                    continue
                elif filtered_mask.mean()<0.4 and pos_pair_cnt>=10: # please get pos pair cnt first!!
                    #pdb.set_trace()
                    pair_mat[pair_cnt,:] = [xmin, ymin, xmax, ymax]
                    label_out[pair_cnt,:] = 1 # cause the reference is positive
                    pair_cnt = pair_cnt + 1

            #pdb.set_trace()
            # concate pos_mat for each row in pair_mat to generate patch pairs
            for fillidx in range(0,num_sample):
                position_out[fillidx,0:4] = pos_mat[0,:]
                position_out[fillidx,4:8] = pair_mat[fillidx,:]

            #pdb.set_trace()
        #dataNew = './matlab/location_box_oneimg.mat' 
        #scio.savemat(dataNew, {'position_out':position_out})

        return position_out,label_out


    def get_features_from_position(self,feature_all,sample_p,forward_img_size):
        # feature_all = 1:128:H*W(H and W are image size)
        # sample_p = 20*8*h*w(h and w are subsample size)
        # forward_img_size =[3,3]
        num_channel_in = feature_all.shape[1]
        num_channel_out = 2*num_channel_in
        num_batch_out = sample_p.shape[0]
        num_row_out = forward_img_size[0]
        num_col_out = forward_img_size[1]

        features = np.zeros((num_batch_out,num_channel_out,num_row_out,num_col_out))

        noutpixel = self.out_pixel
        for ind in range(0,num_batch_out):
            loc_vec = sample_p[ind,:]
            xmin1 = loc_vec[0] 
            ymin1 = loc_vec[1]
            xmax1 = loc_vec[2]
            ymax1 = loc_vec[3]
            xmin2 = loc_vec[4]
            ymin2 = loc_vec[5]
            xmax2 = loc_vec[6]
            ymax2 = loc_vec[7]

            # resize feature pair for each channel
            # actually it is down sample     
            yinds1 = (np.linspace(ymin1, ymax1, noutpixel)+0.5).astype(np.uint32)
            xinds1 = (np.linspace(xmin1, xmax1, noutpixel)+0.5).astype(np.uint32)
            yinds2 = (np.linspace(ymin2, ymax2, noutpixel)+0.5).astype(np.uint32)
            xinds2 = (np.linspace(xmin2, xmax2, noutpixel)+0.5).astype(np.uint32)

            tmp_feature1_rows = feature_all[:,:,yinds1,:] #20*128*3*3
            tmp_feature1 = tmp_feature1_rows[:,:,:,xinds1]
            tmp_feature2_rows = feature_all[:,:,yinds2,:] #20*128*3*3s
            tmp_feature2 = tmp_feature1_rows[:,:,:,xinds2]

            tmp_feature = np.concatenate((tmp_feature1,tmp_feature2),axis=1) # concate on channels,20*256*3*3
            features[ind,:,:,:]=tmp_feature

        return features


class SubPointLayer(caffe.Layer):
    def setup(self, bottom, top):
        #params = eval(self.param_str)
        self.width_ratio = 0.2 #size of subsample
        self.height_ratio = 0.2
        self.wh_ratio_notarget = 0.1 # when no target on image. the ratio r.r.d img width(after conv3),5*8
        self.candibox_ratio = 0.5 # range to generate subsample
        self.point_num = 180 # 10 pairs of (+-) pair; 5 pairs of (+.+); 5 pairs of (- -)
        self.out_pixel = 1
        np.random.seed(7)




    def reshape(self, bottom, top):
        # Input shape
        self.data_shape = bottom[0].data.shape
        self.batch_num = self.data_shape[0]
        self.channel_num = self.data_shape[1]
        self.img_height = self.data_shape[2]
        self.img_width = self.data_shape[3]
        if self.data_shape[0]>1:
            raise ValueError('The batch num of bottom blob should be one!')

        self.label_shape = bottom[1].data.shape
        #pdb.set_trace()
        # Output shape
        self.batch_label_num = self.label_shape[0]*self.point_num
        self.batch_num_out = self.batch_num*self.point_num
        self.channel_num_out = self.channel_num*2
        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(self.batch_num_out,self.channel_num_out,self.out_pixel,self.out_pixel)#self.out_pixel=1
        #top[1].reshape(self.batch_label_num,1,1,1)
        top[1].reshape(self.batch_label_num,1)# 180*1
        #top[1].reshape(*bottom[1].data.shape)

    def forward(self, bottom, top):
        # bottom[0]: img features after conv3 but before cropping sample patches
        # bottom[1]: 0 1label after conv3
        # top[0]: points (in pair, concated in axis1)
        # top[1]: 0/1 labels corresponding to top[0],0 means same class; 1 means different class
       
        # step1: get subsample positions
        features = bottom[0].data
        labels = bottom[1].data

        sample_p, sample_l = self.get_points(features, labels, self.img_width, self.img_height) # sample_p and sample_l are numpy matrix

        # step2: get subfeature maps from sample_p, that's same as color images
        resized_features = self.get_features_from_point_position(features,sample_p)# 20*256*3*3

        # step4: assign output, concate matrix on axis 1
        #label_out = sample_l[...,np.newaxis,np.newaxis]# TODO: CHECK SHAPE OF SAMPLE_L
        label_out = sample_l
        top[0].data[...] = resized_features
        top[1].data[...] = label_out

    def backward(self, top, propagate_down, bottom):
        pass

    def get_points(self,features, label, img_width, img_height):
        # point_p = [x1 y1 x2 y2; x_1,y_1,x_2,y_2]
        # point_l = [0;1]
        num_pos_point = self.point_num
        num_neg_point = self.point_num
		# define positions and labels of generated patch pair boxes
        point_p = np.zeros((self.point_num, 4),dtype=np.uint32)
        point_l = np.zeros((self.point_num, 1),dtype=np.uint32)
        pos_cnt = 0
        neg_cnt = 0
        #pdb.set_trace()
        # get sample range: around bounding boxes
        labelnp = np.array(label, dtype=np.uint8).squeeze() 

        labelnp[labelnp>0] = 1
        rows, cols = np.nonzero(labelnp)
        if np.count_nonzero(rows)>20: # large target in this image
            rowmin = rows.min()
            rowmax = rows.max()
            colmin = cols.min()
            colmax = cols.max()
            maskwidth = colmax-colmin+1
            maskheight = rowmax-rowmin+1
            # get sample area: bigger than bounding box area
            sample_xmin = int(max(1, round(colmin-self.candibox_ratio*maskwidth)))
            sample_ymin = int(max(1, round(rowmin-self.candibox_ratio*maskheight)))
            sample_xmax = int(min(img_width-1, round(colmax+self.candibox_ratio*maskwidth)))
            sample_ymax = int(min(img_height-1, round(rowmax+self.candibox_ratio*maskheight)))

            #pdb.set_trace()
            # generate subsamples
            pos_mat = np.zeros((num_pos_point,2),dtype=np.uint32)
            neg_mat = np.zeros((num_neg_point,2),dtype=np.uint32)
     
            while(pos_cnt<num_pos_point or neg_cnt<num_neg_point):
                # generate a new point
                x = int(np.random.randint(sample_xmin, sample_xmax))
                y = int(np.random.randint(sample_ymin, sample_ymax))

                # judge the label of this new point
                point_label = labelnp[y,x]     

                if point_label>0: # positive point
                    if pos_cnt<num_pos_point:
                        # get positive subsample
                        pos_mat[pos_cnt,:] = [x,y]
                        pos_cnt = pos_cnt+1
                elif point_label==0:
                    if neg_cnt<num_neg_point:
                        # get negative point
                        neg_mat[neg_cnt,:] = [x,y]
                        neg_cnt = neg_cnt + 1

            # arrange subsamples into sample_position which include positions in pair
            diff_cnt_thred = int(round(self.point_num*0.5))# the num of diff patch pairs
            #pdb.set_trace()
            # part1:[pos neg] or [neg pos] pair
            pos_mat_for_part1 = pos_mat[0:diff_cnt_thred]
            neg_mat_for_part1 = neg_mat[0:diff_cnt_thred]
            tmp_data_diffa = np.concatenate((pos_mat_for_part1[0:round(diff_cnt_thred*0.5),:],
    neg_mat_for_part1[0:round(diff_cnt_thred*0.5),:]),axis=1) #5*8
            tmp_data_diffb = np.concatenate((neg_mat_for_part1[round(diff_cnt_thred*0.5):,:],
    pos_mat_for_part1[round(diff_cnt_thred*0.5):,:]),axis=1)#5*8
            data_part1 = np.concatenate((tmp_data_diffa,tmp_data_diffb),axis=0)
            label_part1 = np.ones((diff_cnt_thred,1),dtype = np.int32) # label=1
     
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
            label_part2 = np.zeros((npairpos+npairneg,1),dtype=np.int32) # label=0

            # put part1 and part2 together
            # pay attention to the label!
            point_p = np.concatenate((data_part1,data_part2),axis = 0)
            point_l = np.concatenate((label_part1,label_part2), axis=0)
        
        else: # no target or very tiny target in this image
            # randomly sample 40 patches in whole image area and compare the IOU with ground truth
            # when dividing 40 patches into patch pairs,
            # the minority part has higher priority to form diff pair
            # FOR EXAMPLE: pos patch=2;neg patch = 38;
            # Then first two pairs would be [pos neg] with label 1
            # The remains would be [neg neg] with label 0
            
            sample_xmin = 1
            sample_ymin = 1
            sample_xmax = img_width-1
            sample_ymax = img_height-1

            ###pdb.set_trace()
            # generate subsamples
            # they are large enough to store all posible locations
            pos_mat = np.zeros((self.point_num*2,2),dtype=np.uint32)
            neg_mat = np.zeros((self.point_num*2,2),dtype=np.uint32)
            pos_cnt = 0
            neg_cnt = 0
            for idx_whole in range(0,self.point_num*2):
                # generate a point
                x = np.random.randint(sample_xmin, sample_xmax)
                y = np.random.randint(sample_ymin, sample_ymax)

                # judge the label of this new subsample
                point_label = labelnp[y,x]
 
                if point_label>0:
                    pos_mat[pos_cnt,:] = [x, y]
                    pos_cnt = pos_cnt+1
                elif point_label==0:
                    # get negative subsample
                    neg_mat[neg_cnt,:] = [x, y]
                    neg_cnt = neg_cnt + 1

            pos_mat[~np.all(pos_mat==0,axis=1)] # double check
            neg_mat[~np.all(neg_mat==0,axis=1)]
            if pos_cnt==0: # all patches are negative
                point_p = np.concatenate((neg_mat[0:self.point_num,:],neg_mat[self.point_num:self.point_num*2,:]),axis = 1)
                point_l = np.zeros((self.point_num,1),dtype=np.int32) # label=0
            elif neg_cnt>pos_cnt:
                #pdb.set_trace()
                point_p[0:pos_cnt,:] = np.concatenate((pos_mat[0:pos_cnt,:],neg_mat[0:pos_cnt,:]),axis=1)# double check
                label_diff = np.ones((pos_cnt,1),dtype = np.int32)
                rest_num = neg_cnt - pos_cnt
                if rest_num%2 == 1:
                    raise Exception('rest_num should be even number')
                middle_ind = pos_cnt+ 0.5*rest_num
                #pdb.set_trace()
                point_p[pos_cnt:self.point_num,:] = np.concatenate((neg_mat[pos_cnt:middle_ind,:],neg_mat[middle_ind:self.point_num*2-pos_cnt,:]),axis=1)# double check
                label_same = np.zeros((0.5*rest_num,1),dtype = np.int32)
                #pdb.set_trace()
                point_l = np.concatenate((label_diff,label_same),axis=0)
            else: #pos_cnt>neg_cnt
                point_p[0:neg_cnt,:] = np.concatenate((pos_mat[0:neg_cnt,:],neg_mat[0:neg_cnt,:]),axis=1)# double check
                label_diff = np.ones((neg_cnt,1),dtype = np.int32)
                rest_num = pos_cnt - neg_cnt
                if rest_num%2 == 1:
                    raise Exception('rest_num should be even number')
                middle_ind = neg_cnt+ 0.5*rest_num
                point_p[neg_cnt:self.point_num,:] = np.concatenate((pos_mat[neg_cnt:middle_ind,:],neg_mat[middle_ind:neg_cnt,:]),axis=1)# double check
                label_same = np.zeros((0.5*rest_num,1),dtype = np.int32)
                #pdb.set_trace()
                point_l = np.concatenate((label_diff,label_same),axis=0)

        #dataNew = './matlab/location_point_oneimg_train.mat' 
        #scio.savemat(dataNew, {'position_out':point_p})
        #labelNew = './matlab/label_point_oneimg_train.mat' 
        #scio.savemat(labelNew, {'pred_label':point_l})
        return point_p, point_l

    def get_features_from_point_position(self,feature_all,sample_p):
        # feature_all = 1:128:H*W(H and W are image size)
        # sample_p = 180*4*1*1(h=1 and w=1 are subsample size)

        num_channel_in = feature_all.shape[1]
        num_channel_out = 2*num_channel_in
        num_batch_out = sample_p.shape[0] #180
        num_row_out = 1
        num_col_out = 1
        #pdb.set_trace()
        features = np.zeros((num_batch_out,num_channel_out,num_row_out,num_col_out))

        noutpixel = self.out_pixel
        for ind in range(0,num_batch_out): # for each point in 180 points
            loc_vec = sample_p[ind,:]
            x1 = loc_vec[0] 
            y1 = loc_vec[1]
            x2 = loc_vec[2] 
            y2 = loc_vec[3]

            tmp_feature1 = feature_all[:,:,y1,x1] #180*128*1*1
            tmp_feature2 = feature_all[:,:,y2,x2] #180*128*1*1
            feature1 = tmp_feature1[...,np.newaxis,np.newaxis]  
            feature2 = tmp_feature2[...,np.newaxis,np.newaxis]  
            #pdb.set_trace()
            tmp_feature = np.concatenate((feature1,feature2),axis=1) # concate on channels,20*256*3*3
            features[ind,:,:,:]=tmp_feature

        return features

class SubPointLayerTest(caffe.Layer):
    def setup(self, bottom, top):
        # set the range for sampling point
        #self.width_ratio = 0.2 #size of subsample
        #self.height_ratio = 0.2
        self.wh_ratio_notarget = 0.1 # when no target on image. the ratio r.r.d img width(after conv3),5*8
        self.candibox_ratio = 0.5 # range to generate subsample regarding the size of bounding box
        self.sample_num = 180 # 180 pairs points
        self.out_pixel = 1 # the data into next layer is 256*3*3
        #self.sample_num = params.get('sample_num', 1024) 
        np.random.seed(7)


    def reshape(self, bottom, top):

        self.data_shape = bottom[0].data.shape
        self.batch_num = self.data_shape[0]*self.sample_num
        self.channel_num = self.data_shape[1]
        self.img_height = self.data_shape[2]
        self.img_width = self.data_shape[3]
        if self.data_shape[0]>1:
            raise ValueError('The batch num of bottom blob should be one!')

        self.label_shape = bottom[1].data.shape
        self.batch_label_num = self.label_shape[0]*self.sample_num

        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(self.batch_num,self.channel_num*2,self.out_pixel,self.out_pixel)#180*256*1*1
        #top[1].reshape(self.batch_label_num,1,1,1)
        top[1].reshape(self.batch_label_num,1) #180*1
        #top[1].reshape(*bottom[1].data.shape)
        
    def forward(self, bottom, top):
        # bottom[0]: img features after conv3 but before cropping sample patches
        # bottom[1]: 0 1label after conv3
        # top[0]: cropped sample patches (in pair, concated in axis1)
        
        # step1: get subsample positions
        features = bottom[0].data #1*128*90*159
        labels = bottom[1].data #1*1*90*159

        sample_p, sample_l = self.get_point_by_positive_seed(features, labels, self.img_width, self.img_height) # set positive_seed in this function

        # step2: get subfeature maps from sample_p, that's same as color images
        resized_features = self.get_features_from_point_position(features,sample_p)# 20*256*3*3

        # step4: assign output, concate matrix on axis 1
        label_out = sample_l
        top[0].data[...] = resized_features
        top[1].data[...] = label_out

    def backward(self, top, propagate_down, bottom):
        pass

    def get_point_by_positive_seed(self,features, label, img_width, img_height):
        # position_out = [x1 y1,x2 y2; x_1 y_1,x_2 y_2]
        # label_out = [0;1]
        # the first point is the seed which is positive here
        # num_sample =180
        num_sample = self.sample_num 
        label_out = np.zeros((num_sample,1),dtype=np.int32)
        position_out = np.zeros((num_sample,4),dtype = np.uint32)
        # generate seed bounding box location
        # get sample range: around bounding boxes
        labelnp = np.array(label, dtype=np.uint8).squeeze() 
        labelnp[labelnp>0] = 1
        rows, cols = np.nonzero(labelnp)    
        #pdb.set_trace()
        # get a positive seed bounding box
        if np.count_nonzero(rows)>20: # must meet this condition
            # get sampling range of points: 
            # int in [sample_xmin,sample_xmax] and [sample_ymin, sample_ymax]
            rowmin = rows.min()
            rowmax = rows.max()
            colmin = cols.min()
            colmax = cols.max()
            maskwidth = colmax-colmin+1
            maskheight = rowmax-rowmin+1
            sample_xmin = int(max(1, round(colmin-self.candibox_ratio*maskwidth)))
            sample_ymin = int(max(1, round(rowmin-self.candibox_ratio*maskheight)))
            sample_xmax = int(min(img_width-1, round(colmax+self.candibox_ratio*maskwidth)))
            sample_ymax = int(min(img_height-1, round(rowmax+self.candibox_ratio*maskheight)))

            #pdb.set_trace()
            # generate one positive point as seed,
            # and randomly generate the rest as pair for the seed
            # [seed, pair1; seed, pair2]
            pos_mat = np.zeros((1,2),dtype=np.uint32)
            pair_mat = np.zeros((num_sample,2),dtype=np.uint32)
            pos_cnt = 0
            pair_cnt = 0
            pos_pair_cnt = 0

            while(pos_cnt<1 or pair_cnt < num_sample):
                # generate a point
                x = np.random.randint(sample_xmin, sample_xmax)
                y = np.random.randint(sample_ymin, sample_ymax)

                # judge the label of this new point
                # and classify them into groups
                point_label = labelnp[y,x]     

                if point_label>0 and pos_cnt<1:
                    # get a positive subsample as the seed
                    #pdb.set_trace()
                    pos_mat[pos_cnt,:] = [x, y]
                    pos_cnt = pos_cnt+1 
                    continue
                elif point_label>0 and pos_pair_cnt <round(0.5*num_sample): # get a positive pair
                    #pdb.set_trace()
                    pair_mat[pair_cnt,:] = [x, y]
                    label_out[pair_cnt,:] = 0 # cause the reference is positive label_out 
                    pair_cnt = pair_cnt + 1
                    pos_pair_cnt = pos_pair_cnt + 1
                    continue
                elif point_label==0 and pos_pair_cnt>=round(0.5*num_sample): 
                    # All pos pair have been got!!
                    #pdb.set_trace()
                    pair_mat[pair_cnt,:] = [x, y]
                    label_out[pair_cnt,:] = 1 # cause the reference is positive
                    pair_cnt = pair_cnt + 1

            #pdb.set_trace()
            # concate pos_mat for each row in pair_mat to generate patch pairs
            for fillidx in range(0,num_sample):
                position_out[fillidx,0:2] = pos_mat[0,:]
                position_out[fillidx,2:4] = pair_mat[fillidx,:]

            #pdb.set_trace()
            dataNew = './matlab/location_point_oneimg.mat' 
            scio.savemat(dataNew, {'position_out':position_out})

        return position_out,label_out

    def get_features_from_point_position(self,feature_all,sample_p):
        # feature_all = 1:128:H*W(H and W are image size)
        # sample_p = 180*2*1*1(h=1 and w=1 are subsample size)

        num_channel_in = feature_all.shape[1]
        num_channel_out = 2*num_channel_in
        num_batch_out = sample_p.shape[0] #180
        num_row_out = 1 # since it is point
        num_col_out = 1
        #pdb.set_trace()
        features = np.zeros((num_batch_out,num_channel_out,num_row_out,num_col_out))

        noutpixel = self.out_pixel
        for ind in range(0,num_batch_out): # for each point in 180 points
            loc_vec = sample_p[ind,:]
            x1 = loc_vec[0] 
            y1 = loc_vec[1]
            x2 = loc_vec[2] 
            y2 = loc_vec[3]

            tmp_feature1 = feature_all[:,:,y1,x1] #180*128*1*1
            tmp_feature2 = feature_all[:,:,y2,x2] #180*128*1*1
            feature1 = tmp_feature1[...,np.newaxis,np.newaxis]  
            feature2 = tmp_feature2[...,np.newaxis,np.newaxis]  
            #pdb.set_trace()
            tmp_feature = np.concatenate((feature1,feature2),axis=1) # concate on channels,20*256*3*3
            features[ind,:,:,:]=tmp_feature

        return features
