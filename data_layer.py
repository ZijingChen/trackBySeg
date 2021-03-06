import caffe

import scipy.io as sio
import numpy as np
import cv2
import random
import glob
import os.path as osp
import pdb

class DAVISDataLayer(caffe.Layer):
    def setup(self, bottom, top):
        # config
        params = eval(self.param_str)
        self.data_dir = params['data_dir'] # /path/to/DAVIS
        self.seed = params.get('seed', None)
        self.imsz = [480, 832]
        self.gtsz = [480, 832]
        self.aux_gtsz = [30, 52]
        
        image_set = osp.join(self.data_dir, 'ImageSets', '480p', 'train.txt')
        
        self.im_paths = []; self.gt_paths = [];
        with open(image_set) as f:
            for line in f:
                names = line.strip().split()
                self.im_paths.append(self.data_dir + names[0] )
                self.gt_paths.append(self.data_dir + names[1] )
                
        self.num_images = len(self.im_paths);
        self.intervals = [1,2,4]
        self.num_intervals = len(self.intervals)
      
        #self.rand_im_id = [None] * self.batch_num
        #self.crop_param = np.zeros((self.batch_num, 4), np.float32) # crop coord, crop sz
        #self.aug_mirror = [None] * self.batch_num
        #self.aug_color = [None] * self.batch_num # 0: gaussian noise; 1 dropout; 2: hsv; 3: grayscale
        #self.aug_affine = [None] * self.batch_num # 0: h = 1.5 * h; 1: w = 1.5 * w;

        # three tops: data and label
        #if len(top) != 4:
        #    raise Exception("Need to define 4 tops: data, 3d data, and labels.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # randomization: seed
        random.seed(self.seed)

    def reshape(self, bottom, top):
        # pick random images and random interval
        #
        while True:
            interval = self.intervals[random.randint(0,self.num_intervals-1)]
            prev_id = random.randint(0, self.num_images - 1 - interval)
            cur_id = prev_id + interval
            if self.im_paths[prev_id].split('/')[3] == self.im_paths[cur_id].split('/')[3]: # if same video
                self.prev_impath = self.im_paths[prev_id]
                self.cur_impath = self.im_paths[cur_id]
                self.prev_gtpath = self.gt_paths[prev_id]
                self.cur_gtpath = self.gt_paths[cur_id]
                break;
        
        # load image + label image pair
        self.prepare_input()
       
        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(*self.prev_im.shape)
        top[1].reshape(*self.cur_im.shape)
        top[2].reshape(*self.prev_gt.shape)
        top[3].reshape(*self.diff_gt.shape)
        #top[4].reshape(*self.aux_prev_gt.shape)
        top[4].reshape(*self.aux_cur_gt.shape)


    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.prev_im
        top[1].data[...] = self.cur_im
        top[2].data[...] = self.prev_gt
        top[3].data[...] = self.diff_gt
        #top[4].data[...] = self.aux_prev_gt
        top[4].data[...] = self.aux_cur_gt

    def backward(self, top, propagate_down, bottom):
        pass

    def prepare_input(self):
        # random augmentation method
        #self.aug_color[i] = random.randint(0, 4) # 0: orig; 1: gaussian noise; 2 dropout; 3: hsv; 4: grayscale
        #self.aug_affine[i] = random.randint(0, 4) # 0: orig; 1: h = 1.2 * h; 2: w = 1.5 * w; 3: h = 0.8 * h; 4: w = 0.65 * w
            
        cv2.namedWindow("img")
        self.prev_im = cv2.imread(self.prev_impath)
        self.cur_im = cv2.imread(self.cur_impath)
        self.prev_im = cv2.resize(self.prev_im, (self.imsz[1], self.imsz[0]), interpolation=cv2.INTER_CUBIC)
        self.cur_im = cv2.resize(self.cur_im, (self.imsz[1], self.imsz[0]), interpolation=cv2.INTER_CUBIC)
        
        self.prev_gt = cv2.imread(self.prev_gtpath, cv2.IMREAD_UNCHANGED)
        self.cur_gt = cv2.imread(self.cur_gtpath, cv2.IMREAD_UNCHANGED)
        self.prev_gt = cv2.resize(self.prev_gt, (self.gtsz[1], self.gtsz[0]), interpolation=cv2.INTER_NEAREST)
        self.cur_gt = cv2.resize(self.cur_gt, (self.gtsz[1], self.gtsz[0]), interpolation=cv2.INTER_NEAREST)
        self.prev_gt[self.prev_gt > 0] = 1
        self.cur_gt[self.cur_gt > 0] = 1
            
        # random shift the cur frame
        shift_x = random.randint(-2, 2); 
        shift_y = random.randint(-2, 2); 
        if shift_x < 0:
            self.cur_im[:,0:shift_x,:] = self.cur_im[:,-shift_x:,:]
            self.cur_im[:,shift_x:,:] = 0
            
            self.cur_gt[:,0:shift_x] = self.cur_gt[:,-shift_x:]
            self.cur_gt[:,shift_x:] = 255
        elif shift_x > 0:
            self.cur_im[:,shift_x:,:] = self.cur_im[:,0:-shift_x,:]
            self.cur_im[:,0:shift_x,:] = 0
            
            self.cur_gt[:,shift_x:] = self.cur_gt[:,0:-shift_x]
            self.cur_gt[:,0:shift_x] = 255
        if shift_y < 0:
            self.cur_im[0:shift_y,:,:] = self.cur_im[-shift_y:,:,:]
            self.cur_im[shift_y:,:,:] = 0
            
            self.cur_gt[0:shift_y,:] = self.cur_gt[-shift_y:,:]
            self.cur_gt[shift_y:,:] = 255
        elif shift_y > 0:
            self.cur_im[shift_y:,:,:] = self.cur_im[0:-shift_y,:,:]
            self.cur_im[0:shift_y:,:,:] = 0
            
            self.cur_gt[shift_y:,:] = self.cur_gt[0:-shift_y,:]
            self.cur_gt[0:shift_y,:] = 255
            
        
        self.aux_prev_gt = cv2.resize(self.prev_gt, (self.aux_gtsz[1], self.aux_gtsz[0]), interpolation=cv2.INTER_NEAREST)
        self.aux_cur_gt = cv2.resize(self.cur_gt, (self.aux_gtsz[1], self.aux_gtsz[0]), interpolation=cv2.INTER_NEAREST)
        # mirror the image
        if random.randint(0, 1) == 0:
            self.prev_im = np.fliplr(self.prev_im)
            self.cur_im = np.fliplr(self.cur_im)
            self.prev_gt = np.fliplr(self.prev_gt)
            self.cur_gt = np.fliplr(self.cur_gt)
            self.aux_prev_gt = np.fliplr(self.aux_prev_gt)
            self.aux_cur_gt = np.fliplr(self.aux_cur_gt)
            
        #TODO: crop the foreground window
               
        # augmentation in hsv 
        hue_noise = 5*np.random.randn()
        
        hsv = cv2.cvtColor(self.prev_im, cv2.COLOR_BGR2HSV)
        hue = np.array(hsv[:,:,0], np.float)
        hue[...] = np.remainder(hue+hue_noise, 180)
        hsv[:,:,0] = hue.astype(np.uint8); 
        self.prev_im = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        hsv = cv2.cvtColor(self.cur_im, cv2.COLOR_BGR2HSV)
        hue = np.array(hsv[:,:,0], np.float)
        hue[...] = np.remainder(hue+hue_noise, 180)
        hsv[:,:,0] = hue.astype(np.uint8); 
        self.cur_im = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        #self.prev_im[self.prev_im>255]=255; self.prev_im[self.prev_im<0] = 0;
        #self.cur_im[self.cur_im>255]=255; self.cur_im[self.cur_im<0] = 0;
        #cv2.imshow("img", self.prev_im)
        #cv2.waitKey(0)
        #cv2.imshow("img", self.cur_im)
        #cv2.waitKey(0)
        #cv2.imshow("img", self.prev_gt)
        #cv2.waitKey(0)
        #cv2.imshow("img", self.cur_gt)
        #cv2.waitKey(0)
        
        # gt for diff
        self.diff_gt = np.zeros( self.cur_gt.shape, np.float32) + 255
        self.diff_gt[ (self.prev_gt == 0) & (self.cur_gt == 1) ] = 2
        self.diff_gt[ (self.prev_gt == self.cur_gt) ] = 1
        self.diff_gt[ (self.prev_gt == 1) & (self.cur_gt == 0) ] = 0
        
        #tmp = np.zeros( (self.diff_gt.shape[0],self.diff_gt.shape[1],3), np.float32)
        #tmp[(self.diff_gt==0),0] = 255; tmp[(self.diff_gt==1),1] = 255; tmp[(self.diff_gt==2),2] = 255;
        #cv2.imshow("img", tmp)
        #cv2.waitKey(0)
        
        # prepare data
        self.prev_im = self.prev_im.astype(np.float32, copy=False) - np.array((104,117,124), np.float32)[np.newaxis, np.newaxis, :]
        self.prev_im = self.prev_im + 10 * np.random.randn()
        self.prev_im = self.prev_im.transpose( (2,0,1) )[np.newaxis,...]
        
        self.cur_im = self.cur_im.astype(np.float32, copy=False) - np.array((104,117,124), np.float32)[np.newaxis, np.newaxis, :]
        self.cur_im = self.cur_im + 10 * np.random.randn()
        self.cur_im = self.cur_im.transpose( (2,0,1) )[np.newaxis,...]
        
        self.diff_gt = self.diff_gt[np.newaxis, np.newaxis, ...].astype(np.float32, copy=False)
        
        self.aux_prev_gt = self.aux_prev_gt[np.newaxis, np.newaxis, ...].astype(np.float32, copy=False)
        self.aux_cur_gt = self.aux_cur_gt[np.newaxis, np.newaxis, ...].astype(np.float32, copy=False)
        self.prev_gt = self.prev_gt[np.newaxis, np.newaxis, ...].astype(np.float32, copy=False)
        self.cur_gt = self.cur_gt[np.newaxis, np.newaxis, ...].astype(np.float32, copy=False)
            


class DAVISOFDataLayer(caffe.Layer):
    def setup(self, bottom, top):
        # config
        params = eval(self.param_str)
        self.data_dir = params['data_dir'] # /path/to/DAVIS
        self.seed = params.get('seed', None)
        self.imsz = [480, 832]
        self.gtsz = [480, 832]
        
        image_set = osp.join(self.data_dir, 'ImageSets', '480p', 'train.txt')
        
        self.im_paths = []; self.gt_paths = [];
        with open(image_set) as f:
            for line in f:
                names = line.strip().split()
                self.im_paths.append(self.data_dir + names[0] )
                self.gt_paths.append(self.data_dir + names[1] )
                
        self.num_images = len(self.im_paths);
        self.intervals = [1,2,4]
        self.num_intervals = len(self.intervals)

        # three tops: data and label
        #if len(top) != 4:
        #    raise Exception("Need to define 4 tops: data, 3d data, and labels.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # randomization: seed
        random.seed(self.seed)

    def reshape(self, bottom, top):
        # pick random images and random interval
        #
        while True:
            interval = self.intervals[random.randint(0,self.num_intervals-1)]
            prev_id = random.randint(0, self.num_images - 1 - interval)
            cur_id = prev_id + interval
            if self.im_paths[prev_id].split('/')[-2] == self.im_paths[cur_id].split('/')[-2]: # if same video
                video_name = self.im_paths[cur_id].split('/')[-2]

                cur_impath = self.im_paths[cur_id]
                cur_frame = int(cur_impath.split('/')[-1].split('.')[0]) + 1

                self.incre_flow_path = osp.join(self.data_dir, 'OpticalFlowImages_wholeset', 'incre', 'img', video_name, 'acc%d' % interval, '%05d.png'%cur_frame)
                self.decre_flow_path = osp.join(self.data_dir, 'OpticalFlowImages_wholeset', 'decre', 'img', video_name, 'acc%d' % interval, '%05d.png'%cur_frame)
                if not osp.exists(self.incre_flow_path) or not osp.exists(self.decre_flow_path):
                    continue;
                self.prev_gtpath = self.gt_paths[prev_id]
                self.cur_gtpath = self.gt_paths[cur_id]
                break;
        
        # load image + label image pair
        self.prepare_input()
       
        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(*self.incre_flow.shape)
        top[1].reshape(*self.decre_flow.shape)
        top[2].reshape(*self.prev_gt.shape)
        top[3].reshape(*self.cur_gt.shape)


    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.incre_flow
        top[1].data[...] = self.decre_flow
        top[2].data[...] = self.prev_gt
        top[3].data[...] = self.cur_gt

    def backward(self, top, propagate_down, bottom):
        pass

    def prepare_input(self):
            
        #cv2.namedWindow("img")
        self.incre_flow = cv2.imread(self.incre_flow_path)
        self.decre_flow = cv2.imread(self.decre_flow_path)
        self.incre_flow = cv2.resize(self.incre_flow, (self.imsz[1], self.imsz[0]), interpolation=cv2.INTER_CUBIC)
        self.decre_flow = cv2.resize(self.decre_flow, (self.imsz[1], self.imsz[0]), interpolation=cv2.INTER_CUBIC)
        
        self.prev_gt = cv2.imread(self.prev_gtpath, cv2.IMREAD_UNCHANGED)
        self.cur_gt = cv2.imread(self.cur_gtpath, cv2.IMREAD_UNCHANGED)
        self.prev_gt = cv2.resize(self.prev_gt, (self.gtsz[1], self.gtsz[0]), interpolation=cv2.INTER_NEAREST)
        self.cur_gt = cv2.resize(self.cur_gt, (self.gtsz[1], self.gtsz[0]), interpolation=cv2.INTER_NEAREST)
        self.prev_gt[self.prev_gt > 0] = 1
        self.cur_gt[self.cur_gt > 0] = 1

        # mirror the image
        if random.randint(0, 1) == 0:
            self.incre_flow = np.fliplr(self.incre_flow)
            self.decre_flow = np.fliplr(self.decre_flow)
            self.prev_gt = np.fliplr(self.prev_gt)
            self.cur_gt = np.fliplr(self.cur_gt)
            
        # prepare data
        self.incre_flow = self.incre_flow.astype(np.float32, copy=False) - np.array((104,117,124), np.float32)[np.newaxis, np.newaxis, :]
        self.incre_flow= self.incre_flow + 10 * np.random.randn()
        self.incre_flow = self.incre_flow.transpose( (2,0,1) )[np.newaxis,...]
        
        self.decre_flow = self.decre_flow.astype(np.float32, copy=False) - np.array((104,117,124), np.float32)[np.newaxis, np.newaxis, :]
        self.decre_flow = self.decre_flow + 10 * np.random.randn()
        self.decre_flow = self.decre_flow.transpose( (2,0,1) )[np.newaxis,...]
        
        self.prev_gt = self.prev_gt[np.newaxis, np.newaxis, ...].astype(np.float32, copy=False)
        self.cur_gt = self.cur_gt[np.newaxis, np.newaxis, ...].astype(np.float32, copy=False)
            
