#set path
import sys
import os.path as osp
CURDIR = osp.dirname(__file__)
sys.path.insert(0, osp.join(CURDIR, 'OpticalFlowToolkit'))

from skimage.segmentation import slic, mark_boundaries
# import packages
import caffe
import cv2
import numpy as np
import time, glob, os
import pdb
from lib import flowlib as fl

# set up caffe
caffe.set_mode_gpu()
caffe.set_device(1)

net = caffe.Net('./model/deploy_optical_flow.prototxt', './snapshots/params_optical_flow.caffemodel',caffe.TEST)

# params and buffer
im_w = 832; im_h = 480;
intervals = [1,2,4]
#intervals = [1]
alphas = [0.2, 0.5, 1]
num_int = len(intervals)
queue_len = (intervals[-1] + 1)
prev_fg = [None] * queue_len
prev_ptr = 0

# image list
data_dir = './DAVIS'
image_set = osp.join(data_dir, 'ImageSets', '480p', 'valblackswan.txt')
image_set = osp.join(data_dir, 'ImageSets', '480p', 'trainbear.txt')
#image_set = osp.join(data_dir, 'ImageSets', '480p', 'val.txt')
im_paths = []; gt_paths = [];
with open(image_set) as f:
    for line in f:
        names = line.strip().split()
        im_paths.append(data_dir + names[0] )
        gt_paths.append(data_dir + names[1] )
num_images = len(im_paths);
video_name = im_paths[0].split('/')[-2]

optiflo_dir = './DAVIS/OpticalFlowImages_wholeset'
#optiflo_dir = '/home/zichen/PythonPrograms/dataset/DAVIS/OpticalFlowImages_wholeset/incre/flo'

frmi = 0;
while True:
    if frmi > num_images:
        print 'complete!'
        break;
        
    # first frame
    gt = cv2.imread(gt_paths[frmi], cv2.IMREAD_UNCHANGED)
    prev_fg[0] = cv2.resize(gt, (im_w,im_h), interpolation=cv2.INTER_NEAREST)
    prev_fg[0][prev_fg[0]>0] = 1
    prev_ptr = 1
    
    fg_mask = prev_fg[(prev_ptr - 1) % queue_len]
    fg_y, fg_x = np.where(fg_mask == 1)
    min_x = max(np.min(fg_x), 0); max_x = min(np.max(fg_x), im_w - 1);
    min_y = max(np.min(fg_y), 0); max_y = min(np.max(fg_y), im_h - 1);
    
    ctrx = (max_x + min_x) * 0.5; ctry = (max_y + min_y) * 0.5;
    w = (max_x - min_x); h = (max_y - min_y);
    
    # following frames
    i = 0;
    while True:
        i = i + 1
        video_name_ = im_paths[frmi + i].split('/')[-2]
        if video_name_ != video_name:
            video_name = video_name_;
            queue_len = (intervals[-1] + 1)
            prev_fg = [None] * queue_len
            prev_ptr = 0
            frmi = frmi + i
            break;
        
        fg_mask = prev_fg[(prev_ptr - 1) % queue_len]
        # find coordinates
        fg_y, fg_x = np.where(fg_mask == 1)
        if len(fg_y) > 0:
            min_x = max(np.min(fg_x), 0); max_x = min(np.max(fg_x), im_w - 1);
            min_y = max(np.min(fg_y), 0); max_y = min(np.max(fg_y), im_h - 1);
            ctrx = 0.2 * ctrx + 0.8 * (max_x + min_x) * 0.5;
            ctry = 0.2 * ctry + 0.8 * (max_y + min_y) * 0.5;
            w = max(0.2 * w + 0.8 * (max_x - min_x), 244); 
            h = max(0.2 * h + 0.8 * (max_y - min_y), 244);
            
            min_x = max(ctrx - w * 0.5 - 50, 0); max_x = min(ctrx + w * 0.5 + 50, im_w);
            min_y = max(ctry - h * 0.5 - 50, 0); max_y = min(ctry + h * 0.5 + 50, im_h);
        
        # prepare frames at different intervals
        for int_i in range(num_int - 1, -1, -1):
            if i >= intervals[int_i]:
                int_ = int_i + 1

                net.blobs['incre_flow'].reshape(int_,3, im_h, im_w)
                net.blobs['decre_flow'].reshape(int_,3, im_h, im_w)
                net.blobs['prev_fg'].reshape(int_, 1, *prev_fg[0].shape)
                
                for prev_i in range(int_):
                    #pdb.set_trace()
                    interval_ = intervals[prev_i]
                    incre_flow_path = osp.join(optiflo_dir, 'incre', 'img', video_name, 'acc%d' % interval_, '%05d.png'%(i));
                    decre_flow_path = osp.join(optiflo_dir, 'decre', 'img', video_name, 'acc%d' % interval_, '%05d.png'%(i));
                    
                    incre_flow = cv2.imread(incre_flow_path)
                    decre_flow = cv2.imread(decre_flow_path)
                    incre_flow = cv2.resize(incre_flow, (im_w, im_h), interpolation=cv2.INTER_CUBIC)
                    decre_flow = cv2.resize(decre_flow, (im_w, im_h), interpolation=cv2.INTER_CUBIC)

                    incre_flow = incre_flow.astype(np.float32, copy=False) - np.array((104,117,124), np.float32)[np.newaxis, np.newaxis, :]
                    incre_flow = incre_flow.transpose( (2,0,1) )[np.newaxis,...]
                    
                    decre_flow = decre_flow.astype(np.float32, copy=False) - np.array((104,117,124), np.float32)[np.newaxis, np.newaxis, :]
                    decre_flow = decre_flow.transpose( (2,0,1) )[np.newaxis,...]

                    ptr_ = (prev_ptr - interval_) % queue_len
                    net.blobs['incre_flow'].data[prev_i, ...] = incre_flow
                    net.blobs['decre_flow'].data[prev_i, ...] = decre_flow
                    net.blobs['prev_fg'].data[prev_i, ...] = 0#prev_fg[ptr_]
                break;
         
        t0 = time.time()
        net.forward()
        print 'Time({:d}'.format(i) + ') {0:.2f}'.format(time.time() - t0)

        pred_rst = net.blobs['pred'].data[...]
        
        #apply diff
        pred_fg = np.zeros( prev_fg[0].shape )
        sum_alpha = 0.
        for ni in range(pred_rst.shape[0]):
            pred_ = pred_rst[ni,1,:,:]
            cv2.imshow("img", np.array(pred_ * 255, np.uint8))
            cv2.waitKey(0)
            #pred_[:min_y, :] = 0; pred_[max_y:, :] = 0;
            #pred_[:,:min_x] = 0; pred_[:,max_x:] = 0;
            
            fg_mask = pred_
           
            pred_fg = pred_fg + alphas[ni] * fg_mask
            sum_alpha = sum_alpha + alphas[ni]
        pred_fg = pred_fg / max(sum_alpha, alphas[0]) #avg
        
        # overall all
        cur_fg = pred_fg
        
        #cur_fg = cv2.blur(cur_fg, (15,15))
        
        # update 
        cur_fg[cur_fg < 0.9] = 0
        cur_fg[cur_fg > 0.9] = 1
        prev_fg[prev_ptr] = cur_fg
        prev_ptr = (prev_ptr + 1) % queue_len
        
        # visualize
        im = cv2.imread(im_paths[frmi + i]).astype(np.float32, copy=False)
        im = cv2.resize(im, (im_w,im_h), interpolation=cv2.INTER_CUBIC)
        im[:,:,1] = im[:,:,1] + 128 * cur_fg
        im[im>255] = 255
        cv2.imshow("img", im.astype(np.uint8))
        cv2.waitKey(0)
        
        #write results
        #if not osp.exists('./results/'+video_name):
        #    os.makedirs('./results/'+video_name)
        #cv2.imwrite('./results/'+video_name+'/%05d.png'%(i), np.array(cur_fg, np.float32))

