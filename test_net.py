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
from skimage.segmentation import slic

# set up caffe
caffe.set_mode_gpu()
caffe.set_device(0)

net = caffe.Net('./model/deploy.prototxt', './snapshots/params.caffemodel',caffe.TEST)

# params and buffer
im_w = 832; im_h = 480;
fl_w = 416; fl_h = 240;
intervals = [1,4,8,16,32]
alphas = [0.5, 0.8, 0.8, 1., 1.]
fl_intervals = [1,2,4,8]
fl_alphas = [0.5,0.8,0.8,1]
#intervals = [1]
num_int = len(intervals)
num_fl_int = len(fl_intervals)
queue_len = (intervals[-1] + 1)
prev_im = [None] * queue_len
prev_fg = [None] * queue_len
prev_ptr = 0
n_segments_ = 200
fg_thr = 20
fl_scr_thr = 0.2

# image list
data_dir = './DAVIS'
#image_set = osp.join(data_dir, 'ImageSets', '480p', 'valblackswan.txt')
image_set = osp.join(data_dir, 'ImageSets', '480p', 'val2.txt')
im_paths = []; gt_paths = [];
with open(image_set) as f:
    for line in f:
        names = line.strip().split()
        im_paths.append(data_dir + names[0] )
        gt_paths.append(data_dir + names[1] )
num_images = len(im_paths);
video_name = im_paths[0].split('/')[4]

# optiflo_dir should contain '/decre/img/videoname/acc[1-8]/*.png'
optiflo_dir = './DAVIS/OpticalFlowImages_wholeset/'

#taus = [0.1, 0.2, 0.3]
taus = [0.2]
for tau_i in range(0,1):#
    tau = taus[tau_i]
    print str(tau)
    frmi = 0;
    if not osp.exists('./results_czj/%d/'%(tau_i)):
        os.makedirs('./results_czj/%d/'%(tau_i))
    while True:
        if frmi >= num_images:
            text_file.close()
            print 'complete!'
            break;
            
        # first frame
        im = cv2.imread(im_paths[frmi], cv2.IMREAD_UNCHANGED)
        im = np.array(im, np.float32)
        im = cv2.resize(im, (im_w,im_h), interpolation=cv2.INTER_CUBIC)
        prev_im[0] = (im - np.array((104,117,124), np.float32)[np.newaxis, np.newaxis, :])
        prev_im[0] = prev_im[0].transpose( (2,0,1) )

        gt = cv2.imread(gt_paths[frmi], cv2.IMREAD_UNCHANGED)
        prev_fg[0] = cv2.resize(gt, (im_w,im_h), interpolation=cv2.INTER_NEAREST)
        prev_fg[0][prev_fg[0]>0] = 1
        prev_ptr = 1
        
        fg_mask = prev_fg[0]
        fg_y, fg_x = np.where(fg_mask > 0)
        min_x = max(np.min(fg_x), 0); max_x = min(np.max(fg_x), im_w - 1);
        min_y = max(np.min(fg_y), 0); max_y = min(np.max(fg_y), im_h - 1);
        min_x = np.min(fg_x); max_x = np.max(fg_x);
        min_y = np.min(fg_y); max_y = np.max(fg_y);
        
        ctrx = (max_x + min_x) * 0.5; ctry = (max_y + min_y) * 0.5;
        w = (max_x - min_x); h = (max_y - min_y);
        
        #text_file = open('./results_czj/%d/bbox_'%(tau_i)+video_name + '.txt', 'w')
        # following frames
        i = 0;
        while True:
            i = i + 1
            if frmi + i > num_images:
                text_file.close()
                break;
            video_name_ = im_paths[frmi + i].split('/')[4]
            if video_name_ != video_name:
                video_name = video_name_;
                queue_len = (intervals[-1] + 1)
                prev_im = [None] * queue_len
                prev_fg = [None] * queue_len
                prev_ptr = 0
                frmi = frmi + i
                text_file.close()
                break;
            
            # read color image
            im = cv2.imread(im_paths[frmi + i]).astype(np.float32, copy=False)
            
            im = cv2.resize(im, (im_w,im_h), interpolation=cv2.INTER_CUBIC)
            cur_im = (im - np.array((104,117,124), np.float32)[np.newaxis, np.newaxis, :])
            cur_im = cur_im.transpose( (2,0,1) )
            
            # prepare frames at different intervals
            for int_i in range(num_int - 1, -1, -1):
                if i >= intervals[int_i]:
                    int_ = int_i + 1
                    net.blobs['prev_im'].reshape(int_,*prev_im[0].shape)
                    net.blobs['cur_im'].reshape(int_,*cur_im.shape)
                    net.blobs['prev_fg'].reshape(int_, 1, *prev_fg[0].shape)
                    
                    for prev_i in range(int_):
                        ptr_ = (prev_ptr - intervals[prev_i]) % queue_len
                        net.blobs['prev_im'].data[prev_i, ...] = prev_im[ptr_]
                        net.blobs['cur_im'].data[prev_i, ...] = cur_im
                        
                        prev_fg_ = np.copy(prev_fg[ptr_])
                        prev_fg_[prev_fg_ < 0.5] = 0;
                        prev_fg_[prev_fg_ > 0.5] = 1;
                        net.blobs['prev_fg'].data[prev_i, ...] = prev_fg_
                    break;
             
            t0 = time.time()
            net.forward()
            print 'Time({:d}'.format(i) + ') {0:.2f}'.format(time.time() - t0)

            diff_rst = net.blobs['diff_rst'].data[...]
            
            #apply diff
            # semantic shift
            color_fg = np.zeros( prev_fg[0].shape )
            sum_alpha = 0.
            for ni in range(diff_rst.shape[0]):
                ptr_ = (prev_ptr - intervals[ni]) % queue_len
               
                diff_ = np.zeros( (im_h, im_w, 3) )
                for ci in range(diff_rst.shape[1]):
                    diff_[:,:,ci] = diff_rst[ni,ci,:,:]
                
                diff_[:min_y, :,:] = 0; diff_[max_y:, :,:] = 0;
                diff_[:,:min_x,:] = 0; diff_[:,max_x:,:] = 0;
                fg_mask = np.copy(prev_fg[ptr_])
                
                # pass probability
                #fg_mask = np.multiply(prev_fg[ptr_], diff_[:,:,1] + diff_[:,:,2]) + \
                #    np.multiply(1. - prev_fg[ptr_], diff_[:,:,2])
                #fg_mask = np.sqrt(fg_mask)
                
                # decide
                grow = (diff_[:,:,2] > tau) & (diff_[:,:,2] > diff_[:,:,0])
                decrease = (diff_[:,:,0] > tau) & (diff_[:,:,0] > diff_[:,:,2])
                fg_mask[grow] = 1
                fg_mask[decrease] = 0
               
                color_fg = color_fg + alphas[ni] * fg_mask
                sum_alpha = sum_alpha + alphas[ni]
            color_fg = color_fg / max(sum_alpha, alphas[0]) #avg
            
            # smooth label
            #prev_fg_ = cv2.resize(prev_fg[(prev_ptr - 1) % queue_len], (104,60), interpolation=cv2.INTER_NEAREST )
            #prev_fg_ = cv2.GaussianBlur(prev_fg_ * 255., (21,21), 50)
            #prev_fg_[prev_fg_ > 1] = 1.
            #spatial_smooth = cv2.resize(prev_fg_, (832,480), interpolation=cv2.INTER_NEAREST )
            #cur_fg = np.multiply(cur_fg, spatial_smooth)
            
            cur_fg = cv2.blur(color_fg, (21,21))
            
            # read optical flow
            tmp_fg = cv2.resize(cur_fg, (fl_w, fl_h), interpolation=cv2.INTER_CUBIC)
            cluster_img = np.zeros(tmp_fg.shape)
            sum_alpha = 0
            for int_i in range(num_fl_int):
                if i >= fl_intervals[int_i]:
                
                    tmp_cluster_img = np.zeros( (fl_h, fl_w))
                    int_ = fl_intervals[int_i]
                    optiflo_file = osp.join(optiflo_dir, 'decre', 'img', video_name, 'acc%d'%(int_), '%05d.png'%(i))
                    flow_acc1 = cv2.imread(optiflo_file)
                    flow_acc1 = cv2.resize(flow_acc1, (fl_w,fl_h), interpolation=cv2.INTER_CUBIC)
                    segments = slic(flow_acc1, n_segments=n_segments_)
                    
                    # cluster, merge and filter
                    fg_segments = [False] * n_segments_
                    masks = [None] * n_segments_
                    
                    bg_feat = np.zeros( (1,1,3) );
                    bg_n = 0
                    for si in range(n_segments_):
                        mask = (segments == si)
                        if np.sum(mask) < 1:
                            fg_segments[si] = False
                            continue
                            
                        scr = np.mean(tmp_fg[mask])
                        masks[si] = mask
                        if scr > fl_scr_thr:
                            fg_segments[si] = True
                        else:
                            fg_segments[si] = False
                            bg_feat = bg_feat + np.mean(flow_acc1[mask, :])
                            bg_n = bg_n + 1
                    bg_feat = bg_feat / max(bg_n, 1.0)
                    
                    for si in range(n_segments_):
                        if fg_segments[si]:
                            mean_feat = np.mean(flow_acc1[masks[si], :])
                            if np.mean(np.abs(mean_feat - bg_feat)) > fg_thr:
                                tmp_cluster_img[masks[si]] = 1
                            
                    cluster_img = cluster_img + tmp_cluster_img * fl_alphas[int_i]
                    sum_alpha = sum_alpha + fl_alphas[int_i]
                    
            cluster_img = cluster_img / max(sum_alpha, fl_alphas[0]) #avg
            cur_fg = cv2.resize(cluster_img, (im_w, im_h), interpolation=cv2.INTER_NEAREST)    
            
            # update 
            ptr_ =  (prev_ptr - 1) % queue_len
            cur_fg = 0.8 * cur_fg + 0.2 * prev_fg[ptr_]
            
            fg_mask = np.copy(cur_fg)
            # find coordinates
            fg_y, fg_x = np.where(fg_mask > 0.1)
            if len(fg_y) > 0:
                min_x = max(np.min(fg_x), 0); max_x = min(np.max(fg_x), im_w - 1);
                min_y = max(np.min(fg_y), 0); max_y = min(np.max(fg_y), im_h - 1);
                ctrx = 0.2 * ctrx + 0.8 * (max_x + min_x) * 0.5;
                ctry = 0.2 * ctry + 0.8 * (max_y + min_y) * 0.5;
                w = max(0.2 * w + 0.8 * (max_x - min_x), 150); 
                h = max(0.2 * h + 0.8 * (max_y - min_y), 150);
                
                min_x = max(ctrx - w * 0.5 - 100, 0); max_x = min(ctrx + w * 0.5 + 100, im_w);
                min_y = max(ctry - h * 0.5 - 100, 0); max_y = min(ctry + h * 0.5 + 100, im_h);
                min_x = int(min_x); min_y = int(min_y); 
                max_x = int(max_x); max_y = int(max_y);

            prev_fg[prev_ptr] = cur_fg
            prev_im[prev_ptr] = cur_im
            prev_ptr = (prev_ptr + 1) % queue_len
            
            x1 = int(max(ctrx - w * 0.5,0)); x2 = int(min(ctrx + w * 0.5,im_w));
            y1tmp = max(ctry - h * 0.5,0); y2tmp = min(ctry + h * 0.5,im_h);
            y1 = int(y1tmp*854./832.0); y2 = int(y2tmp*854./832.0);
            # visualize
            im[:,:,1] = im[:,:,1] + 255 * cur_fg
            im[im>255] = 255
            #cv2.rectangle(im, (x1, y1), (x2, y2), color=(0,0,255))
            cv2.imshow("img", im.astype(np.uint8))
            cv2.waitKey(0)
            
            #write results
            #if not osp.exists('./results_czj/%d/'%(tau_i)+video_name):
            #    os.makedirs('./results_czj/%d/'%(tau_i)+video_name)
#            pdb.set_trace()
            #fg_img_ = cv2.resize(cur_fg,(854,480), interpolation=cv2.INTER_CUBIC);
            #fg_img_ = fg_img_*255
            #fg_img_[fg_img_ > 255] = 255
            #cv2.imwrite('./results_czj/%d/'%(tau_i)+video_name+'/%05d.png'%(i), np.array(fg_img_, np.uint8))
            #text_file.write('%d %d %d %d\n' %(x1,y1,x2,y2))

