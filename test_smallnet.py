import numpy as np
from PIL import Image
import pdb
import os
import string
import scipy.io as scio  
import scipy
import time
import caffe
import traceback
import cv2
import sys
import docopt
import math 

caffe.set_mode_gpu()
caffe.set_device(1)

# Test img is specified by txt file

dataset_dir = '/data/zichen/PythonPrograms/dataset/DAVIS/'
test_inds_txt = '/data/zichen/PythonPrograms/dataset/DAVIS/ImageSets/480p/validate_model/train_oneimg.txt'#valblackswan.txt' use trainimg to test
save_root = './results/'
model_root = '/data/zichen/PythonPrograms/DeepLearning/Segmentation/VideoSeg/MotionSeg/TrackSeg_cap_general/'
imsz = [360,636] #[848,480] 
# choose model from:
# For models trained by patch samples:
# smallnet_oneimg/smallnet_on_duoset/smallnet_on_validateset
# For models trained by point samples:
# train_smallnet_points_iter_1000/
# all models are stored bellow: TrackSeg_cap_general/model/
net = caffe.Net('./model/deploy_smallnet.prototxt', model_root+'train_smallnet_points_iter_1000.caffemodel', caffe.TEST) 
color_img_mean = np.array((104.00699, 116.66877, 122.67892), dtype=np.float32)
#weight11 = net.params['conv1_1'][0].data
#weight12 = net.params['net2_conv1_1'][0].data
#pdb.set_trace()
#read train_inds_txt into img name list
img_path = []
label_path = []
f_ = open(test_inds_txt, 'r')
for line in f_:
    img_path.append(line.strip().split()[0])
    label_path.append(line.strip().split()[1])
f_.close()
test_frame_num = len(img_path)

# test on the 1st line of txt file
# read image
im1 = Image.open(dataset_dir+img_path[0])
in_1 = np.array(im1, dtype=np.uint8)

in_tmp1 = scipy.misc.imresize(in_1, imsz, interp='nearest', mode=None)
in_tmp1 = in_tmp1[:,:,::-1] # RGB->BGR
in_tmp1_float = in_tmp1.astype(np.float32)
in_tmp1_nomean = in_tmp1_float - color_img_mean
in_transpose = in_tmp1_nomean.transpose((2,0,1))
#in_ = in_transpose[np.newaxis,...]
in_ = in_transpose

net.blobs['data'].reshape(1, *in_.shape)
net.blobs['data'].data[...] = in_


## read label
label_im = cv2.imread(dataset_dir+label_path[0],0)
label_np = np.array(label_im, dtype = np.int32)
label_resized = scipy.misc.imresize(label_np, imsz, interp='nearest', mode=None)
#in_label = label_resized[np.newaxis, np.newaxis, ...]
in_label = label_resized[np.newaxis,...]
net.blobs['label'].reshape(1, *in_label.shape)
net.blobs['label'].data[...] = in_label




# run net and take argmax for prediction
net.forward()



# analysis classification result

predict_prob = np.array(net.blobs['softmax'].data, dtype=np.float32)
#predict_label = np.argmax(predict_prob,axis=1)
real_label = np.array(net.blobs['sublabel'].data,dtype = np.int32)


pred_label = np.argmax(predict_prob,axis=1)
dataNew = './matlab/pred_label_point_oneimg.mat' 
#dataNew = './matlab/pred_label_box_oneimg.mat' 
scio.savemat(dataNew, {'pred_label':pred_label})
pred_label = pred_label[...,np.newaxis]
diff_label = np.absolute(pred_label - real_label)
rto = float(diff_label.sum())/float(diff_label.shape[0])
pdb.set_trace()




