import numpy as np
from PIL import Image
import pdb
import os

import caffe
# parallel parallel parallel parallel parallel
caffe.set_mode_gpu()
caffe.set_device(0)

solver = caffe.SGDSolver('./model/solver_smallnet.prototxt')
#pdb.set_trace()
PROJ_DIR = '/data/zichen/PythonPrograms/DeepLearning/Segmentation/FCNsegmentation'
SAVE_DIR = 'run_best_perf_to_copy/be_copied'
DEPLOY_FILE = os.path.join(PROJ_DIR, SAVE_DIR, 'deploy.prototxt')
SAVED_MODEL_FILE = os.path.join(PROJ_DIR, SAVE_DIR, 'fcn8s.caffemodel')
# load weight
net0 = caffe.Net(DEPLOY_FILE, SAVED_MODEL_FILE, caffe.TEST)
params0 = net0.params.keys()# VGG net

#weight11 = solver.net.params['conv1_1'][0].data
#weight12 = solver.net.params['net2_conv1_1'][0].data
#pdb.set_trace()

for pr in params0:
    if pr=='conv4_1':
        #pdb.set_trace()
        break;
    for i in xrange(len(net0.params[pr])):
        print(pr)
        print(i)
        solver.net.params[pr][i].data[...] = net0.params[pr][i].data[...]
print('Weights coping finished.')
#weight21 = solver.net.params['conv1_1'][0].data
#weight22 = solver.net.params['net2_conv1_1'][0].data
#pdb.set_trace()
# run net and take argmax for prediction
solver.step(20000)
solver.net.save('smallnet_pt_bear.caffemodel')
#weight31 = solver.net.params['conv1_1'][0].data
#weight32 = solver.net.params['net2_conv1_1'][0].data
#pdb.set_trace()
