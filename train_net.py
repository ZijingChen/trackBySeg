import caffe

caffe.set_mode_gpu()
caffe.set_device(1)

solver = caffe.SGDSolver('./model/solver_optical_flow.prototxt')

stepsize = 1000
for epoch in range(0,50):
  solver.step(stepsize)
  solver.net.save('./snapshots/params_optical_flow.caffemodel'.format(epoch))
    
