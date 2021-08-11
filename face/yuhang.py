import numpy as np
import cv2
from skimage import transform as trans
from utils.detector import Detector
import shutil
import pickle
import pdb
import sys
import os
import argparse
import mxnet as mx

def preprocess(img_path):
  img = cv2.imread(img_path, cv2.IMREAD_COLOR)
  img = img[...,::-1]
  img_size = [112,112,3]

  model_path = './model/mtcnn'
  detector = Detector(model_path, gpu_fraction=0.5)

  results = detector.detect_face(img, debug=False)
  if len(results)==1:
    result = results[0]

    bbox_list = result['bbox']
    det_lmk = np.array([result['left_eye'],result['right_eye'],result['nose'],result['left_mouth'],result['right_mouth']])
    src = np.array([
      [30.2946, 51.6963],
      [65.5318, 51.5014],
      [48.0252, 71.7366],
      [33.5493, 92.3655],
      [62.7299, 92.2041]], dtype=np.float32)
    if img_size[1] == 112:
      src[:,0] += 8.0
    tform = trans.SimilarityTransform()
    tform.estimate(det_lmk, src)
    M = tform.params[0:2,:]
    warped = cv2.warpAffine(img,M,(img_size[1],img_size[0]), borderValue = 0.0)
  else:
    print('no face detected or more than one face detected')
  return warped, bbox_list

class Extractor(object):
    def __init__(self, prefix, epoch, gpu_list=(0, ), batch_size=32, out_name='fc1_output'):
        ctx = []
        if len(gpu_list) == 0:
            ctx = [mx.cpu()]
        else:
            for gpu_id in gpu_list:
                ctx.append(mx.gpu(gpu_id))

        sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
        all_layers = sym.get_internals()
        sym = all_layers[out_name]
        self.shape = (batch_size, 3, 112, 112)
        self.model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
        self.model.bind(data_shapes=[('data', (batch_size, 3, 112, 112))])
        self.model.set_params(arg_params, aux_params)

    def forward(self, data):
        self.model.forward(data, is_train=False)
        fea = self.model.get_outputs()[0].asnumpy()
        return fea

if __name__ == '__main__':
    img_path = '/home/gehen/PycharmProjects/face/images/Alejandro_Avila_0001.jpg'
    model_path = './model/model-r50-google/model,00'
    prefix, epoch = model_path.split(',')
    epoch = int(epoch)
    gpu_list = [0]
    batch_size = 1
    warped1, bbox = preprocess('/home/gehen/PycharmProjects/face/images/Alejandro_Avila_0001.jpg')
    # warped2, bbox = preprocess('/home/gehen/PycharmProjects/face/images/Alejandro_Avila_0002.jpg')
    warped3, bbox = preprocess('/home/gehen/PycharmProjects/face/Anders_Ebbeson_0003.jpg')

    # cv2.imwrite('image2.jpg',warped1)
    warped1 = np.transpose(warped1,(2,0,1))
    # warped2 = np.transpose(warped2,(2,0,1))
    warped3 = np.transpose(warped3,(2,0,1))

    ext = Extractor(prefix, epoch, gpu_list, batch_size)
    # #import pdb
    # #pdb.set_trace()
    fea = ext.forward(mx.io.DataBatch(data=(mx.nd.array([warped1,warped3]),)))
    print(fea)

