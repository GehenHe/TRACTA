# coding=utf-8
#!/usr/bin/python
# -*- coding: utf-8 -*-

## extract person re-id feature with a pre-trained caffemodel
import _init_paths
import numpy as np
import sys, os
import argparse
import time
import cv2
import caffe

def load_ReID_net():
    """
    载入ReID网络模型
    :return: 载入的模型
    """
    # Note: change the path according to your environment
    gpu_id = 0
    caffe.set_device(gpu_id)
    caffe.set_mode_gpu()
    print 'use gpu: ' + str(0)
    this_dir = os.path.realpath(__file__).rsplit('/', 1)[0]
    extraction_path = os.path.join(this_dir, 'extract_re-id')
    net_file = extraction_path + '/deploy.prototxt'
    caffe_model = extraction_path + '/CUHK03_spgt_iter_36000.caffemodel'
    mean_file = extraction_path + '/CUHK03_mean_train.npy'
    net = caffe.Net(net_file, caffe_model, caffe.TEST)
    return net

def extract(net, image, detections, frame_idx):
    """
    根据检测框提取特征
    :param net: 载入的模型
    :param image: 输入的图片
    :param detections: 检测框
    :param frame_idx: 帧的ID
    :return: 各个检测框中的特征
    """
    # Note: change the path according to your environment
    this_dir = os.path.realpath(__file__).rsplit('/', 1)[0]
    extraction_path = os.path.join(this_dir, 'extract_re-id')
    mean_file = extraction_path + '/CUHK03_mean_train.npy'
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', np.load(mean_file)[0].mean(1).mean(1))
    transformer.set_raw_scale('data', 255)
    transformer.set_channel_swap('data', (2,1,0))

    rows = []
    detections_out = []
    features = []
    bbox_num = detections.shape[0]
    if bbox_num == 0:
        features = []
    else:
        start = time.time()
        for bbox_idx in xrange(bbox_num):
            tl_x = max(0,int(detections[bbox_idx][0]))
            tl_y = max(0,int(detections[bbox_idx][1]))
            br_x = min(image.shape[1]-1,int(detections[bbox_idx][2]))
            br_y = min(image.shape[0]-1,int(detections[bbox_idx][3]))
            if tl_x == br_x:
                br_x = br_x+1
            if tl_y == br_y:
                br_y = br_y+1
            confidence = detections[bbox_idx][4]
            bbox = [frame_idx, -1, tl_x, tl_y, br_x-tl_x+1, br_y-tl_y+1, confidence, -1, -1, -1]
            rows.append(bbox)

            # extract feature of per bbox
            ## crop detection object from original image
            im_crop = image[tl_y:br_y, tl_x:br_x]
            ## resize to input
            im_crop2 = cv2.resize(im_crop, (100, 250))
            im_crop3 = im_crop2[10:240, 10:90]
            ## trasnfer opencv image to caffe.io.load_image
            im_input = cv2.cvtColor(im_crop3, cv2.COLOR_BGR2RGB)
            im_input = im_input/float(255)
            ## input to caffe
            net.blobs['data'].data[...] = transformer.preprocess('data', im_input)
            out = net.forward()
            res = out['concat_3_a'][0]
            res =res.tolist()
            features.append(res)
            #res = net.blobs['norm_a'].data[0]
            #res = res.tolist()
            #features.append([x[0][0] for x in res])
        rows = np.array(rows)
        end = time.time()
        # print 'extract per frame time: {}, with {} bboxes'.format((end-start), bbox_num)
    detections_out += [np.r_[(row, feature)] for row, feature
                       in zip(rows, features)]
    detections_out = np.array(detections_out)
    return detections_out

