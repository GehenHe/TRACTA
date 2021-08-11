#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 1/18/18 10:36 AM
# @Author  : gehen
# @File    : pedestiran_det.py
# @Software: PyCharm Community Edition


import _init_paths
from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms
from utils.timer import Timer
import tensorflow as tf
import numpy as np
import os
import cv2
import argparse
from nets.resnet_v1 import resnetv1

class PedestrianDet:
    def __init__(self,model_name,conf_thresh=0.8,nms_thresh=0.5):
        self.model_list = ['coco-res101','voc07-res101']
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.model_name = model_name
        self.sess,self.net = self._load_net()

    def _load_net(self):
        assert(self.model_name in self.model_list),'model {} is not defined'.format(self.model_name)

        cfg.TEST.HAS_RPN = True
        tfmodel_dir = os.path.join('model/detection/{}'.format(self.model_name))
        model_name = os.listdir(tfmodel_dir)[0].split('.')[0]
        tfmodel = os.path.join(tfmodel_dir,model_name+'.ckpt')

        tfconfig = tf.ConfigProto(allow_soft_placement=True)
        tfconfig.gpu_options.allow_growth = True
        sess = tf.Session(config=tfconfig)

        if self.model_name == 'coco-res101':
            net = resnetv1(num_layers=101)
            net.create_architecture("TEST", 81,
                                    tag='default', anchor_scales=[4, 8, 16, 32])
            saver = tf.train.Saver()
            saver.restore(sess, tfmodel)

        elif self.model_name == 'voc07-res101':
            net = resnetv1(num_layers=101)
            net.create_architecture("TEST", 21,
                                    tag='default', anchor_scales=[8, 16, 32])
            saver = tf.train.Saver()
            saver.restore(sess, tfmodel)
        return sess,net

    def det_person(self,im):
        scores, boxes = im_detect(self.sess, self.net, im)
        if self.model_name == 'voc07-res101':
            cls_ind = 15
            cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes,
                              cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, self.nms_thresh)
            dets = dets[keep, :]
            if dets is not None:
                inds = np.where(dets[:, -1] >= self.conf_thresh)[0]
                if len(inds) == 0:
                    return
                detection = dets

        if self.model_name == 'coco-res101':
            cls_ind = 1
            cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes,
                              cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, self.nms_thresh)
            dets = dets[keep, :]
            if dets is not None:
                inds = np.where(dets[:, -1] >= self.conf_thresh)[0]
                if len(inds) == 0:
                    return
                detection = dets[inds]
        return detection



# if __name__=='__main__':
#     ped_det = PedestrianDet('coco-res101')
#     # ped_det = PedestrianDet('voc07-res101')
#     image = cv2.imread('/home/gehen/PycharmProjects/multi_view_tracking/data/EPFL/terrace1-c2/img1/000903.jpg')
#     image2 = cv2.imread('/home/gehen/PycharmProjects/multi_view_tracking/data/EPFL/terrace1-c2/img1/000603.jpg')
#
#     image_batch = [image]
#     detection = ped_det.det_person(image)
#     print detection.shape