#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-5-4 上午11:22
# @Author  : Aries
# @Site    : 
# @File    : Pedestrain_det_msk.py
# @Software: PyCharm

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import logging
import os
import sys
import time
import numpy as np

from caffe2.python import workspace
# import detectron.init_mask_path
from core.config import assert_and_infer_cfg
from core.config import cfg
from core.config import merge_cfg_from_file
from utils.io import cache_url
from utils.timer import Timer
from scipy.spatial.distance import euclidean
import core.test_engine as infer_engine
import datasets.dummy_datasets as dummy_datasets
import utils.c2 as c2_utils
import utils.logging
import utils.vis as vis_utils
import pycocotools.mask as mask_util

c2_utils.import_detectron_ops()
# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

class Pedestrain_Det_Msk:
    def __init__(self,model_name,conf_thresh=0.90,nms_thresh=0.5,task='mask',gpu_num=1):
        """
        mask rcnn 检测初始化
        :param model_name: 模型名字-->目前只有coco-res101
        :param conf_thresh: 阈值
        :param nms_thresh: mns阈值
        :param task: 分为'mask'和'kp',分别用于检测mask和人体关键点,本工程中使用mask
        :param gpu_num: gpu数目
        """
        print ('load detectron framework')
        self.model_list = ['coco-res101']
        self.gpu_num = gpu_num
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.model_name = model_name
        assert model_name in self.model_list,'model name {} is not in model list {}'.format(model_name,self.model_list)
        self.model = self.load_model(model_name,task)

    def load_model(self,model_name,task='mask'):
        """
        读取模型参数
        :param model_name:模型名字
        :param task: 默认'mask'
        :return: 载入的模型
        """
        cfg.NUM_GPUS = self.gpu_num
        if model_name == 'coco-res101':

            ## model for keypoint
            if task == 'kp':
                weights_dir = 'detectron/model/coco/m odel_final.pkl'
                cfg_dir = 'detectron/configs/12_2017_baselines/e2e_keypoint_rcnn_R-101-FPN_1x.yaml'

            ## model for mask
            if task == 'mask':
                weights_dir = 'detectron/model/coco/R-101-FPN-2x.pkl'
                cfg_dir = 'detectron/configs/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml'

            merge_cfg_from_file(cfg_dir)
            weights = cache_url(weights_dir, cfg.DOWNLOAD_CACHE)
            assert_and_infer_cfg(cache_urls=False)
            model = infer_engine.initialize_model_from_cfg(weights)
            return model
        else:
            return None


    def det_person(self,image):
        """
        行人检测
        :param image:输入的图片
        :return: 行人检测框和分割信息
        """
        timers = defaultdict(Timer)
        with c2_utils.NamedCudaScope(0):
            cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
                self.model, image, None, timers=timers
            )
        temp = cls_boxes[1]
        # index = np.where(temp[:,-1]>=self.conf_thresh)
        index = np.where(temp[:, -1] >= self.conf_thresh)
        index = index[0].tolist()
        person_det = temp[index]
        temp = cls_segms[1]
        person_seg = [temp[i] for i in index]
        return person_det,person_seg

    def cal_point(self,image,name='top'):
        """
        行人检测以及头/脚点的检测
        :param image: 输入图像
        :param name: 头点/脚点/头脚点,本工程中检测头脚点
        :return: 检测框以及头脚点
        """
        dets = []
        point_list = []
        ## 检测脚点
        if name == 'bottom':
            dets,segms = self.det_person(image)
            num = len(segms)
            if num>0:
                masks = mask_util.decode(segms)
                for i in range(num):
                    e = masks[:,:,i]
                    [x, y] = np.where(e != 0)
                    num = x.shape[0]
                    cal_num = max(int(num * 0.1), 1)
                    indexs = np.argsort(-x)[0:cal_num]
                    x_list = x[indexs]
                    y_list = y[indexs]
                    side_num = max(int(cal_num * 0.2), 1)
                    left_list = np.argsort(y_list)[0:side_num]
                    right_list = np.argsort(-y_list)[0:side_num]
                    center_x = int(np.mean(x_list[np.append(left_list, right_list)]))
                    center_y = int(np.mean(y_list[np.append(left_list, right_list)]))
                    point_list.append([center_x,center_y])

        ## 检测头点
        elif name == 'top':
            dets, segms = self.det_person(image)
            num = len(segms)
            if num > 0:
                masks = mask_util.decode(segms)
                for i in range(num):
                    e = masks[:, :, i]
                    [x, y] = np.where(e != 0)
                    num = x.shape[0]
                    cal_num = max(int(num * 0.01), 1)
                    indexs = np.argsort(x)[0:cal_num]
                    x_list = x[indexs]
                    y_list = y[indexs]
                    side_num = max(int(cal_num * 0.2), 1)
                    left_list = np.argsort(y_list)[0:side_num]
                    right_list = np.argsort(-y_list)[0:side_num]
                    center_x = int(np.mean(x_list[np.append(left_list, right_list)]))
                    center_y = int(np.mean(y_list[np.append(left_list, right_list)]))
                    point_list.append([center_x, center_y])
        ## 检测头脚点
        elif name == 'top_bottom':
            dets,segms = self.det_person(image)
            num = len(segms)
            if num>0:
                masks = mask_util.decode(segms)
                for i in range(num):
                    e = masks[:,:,i]
                    [x, y] = np.where(e != 0)
                    num = x.shape[0]
                    cal_num = max(int(num * 0.1), 1)
                    indexs = np.argsort(-x)[0:cal_num]
                    x_list = x[indexs]
                    y_list = y[indexs]
                    side_num = max(int(cal_num * 0.2), 1)
                    left_list = np.argsort(y_list)[0:side_num]
                    right_list = np.argsort(-y_list)[0:side_num]
                    bot_center_x = int(np.mean(x_list[np.append(left_list, right_list)]))
                    bot_center_y = int(np.mean(y_list[np.append(left_list, right_list)]))

                    cal_num = max(int(cal_num/10),1)
                    indexs = np.argsort(x)[0:cal_num]
                    x_list = x[indexs]
                    y_list = y[indexs]
                    side_num = max(int(cal_num * 0.2), 1)
                    left_list = np.argsort(y_list)[0:side_num]
                    right_list = np.argsort(-y_list)[0:side_num]
                    top_center_x = int(np.mean(x_list[np.append(left_list, right_list)]))
                    top_center_y = int(np.mean(y_list[np.append(left_list, right_list)]))

                    total = np.sum(e)
                    h = euclidean([top_center_x,top_center_y],[bot_center_x,bot_center_y])
                    ratio = float(h*h/total)
                    point_list.append([[top_center_x,top_center_y],[bot_center_x,bot_center_y],ratio])
        return dets,point_list

    def vis_mask(self,image,frame_idx,save_dir):
        """
        行人检测到的mask可视化
        :param image: 输入图像
        :param frame_idx: 帧的ID
        :param save_dir: 保存路径
        :return:
        """
        im_name = '{}'.format(frame_idx)
        timers = defaultdict(Timer)
        with c2_utils.NamedCudaScope(0):
            cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
                self.model, image, None, timers=timers
            )
        cls_boxes = [cls_boxes[1]]

        vis_utils.vis_one_image(
            image[:, :, ::-1],  # BGR -> RGB for visualization
            im_name,
            save_dir,
            cls_boxes,
            cls_segms,
            cls_keyps,
            dataset=None,
            box_alpha=0.3,
            show_class=True,
            thresh=self.conf_thresh,
            kp_thresh=2,
            ext='jpg'
        )



if __name__ == '__main__':
    detector = Pedestrain_Det_Msk('coco-res101',0.9,task='mask')
    img_dir = '/images/test_place/multi_camera/112'
    save_dir = '/images/temp'
    img_list = os.listdir(img_dir)
    for i in range(600,2000,5):
        if i%100==0:
            print (i)
        # image = cv2.imread('{}/{:0>6}.jpg'.format(img_dir,i))
        image = cv2.imread('{}/{}.jpg'.format(img_dir,i))
        # detector.vis_mask(image,i,save_dir)
        bboxs,points = detector.cal_point(image,'top_bottom')
        # for idx,bbox in enumerate(bboxs):
        #     point = points[idx]
        #     score = bbox[-1]
        #     bbox = map(int,bbox[0:4])
        #     image = cv2.rectangle(image,(bbox[0],bbox[1]),(bbox[2],bbox[3]),[0,255,0],4)
        #     image = cv2.circle(image,(point[1],point[0]),4,[0,255,0],4)
        #
        # cv2.imwrite('{}/{}.jpg'.format(save_dir,i),image)

