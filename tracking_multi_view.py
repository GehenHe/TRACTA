# -*- coding: UTF-8 -*-
# vim: expandtab:ts=4:sw=4
from __future__ import division, print_function, absolute_import, print_function

import argparse
import os
import time
import cv2
import numpy as np
import math
import sys
import multiprocessing
from ctypes import c_wchar_p
this_dir = os.path.realpath(__file__).rsplit('/', 1)[0]
extract_path = os.path.join(this_dir, 'reID_caffe')
sys.path.append(this_dir)
sys.path.append(extract_path)

from extract_per_frame import load_ReID_net, extract
from preprocess.visualization import draw_tracker
from preprocess.homography_matrix import terrace_H,test_all
from application_util import preprocessing
import application_util.cal_track_dist as cal_track_dist
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
# from tf_faster_rcnn.pedestiran_det import PedestrianDet
from detectron.pedestrain_det_msk import Pedestrain_Det_Msk
from application_util.tracker_match import ID_Match
from top_view_visual import Top_Visual
from scipy.spatial.distance import euclidean
import requests
from self_config.self_config import self_cfg
from self_config.self_config import merge_cfg_from_file

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(filename)s:%(lineno)4d: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)



class TrackObject:
    def __init__(self,display=False):
        self.display = display
        self.caffeReIDNet = load_ReID_net()

        self.tracker_list = self_cfg.TEST.tracker_list
        self.view_num = len(self.tracker_list)

    ## 检测相关参数初始化
        self.min_confidence = self_cfg.TEST.min_confidence
        self.nms_max_overlap = self_cfg.TEST.nms_max_overlap
        self.min_detection_height = self_cfg.TEST.min_detection_height
        self.detector = Pedestrain_Det_Msk(self_cfg.TEST.detector_name, self.min_confidence)

    ## 单摄像头追踪参数初始化
        self.max_age = self_cfg.TEST.tracker_max_age
        self.n_init = self_cfg.TEST.tracker_n_init
        self.nn_budget = self_cfg.TEST.nn_budget
        self.matching_threshold = self_cfg.TEST.nn_matching_threshold
        self.max_iou_distance = self_cfg.TEST.tracker_max_iou_distance
        self.height = self_cfg.TEST.init_height
        ## 宽高比和更新身高阈值
        self.wh_thresh1 = self_cfg.TEST.wh_thresh1
        self.wh_thresh2 = self_cfg.TEST.wh_thresh2
        self.h_thresh = self_cfg.TEST.h_thresh
        self.width_ratio = self_cfg.TEST.width_ratio
        ## 抑制匹配阈值
        self.gating_dim = self_cfg.TEST.gating_thresh

    ## 基本信息参数初始化
        ## 图片resize比例
        self.resize_ratio = self_cfg.TEST.resize_ratio
        self.image_width = self_cfg.TEST.image_width
        self.image_height = self_cfg.TEST.image_height
        self.place_retion = self_cfg.TEST.place_region
        self.height_list = self_cfg.TEST.height_list
        self.time_since_update = self_cfg.TEST.time_since_update

        ## two trackers for prematching
        multi_trackers = []
        for i in range(self.view_num):
            metric = nn_matching.NearestNeighborDistanceMetric(metric='cosine', matching_threshold=self.matching_threshold, budget=self.nn_budget)
            multi_trackers.append(Tracker(metric,max_iou_distance=self.max_iou_distance, max_age=self.max_age, n_init=self.n_init,\
                                          tracker_id=self.tracker_list[i],height=self.height,gating_dim=self.gating_dim))
        self.multi_trackers = multi_trackers

        ## homography matrix
        self.homography_matrix = test_all()



    def create_detections(self, detection_mat, frame_idx, min_height=0):
        """Create detections for given frame index from the raw detection matrix.

        Parameters
        ----------
        detection_mat : ndarray
        Matrix of detections. The first 10 columns of the detection matrix are
        in the standard MOTChallenge detection format. In the remaining columns
        store the feature vector associated with each detection.
        frame_idx : int
        The frame index.
        min_height : Optional[int]
        A minimum detection bounding box height. Detections that are smaller
        than this value are disregarded.

        Returns
        -------
        List[tracker.Detection]
        Returns detection responses at given frame index.

        """
        detection_list = []
        if len(detection_mat) != 0:
            frame_indices = detection_mat[:, 0].astype(np.int)
            mask = frame_indices == frame_idx
            for row in detection_mat[mask]:
                bbox, confidence, feature = row[2:6], row[6], row[10:]
                detection_list.append(Detection(bbox, confidence, feature))
        return detection_list

    def multi_view_detection(self,images):
        """
        行人检测
        :param images: 多路输入的图片
        :return: 检测框以及对应的头点和脚点
        """
        detection_list = []
        point_list = []
        for view_idx,image in enumerate(images):
            det_frRCNN,points = self.detector.cal_point(image,'top_bottom')
            if det_frRCNN is not None:
                idx = np.where(det_frRCNN[:, -1] >= self.min_confidence)[0]
                det_frRCNN = det_frRCNN[idx, :]
                detections = np.asarray(det_frRCNN)
                point_list.append(points)
            else:
                # detections = np.reshape(np.asarray([]),[0,5])
                detections = None
                point_list.append([])
            detection_list.append(detections)
        return detection_list,point_list


    def multi_view_matching(self,images,detection_list,frame_idx,point_list=None):
        """
        单摄像头追踪,主要完成单摄像头下的追踪并将追踪到的人投影到世界坐标系上
        :param images: 多路输入的图片
        :param detection_list: 检测框结果
        :param frame_idx: 当前帧的ID
        :param point_list: 头点和脚点的结果
        :param wh_thresh1: 宽高比双阈值中较大的阈值
        :param wh_thresh2: 宽高比双阈值中较小的阈值
        :param h_thresh: 通过宽高比判定是否更新身高的阈值
        :param ratio_thresh: 通过屏幕的区域判定是否更新身高的阈值
        :return: 各个摄像头下的追踪到的人投影到世界坐标系下的点,格式为 --> [{camera_id1:person_id1:[point2,1],...},{camera_id2:person_id1:[point2,1]}]
        """
        ## 单视角下的追踪,将当前检测到的框与tracker进行匹配,并将检测框与头脚点更新到对应的track中
        for view_idx,det_frRCNN in enumerate(detection_list):
            image = images[view_idx]
            if det_frRCNN is not None:
                idx = np.where(np.logical_and(det_frRCNN[:, -1] >= self.min_confidence , (det_frRCNN[:,3]-det_frRCNN[:,1])>=self.min_detection_height))[0]
                det_frRCNN = det_frRCNN[idx, :]
                det_fea = extract(self.caffeReIDNet, image, det_frRCNN, frame_idx)
                detections = self.create_detections(
                    det_fea, frame_idx, self.min_detection_height)
                boxes = np.array([d.tlwh for d in detections])
                scores = np.array([d.confidence for d in detections])
                indices = preprocessing.non_max_suppression(boxes, self.nms_max_overlap, scores)
                detections = [detections[i] for i in indices]
                if point_list is None:
                    self.multi_trackers[view_idx].predict()
                    self.multi_trackers[view_idx].update(detections)
                else:
                    points = np.array(point_list[view_idx])
                    points = points[idx]
                    points = [points[i] for i in indices]
                    self.multi_trackers[view_idx].predict()
                    self.multi_trackers[view_idx].update(detections,points)
            else:
                self.multi_trackers[view_idx].predict()

        ## 将各个视角下的行人投影到世界坐标系
        send_info = {}
        for index,tracker_view in enumerate(self.multi_trackers):
            send_info.setdefault(tracker_view.tracker_id,'')
            track_infos = ''
            for track in tracker_view.tracks:
                if not track.is_confirmed() or track.time_since_update > self.time_since_update:
                    continue
                [top,bot,ratio] = track.point
                ## 通过双阈值来确定将头/脚点投影到世界坐标系.如果宽高比(ratio)=>wh_thresh1,此时认为看到全身,将误差更小的脚点投影世界坐标系
                if ratio>=self.wh_thresh1:                                                                      # use foot to top-view
                    [top1, cov] = cal_track_dist.img_to_world([top[1] / self.resize_ratio, top[0] / self.resize_ratio],
                                                              self.homography_matrix[tracker_view.tracker_id],
                                                              track.height)
                    bbox = track.to_tlwh()
                    y = int(bbox[1] + bbox[3])
                    bot1 = [y, bot[1]]
                    [bot1, cov] = cal_track_dist.img_to_world([bot1[1] / self.resize_ratio, bot1[0] / self.resize_ratio],
                                                              self.homography_matrix[tracker_view.tracker_id], 0)
                    if not self.check_place(bot1,self.place_retion) and self.check_place(top1,self.place_retion):
                        point = top1
                    else:
                        point = bot1

                ## 通过双阈值来确定将头/脚点投影到世界坐标系.wh_thresh2<=ratio<wh_thresh,此时认为头脚点都较为可靠,\
                ## 在头脚点中,选取距离历史点距离较小的点作为投影到世界坐标系的点
                elif ratio>=self.wh_thresh2:
                    [top1, cov] = cal_track_dist.img_to_world([top[1] / self.resize_ratio, top[0] / self.resize_ratio],
                                                             self.homography_matrix[tracker_view.tracker_id],track.height)
                    bbox = track.to_tlwh()
                    y = int(bbox[1] + bbox[3])
                    bot1 = [y, bot[1]]
                    [bot1, cov] = cal_track_dist.img_to_world([bot1[1] / self.resize_ratio, bot1[0] / self.resize_ratio],
                                                             self.homography_matrix[tracker_view.tracker_id],0)
                    if track.hist_point is None:
                        if not self.check_place(bot1, self.place_retion) and self.check_place(top1,self.place_retion):
                            point = top1
                        else:
                            point = bot1
                    else:
                        dist1 = euclidean(top1,track.hist_point)
                        dist2 = euclidean(bot1,track.hist_point)
                        if dist1>=dist2:
                            if not self.check_place(bot1, self_cfg.TEST.place_region) and self.check_place(top1,
                                                                                                           self_cfg.TEST.place_region):
                                point = top1
                            else:
                                point = bot1
                        else:
                            if not self.check_place(top1, self.place_retion) and self.check_place(bot1,self.place_retion):
                                point = bot1
                            else:
                                point = top1

                ## 通过双阈值来确定将头/脚点投影到世界坐标系.如果宽高比(ratio)<wh_thresh2,此时认为看到半身,将头点投影到射界坐标系
                else:
                    [top1, cov] = cal_track_dist.img_to_world([top[1] / self.resize_ratio, top[0] / self.resize_ratio],
                                                              self.homography_matrix[tracker_view.tracker_id],
                                                              track.height)
                    bbox = track.to_tlwh()
                    y = int(bbox[1] + bbox[3])
                    bot1 = [y, bot[1]]
                    [bot1, cov] = cal_track_dist.img_to_world([bot1[1] / self.resize_ratio, bot1[0] / self.resize_ratio],
                                                              self.homography_matrix[tracker_view.tracker_id], 0)
                    if not self.check_place(top1,self.place_retion) and self.check_place(bot1,self.place_retion):
                        point = bot1
                    else:
                        point = top1
                track_infos = track_infos+'{}:{},{};'.format(track.track_id,point[0,0],point[1,0])
                # send_info[tracker_view.tracker_id][track.track_id] = [point, 1]

                ## 更新身高,如果宽高比ratio>=h_thresh,并且框位于图像中间位置(距离左右两边的距离大于比例阈值),则更新身高
                if ratio>=self.h_thresh and bot[1]>=self_cfg.TEST.image_width*self.width_ratio and bot[1]<=self_cfg.TEST.image_width*(1-self.width_ratio):
                    bbox = track.to_tlwh()
                    y = int(bbox[1] + bbox[3])
                    bot1 = track.point[1]
                    bot = [y, bot1[1]]
                    [world_point, cov] = cal_track_dist.img_to_world([bot[1] / self.resize_ratio, bot[0] / self.resize_ratio],
                                                             self.homography_matrix[tracker_view.tracker_id], 0)
                    track.height = self.update_height([top[1]/ self.resize_ratio,top[0]/ self.resize_ratio],world_point,track.height,self.homography_matrix[tracker_view.tracker_id])
                    # print ('for id :{} height is {}'.format(track.track_id,track.height))

                ## single view head and foot point
                # top = track.point[0]
                # [top, cov] = cal_track_dist.img_to_world([top[1] / self.resize_ratio, top[0] / self.resize_ratio],
                #                                          self.homography_matrix[tracker_view.tracker_id],track.height)
                # bot1 = track.point[1]
                # bbox = track.to_tlwh()
                # y = int(bbox[1] + bbox[3])
                # bot = [y,bot1[1]]
                # [bot, cov] = cal_track_dist.img_to_world([bot[1] / self.resize_ratio, bot[0] / self.resize_ratio],
                #                                          self.homography_matrix[tracker_view.tracker_id],0)
                #
                # send_info[tracker_view.tracker_id][track.track_id] = [bot, 1]
                # send_info[tracker_view.tracker_id][track.track_id+1] = [top, 1]
            send_info[tracker_view.tracker_id] = track_infos
        return send_info

    def check_place(self,point,region):
        """
        用于检验点是否在活动场地中
        :param point: 投影到的世界坐标系中的点
        :param region: 场地区域(长方形)
        :return: bool,如果在场地中则返回True,反之亦然
        """
        if point[0]>=region[0] and point[0]<=region[2]\
            and point[1]>=region[1] and point[1]<=region[3]:
            return True
        else:
            return False

    def draw_trackers(self, frame_idx, images, output_dir=None, is_det=False, detections=None,match_info=None):
        """
        用于可视化多视角追踪结果
        :param frame_idx:当前帧ID
        :param images: 多路输入的图像
        :param output_dir: 图像保存路径,default=None,这种情况下只显示,不保存结果
        :param is_det:是否显示检测框
        :param detections:检测结果
        :param match_info:多视角匹配结果,default=None,这种情况下只显示各个视角分别追踪的效果;反之,则显示多视角匹配后的结果
        :return:
        """
        # save_dir = os.path.join(output_dir, 'visual_temp')
        # if not os.path.exists(save_dir):
        #     os.makedirs(save_dir)
        view_num = len(images)
        row_num = int(math.ceil(np.sqrt(view_num)))
        show_list = []
        for view,image in enumerate(images):
            # image_show = draw_tracker(image,self.multi_trackers[view].tracks)              # draw not match
            image_show = draw_tracker(image,self.multi_trackers[view].tracks,self.tracker_list[view],match_info=match_info,time_since_update=self.time_since_update)     # draw id matching
            if is_det:
                det = detections[view]
                for idx,bbox in enumerate(det):
                    image_show = cv2.rectangle(image_show,(bbox[0],bbox[1]),(bbox[2],bbox[3]),[0,0,0],1)
            show_list.append(image_show)
        show_list+=[np.zeros(image.shape) for i in range(row_num*row_num-len(show_list))]
        for index in range(int(math.ceil(view_num/row_num))):
            if index == 0:
                image_show = np.concatenate(show_list[index*row_num:(index+1)*row_num],axis=1)
            else:
                temp = np.concatenate(show_list[index*row_num:(index+1)*row_num],axis=1)
                image_show = np.concatenate([image_show,temp],axis=0)
        image_show = image_show[:,:,[2,1,0]].astype(np.uint8)
        cv2.imshow('test',image_show)
        cv2.waitKey(1)
        # if output_dir is not None:
        #     cv2.imwrite('{}/{}.jpg'.format(save_dir,frame_idx),image_show)

    def update_height(self,view_head,world_foot,height,H):
        """
        更新身高
        :param view_head: 人的头点的像素坐标
        :param world_foot:人的脚点的世界坐标
        :param height: 原来身高
        :param H: 映射矩阵
        :return: 更新后的身高
        """
        h_list = [height+5*(i-1) for i in range(3)]
        h_list= list(set(h_list).intersection(set(self.height_list)))
        h_list.sort()
        dist = []
        for h in h_list:
            world_head,cov = cal_track_dist.img_to_world(view_head,H,h)
            dist.append(euclidean(world_foot,world_head))
        index = dist.index(min(dist))
        new_height = h_list[index]
        return new_height

    def info_reformat(self,frame_infos):
        frame_list = []
        # if isinstance(frame_infos,dict):
        for tracker_id in frame_infos:
            frame_dict = {}
            frame_dict.setdefault(tracker_id,{})
            data = frame_infos[tracker_id]
            if len(data)>0:
                items = data.split(';')
                for item in items:
                    if len(item)<1:
                        continue
                    temp_item = item.split(':')
                    track_id = int(float(temp_item[0]))
                    temp_info = temp_item[1].split(',')
                    temp_info = map(float,temp_info)
                    point = np.array(temp_info).reshape(2,-1)
                    frame_dict[tracker_id][track_id] = [point,1]
            frame_list+=[frame_dict]

        return frame_list


    def _serialize(self, value):
        #if isinstance(value, list):
        return [x.tolist() for x in value]

    def send_info(self, frame_idx):
        current_info = self.tracking_results[0][frame_idx]
        info = {k: self._serialize(v) for k, v in current_info.items()}
        # print(len(current_info))
        try:
            requests.post('http://127.0.0.1:8000/track/info', json=info)
        except:
            pass
        return current_info


def running(tracker, images, frame_idx, output_dir, process_rate=1, cnt=0):
    if cnt % process_rate == 0:
        det_start = time.time()
        assert len(images)==tracker.view_num,'images number is not equal to trackers\' number: {}:{}'.format(len(images),tracker.view_num)
        detections,points = tracker.multi_view_detection(images)
        det_end = time.time()
        print (frame_idx)

        track_start = time.time()
        frame_infos = tracker.multi_view_matching(images, detections, frame_idx,points)
        track_end = time.time()

        # convert info format
        frame_infos = tracker.info_reformat(frame_infos)

        # match_start = time.time()
        match_info,send_info = id_match.cal_point(frame_infos)
        top_visual.visual(send_info,'top')
        # match_end = time.time()
        #
        # print ('det: {:.3f}s,\t track: {:.3f}s \t match: {:.3f}s'.format(det_end-det_start,\
        #                                                              track_end-track_start,match_end-match_start))
        # print ('det: {:.3f}s,\t track: {:.3f}s \t'.format(det_end-det_start,\
        #                                                              track_end-track_start))

        tracker.draw_trackers(frame_idx, images, output_dir, False, detections,match_info)



def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default=os.path.expanduser('~/tracking'))
    parser.add_argument('--process_rate', type=int, default=5)
    parser.add_argument('--camera_num', type=int, default=2)
    parser.add_argument('--min_confidence', type=int, default=0.8)
    parser.add_argument('--input', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=os.path.join(this_dir, 'output'))
    parser.add_argument('--type', type=str, default='image')
    parser.add_argument('--host', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--cfg',dest='cfg',default="./self_config/test.ymal",type=str)
    return parser.parse_args(argv)

def read_camera(image,input,lock):
    print('start read video:' + input)
    cap = cv2.VideoCapture(input)
    while True:
        ret, frame = cap.read()
        if not ret:
            print(input + 'not ret')
            break
        if frame is None:continue
        else:
            lock.acquire()
            image.value = frame
            lock.release()


def main(args):
    tracker = TrackObject()
    if args.type == 'image':
        print ('reading image from file')
        # image
        # img_dir = args.input
        img_dir = '/images/test_place/multi_camera'
        # video_list = ['111','112','113','114']
        video_list = ['111','112','113']
        # img_list = os.listdir(img_dir)
        # img_list.sort()
        cnt = 0
        for frame_idx in range(10000, 12250):
            if frame_idx == 10665:
                pass
            images = [cv2.imread('{}/{}/{}.jpg'.format(img_dir,video,frame_idx)) for video in video_list]
            new_shape = (int(self_cfg.TEST.image_width*self_cfg.TEST.resize_ratio), int(self_cfg.TEST.image_height*self_cfg.TEST.resize_ratio))
            images = [cv2.resize(image, new_shape) for image in images]
            images = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images]
            running(tracker, images, frame_idx, args.output_dir,args.process_rate,cnt)
            cnt = (cnt + 1) % args.process_rate
    else:
        # video
        print ('reading images from IP camera')
        manager = multiprocessing.Manager()
        pool = multiprocessing.Pool(processes=len(self_cfg.TEST.cameras))
        video_images = list()
        locks = list()
        for i in range(0, len(self_cfg.TEST.cameras)):
            video_images.append(manager.Value(c_wchar_p, 'ss'))
            locks.append(manager.Lock())
            pool.apply_async(read_camera, (video_images[i],self_cfg.TEST.cameras[i],locks[i],))
        pool.close()

        cnt = 0
        frame_idx = 0
        while True:
            status_wait = False
            frames = []
            for i in range(0, len(self_cfg.TEST.cameras)):
                locks[i].acquire()
                frames.append(video_images[i].value)
                locks[i].release()
                if len(frames[i]) == 2:
                    status_wait = True
                    break
            if status_wait:
                continue
            new_shape = (int(self_cfg.TEST.image_width * self_cfg.TEST.resize_ratio),\
                         int(self_cfg.TEST.image_height * self_cfg.TEST.resize_ratio))
            images = np.array(frames)
            images = [cv2.resize(image,new_shape) for image in images]
            images = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images]
            # cv2.imwrite('/images/test_place/multi_camera/111/{}.jpg'.format(frame_idx), images[0])
            # cv2.imwrite('/images/test_place/multi_camera/112/{}.jpg'.format(frame_idx),images[1])
            # cv2.imwrite('/images/test_place/multi_camera/113/{}.jpg'.format(frame_idx),images[2])
            # cv2.imwrite('/images/test_place/multi_camera/114/{}.jpg'.format(frame_idx), images[3])
            running(tracker, images, frame_idx, args.output_dir,args.process_rate,cnt)
            cnt = (cnt + 1) % args.process_rate
            frame_idx+=1
            if cv2.waitKey(20) & 0xFF == ord('q'): break
        pool.terminate()

if __name__ == "__main__":
    args = parse_arguments(sys.argv[1:])
    merge_cfg_from_file(args.cfg)
    id_match = ID_Match()
    top_visual = Top_Visual()
    main(args)
