#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-5-18 上午9:58
# @Author  : Aries
# @Site    : 
# @File    : tracker_match.py
# @Software: PyCharm

import application_util.cal_track_dist as cal_track_dist
from application_util.nmf import assign_H
import numpy as np
from application_util.global_id_map import Global_ID_Map
from self_config.self_config import self_cfg


# This part is mainly for distributed host multi-camera-tracking task.
class ID_Match:
    def __init__(self):
        """
        多摄像头匹配
        :param max_length: 保存历史信息的长度
        :param out_area: 出场地的区域
        """
        self.tracker_list = None
        self.max_result_length = self_cfg.TEST.max_length
        self.frame_idx = 0
        self.out_area = self_cfg.TEST.out_area
        self.vote_method = self_cfg.TEST.vote_method
        self.dist_length = self_cfg.TEST.dist_length
        self.search_length = self_cfg.TEST.search_length
        if len(self.out_area)>0:
            assert len(self.out_area)==4,'out_area must in form of [xmin,ymin,xmax,ymax],if no edge form as -> [xmin,ymin,inf,inf]'


    def updata_infos(self,infos):
        """
        将多路摄像头的结果追踪得到的信息保存
        :param infos:
        :return:
        """
        if self.tracker_list is None:
            tracker_list = [info.keys()[0] for info in infos]
            self.tracker_list = tracker_list
            self.view_num = len(self.tracker_list)
            self.global_id_map = Global_ID_Map(self.tracker_list,max_length=self.max_result_length,search_length=self.search_length)
            self.tracking_results = {key: {} for key in self.tracker_list}
        assert len(infos)==self.view_num,'update info\'s length: {} is not equal to camera number: {}'.format(len(infos),self.view_num)
        for tracker_id in self.tracker_list:
            self.tracking_results[tracker_id].setdefault(self.frame_idx,{})
        for info in infos:
            tracker_id = info.keys()[0]
            # index = self.tracker_list.index(tracker_id)
            tracks_info = info[tracker_id]
            for track_id in tracks_info:
                data = tracks_info[track_id]
                self.tracking_results[tracker_id][self.frame_idx].setdefault(track_id,data)
                if len(self.tracking_results[tracker_id]) > self.max_result_length:
                    self.tracking_results[tracker_id].pop(min(self.tracking_results[tracker_id].keys()))


    def match_id(self,infos):
        """
        将多路摄像头的追踪结果进行匹配,并根据匹配结果进行 global id的分配
        :param infos: 多路摄像头的追踪结果
        :return: 匹配结果  如 --> {'view_id':[[(1,2),(2,3)],[(1,3),(3,2)],'global_id':[1,3]}
        """
        ## 将当前帧的追踪结果保存到tracking results中
        self.updata_infos(infos)

        ## 在时间尺度和空间尺度上计算轨迹与轨迹之间的距离
        dist_matrix, num_list = cal_track_dist.cal_dist_matrix(self.tracking_results,dist_length=self.dist_length)

        ## 用得到的距离度量矩阵来求得分配矩阵
        assign_matrix, k = assign_H(dist_matrix, num_list)

        ## 验证分配矩阵是否有错,并将分配矩阵格式转换为list格式
        sum_view = np.asarray([sum(num_list[0:i]) for i in range(len(num_list) + 1)])
        id_list = np.array([sorted(self.tracking_results[i][self.frame_idx].keys()) for i in self.tracking_results.keys()])
        id_num = sum(map(len, id_list))
        assert id_num == sum(num_list), 'tracker length is {}, assign matrix is {}'.format(id_num, sum(num_list))
        view_id_list = []
        for j in range(min(assign_matrix.shape)):
            i_index = list(np.where(assign_matrix[:, j] == 1)[0])
            view_index = self.search_view_index(sum_view, i_index)
            tracker_index = np.array(self.tracking_results.keys())[view_index]
            tracker_index = tracker_index.tolist()
            id_index = list(i_index - sum_view[view_index])
            assert len(id_index) == len(view_index), 'view index can not match id index'
            zip_index = zip(view_index, id_index)
            view_id = [id_list[item[0]][item[1]] for item in zip_index]
            view_id_index = zip(tracker_index, view_id)
            view_id_list.append(view_id_index)

        ## 根据当前帧的匹配结果分配global id
        self.global_id_map.assign_global_id(self.frame_idx, view_id_list)

        global_info = self.global_id_map.view_id_map[self.frame_idx]
        return global_info

    def cal_point(self,infos):
        """
        将多路摄像头进行匹配并分配global id,并将匹配到的世界点进行融合
        :param infos: 多路摄像头追踪信息
        :return: 匹配结果和融合后的点
        """
        ## 多摄像头匹配并分配 global id
        global_info = self.match_id(infos)
        dict_infos = self.list2dict(infos)
        point_infos = {}
        global_ids = self.global_id_map.id_map[self.frame_idx]['global_id']
        view_id_list = self.global_id_map.id_map[self.frame_idx]['view_id']

        ## 将匹配到一起的世界点进行融合
        for idx,global_id in enumerate(global_ids):
            view_ids = view_id_list[idx]
            matching_points = [dict_infos[tracker_id][view_id][0] for tracker_id,view_id in view_ids]
            matching_points = np.asarray(matching_points)
            point = np.mean(matching_points,0)
            point = point.astype(int)
            point_infos[global_id] = point

        ##　判定是否在出口区域,如果在的话将不回收ID
        self.check_out(point_infos)
        self.frame_idx += 1
        return global_info,point_infos

    def check_out(self,point_infos):
        """
        判断世界点是否在出口区域,如果在的话这个ID将不会被回收
        :param point_infos: 融合后的点的信息
        :return:
        """
        for id in point_infos:
            point = point_infos[id]
            if point[0]>=float(self.out_area[0]) and point[0]<=float(self.out_area[2])\
                 and point[1]>=float(self.out_area[1]) and point[1]<=float(self.out_area[3]):
                if id in self.global_id_map.pop_list:
                    continue
                self.global_id_map.pop_list.append(id)

    def search_view_index(self,sum_list,i_list):
        """
        模块功能函数,用于搜索分配矩阵中第i行所属的摄像头index
        :param sum_list: 各个摄像头下分别检测到的行人数迭代累加和
        :param i_list: 分配矩阵中的第i个人
        :return:
        """
        if isinstance(i_list,list):
            view_index_list = [np.where(sum_list>i)[0][0]-1 for i in i_list]
            return view_index_list
        else:
            return np.where(sum_list>i_list)[0][0]-1

    def list2dict(self,infos):
        """
        infos从list格式转换到dict格式
        :param infos: 单摄像头追踪结果
        :return: 字典格式的infos
        """
        dict_infos = {}
        for info in infos:
            tracker_id = info.keys()[0]
            dict_infos.setdefault(tracker_id, {})
            for view_id in info[tracker_id].keys():
                data = info[tracker_id][view_id]
                dict_infos[tracker_id][view_id] = data
        return dict_infos

    def multiview_locate(self,new_xy_list):
        """
        将多视角下匹配到一起的点融合估计为一个新的点
        :param new_xy_list: 匹配到一起的点
        :return: 融合后的点的坐标
        """
        temp_sum = 0
        for i in range(len(new_xy_list)):
            u = new_xy_list[i][0][0]
            cov = np.mat(new_xy_list[i][0][1])
            temp_sum = temp_sum + cov.I
        sigma_MV = temp_sum.I
        temp_sum = np.zeros([2, 1])
        for i in range(len(new_xy_list)):
            u = new_xy_list[i][0][0]
            cov = np.mat(new_xy_list[i][0][1])
            temp_sum = temp_sum + cov.I * sigma_MV * u.T
        u_MV = temp_sum
        return u_MV, sigma_MV



if __name__ == '__main__':
    pass