#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-4-16 上午9:58
# @Author  : Aries
# @Site    : 
# @File    : global_id_map.py
# @Software: PyCharm

import numpy as np
from collections import Counter
from global_id import Global_ID


class Next_ID_List:
    def __init__(self,length=20):
        self.id_list = range(1,length+1)

    def next_id(self):
        data = self.id_list.pop(0)
        self.id_list+=[self.id_list[-1]+1]
        return data

    def recover(self,id):
        self.id_list.append(id)
        self.id_list.sort()

class IDState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.

    """

    Tentative = 1
    Confirmed = 2
    Deleted = 3

class Global_ID_Map:
    '''
     Attributes
    ----------
    id_map: dict with frame index as keys, each value is consist of 'view_id','global_id'
            view_id: [[real_id_list1],[real_id_list2],...] , real_id_list: [[view_index,view_id],[view_index,view_id]]
            global_id [[global_id1],[global_id2],....]

    '''


    def __init__(self,tracker_list,max_length=200,search_length=50,vote_method='vote'):
        """
        初始化
        :param tracker_list: 所有摄像头的id组成的list
        :param max_length: 结果保存长度
        :param search_length: 分配global id时,进行搜索的长度
        """
        self.tracker_list = tracker_list
        self.view_num = len(tracker_list)
        self.next_id_list = Next_ID_List()
        self.id_map = {}
        self.view_id_map = {}
        self.max_length = max_length
        self.search_length = search_length
        self.frame_idx = 0
        self.pop_list = []
        self.vote_method = vote_method

    def assign_global_id(self,frame_idx,view_id_list):
        """
        根据当前帧的匹配结果进行global id的分配
        :param frame_idx: 当前帧id
        :param view_id_list: 当前帧匹配结果
        :return:
        """
        self.frame_idx = frame_idx
        self.update_id_map(frame_idx,view_id_list)
        frame_index = range(frame_idx-self.search_length+1,frame_idx)
        inter_index = list(set(frame_index).intersection(set(self.id_map.keys())))
        global_id_list = []
        frame_list = []
        top_list = []
        view_id_set = self.id_map[self.frame_idx]['view_id']

        ## 根据被匹配到一起的各个视角下的ID(合称为ID组),进行投票来获得初步的global id
        for view_id_list in view_id_set:
            global_id,frame,top = self.search_global_id(view_id_list,inter_index)
            global_id_list+=[global_id]
            frame_list+=[frame]
            top_list+=[top]

        ## 检查上一步分配的初步global id是否有冲突,如果发生冲突(多个ID组被分配到一个global id),依据一定规则进行重新分配global id
        id_set = list(set(global_id_list))                ## remove duplicate global id in one frame
        while(len(id_set)!=len(global_id_list)):
            frame_list = np.array(frame_list)
            for id in id_set:
                ## 寻找哪些ID组发生了冲突
                indexs = self.duplicates(global_id_list,id)
                if len(indexs)>1:
                    ## 根据投票数来决定冲突的ID组里,哪个ID组需要重新分配global id.此处,我们选择投票数最少的ID组重新分配它的global id
                    frame_set = frame_list[indexs]
                    num = np.sum(frame_set==np.min(frame_set))
                    if num<=1:
                        change_index = np.argmin(frame_list[indexs])
                        change_index = indexs[change_index]
                    else:
                        change_index = np.argmin(map(len,np.array(view_id_set)[indexs]))
                        change_index = indexs[change_index]

                    ## 需要重新分配global id的ID组,以投票数次高的global id作为初始global id进入下一次的检查直至没有冲突
                    top = top_list[change_index]
                    global_id,frame,top = self.search_global_id(view_id_set[change_index],inter_index,top+1)
                    global_id_list[change_index] = global_id
                    frame_list[change_index] = frame
                    top_list[change_index] = top
            id_set = list(set(global_id_list))

        self.id_map.setdefault(self.frame_idx,{})
        self.id_map[self.frame_idx]['global_id'] = global_id_list
        self.update_view_id_map(frame_idx)

        ## 检查ID以判定是否要将ID回收
        self.check_ID()

    def check_ID(self,ratio=0.2):
        """
        检查ID以判定是否要将ID回收
        :param ratio: global id 被命中的次数占搜索长度的最小比例
        :return:
        """
        frame_count = {}
        hit_count = {}
        frame_list = range(self.frame_idx-self.search_length+1,self.frame_idx+1)
        frame_list = set(frame_list).intersection(set(self.id_map.keys()))
        for frame in frame_list:
            view_id = self.id_map[frame]['view_id']
            global_id = self.id_map[frame]['global_id']
            assert len(global_id)==len(view_id),'length of global-id:{}  is not equal to view-id:{}'.format(len(global_id),len(view_id))
            for idx,id in enumerate(global_id):
                hit_count.setdefault(id,0)
                num = len(view_id[idx])
                hit_count[id]+=num
                frame_count[id] = frame
        for id in frame_count:
            ## 根据两点来判断是否回收ID:1,global id在搜索范围内,被命中次数/搜索长度小于ratio 且
            ##                      2,global id最近被命中的时间距离当前帧大于 搜索长度*(1-ratio)
            if float(hit_count[id])/self.search_length<=ratio and \
                    1-(float(self.frame_idx-frame_count[id])/self.search_length)<=ratio:
                ## 判断是否是从出口区域出去的ID
                if id in self.pop_list:
                    continue
                self.next_id_list.recover(id)

    def update_view_id_map(self,frame_idx):
        """
        更新view_id_map,与id_map的区别在于这个是根据摄像头ID和行人ID找其对应的global id
        :param frame_idx: 当前帧ID
        :return:
        """
        self.view_id_map.setdefault(frame_idx,{})
        self.view_id_map[frame_idx] = {key:{} for key in self.tracker_list}
        for idx,items in enumerate(self.id_map[frame_idx]['view_id']):
            global_id = self.id_map[frame_idx]['global_id'][idx]
            for item in items:
                view = item[0]
                view_id = item[1]
                self.view_id_map[frame_idx][view][view_id] = global_id
        if len(self.view_id_map.keys())>self.max_length:
            index = min(self.id_map.keys())
            self.view_id_map.pop(index)


    def duplicates(selfh,lst,item):
        return [i for i,x in enumerate(lst) if x==item]

    def update_id_map(self,frame_idx,view_id_list):
        """
        更新 id_map
        :param frame_idx: 当前帧ID
        :param view_id_list: 当前帧匹配结果
        :return:
        """
        self.id_map.setdefault(frame_idx,{})
        self.id_map[frame_idx]['global_id'] = []
        self.id_map[frame_idx]['view_id'] = view_id_list
        if len(self.id_map.keys())>self.max_length:
            index = min(self.id_map.keys())
            self.id_map.pop(index)

    def search_global_id(self,view_id_list,inter_index,top=1):
        """
        根据被匹配到一起的点的历史信息,投票获得global id作为这些点的global id
        :param view_id_list: 被匹配到一起的点
        :param inter_index: 查找历史信息的帧的范围
        :param top: 将投票结果进行排序,选取对应排序的id. 如top=1,则选取投票数最高的ID;top=2则是次高
        :param method: 搜索时间窗的权重,'vote':权重相同;'gaussian':较早的帧有更高的权重;'revert-gaussian':距离当前帧较近的有更高的权重
        :return: global_id:投票获得的global id; frame: 票数 ; top: 当前的global id所对应的rank排序
        """
        vote_list = []
        frame_list = []
        for view_id in view_id_list:
            for inter_frame in inter_index:
                inter_data = self.id_map[inter_frame]['view_id']
                index = self.search_real_id(inter_data, view_id)
                if index is not None:
                    vote_list.append(self.id_map[inter_frame]['global_id'][index])
                    frame_list.append(inter_frame)
        if len(vote_list)==0:
            global_id = self.next_id_list.next_id()
            frame = 0
        elif self.vote_method == 'vote':
            ## vote use the most common item
            id_count = Counter(vote_list)
            items = id_count.items()
            items = id_count.most_common(len(items))

            if len(items) == 0:
                global_id = self.next_id_list.next_id()
                frame = 0
            elif top<=len(items):
                global_id = items[top-1][0]
                frame = items[top-1][1]
            else:
                global_id = self.next_id_list.next_id()
                frame = 0

        elif self.vote_method == 'gaussian':
            ## vote use gaussian sliding window
            vote_set = list(set(vote_list))
            vote_list = np.array(vote_list)
            vote_dict = {}
            frame_list = np.array(frame_list)
            for id in vote_set:
                idx = np.where(vote_list==id)
                count = float(np.sum(self.gau(self.frame_idx-frame_list[idx],self.search_length,self.search_length/2)))
                vote_dict[id] = count
            vote_dict = sorted(vote_dict.items(),key=lambda x:x[1],reverse=True)
            if len(vote_dict)==0:
                global_id = self.next_id_list.next_id()
                frame = 0
            elif top<=len(vote_dict):
                item = vote_dict[top-1]
                global_id = item[0]
                frame = item[1]
            else:
                global_id = self.next_id_list.next_id()
                frame = 0

        elif self.vote_method == 'revert-gaussian':
            ## vote use revert gaussian sliding window
            vote_set = list(set(vote_list))
            vote_list = np.array(vote_list)
            vote_dict = {}
            frame_list = np.array(frame_list)
            for id in vote_set:
                idx = np.where(vote_list==id)
                count = float(np.sum(self.gau(self.frame_idx-frame_list[idx],0,self.search_length/2)))
                vote_dict[id] = count
            vote_dict = sorted(vote_dict.items(),key=lambda x:x[1],reverse=True)
            if len(vote_dict)==0:
                global_id = self.next_id_list.next_id()
                frame = 0
            elif top<=len(vote_dict):
                item = vote_dict[top-1]
                global_id = item[0]
                frame = item[1]
            else:
                global_id = self.next_id_list.next_id()
                frame = 0
        return global_id,frame,top

    def gau(self,x,mu,std):
        """
        高斯函数
        :param x: 输入
        :param mu: 均值
        :param std: 标准差
        :return:
        """
        return np.exp(-np.power(x-mu,2.)/(2*np.power(std,2.)))


    def search_real_id(self,inter_data,view_id):
        """
        寻找view_id所对应的索引
        :param inter_data: 各个视角下的view id组成的list
        :param view_id: view id
        :return:view id 所对应的索引
        """
        for index,id_info in enumerate(inter_data):
            if view_id in id_info:
                return index
        return None




