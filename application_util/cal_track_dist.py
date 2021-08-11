#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 3/19/18 2:54 PM
# @Author  : gehen
# @File    : cal_track_dist.py
# @Software: PyCharm Community Edition

from scipy.spatial.distance import euclidean
from operator import xor
import numpy as np
import cv2
np.set_printoptions(precision=4)

def pred_dot(H,xy,r):
    """

    :param H:
    :param xy:
    :param r:
    :return:
    """
    m0 = np.asarray(xy).T
    S0 = np.asarray([[r*r, 0], [0, r*r]])
    nu = 2
    k = 1
    v = []
    w = []
    for i in range(nu * 2 + 1):
        if i == 0:
            v.append(m0)
            w.append((float(k) / (k + nu)))
        elif i <= nu:
            vi = m0 + np.sqrt((k + nu) * S0)[i - 1].T
            v.append(vi)
            w.append(float(k) / (2 * (k + nu)))
        elif i > nu:
            vi = m0 - np.sqrt((k + nu) * S0)[i / 2 - 1].T
            v.append(vi)
            w.append(float(k) / (2 * (k + nu)))
    gesai = []
    for i in range(nu * 2 + 1):
        temp = np.concatenate((v[i], np.asarray([1]))).reshape(3, 1)
        gesai_i = H[0:2, :] * temp / (H[2] * temp)/10
        gesai.append(gesai_i)
    u0 = sum([gesai[i] * w[i] for i in range(len(gesai))])
    cov = sum([w[i] * (gesai[i] - u0) * (gesai[i] - u0).reshape(1, 2) for i in range(len(gesai))])

    return u0,cov

def img_to_world(point,cali_H,height):
    point = np.asarray(point,dtype=np.float32).reshape([1,1,-1])
    H = cali_H[0][height]
    dist = cali_H[1]
    k = cali_H[2]
    undist_point = cv2.undistortPoints(point, k, dist, P=k)
    input = np.mat([undist_point[0][0][0],undist_point[0][0][1],1])
    world = H*input.T
    world = world/world[2]
    world = world[0:2]/10
    return [world,1]
    # input = np.mat([undist_point[0][0][0], undist_point[0][0][1], 1])
    # return pred_dot(H,input,r)

def generate_sequence(tracking_results=None, queue_length=21):
    index = tracking_results.keys()[0]
    frame_idx = max(tracking_results[index].keys())
    start_frame = frame_idx - queue_length
    frame_list = range(start_frame + 1, frame_idx + 1)
    data = []
    for tracker_id in tracking_results:
        tracker = tracking_results[tracker_id]
        view_data = []
        id_list = tracker[frame_idx].keys()
        id_list.sort()
        for id in id_list:
            id_data = []
            for frame in frame_list:
                if frame not in tracker.keys():
                    id_data.append([])
                    continue
                if id in tracker[frame].keys():
                    id_data.append(tracker[frame][id][0])
                else:
                    id_data.append([])
            view_data.append(id_data)
        data.append(view_data)
    return data

def cal_track_dist(seq1,seq2,conf_dist=50):
    assert len(seq1)==len(seq2),"tracklet length is not equal when calculate distance!"
    dist = 0.0
    miss = 0
    count = 0
    for i in range(len(seq1)):
        if isinstance(seq1[i], list) and isinstance(seq2[i], list):
            continue
        elif xor(not isinstance(seq1[i], list), not isinstance(seq2[i], list)):
            miss += 1
            count+=1
        elif not isinstance(seq1[i], list) and not isinstance(seq2[i], list):
            dist += euclidean(seq1[i], seq2[i])
            count += 1

    # if miss > 50:
    #     dist += 1000
    # if miss > 15:
    #     dist += 2000
    # if miss > 5:
    #     dist += 300

    dist += miss*conf_dist
    dist = dist/count
    return dist


# def cal_similarity(tracking_results,sigma=20):
#     data_seq = generate_sequence(tracking_results)
#     smooth_seq = trajctory_smooth(data_seq)
#     view_list = [len(data_seq[i]) for i in range(len(data_seq))]
#     det_num = sum(view_list)
#     sim_matrix = np.zeros([det_num,det_num])
#     # data_seq = [smooth_seq[i][j] for i in range(len(smooth_seq)) for j in range(len(smooth_seq[i]))]
#     for i in range(det_num):
#         for j in range(i+1,det_num):
#             dist = cal_track_dist(smooth_seq[i],smooth_seq[j])
#             sim_matrix[i][j] = np.exp(-pow(dist,2)/(2*pow(sigma,2)))
#     sim_matrix = sim_matrix+sim_matrix.T
#     count = 0
#     for view in view_list:
#         sim_matrix[count:view+count,count:view+count] = np.eye(view)
#         count += view
#     return sim_matrix,view_list

def cal_dist_matrix(tracking_results,dist_length=21):
    data_seq = generate_sequence(tracking_results,dist_length)
    smooth_seq = trajctory_smooth(data_seq)
    view_list = [len(data_seq[i]) for i in range(len(data_seq))]
    det_num = sum(view_list)
    dist_matrix = np.zeros([det_num,det_num])
    # data_seq = [smooth_seq[i][j] for i in range(len(smooth_seq)) for j in range(len(smooth_seq[i]))]
    for i in range(det_num):
        for j in range(i+1,det_num):
            dist = cal_track_dist(smooth_seq[i],smooth_seq[j])
            dist_matrix[i][j] = dist
    dist_matrix = dist_matrix+dist_matrix.T
    return dist_matrix,view_list

def trajctory_smooth(track_datas,conf_dist=10000):

    kalman = cv2.KalmanFilter(4,2)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.003
    kalman.measurementNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * 1

    track_data_seq = [view_data[i] for view_data in track_datas for i in range(len(view_data))]
    smooth_data_track = []
    for track_data in track_data_seq:
        start = 0
        smooth_data = []
        for idx,data in enumerate(track_data[1:]):
            smooth_data.append([])

            # init the kalman filter in the first frame
            if len(data)>0:
                if not start:
                    x = data[0]
                    y = data[1]
                    if len(track_data[idx])>0:
                        vx = data[0]-track_data[idx][0]
                        vy = data[1]-track_data[idx][1]
                    else:
                        vx = 0
                        vy = 0
                    kalman.statePre = np.asarray([x, y, vx, vy], np.float32)
                    start = 1
                    # smooth_data[idx] = np.asarray([x,y],np.float32)
                    smooth_data[idx] = np.asarray(data,np.float32)
                    kalman.correct(smooth_data[idx])


                    # on the believe of detect data is correct when dist<=conf_dist
                    # elif len(data)>0 and start and euclidean(data,smooth_data[idx-1])<=conf_dist:
                else:
                    predict_data = kalman.predict()[0:2]
                    if euclidean(data,predict_data)<=conf_dist:        # use the predicted point to check
                                                                                                        # whether the detect point is contiously
                        smooth_data[idx] = np.asarray(data,np.float32)
                        temp_data = np.asarray([smooth_data[idx][0],smooth_data[idx][1]],np.float32)
                        kalman.correct(temp_data)
                    else:
                        temp_data = np.asarray([data[0], data[1]], np.float32)
                        kalman.correct(temp_data)[0:2]
                        smooth_data[idx] = kalman.predict()[0:2]


            # elif len(data)==0 and start:                                                            # in the case of miss detection
            #     pred = kalman.predict()[0:2]
            #     smooth_data[idx] = pred
            #     temp_data = np.asarray([smooth_data[idx][0], smooth_data[idx][1]],np.float32)
            #     kalman.correct(temp_data)

        smooth_data_track.append(smooth_data)
    return smooth_data_track








