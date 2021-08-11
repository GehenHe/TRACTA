#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 1/17/18 10:36 AM
# @Author  : gehen
# @File    : visualization.py
# @Software: PyCharm Community Edition

import cv2
import colorsys
import numpy as np
from PIL import Image

def create_unique_color_float(tag, hue_step=0.41):
    """Create a unique RGB color code for a given track id (tag).

    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.

    Parameters
    ----------
    tag : int
        The unique target identifying tag.
    hue_step : float
        Difference between two neighboring color codes in HSV space (more
        specifically, the distance in hue channel).

    Returns
    -------
    (float, float, float)
        RGB color code in range [0, 1]

    """
    h, v = (tag * hue_step) % 1, 1. - (int(tag * hue_step) % 4) / 5.
    r, g, b = colorsys.hsv_to_rgb(h, 1., v)
    return r, g, b


def create_unique_color_uchar(tag, hue_step=0.41):
    """Create a unique RGB color code for a given track id (tag).

    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.

    Parameters
    ----------
    tag : int
        The unique target identifying tag.
    hue_step : float
        Difference between two neighboring color codes in HSV space (more
        specifically, the distance in hue channel).

    Returns
    -------
    (int, int, int)
        RGB color code in range [0, 255]

    """
    r, g, b = create_unique_color_float(tag, hue_step)
    return int(255*r), int(255*g), int(255*b)

def view_to_top(H,xy):
    temp = H*np.mat([xy[0],xy[1],1]).T
    temp = temp/temp[2]
    return temp[0:2]

def draw_single_view(image,tracks):
    if len(tracks)>0:
        for track in tracks:
            if not track.is_confirmed() or track.hit_streak==0:
                continue
            person_id = track.track_id

            id_color = create_unique_color_uchar(person_id)
            bbox = map(int,track.to_tlwh())
            image = cv2.rectangle(image,(bbox[0],bbox[1]),(bbox[2]+bbox[0],bbox[3]+bbox[1]),id_color,4)

            image = cv2.putText(image,'{}'.format(person_id),(bbox[0],bbox[1]),cv2.FONT_ITALIC,0.8,[0,0,0],2)
            point = track.point
            if point is not None:
                image = cv2.circle(image, (point[1], point[0]), 4, [0, 255, 0], 4)
    return image


def draw_tracker(image,tracks,tracker_id=None,match_info=None,time_since_update=1):
    if len(tracks)>0:
        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > time_since_update:
                continue
            if match_info is not None:
                person_id = match_info[tracker_id][track.track_id]
            else:
                person_id = track.track_id
            id_color = create_unique_color_uchar(person_id)
            bbox = map(int,track.to_tlwh())
            image = cv2.rectangle(image,(bbox[0],bbox[1]),(bbox[2]+bbox[0],bbox[3]+bbox[1]),id_color,4)
            image = cv2.putText(image,'{}'.format(person_id),(bbox[0],bbox[1]),cv2.FONT_ITALIC,0.8,[0,0,0],2)
            point = track.point
            if point is not None:
                top_point = point[0]
                bot_point = point[1]
                image = cv2.circle(image, (top_point[1], top_point[0]), 4, [0, 255, 0], 4)
                image = cv2.circle(image, (bot_point[1], bot_point[0]), 4, [0, 255, 0], 4)
    return image

def search_id_index(view_id_list,view_id):
    for index,view_id_index in enumerate(view_id_list):
        if view_id in view_id_index:
            return index
    assert 'view_id not in view_id_list'


def draw_tracker_global(image,tracks,frame_id_map):
    if len(tracks)>0:
        for track in tracks:
            if not track.is_confirmed() or track.hit_streak==0 or track.time_since_update>1:
                continue
            person_id = track.track_id
            id_color = create_unique_color_uchar(person_id)
            bbox = map(int,track.to_tlwh())
            image = cv2.rectangle(image,(bbox[0],bbox[1]),(bbox[2]+bbox[0],bbox[3]+bbox[1]),id_color,4)
            image = cv2.putText(image,'{}'.format(person_id),(bbox[0],bbox[1]),cv2.FONT_ITALIC,0.8,[0,0,0],2)
    return image

def generate_top_view(frame_idx,tracking_results,time_length=100):
    black_board = np.zeros([2400,2400,3])
    frame_range = range(frame_idx - time_length, frame_idx)
    for view,view_result in enumerate(tracking_results):
        for person_id in view_result[frame_idx]:
            # dot_xy = view_result[frame_idx][person_id]
            # dot_xy = [dot_xy[0]/2+100,dot_xy[1]/2+350]
            # dot_xy[0]+=10
            # black_board = cv2.putText(black_board, '{}-{}'.format(view, person_id),tuple(dot_xy),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,[255,255,255],2)
            for frame in frame_range:
                if frame not in view_result.keys():
                    continue
                frame_data = view_result[frame]
                if person_id in view_result[frame]:
                    color = create_unique_color_uchar(person_id)
                    xy = frame_data[person_id]
                    xy = [int(xy[0][0]+200),int(xy[0][1]+700)]
                    xy = tuple(xy)
                    black_board = cv2.circle(black_board, xy, 5, color, 7)
    black_board = black_board[600:1600, 900:1800, :]
    return black_board




