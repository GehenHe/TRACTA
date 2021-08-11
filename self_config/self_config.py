# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from ast import literal_eval
from self_config.self_collections import AttrDict
import copy
import logging
import numpy as np
import os
import os.path as osp
import yaml

logger = logging.getLogger(__name__)

__C = AttrDict()
# Consumers can get config by:
#   from core.config import cfg
self_cfg = __C  # type: AttrDict

__C.TRAIN = AttrDict()
__C.TRAIN.MAX_SIZE = 1000

__C.TEST = AttrDict()

## 检测相关参数
__C.TEST.detector_name = b''                # 检测器名字,'coco-res101'
__C.TEST.min_detection_height = 0           # 最小检测框高度
__C.TEST.min_confidence = 0.8               # 最小检测框阈值
__C.TEST.nms_max_overlap = 0.75             # nms 阈值

## 单摄像头追踪相关参数
__C.TEST.tracker_list = []                  # 摄像头所对应的key值,此处key值应当相机内外参的key值相对应,并且顺序应当与读入图片顺序相对应
__C.TEST.tracker_n_init = 10                # 初始化一个tracker所需要的帧数,该值越大,代表初始化一条轨迹需要的帧数越多
__C.TEST.tracker_max_age = 200              # 轨迹消失的最长时间,如果大于这个阈值仍然没有一次匹配上,则删除这条轨迹
__C.TEST.nn_budget = 100                    # 保存的特征的长度
__C.TEST.gating_thresh = 5                  # 卡方分布索引值,为1-10的整数.该阈值用于剔除匈牙利算法匹配时的一些不可能连接边;该值越小,轨迹的连续性要求越高,\
                                            # 该值越大,轨迹的连续性要求越小.如对每帧做处理时,该值可设为2,而每隔五帧做处理时,该值可设置为5.
__C.TEST.tracker_max_iou_distance = 0.7     # position信息做匹配的最大距离,如果匹配到的结果大于这个阈值,则认为没有匹配上
__C.TEST.nn_matching_threshold = 0.8        # appearance信息做匹配的最大距离,如果匹配到的结果大于这个阈值,则认为没有匹配上
    ## 下面参数不建议轻易更改
__C.TEST.init_height = 175                  # 初始化的身高
__C.TEST.wh_thresh1 = 5                     # 宽高比双阈值筛选中较大的阈值
__C.TEST.wh_thresh2 = 4                     # 宽高比双阈值筛选中较小的阈值
__C.TEST.h_thresh = 4.8                     # 估计身高的宽高比阈值
__C.TEST.width_ratio = 0.2                  # 估计身高的图像区域阈值


## 多摄像头匹配相关参数
__C.TEST.search_length = 50                 # 选取global id时的时间窗口长度
__C.TEST.vote_method = 'vote'               # 投票选取global id的时间窗口权重,分别为'vote','gaussian','revert-gaussian'
__C.TEST.dist_length = 21                   # 计算两条轨迹间的距离的时间窗口长度,大于2的整数值
__C.TEST.max_length = 200                   # 保存结果长度


## 显示相关参数
__C.TEST.show_plane = [3000,3000]           # 画板大小
__C.TEST.top_length = 100                   # 顶视图显示的长度
__C.TEST.convert_method = [350,350,2]       # 世界坐标系转换到显示坐标系的方法,如[dx,dy,times]--> x=(x0+dx)*times, y=(y0+dy)*times
__C.TEST.revert = True                      # 是否将x,y坐标翻转
__C.TEST.is_heat = True                     # 是否显示热力图
__C.TEST.heatmap_size = 50                  # 热力图块的大小
__C.TEST.heat_length = 100                  # 热力图显示的历史帧数
__C.TEST.heat_gap = 10                      # 热力图更新间隔
__C.TEST.heat_vmax = 20                     # 热力图最大被命中次数
__C.TEST.time_since_update = 1              # 显示预测的帧数


## 基本信息
__C.TEST.resize_ratio = 0.5                 # 图片resize比例,0.5将图片resize为一般大小
__C.TEST.image_width = 1280                 # 输入图片宽度
__C.TEST.image_height = 720                 # 输入图片高度
__C.TEST.out_area = []                      # 出口区域:[xmin,ymin,xmax,ymax],如果某一边无边界,则改为[xmin,-inf,inf,ymax]
__C.TEST.place_region = []                  # 活动区域:[xmin,ymin,xmax,ymax],如果对边界不加约束,则改为[-inf,-inf,inf,inf]
__C.TEST.cameras = ()                       # 输入的摄像头ID list,如 [1,2,3],与映射矩阵的key值相对应
__C.TEST.camera_wait_time = 0.2
__C.TEST.height_list = [0]+range(150,201,5) # 高度list

# ---------------------------------------------------------------------------- #
# Deprecated options
# If an option is removed from the code and you don't want to break existing
# yaml configs, you can add the full config key as a string to the set below.
# ---------------------------------------------------------------------------- #
_DEPCRECATED_KEYS = {'FINAL_MSG', 'MODEL.DILATION', 'ROOT_GPU_ID', 'RPN.ON', 'TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED',
                     'TRAIN.DROPOUT', 'USE_GPU_NMS', 'TEST.NUM_TEST_IMAGES'}

def merge_cfg_from_cfg(cfg_other):
    """Merge `cfg_other` into the global config."""
    _merge_a_into_b(cfg_other, __C)

def merge_cfg_from_file(cfg_filename):
    """Load a yaml config file and merge it into the global config."""
    if cfg_filename == "":
        return
    with open(cfg_filename, 'r') as f:
        yaml_cfg = AttrDict(yaml.load(f))
    _merge_a_into_b(yaml_cfg, __C)

def _merge_a_into_b(a, b, stack=None):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    assert isinstance(a, AttrDict), 'Argument `a` must be an AttrDict'
    assert isinstance(b, AttrDict), 'Argument `b` must be an AttrDict'

    for k, v_ in a.items():
        full_key = '.'.join(stack) + '.' + k if stack is not None else k
        # a must specify keys that are in b
        if k not in b:
            if _key_is_deprecated(full_key):
                continue
            else:
                raise KeyError('Non-existent config key: {}'.format(full_key))

        v = copy.deepcopy(v_)
        v = _decode_cfg_value(v)
        v = _check_and_coerce_cfg_value_type(v, b[k], k, full_key)

        # Recursively merge dicts
        if isinstance(v, AttrDict):
            try:
                stack_push = [k] if stack is None else stack + [k]
                _merge_a_into_b(v, b[k], stack=stack_push)
            except BaseException:
                raise
        else:
            b[k] = v


def _key_is_deprecated(full_key):
    if full_key in _DEPCRECATED_KEYS:
        logger.warn(
            'Deprecated config key (ignoring): {}'.format(full_key)
        )
        return True
    return False

def _decode_cfg_value(v):
    """Decodes a raw config value (e.g., from a yaml config files or command
    line argument) into a Python object.
    """
    # Configs parsed from raw yaml will contain dictionary keys that need to be
    # converted to AttrDict objects
    if isinstance(v, dict):
        return AttrDict(v)
    # All remaining processing is only applied to strings
    if not isinstance(v, basestring):
        return v
    # Try to interpret `v` as a:
    #   string, number, tuple, list, dict, boolean, or None
    try:
        v = literal_eval(v)
    # The following two excepts allow v to pass through when it represents a
    # string.
    #
    # Longer explanation:
    # The type of v is always a string (before calling literal_eval), but
    # sometimes it *represents* a string and other times a data structure, like
    # a list. In the case that v represents a string, what we got back from the
    # yaml parser is 'foo' *without quotes* (so, not '"foo"'). literal_eval is
    # ok with '"foo"', but will raise a ValueError if given 'foo'. In other
    # cases, like paths (v = 'foo/bar' and not v = '"foo/bar"'), literal_eval
    # will raise a SyntaxError.
    except ValueError:
        pass
    except SyntaxError:
        pass
    return v

def _check_and_coerce_cfg_value_type(value_a, value_b, key, full_key):
    """Checks that `value_a`, which is intended to replace `value_b` is of the
    right type. The type is correct if it matches exactly or is one of a few
    cases in which the type can be easily coerced.
    """
    # The types must match (with some exceptions)
    type_b = type(value_b)
    type_a = type(value_a)
    if type_a is type_b:
        return value_a

    # Exceptions: numpy arrays, strings, tuple<->list
    if isinstance(value_b, np.ndarray):
        value_a = np.array(value_a, dtype=value_b.dtype)
    elif isinstance(value_b, basestring):
        value_a = str(value_a)
    elif isinstance(value_a, tuple) and isinstance(value_b, list):
        value_a = list(value_a)
    elif isinstance(value_a, list) and isinstance(value_b, tuple):
        value_a = tuple(value_a)
    else:
        raise ValueError(
            'Type mismatch ({} vs. {}) with values ({} vs. {}) for config '
            'key: {}'.format(type_b, type_a, value_b, value_a, full_key)
        )
    return value_a

def get_output_dir(datasets, training=True):
    """Get the output directory determined by the current global config."""
    assert isinstance(datasets, (tuple, list, basestring)), \
        'datasets argument must be of type tuple, list or string'
    is_string = isinstance(datasets, basestring)
    dataset_name = datasets if is_string else ':'.join(datasets)
    tag = 'train' if training else 'test'
    # <output-dir>/<train|test>/<dataset-name>/<model-type>/
    outdir = osp.join(__C.OUTPUT_DIR, tag, dataset_name, __C.MODEL.TYPE)
    if not osp.exists(outdir):
        os.makedirs(outdir)
    return outdir