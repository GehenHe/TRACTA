#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-4-16 上午9:58
# @Author  : Aries
# @Site    :
# @File    : global_id_map.py
# @Software: PyCharm

import numpy as np
from collections import Counter

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

class Global_ID:
    '''
     Attributes
    ----------
        state : IDState
            The current global id state.

    '''


    def __init__(self,global_id):
        self.state = IDState.Tentative
        self.id = global_id

    def mark_confirmed(self):
        self.state = IDState.Confirmed



