# ---------------------------------------------- #
# Copyright (c) 2018-present, Facebook, Inc.
# https://github.com/facebookresearch/QuaterNet
# ---------------------------------------------- #

import numpy as np
import torch
from common.quaternion import qeuler_np, qfix

class MocapDataset:
    """
    Dataset object for Motion Capture
    Attributes
    ----------
        * _data : motion capture data
        * _fps  : sampling rate (frames per second)
        * _use_gpu  : flag to use GPU
        * _skeleton : skeleton object
    Methods
    -------
        *__getitem(__() :
        * _load() :
        * _mirror_sequence():
        * cuda()  :
        * downsample() :
        * mirror() :
        * compute_euler_angles() :
        * compute_positions() :
        * subjects() :
        * subject_actions() :
        * all_actions() :
        * fps() :
        * skeleton() :
    """

    def __init__(self, path, skeleton, fps):
        """
        MocapDataset initiaizer
        Input
        -----
            * path : dataset path (.npz file format)
            * skeleton : skeleton object
                - Skeleton  defines offsets, parents, children, etc.
            * fps : sampling rate (frames per second)
        Output
        ------
            None
        """

        # initialize attributes
        self._data = self._load(path)
        self._skeleton = skeleton
        self._fps = fps
        self._use_gpu = False
    
    def cuda(self):
        """
        Send skeleton to CUDA memory.
        Input
        -----
            None
        Output
        ------
            * self
        """

        self._use_gpu = True
        self._skeleton.cuda()

        return self
    
    def _load(self, path):
        """
        Read datset and return it as a dictionary.
        Input
        -----
            * path : dataet path (.npz file format)
        Output
        ------
            * result : Nested dictionaries
            
            result -> subjects -> actions -> rotations, trajectory
        """
        
        # dictiorary with subjects
        result = {}

        data = np.load(path, 'r')
        for i, (trajectory, rotations, subject, action) in enumerate(
            zip( data['trajectories'] ,
                 data['rotations']    ,
                 data['subjects']     ,
                 data['actions']      )
        ):
            
            # every subject is dictionary of actions
            if subject not in result:
                result[subject] = {}
            
            # every action is a dictionary with two keys:
            # rotations and trajectory
            result[subject]['action'] = {
                'rotations' : rotations,
                'trajectory' : trajectory
            }
        
        return result