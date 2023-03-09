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
            * path : dataset path (.npz file format)
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
    
    def downsample(self, factor, keep_strides = True):
        """
        Downsample data by an integer factor, keeping all strides of the data
        if keep_strides == True.
        The sequences will be replaced by their downsampled version, whose actions
        will have '_d0', ..., '_dn' appened to their name
        Input
        -----
            * factor : downsamplig ratio (fps must be divisible by the factor).
            * keep_strides : flag
        Output
        ------
            None (in-place operator)
        """

        assert self._fps % factor == 0

        # for every subject
        for subject in self._data.keys():
            new_actions = {}

            # for every action
            for action in list( subject.keys() ):

                # to keep all strides
                for idx in range(factor):
                    tup = {}

                    # for rotations and trayectory
                    for k in self._data[subject][action].key():
                        tup[k] = self._data[subject][action][k][idx::factor]
                    
                    new_actions[ action + '_d' + str(idx) ] = tup
                    
                    # In case we do not not want all strides
                    if not keep_strides:
                        break
            
            # replace subject actions
            self._data[subject] = new_actions
        
        # update sampling rate
        self._fps //= factor

    def _mirror_sequence(self, sequence):
        """
        Description
        Input
        -----
            * sequence : 
        Output
        ------
            mirror sequence
        """

        mirrored_rotations  = sequence['rotations'].copy()
        mirrored_trajectory = sequence['trajectory'].copy()

        joints_left = self._skeleton.joints_left()
        joints_right = self._skeleton.joints_right()

        # flip left/right joints
        mirrored_rotations[:, joints_left] = sequence['rotations'][:, joints_right]
        mirrored_rotations[:, joints_right] = sequence['rotations'][:, joints_left]

        mirrored_rotations[:, :, [2,3]] *= -1
        mirrored_trajectory[:, 0]*= -1

        return {
            'rotations' : qfix(mirrored_rotations),
            'trajectory': mirrored_trajectory
        }