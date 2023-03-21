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
        *__getitem__
        * _load
        * _mirror_sequence
        * cuda
        * downsample
        * mirror
        * compute_euler_angles
        * compute_positions
        * subjects
        * subject_actions
        * all_actions
        * fps
        * skeleton
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
        Send skeleton to CUDA memory; set self._use_cuda = True
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
            for action in list( self._data[subject].keys() ):

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
        Mirror a skeleton motion sequence along the X-axis
        (https://stackoverflow.com/questions/32438252/efficient-way-to-apply-mirror-effect-on-quaternion-rotation)
        Input
        -----
            * sequence : skeleton sequence; Quaternions
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
    
    
    def mirror(self):
        """
        Data augmentation by mirroring every sequence in the data set.
        The mirrored sequences are saved with a "_m" appended to the action name.
        Input
        -----
            None
        Output
        ------
            None (In-plae operator)
        """
        
        # for every subject
        for subject in self._data.keys():
            # for every action
            for action in list( self._data[subject].keys() ):
                
                if '_m' in action: # already mirrored sequence
                    continue

                # Add mirrored sequence 
                self._data[subject][action + '_m'] = self._mirror_sequence(
                    self._data[subject][action]
                )
    
    
    def compute_euler_angles(self, order):
        """
        Compute Euler angles parameterization for every sequence.
        Add a new key to every action in self.data: 'rotations_euler'.
        Input
        -----
            * order : order of rotations in Euler angles
        Output
        ------
            None (In-place operator)
        """

        # for every subject, and for every action
        for subject in self._data.values():
            for action in subject.values():
                # add euler angles parameterization
                action['rotation_euler'] = qeuler_np(
                    action['rotations'], order, use_gpu = self._use_gpu
                )


    def compute_positions(self):
        """
        Forward kinematics to compute global and local positions of the joints.
        Add to new keys to every action in self._data: 'positions_world', 'positions_local'
        Input
        -----
            None
        Output
        ------
            None (In-place operator)
        """

        for subject in self._data.values():
            for action in subject.values():

                # rotations and trajectory as PyTorch tensor
                rotations = torch.from_numpy( action['rotations'].astype('float32') ).unsqueeze(0)
                trajectory = torch.from_numpy( action['trajectory'].astype('float32') ).usqueeze(0)

                # send to CUDA
                if self._use_gpu:
                    rotations.cuda()
                    trajectory.cuda()
                
                action['positions_world'] = self._skeleton.forward_kinematics(rotations, trajectory).squeeze(0).cpu().numpy()

                # Absolute translations across the XY plane are removed
                trajectory[:, :, [0,2]] = 0
                action['positions_local'] = self._skeleton.forward_kinematics(rotations, trajectory).squeeze(0).cpu().numpy()
    

    def __getitem__(self, key):
        """
        Return the subject dictionary specified by key.
        Input
        -----
            * key: subject name in dataset
        Output
        ------
            * subject dictionary ( subject -> actions -> rotations, trajectory, etc )
        """

        return self._data[key]
    

    def subjects(self):
        """
        Return subject names in the dataset object.
        Input
        -----
            None
        Output
        ------
            Subjects; _data dictionary keys.
        """

        return self._data.keys()
    

    def subject_actions(self, subject):
        """
        Return action names of given subject.
        Input
        -----
            subject : subject name; str
        Output
        ------
            Actions; subject dictionary keys.
        """

        return self._data[subject].keys()
    
    
    def all_actions(self):
        """
        Return all the action names in the dataset.
        Input
        -----
            None
        Output
        ------
            list with all the (subject, action) tuples. 
        """

        result = []

        for subject, actions in self._data.items():
            for action in actions.keys():
                result.append( (subject, action) )
        
        return result
    
    
    def fps(self):
        """
        Return  sampling rate (frames per second).
        Input
        -----
            None
        Output
        ------
            Sampling frequency
        """

        return self._fps
    
    
    def skeleton(self):
        """
        Return skeleton.
        Input
        -----
            None
        Output
        ------
            Skeleton
        """

        return self._skeleton