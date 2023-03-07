# ---------------------------------------------- #
# Copyright (c) 2018-present, Facebook, Inc.
# https://github.com/facebookresearch/QuaterNet
# ---------------------------------------------- #

import torch
import numpy as np
from common.quaternion import qmul_np, qmul, qrot

class Skeleton:

    """
    Class to handle skeleton hierarchical data
    Attributes
    ----------
        * d
    Methods
    -------
        * d
    """

    def __init__(self, offsets, parents, joints_left = None, joints_right = None):
        """
        Skeleton initializer
        Input
        -----
            * offsets
            * parents
            * joints_left
            * joints_right
        Output
        ------
            None
        """
        
        assert len(offsets) == len(parents)

        # initialize attributes
        self._offsets = torch.FloatTensor(offsets)
        self._parents = np.array(parents)
        self._joints_left = joints_left
        self._joints_right = joints_right
        self._compute_metadata()

    def cuda(self):
        
        """
        Send offsets to CUDA memory.
        Input
        -----
            None
        Output
        ------
            * Self
        """

        self._offsets = self._offsets.cuda()

        return self
    
    def num_joints(self):
        """
        Return the number of joints in the skeleton
        """

        return self._offsets.shape[0]
    
    def offsets(self):
        """
        Return the offsets of the skeleton
        """

        return self._offsets
    
    def parents(self):
        """
        Return the parents of the skeleton
        """

        return self._parents
    
    def has_children(self):
        """
        Returns a boolean array: 1 if the joint has children, 0 if not
        Input
        -----
            None
        Output
        ------
            Boolean array: 1 if the joint has children, 0 if not
        """

        return self._has_children
    
    def children(self):
        """
        """

        return self._children
    
    def remove_joints(self, joints_to_remove, dataset):
        """
        Remove joints specified in joints_to_remove, both from the skeleton
        object and from the dataset (modified in place). the rotation of removed
        joints are propagated along the kinematic chain (forward kinematics). 
        """

        valid_joints = []

        # select valid joints, remove joints_to_remove
        for joint in range( len( self._parents ) ):
            if joint not in joints_to_remove:
                valid_joints.append(joint)
        
        # Update transformations in the dataset
        for subject in dataset.subjects():
            for action in dataset[subject].keys():
                
                rotations = dataset[subject][action]['rotations']

                for joint in joints_to_remove:
                    for child in self._children[joint]:
                        rotations[:, child] = qmul_np( rotations[:,joint] , rotations[:,child] )
                    rotations[:, joint] = [1, 0, 0,0 ] # Identity quaternion
                
                dataset[subject][action]['rotations'] = rotations[:, valid_joints]
        
        index_offsets = np.zeros( len( self._parents ), dtype = int )
        new_parents = []

        for i , parent in enumerate( self._parents ):
            if i not in joints_to_remove:
                new_parents.append(parent - index_offsets[parent] )
            else:
                index_offsets[i:] += 1
        
        self._parents = np.array(new_parents)
        self._offsets = self._offsets[valid_joints]
        self._compute_metadata()
    
    def forward_kinematics(self, rotations, root_positions):
        """
        Description
        """
        pass

    def _compute_metadata(self):
        """
        Description
        """
        pass
