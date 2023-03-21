# ---------------------------------------------- #
# Copyright (c) 2018-present, Facebook, Inc.
# https://github.com/facebookresearch/QuaterNet
# ---------------------------------------------- #

import torch
import numpy as np
from common.quaternion import qmul_np, qmul, qrot

class Skeleton:

    """
    Class to handle skeleton hierarchical data.
    Attributes
    ----------
        * offsets : joint relative offsets in reference position.
        * parents : parent indices.
        * joints_left : left joints indices.
        * joints_rights : right joints indices.
    Methods
    -------
        * d
    """

    def __init__(self, offsets, parents, joints_left = None, joints_right = None):
        """
        Skeleton initializer
        Input
        -----
            * offsets : joint relative offsets in reference position.
            * parents : parent indices.
            * joints_left   : left joints indices.
            * joints_rights : right joints indices.
            * _children : indices of joint children.
            * _has_children : boolean array, 1 if the joint has children, 0 if not
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
        Return a boolean array: 1 if the joint has children, 0 if not
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
        Return joints' children
        """
        return self._children
    

    def remove_joints(self, joints_to_remove, dataset):
        """
        Remove joints specified in joints_to_remove, both from the skeleton
        object and from the dataset (modified in place). the rotation of removed
        joints are propagated along the kinematic chain (forward kinematics).
        Input
        -----
            * joints_to_remove : joints to remove qiven by indices
            * dataset : dataset object
        Output
        ------
            None
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
        Forward kinematics using root positions and  local rotations.
        Input
        -----
            * rotations     : tensor with dimensions (N, L, J, 4); sequence of quaternions 
            * rootpositions : tensor with dimensions (N, L, 3); root world positions

            N -> batch size, L -> sequence length, J -> number of joints 
        Output
        ------
            Joints world positions
        """
        
        assert len( rotations.shape ) == 4
        assert rotations.shape[-1] == 4

        positions_world = []
        rotations_world = []

        expanded_offsets = self._offsets.expand(
            rotations.shape[0],
            rotations.shape[1],
            self._offsets.shape[0],
            self._offsets.shape[1]
        )

        # parallel along the batch and time dimensions
        for i in range( self._offsets.shape[0] ):
            if self._parents[i] == -1:
                positions_world.append(root_positions)
                rotations_world.append(rotations[:, :, 0])
            else:
                positions_world.append(
                    qrot( rotations_world[ self._parents[i] ],
                          expanded_offsets[:, :, i] ) + positions_world[ self._parents[i] ]
                )
                if self._has_children[i]:
                    rotations_world.append(
                        qmul( rotations_world[ self._parents[i] ], rotations[:, :, i] )
                    )
                else:
                    rotations_world.append(None)
                
        return torch.stack( positions_world, dim = 3 ).permute( 0, 1, 3, 2 )


    def joints_left(self):
        """
        Return skeleton left joints
        """
        return self._joints_left
    

    def joints_right(self):
        """
        Return skeleton right joints
        """
        return self._joints_right


    def _compute_metadata(self):
        """
        Compute self._children and slef._has_children.
        Input
        -----
            None
        Output
        ------
            None
        """
        
        self._has_children = np.zeros( len( self._parents) ).astype(bool)
        for i, parent in enumerate( self._parents ):
            if parent != -1:
                self._has_children[parent] = True
        
        self._children = []
        for i, parent in enumerate( self._parents ):
            self._children.append([])
        for i, parent in enumerate( self._parents ):
            if parent != -1:
                self._children[parent].append(i)