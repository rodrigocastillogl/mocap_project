# ---------------------------------------------- #
# Copyright (c) 2018-present, Facebook, Inc.
# https://github.com/facebookresearch/QuaterNet
# ---------------------------------------------- #

from common.pose_network import PoseNetwork
from common.quaternion import qeuler, qeuler_np, qfix, euler_to_quaternion
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from time import time

class PoseNetworkHierarchy(PoseNetwork):
    """
    Hierarchy QuaterNet model for skeleton position prediction.
    Attributes
    ----------
        * translations_size : (num_outputs) extra inputs/outputs, in addition to joint rotations.
        * controls_size     : (num_controls) extra input features.
        * model_velocities  : flag to add a queternion multiplication block on the
                              RNN output to force the network to model velocities
                              instead of absolute rotations.
        * model    : Quaternet model.
        * use_cuda : flag to use CUDA.
        * prefix_length : Number of frames in the input. 
    """

    def __init__(self, prefix_length, selected_joints = None):
        
        super().__init__(prefix_length = prefix_length ,
                         num_joints = 32               ,
                         num_controls = 0              ,
                         num_outputs = 0               ,
                         model_velocities = True       ,
                         selected_joints = selected_joints )


    def _prepare_next_batch_impl(self, batch_size, dataset, target_length, sequences):
        """
        Load a prepare bacth: input sequence and target sequence.
        Input
        -----
            * batch_size : batch size.
            * dataset : dataset object load data.
            * target_length : target sequence length.
            * sequences : list of sequences in dataset to generate batch.
        Output
        ------
            * buffer_quat  : input sequence.
            * buffer_euler : target sequence.
        """
        
        buffer_in = np.zeros( (batch_size, self.prefix_length + target_length, self.num_joints*4),
                               dtype = 'float32' )

        if self.loss_mode == 'euler':
            # Original loss function (Euler angles L1 distance)
            buffer_out = np.zeros( (batch_size, target_length, self.num_selected_joints*3),
                                    dtype = 'float32' )
        elif self.loss_mode == 'quaternions':
            # Loss function with quaternions cosine distance
            buffer_out = np.zeros( (batch_size, target_length, self.num_selected_joints*4),
                                    dtype = 'float32' )

        sequences = np.random.permutation(sequences)

        batch_idx = 0
        for i, (subject, action) in enumerate(sequences):

            # pick a random chunk from each sequence
            start_idx = np.random.randint(
                0, dataset[subject][action]['rotations'].shape[0] - self.prefix_length - target_length + 1
            )
            mid_idx = start_idx + self.prefix_length
            end_idx = start_idx + self.prefix_length + target_length

            # input sequence as quaternions 
            buffer_in[batch_idx] = dataset[subject][action]['rotations'][start_idx:end_idx].reshape(
                self.prefix_length + target_length, -1
            )
            
            if self.loss_mode == 'euler':
                # Original loss function (Euler angles L1 distance)
                buffer_out[batch_idx] = dataset[subject][action]['rotation_euler'][mid_idx:end_idx, self.selected_joints].reshape(
                                        target_length, -1 )
            elif self.loss_mode == 'quaternions':
                # Loss function with quaternions cosine distance
                buffer_out[batch_idx] = dataset[subject][action]['rotations'][mid_idx:end_idx, self.selected_joints].reshape(
                                        target_length, -1 )

            batch_idx += 1
            if batch_idx == batch_size or i == (len(sequences) - 1):
                yield buffer_in[:batch_idx], buffer_out[:batch_idx]
                batch_idx = 0


    def _loss_impl(self, predicted, expected):
        """
        Loss function.
        Input
        -----
            * predicted : predicted sequence; Quaternions.
            * expected : ground truth sequence.
        Output
        ------
            * Loss
        """

        predicted_quat = predicted.view( predicted.shape[0], predicted.shape[1], -1 , 4 )[:,:, self.selected_joints ,:]

        if self.loss_mode == 'euler':
            # Original loss function (Euler angles L1 distance)
            expected_euler = expected.view( predicted.shape[0], predicted.shape[1], -1, 3 )
            predicted_euler = qeuler(predicted_quat, order = 'zyx', epsilon = 1e-6)
            distance = torch.remainder( predicted_euler - expected_euler + np.pi, 2*np.pi ) - np.pi

        elif self.loss_mode == 'quaternions':
            # Loss function with quaternions cosine distance
            expected_quat = expected.view( predicted.shape[0], predicted.shape[1], -1 , 4 )
            distance = 1 - torch.sum( torch.mul( predicted_quat, expected_quat ), dim = -1 )

        return torch.mean( torch.abs( distance ) )