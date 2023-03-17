# ---------------------------------------------- #
# Copyright (c) 2018-present, Facebook, Inc.
# https://github.com/facebookresearch/QuaterNet
# ---------------------------------------------- #

from common.pose_network import PoseNetwork
from common.quaternion import qeuler, qeuler_np, qfix, euler_to_quaternion
import numpy as np
import torch

class PoseNetworkShortTerm(PoseNetwork):
    """
    Short term QuaterNet model for skeleton position forecasting.
    Attributes
    ----------
        * translations_size : (num_outputs) extra inputs/outputs, in addition to joint rotations.
        * controls_size     : (num_controls) extra input features.
        * model_velocities  : flag to add a queternion multiplication block on the
                              RNN output to force the network to model velocities
                              instead of absolute rotations.
        * model    : Quaternet model.
        * use_cuda : flag to use CUDA.
        * prefix_length : ...

    Methods
    -------
        * cuda() : set use_cuda = True, send model to CUDA.
        * eval() : set model to evaluation mode.
        * _prepare_next_batch_impl() : load and next batch.
        * _loss_impl() : loss fucntion to train.
        * train() : train model.
        * save_weights() : save model weights in a dictionary.
        * load_weights() : load model weights from a dictionary.
        * predict()
    """

    def __init__(self, prefix_length):
        super().__init__(prefix_length = prefix_length ,
                         num_joints = 32               ,
                         num_controls = 0              ,
                         num_outputs = 0               ,
                         model_velocities = True       )


    def _prepare_next_batch_impl(self, batch_size, dataset, target_length, sequences):
        """
        Load a prepare bacth: input sequence and target sequence.
        Input
        -----
            * batch_size : batch size.
            * dataset : dataset object load data.
            * target_length : target sequence length.
            * sequences : 
        Output
        ------
            * buffer_quat  : input sequence.
            * buffer_euler : target sequence.
        """
        
        super()._prepare_next_batch_impl(batch_size, dataset, target_length, sequences)

        buffer_quat = np.zeros(
            (batch_size, self.prefix_length + target_length, 32*4), dtype = 'float32'
        )
        buffer_euler = np.zeros(
            (batch_size, target_length, 32*3), dtype = 'float32'
        )

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
            buffer_quat = dataset[subject][action]['rotations'][start_idx:end_idx].reshape(
                self.prefix_length + target_length, -1
            )
            # target sequence as Euler angles
            buffer_euler = dataset[subject][action]['rotation_euler'][mid_idx:end_idx].reshape(
                target_length, -1
            )

            batch_idx += 1
            if batch_idx == batch_size or i == (len(sequences) - 1):
                yield buffer_quat[:batch_idx], buffer_euler[:batch_idx]
                batch_idx = 0


        def _loss_impl(self, predicted, expected):
            """
            Loss function.
            Input
            -----
                * predicted : predicted sequence; Quaternions.
                * expected : ground truth sequence; Euler angles.
            Output
            ------
                * Loss
            """

            super._loss_impl(predicted, expected)

            predicted_quat = predicted.view( predicted.shape[0], predicted.shape[1], -1 , 4 )
            expected_euler = expected.view(expected.shape[0], expected.shape[1], -1, 3 )
            predicted_euler = qeuler(predicted_quat, order = 'zyx', epsilon = 1e-6)

            # L1 loss angle distance with 2pi wrap-around
            angle_distance = torch.reminder( predicted_euler - expected_euler + np.pi, 2*np.pi ) - np.pi

            return torch.mean( torch.abs(angle_distance) )