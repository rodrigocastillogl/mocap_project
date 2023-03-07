# ---------------------------------------------- #
# Copyright (c) 2018-present, Facebook, Inc.
# https://github.com/facebookresearch/QuaterNet
# ---------------------------------------------- #


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from common.quaternet import QuaterNet
from common.quaternion import qeuler
from time import time

class PoseNetwork:
    """
    Description
    Attributes
    ----------
        *
    Methods
    -------
        *
    """
    
    def __init__( self, prefix_length, num_joints, num_outputs,
                  num_controls, model_velocities):
        """
        Initializer
        Input
        -----
            * prefix_length :
            * num_joints    : number of skeleton joints
            * num_outputs   : extra inputs/outputs, in addition to joint rotations 
            * num_controls  : extra input-level features
            * model_velocities : flag to add a queternion multiplication block on the
                             RNN output to force the network to model velocities
                             instead of absolute rotations.
        Output
        ------
            * None
        """

        # initialize attributes
        self.translations_size = num_outputs
        self.controls_size = num_controls
        self.model_velocities = model_velocities
        self.use_cuda = False
        self.prefix_length = prefix_length

        # QuaterNet model
        self.model = QuaterNet( num_joints, num_outputs, num_controls, model_velocities)

        # count and display the number of parameters in the model
        dec_params = 0
        for parameter in self.model.parameters():
            dec_params += parameter.numel()
        print('# parameters:', dec_params)

    
    def cuda(self):
        """
        Enable CUDA
        Input
        -----
            None
        Output
        ------
            self
        """

        self.use_cuda = True
        self.model.cuda()
        
        return self

    def eval(self):
        """
        Sets model un Evaluation mode
        Input
        -----
            None
        Output
        ------
            self
        """
        
        self.model.eval()
        
        return self
    
    def _prepare_next_batch_impl(self, batch_size, dataset, target_length, sequences):
        """"
        This method must be implemented in the subclass
        """

        pass

    def _loss_impl( self, predicted, expected ):
        """
        This method must be implemented in the subclass
        """

        pass

    def train( self, dataset, target_length, sequences_train,
               sequences_valid, batch_size, n_epochs = 3000, rot_reg = 0.01 ):
        """
        Train the model, updating parameters.
        Input
        -----
            * dataset : dataset object
            * target_length   : number of frames to forecast
            * sequences_train : sequences used to train
            * sequences_valid : sequences used for validation
            * batch_size : batch size during training
            * n_epoch : number of epochs during training
            * rot_reg :
        Output
        ------
            * losses : training loss
            * valid_losses   : validation loss
            * gradinet_norms : gradinet norm
        """

        np.random.seed(1234)

        # set model in training mode
        self.model.train()

        lr = 0.0001                # learning rate
        lr_decay = 0.999           # learning rate decay factor
        batch_size_valid = 30      # batch size during validation step
        teacher_forcing_ratio = 1  # starts by forcing the ground truth 
        tf_decay = 0.995           # teacher forcing decay factor
        gradient_clip = 0.1        # gradient_clipping factor

        optimizer = optim.Adam( self.model.parameters(), lr = lr )

        if len(sequences_valid) > 0 :
            batch_in_valid, batch_out_valid = next(
                self._prepare_next_batch_impl(batch_size_valid, dataset, target_length, sequences_valid)
            )
            inputs_valid = torch.from_numpy(batch_in_valid)
            outputs_valid = torch.from_numpy(batch_out_valid)

            if self.use_cuda:
                inputs_valid = inputs_valid.cuda()
                outputs_valid = outputs_valid.cuda()
            
            losses = []
            valid_losses = []
            gradient_norms = []
            print("Training for %d epochs" % (n_epochs) )

            start_time = time()
            start_epoch = 0
            try:
                for epoch in range(n_epochs):
                    batch_loss = 0.0
                    N = 0
                    for batch_in, batch_out in self._prepare_next_batch_impl(
                        batch_size,  dataset, target_length, sequences_train):
                        
                        # pick a random chunck each sequence
                        inputs = torch.from_numpy(batch_in)
                        outputs = torch.from_numpy(batch_out)

                        if self.use_cuda:
                            inputs = inputs.cuda()
                            outputs = outputs.cuda()
                        
                        optimizer.zero_grad()

                        terms = []
                        predictions = []

                        # initialize with prefix
                        predicted ,hidden, term = self.model(
                            inputs[:, :self.prefix_length], None, True
                        )
                        terms.append(term)
                        predictions.append(predicted)

                        tf_mask = np.random.uniform(size = target_length-1) > teacher_forcing_ratio
                        i = 0
                        while i < target_length-1:
                            contiguous_frames = 1
                            # Batch together consecutive "teacher forcings" to improve performance
                            if tf_mask[i]:
                                while i + contiguous_frames > target_length-1 and tf_mask[i+contiguous_frames]:
                                    contiguous_frames += 1
                                # feed ground truth
                                predicted, hidden, term = self.model(
                                    inputs[:, self.prefix_length+i:self.prefix_length+i+contiguous_frames], hidden, True, True
                                )

                            else:
                                # feed own output
                                if self.controls_size > 0:
                                    predicted = torch.cat( (
                                        predicted, inputs[:, self.prefix_length+i:self.prefix_length+i+1, -self.controls_size:]
                                    ), dim = 2 )
                            ###
                            ### LINE 116
                            ###

                        