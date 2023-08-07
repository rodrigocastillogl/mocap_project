# ---------------------------------------------- #
# Copyright (c) 2018-present, Facebook, Inc.
# https://github.com/facebookresearch/QuaterNet
# ---------------------------------------------- #

from common.pose_network import PoseNetwork
from common.quaternion import qeuler, qeuler_np, qfix, euler_to_quaternion
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch
from tqdm import tqdm
from time import time

class PoseNetworkEnsemble(PoseNetwork):
    """
    Ensemble QuaterNet model for skeleton position prediction.
    Attributes
    ----------
        * translations_size : (num_outputs) extra inputs/outputs, in addition to joint rotations.
        * controls_size     : (num_controls) extra input features.
        * model_velocities  : flag to add a queternion multiplication block on the
                              RNN output to force the network to model velocities
                              instead of absolute rotations.
        * model         : Quaternet model.
        * use_cuda      : flag to use CUDA.
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
            * dataset    : dataset object load data.
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
            buffer_out = np.zeros( (batch_size, target_length, self.num_joints*3),
                                    dtype = 'float32' )
        elif self.loss_mode == 'quaternions':
            # Loss function with quaternions cosine distance
            buffer_out = np.zeros( (batch_size, target_length, self.num_joints*4),
                                    dtype = 'float32' )

        sequences = np.random.permutation(sequences)

        batch_idx = 0
        for i, (subject, action) in enumerate(sequences):

            # pick a random chunk from each sequence
            start_idx = np.random.randint( 0, dataset[subject][action]['rotations'].shape[0] - self.prefix_length - target_length + 1 )
            mid_idx = start_idx + self.prefix_length
            end_idx = start_idx + self.prefix_length + target_length

            buffer_in[batch_idx] = dataset[subject][action]['rotations'][start_idx:end_idx, self.selected_joints].reshape(
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
            * expected : ground truth sequence; Euler angles.
        Output
        ------
            * Loss
        """

        if self.loss_mode == 'euler':
            # Original loss function (Euler angles L1 distance)
            predicted_quat = predicted.view( predicted.shape[0], predicted.shape[1], -1 , 4 )
            expected_euler = expected.view( predicted.shape[0], predicted.shape[1], -1, 3 )
            predicted_euler = qeuler(predicted_quat, order = 'zyx', epsilon = 1e-6)
            distance = torch.remainder( predicted_euler - expected_euler + np.pi, 2*np.pi ) - np.pi

        elif self.loss_mode == 'quaternions':
            # Loss function with quaternions cosine distance
            predicted_quat = predicted.view( predicted.shape[0], predicted.shape[1], -1 , 4 )
            expected_quat = expected.view( predicted.shape[0], predicted.shape[1], -1 , 4 )
            distance = 1 - torch.sum( torch.mul( predicted_quat, expected_quat ), dim = -1 )

        return torch.mean( torch.abs( distance ) )


    def train( self, dataset, target_length, sequences_train, sequences_valid, train_params, file_path = 'training.csv'):
        """
        Train the model, updating parameters.
        Input
        -----
            * dataset : dataset object.
            * target_length   : number of frames to forecast.
            * sequences_train : sequences used to train.
            * sequences_valid : sequences used for validation.
            
            * train_parameter: training hyperparameters...
                - lr : starting learning rate.
                - lr_decay : learning rate decay factor.
                - tf_ratio : starting teacher forcing ratio.
                - tf_decay : teacher forcing decay factor.
                - batch_size : training batch size.
                - batch_size_valid : validation batch size.
                - gd_clip : gradient clip factor.
                - quaternion_reg : regularization parameter to force quaterions to be unitary.
                - n_epochs : number of epochs to train
            
            * file_path: file path to save training results.

        Output
        ------
            * losses : training loss
            * valid_losses   : validation loss
            * gradinet_norms : gradinet norm
        """

        np.random.seed(1234)

        # set model in training mode
        self.model.train()


        # --------- Training hyperparameters -----------
        lr = train_params['lr']
        lr_decay = train_params['lr_decay']
        tf_ratio = train_params['tf_ratio']
        tf_decay = train_params['tf_decay']
        
        batch_size = train_params['batch_size']
        batch_size_valid = train_params['batch_size_valid']

        gd_clip = train_params['gd_clip']
        quaternion_reg = train_params['quaternion_reg']

        n_epochs = train_params['n_epochs']
        # -----------------------------------------------

        optimizer = optim.Adam( self.model.parameters(), lr = lr )

        # ---------------- Validation set ---------------
        if len(sequences_valid) > 0 :
            batch_in_valid, batch_out_valid = next(
                self._prepare_next_batch_impl(batch_size_valid, dataset, target_length, sequences_valid)
            )
            inputs_valid = torch.from_numpy(batch_in_valid)
            outputs_valid = torch.from_numpy(batch_out_valid)

            if self.use_cuda:
                inputs_valid = inputs_valid.cuda()
                outputs_valid = outputs_valid.cuda()
        # -----------------------------------------------

        # Start ---------------- Training ---------------

        losses = []             # training loss per epoch
        valid_losses = []       # validation loss per epoch
        gradient_norms = []     # gradient norm per epoch

        # Start training file
        training_file = open(file_path, 'w')
        if len(sequences_valid) > 0:
            training_file.write('epoch,loss,valid loss,learning rate,teacher forcing ratio\n')
        else:
            training_file.write('epoch,loss,learning rate,teacher forcing ratio\n')

        print("Training for %d epochs" % (n_epochs) )

        try:
            for epoch in tqdm(range(n_epochs)):
                
                batch_loss = 0.0
                N = 0

                for batch_in, batch_out in self._prepare_next_batch_impl(
                    batch_size,  dataset, target_length, sequences_train):


                    # --------------- Training batch ----------------
                    inputs = torch.from_numpy(batch_in)
                    outputs = torch.from_numpy(batch_out)
                    if self.use_cuda:
                        inputs = inputs.cuda()
                        outputs = outputs.cuda()
                    # -----------------------------------------------
                        
                    optimizer.zero_grad()
                    terms = []
                    predictions = []

                    # ------------- Forward propagation -------------
                        
                        # x = inputs[:, :self.prefix_length]
                        # h = None
                        # return_prenorm = True
                        # return_all = False (default)

                    predicted , hidden, term = self.model(
                        inputs[:, :self.prefix_length], None, True
                    )

                    terms.append(term)
                    predictions.append(predicted)

                    tf_mask = np.random.uniform(size = target_length-1) < tf_ratio
                    i = 0
                    while i < target_length-1:

                        contiguous_frames = 1

                        # ---------- Teacher forcings strategy ----------
                        if tf_mask[i]:
                            while (i + contiguous_frames) < (target_length - 1) and tf_mask[ i + contiguous_frames ]:
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
                            predicted, hidden, term = self.model(predicted, hidden, True)
                        # -----------------------------------------------

                        terms.append(term)
                        predictions.append(predicted)
                            
                        if contiguous_frames > 1:
                            predicted = predicted[:,-1:]
                            
                        i += contiguous_frames
                        
                    terms = torch.cat(terms, dim = 1)
                    terms =  terms.view(terms. shape[0], terms.shape[1], -1 , 4)

                    # -----------------------------------------------

                    # -------- Compute loss & Backpropagation -------
                    penalty_loss = quaternion_reg * torch.mean(
                        ( torch.sum(terms**2, dim = 3) - 1 )**2
                    )
                    predictions = torch.cat(predictions, dim = 1)
                    loss = self._loss_impl(predictions, outputs)
                    loss_total = penalty_loss + loss

                    loss_total.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), gd_clip)
                    optimizer.step()
                    # -----------------------------------------------

                    batch_loss += loss.item() * inputs.shape[0]
                    N += inputs.shape[0]

                batch_loss = batch_loss/N
                losses.append(batch_loss)

                # ----------------- Validation ------------------
                if len(sequences_valid) > 0:
                    with torch.no_grad():
                        predictions = []
                        predicted, hidden = self.model(
                            inputs_valid[:, :self.prefix_length]
                        )
                        predictions.append(predicted)
                        for i in range(target_length - 1):
                            # Feed own output
                            if self.controls_size > 0:
                                predicted = torch.cat(
                                    ( predicted, inputs_valid[:, (self.prefix_length+i):(self.prefix_length+i+1), - self.controls_size:]), dim =  2 
                                )
                            
                            predicted, hidden = self.model(predicted, hidden)
                            predictions.append(predicted)
                            
                        predictions = torch.cat( predictions, dim = 1 )
                        loss = self._loss_impl(predictions, outputs_valid)

                        valid_loss = loss.item()
                        valid_losses.append(valid_loss)
                    training_file.write( '%d,%.5e,%.5e,%.5e,%.5e\n' % (epoch + 1, batch_loss, valid_loss, lr, tf_ratio) )
                else:
                    training_file.write( '%d,%.5e,%.5e,%.5e\n' % (epoch + 1, batch_loss, lr, tf_ratio) )
                # -----------------------------------------------

                # -------------- Update aparameters -------------
                tf_ratio *= tf_decay
                lr *= lr_decay
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= lr_decay
                # -----------------------------------------------

        except KeyboardInterrupt:
            print('Training aborted.\n'), training_file.close()
            
        # End ----------------- Training ----------------

        training_file.close()
        return losses, valid_losses, gradient_norms
    

    def predict(self, prefix, target_length):
        """
        Predict a skeleton sequence , given a prefix.
        Input
        -----
            * prefix : input; quaternions.
            * target_length : target sequence length.
        Output
        ------
            * predicted sequence; quaternions.
        """

        assert target_length > 0

        
        with torch.no_grad():
            # quaternions --> euler angle --> quaternions  
            prefix = prefix.reshape( prefix.shape[1], -1 , 4 )
            prefix = qeuler_np( prefix, 'zyx' )
            prefix = qfix( euler_to_quaternion(prefix, 'zyx') )
            inputs = torch.from_numpy(
                prefix.reshape(1, prefix.shape[0], -1).astype('float32')
            )

            # input to cuda
            if self.use_cuda:
                inputs = inputs.cuda()

            # evaluate model
            predicted, hidden = self.model(inputs)
            frames = [predicted]
            for i in range(1, target_length):
                predicted, hidden = self.model(predicted, hidden)
                frames.append(predicted)
            
            # result
            result = torch.cat( frames, dim = 1 )
            
            return result.view( result.shape[0], result.shape[1], -1, 4).cpu().numpy()