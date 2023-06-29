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
from tqdm import tqdm

class PoseNetwork:
    """
    Recurrent model to predict skeleton movement.
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

    Methods
    -------
        * __init__()
        * cuda()
        * eval()
        * _prepare_next_batch_impl()
        * _loss_impl()
        * train()
        * save_weights()
        * load_weights()
    """
    
    def __init__( self, prefix_length, num_joints, num_outputs, 
                  num_controls, model_velocities, selected_joints = None):
        """
        Initializer
        Input
        -----
            * prefix_length : Number of frames in the input. 
            * num_joints    : number of skeleton joints
            * num_outputs   : extra inputs/outputs, in addition to joint rotations 
            * num_controls  : extra input features
            * model_velocities : flag to add a queternion multiplication block on the
                                 RNN output to force the network to model velocities
                                 instead of absolute rotations.
        Output
        ------
            * None
        """

       # ------------ PoseNetwork attributes ------------
        self.translations_size = num_outputs
        self.controls_size = num_controls
        self.model_velocities = model_velocities
        self.use_cuda = False
        self.prefix_length = prefix_length
        self.num_joints = num_joints
        
        if selected_joints:
            self.num_joints = len(selected_joints)
            self.selected_joints = selected_joints
        else:
            self.num_joints = num_joints
            self.selected_joints = list( range(num_joints) )
        # -----------------------------------------------

        # QuaterNet model
        self.model = QuaterNet( self.num_joints, num_outputs, num_controls, model_velocities )

        # print model
        self.print_model()

    def print_model(self):
        """
        Display model information
        Input
        -----
            None
        Output
        ------
            None
        """
        print( '-'*9 + ' MODEL '+ '-'*9 )
        dec_params = 0
        for parameter in self.model.parameters():
            dec_params += parameter.numel()
        print('parameters:', dec_params)
        print('joints:', self.num_joints)
        print( '-'*25 + '\n' )
    
    def cuda(self):
        """
        Enable CUDA.
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
        Sets model in Evaluation mode.
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


    def train( self, dataset, target_length, sequences_train, sequences_valid,
               batch_size, n_epochs = 3000, rot_reg = 0.01, file_path = 'training.txt' ):
        """
        Train the model, updating parameters.
        Input
        -----
            * dataset : dataset object.
            * target_length   : number of frames to forecast.
            * sequences_train : sequences used to train.
            * sequences_valid : sequences used for validation
            * batch_size : batch size during training.
            * n_epoch : number of epochs during training.
            * rot_reg : regularization parameter to force quaterions to be unitary.
        Output
        ------
            * losses : training loss
            * valid_losses   : validation loss
            * gradinet_norms : gradinet norm
        """

        np.random.seed(1234)

        # set model in training mode
        self.model.train()

        # -------------- Hyperparameters ----------------
        lr = 0.001                 # learning rate
        lr_decay = 0.999           # learning rate decay factor
        batch_size_valid = 30      # batch size during validation step
        teacher_forcing_ratio = 1  # starts by forcing the ground truth 
        tf_decay = 0.995           # teacher forcing decay factor
        gradient_clip = 0.1        # gradient_clipping factor
        optimizer = optim.Adam( self.model.parameters(), lr = lr )
        # -----------------------------------------------

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
            training_file.write('epoch, loss, valid loss, learning rate, teacher forcing ratio\n')
        else:
            training_file.write('epoch, loss, learning rate, teacher forcing ratio\n')

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

                    tf_mask = np.random.uniform(size = target_length-1) < teacher_forcing_ratio
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
                    penalty_loss = rot_reg * torch.mean(
                        ( torch.sum(terms**2, dim = 3) - 1 )**2
                    )
                    predictions = torch.cat(predictions, dim = 1)
                    loss = self._loss_impl(predictions, outputs)
                    loss_total = penalty_loss + loss

                    loss_total.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)
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
                    training_file.write( '%d, %.5e, %.5e, %.5e, %.5e\n' % (epoch + 1, batch_loss, valid_loss, lr, teacher_forcing_ratio) )
                else:
                    training_file.write( '%d, %.5e, %.5e, %.5e\n' % (epoch + 1, batch_loss, lr, teacher_forcing_ratio) )
                # -----------------------------------------------

                # -------------- Update aparameters -------------
                teacher_forcing_ratio *= tf_decay
                lr *= lr_decay
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= lr_decay
                # -----------------------------------------------

        except KeyboardInterrupt:
            print('Training aborted.\n'), training_file.close()
            
        # End ----------------- Training ----------------

        training_file.close()
        return losses, valid_losses, gradient_norms
    

    def save_weights(self, model_file):
        """
        Save model weights in a dictionary.
        Input
        -----
            * model_file : file name.
        Output
        ------
            None 
        """
        print('Saving weights to', model_file , '\n')
        torch.save( self.model.state_dict(), model_file )

    
    def load_weights(self, model_file):        
        """
        Load model weights from a dictionary
        Input
        -----
            * model_file : file name.
        Output
        ------
            None
        """
        print('Loadings weights from', model_file, '\n')
        self.model.load_state_dict(
            torch.load( model_file, map_location = lambda storage, loc:storage)
        )