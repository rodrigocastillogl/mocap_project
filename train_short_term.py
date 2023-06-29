# ---------------------------------------------- #
# Copyright (c) 2018-present, Facebook, Inc.
# https://github.com/facebookresearch/QuaterNet
# ---------------------------------------------- #

import torch
from short_term.pose_network_short_term import PoseNetworkShortTerm
from short_term.dataset_h36m import dataset, subjects_train, subjects_valid, subjects_test, short_term_weights_path
from test_short_term import run_evaluation

torch.manual_seed(1234)

if __name__ == '__main__':
    
    # ----------------- Define model -----------------
    model = PoseNetworkShortTerm( prefix_length = 50 )
    if torch.cuda.is_available():
        model.cuda()
    # ------------------------------------------------
    
    # -------------- Training sequences --------------
    # sequences_train: list of (subjet, action) tuples
    #                  to train the model.
    sequences_train = []
    for subject in subjects_train:
        for action in dataset[subject].keys():
            sequences_train.append( (subject, action) )
    # ------------------------------------------------

    
    # ------------- Validation sequences -------------
    # sequences_train: list of (subjet, action) tuples 
    #                  for validation.
    sequences_valid = []
    for subject in subjects_valid:
        for action in dataset[subject].keys():
            sequences_valid.append( (subject, action) )
    # ------------------------------------------------

    # Display message
    print( f'Training on {len(sequences_train)} sequences, validation on {len(sequences_valid)} sequences' )

    # Define prediction target length
    target_length = 10

    # Compute Euler angles in dataset (in case of using the 
    # loss function based on euler angles)
    dataset.compute_euler_angles( order = 'zyx' )
    

    # ----------------- Train model ------------------
    model.train( dataset         ,
                 target_length   ,
                 sequences_train ,
                 sequences_valid ,
                 batch_size = 60 ,
                 n_epochs = 3000 ,
                 file_path = 'training_32joints.txt')
    # ------------------------------------------------

    # Save weights
    model.save_weights(short_term_weights_path)

    # --------------- Model evaluation ---------------
    model.eval()
    with torch.no_grad():
        run_evaluation(model,file_path = 'test_32joints.txt')
    # ------------------------------------------------