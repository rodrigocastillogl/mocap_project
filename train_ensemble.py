import torch
import os
import errno
from models.pose_network_ensemble import PoseNetworkEnsemble
from dataset_h36m import dataset, subjects_train, subjects_valid, subjects_test, short_term_weights_path
from test_ensemble import run_evaluation

#torch.manual_seed(1234)

if __name__ == '__main__':
    
    # ----------- Create results directory -----------
    results_path = 'ensemble_results'
    try:
        os.makedirs(results_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    # ------------------------------------------------

    selected_list = [ [ 0, 1, 2, 6, 7, 11, 12, 13, 16, 17, 24, 25],
                      [ 0, 1, 2, 3, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18, 24, 25, 26], 
                      None ]

    weights_names = [ os.path.join(results_path, f) for f in [ f'weights_{len(selected_list[0])}joints.bin',
                                                               f'weights_{len(selected_list[1])}joints.bin',
                                                               'weights_fullskeleton.bin'
                                                             ] ]
    
    training_files_names = [os.path.join(results_path, f) for f in [ f'training_{len(selected_list[0])}joints.csv',
                                                                     f'training_{len(selected_list[1])}joints.csv',
                                                                     'training_fullskeleton.csv'
                                                                    ] ]
    
    test_files_names = [ os.path.join(results_path, f) for f in [ f'test_{len(selected_list[0])}joints.csv',
                                                                  f'test_{len(selected_list[1])}joints.csv',
                                                                  'test_fullskeleton.csv'
                                                                ] ]
    
    for i in range( len(selected_list) ):
        
        # ----------------- Define model -----------------
        model = PoseNetworkEnsemble( prefix_length = 50 , selected_joints = selected_list[i] )
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
        
        # Compute Euler angles in dataset (in case of using the loss function based on euler angles)
        dataset.compute_euler_angles( order = 'zyx' )
        
        # ----------------- Train model ------------------
        model.train( dataset         ,
                     target_length   ,
                     sequences_train ,
                     sequences_valid ,
                     batch_size = 60 ,
                     n_epochs = 3000 ,
                     file_path = training_files_names[i])
        # ------------------------------------------------
        
        # Save weights
        model.save_weights(weights_names[i])
        
        # --------------- Model evaluation ---------------
        model.eval()
        with torch.no_grad():
            run_evaluation(model,file_path = test_files_names[i])
        # ------------------------------------------------