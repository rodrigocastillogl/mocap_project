import torch
import os
import errno
from models.pose_network_ensemble import PoseNetworkEnsemble
from dataset_h36m import dataset, subjects_train, subjects_valid, subjects_test, short_term_weights_path
from test_ensemble import run_evaluation

torch.manual_seed(1234)

if __name__ == '__main__':
    
    # ----------- Create results directory -----------
    results_path = 'ensemble_results'
    try:
        os.makedirs(results_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    # ------------------------------------------------

    selected_list = [ None,
                      [ 0, 1, 2, 3, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18, 24, 25, 26],
                      [ 0, 1, 2, 6, 7, 11, 12, 13, 16, 17, 24, 25]
                    ]
    
    train_params = { 'lr' : 0.001            ,
                     'lr_decay' : 0.999      ,
                     'tf_ratio' : 1          ,
                     'tf_decay' : 0.995      ,
                     'batch_size' : 60       ,
                     'batch_size_valid' : 30 ,
                     'gd_clip' : 0.1         ,
                     'quaternion_reg' : 0.01 ,
                     'n_epochs' : 3000       }

    weights_names = [ os.path.join(results_path, f) for f in [ 'weights_fullskeleton.bin',
                                                               f'weights_{len(selected_list[1])}joints.bin',
                                                               f'weights_{len(selected_list[2])}joints.bin'
                                                             ]
                    ]
    
    training_files_names = [os.path.join(results_path, f) for f in [ 'training_fullskeleton.csv',
                                                                     f'training_{len(selected_list[1])}joints.csv',
                                                                     f'training_{len(selected_list[2])}joints.csv'
                                                                    ]
                           ]
    
    test_files_names = [ os.path.join(results_path, f) for f in [ 'test_fullskeleton.csv',
                                                                  f'test_{len(selected_list[1])}joints.csv',
                                                                  f'test_{len(selected_list[2])}joints.csv'
                                                                ]
                        ]
    
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
                     train_params    ,
                     file_path = training_files_names[i])
        # ------------------------------------------------
        
        # Save weights
        model.save_weights(weights_names[i])
        
        # --------------- Model evaluation ---------------
        model.eval()
        with torch.no_grad():
            run_evaluation(model,file_path = test_files_names[i], directory_path = 'ensemble_results/joints_errors' )
        # ------------------------------------------------