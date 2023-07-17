# ---------------------------------------------- #
# Copyright (c) 2018-present, Facebook, Inc.
# https://github.com/facebookresearch/QuaterNet
# ---------------------------------------------- #

import torch
import os
import errno
import numpy as np
from common.mocap_dataset import MocapDataset
from common.quaternion import qeuler_np
from models.pose_network_ensemble import PoseNetworkEnsemble
from dataset_h36m import dataset, subjects_test, short_term_weights_path
from tqdm import tqdm

torch.manual_seed(1234)

def find_indices_srnn( data, action, subject, num_seeds, prefix_length, target_length ):
    """
    Given a data map generated with build_sequence_map_srnn(), return the starting indices
    to get input prefixes to test the model.
    From: https://github.com/una-dinosauria/human-motion-prediction
    Input
    -----
        * data   : dataset generated with build_sequence_map_srnn()
        * action : action name.
        * subject   : subject number.
        * num_seeds : sequences generated.
        * prefix_length : input sequence length.
        * target_length : output sequence length.
    Output
    ------
        * idx : starting indices for test prefixes.
    """

    rnd = np.random.RandomState(1234567890)

    # A subject performs the same action twice in the Human3.6M dataset.
    # since actions were downsampled by a factor of 2, keeping all strides,
    # there are two sequences for every action
    T1 = data[(subject, action, 1)].shape[0]
    T2 = data[(subject, action, 2)].shape[0]

    idx = []
    for i in range(num_seeds//2):
        idx.append( rnd.randint(16, T1 - prefix_length - target_length) )
        idx.append( rnd.randint(16, T2 - prefix_length - target_length) )

    return idx


def build_sequence_map_srnn(data):
    """
    Return rotations sequences to test.
    From: https://github.com/una-dinosauria/human-motion-prediction
    Input
    -----
        * data : dataset object.
        (dictionary) data -> subject -> action -> rotations, trajectory, etc.
    Output
    ------
        * out : rotations sequence in a dictionary.
        (subject number, action name, action number) -> rotations
    """

    out = {}
    for subject in data.subjects():
        for action, seq in data[subject].items():
            # do nothing if: it is not a repeated action (or downsampled) or it is mirrored.
            if not '_d0' in action or '_m' in action:
                continue
            # action in 'data' only if it is a repeated action (or downsampled).
            act, sub, _ = action.split('_')
            out[ ( int(subject[1:]), act, int(sub) ) ] = seq['rotations']
    
    return out


def get_test_data(data, action, subject):
    """
    Generate test sequences chunks.
    From: https://github.com/una-dinosauria/human-motion-prediction
    Input
    -----
        * data    : dataset object.
        * action  : action name.
        * subject : subject number.
    Output
    ------
        * out : test chunks.
    """

    seq_map = build_sequence_map_srnn(data)
    num_seeds = 8
    prefix_length = 50
    target_length = 100
    indices = find_indices_srnn(seq_map, action, subject, num_seeds, prefix_length, target_length)

    # since actions were downsampled by a factor of 2, keeping all strides,
    # there are two sequences for every action 
    seeds = [ ( action, (i%2)+1, indices[i] ) for i in range(num_seeds) ]

    out = []
    for i in range(num_seeds):
        _, subsequence, idx = seeds[i]
        idx = idx + prefix_length
        chunk = seq_map[ (subject, action, subsequence) ]
        chunk = chunk[ (idx-prefix_length):(idx+target_length), : ]
        out.append( (
            chunk[0:(prefix_length-1), :],                                 # Input
            chunk[(prefix_length-1):(prefix_length+target_length-1), :],   # Target
            chunk[prefix_length:, :]                                       # ??
        ) )

    return out


def evaluate(model, test_data):
    """
    Run evaluation of the model.
    Input
    -----
        * model     : pre-trained ShortTermPoseNetwork model.
        * test_data : test data generated with get_test_data().
    Output
    ------
        * errors : Root Mean Squared Error (Euler angles).
    """

    errors = []
    errors_joint = np.zeros( ( len(test_data), np.max(frame_targets) + 1, model.num_joints ) )
    for i, d in enumerate(test_data):
        source = np.concatenate( (d[0][:,model.selected_joints], d[1][:1,model.selected_joints]), 
                                 axis = 0).reshape( -1, model.num_joints*4 )
        target = d[2][:,model.selected_joints].reshape(-1, model.num_joints*4)

        if model is None:
            target_predicted = np.tile( source[-1], target.shape[0] ).reshape(-1, model.num_joints*4)
        else:
            target_predicted = model.predict(
                np.expand_dims(source, 0), target_length = np.max(frame_targets) + 1 
            ).reshape(-1, model.num_joints*4)
            
        target = qeuler_np( target[:target_predicted.shape[0]].reshape(-1,4), 'zyx' ).reshape(-1, model.num_joints, 3)
        target_predicted = qeuler_np( target_predicted.reshape(-1,4), 'zyx').reshape(-1, model.num_joints, 3)
        
        e_joint = np.sqrt( np.sum( (target_predicted - target)**2, axis = 2 ) )
        e = np.sqrt( np.sum( (target_predicted.reshape(-1, model.num_joints*3)[:,3:] - target.reshape(-1, model.num_joints*3)[:,3:] )**2, axis = 1 ) )

        errors.append(e)
        errors_joint[i,:,:] = e_joint
    errors = np.mean( np.array(errors), axis = 0 )
    errors_joint = np.mean( errors_joint, axis = 0 )
    
    return errors, errors_joint


frame_targets = [1, 3, 7, 9, 14, 19, 24, 49, 74, 99] # 80, 160, 320, and 400 ms (at 25 Hz)
all_errors = np.zeros((15, 100))

def print_results(action, errors):
    """
    Display errors during evaluation.
    Input
    -----
        * action : action evaluated.
        * errors : errors array.
    Output
    ------
        None
    """

    print(action)
    for f, e in zip(frame_targets, errors[frame_targets] ):
        print( (f+1)/25*1000 , 'ms: ', e)
    print()


def run_evaluation( model = None, file_path = 'test.csv', directory_path = ''):
    """
    Evaluate model and display results.
    Input
    -----
        * model : model to evaluate
    Output
    ------
        None
    """

    # ----------- Create results directory -----------
    if directory_path != '':
        try:
            os.makedirs(directory_path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    # ------------------------------------------------

    actions = [ 'walking', 'eating', 'smoking', 'discussion', 'directions', 'greeting',
               'phoning', 'posing', 'purchases', 'sitting', 'sittingdown', 'takingphoto',
               'waiting', 'walkingdog', 'walkingtogether']

    # Open test file
    test_file = open(file_path, 'w')
    test_file.write('subject,action,time(ms),error\n')

    print('Testing on subjects: ', subjects_test)
    for subject_test in tqdm(subjects_test):
        for idx, action in enumerate( actions ):
            test_data = get_test_data( dataset, action, int(subject_test[1:]) )
            errors, errors_joint = evaluate(model, test_data)
            all_errors[idx] = errors
            for f, e in zip(frame_targets, errors[frame_targets] ):
                test_file.write( '%s,%s,%d,%.5e\n' % ( subject_test, action, (f+1)/25*1000, e) )
            
            # ---------- Write errors per joint ---------- #
            file = open( os.path.join(directory_path, 'errors_{errors_joint.shape[1]:d}joints_{action}.csv'), 'w' )
            file.write('frame,')
            for i in range( len(model.selected_joints)-1 ):
                file.write(f'joint{model.selected_joints[i]:0>2d},')
            file.write(f'joint{model.selected_joints[-1]:0>2d}\n')
            for frame in range( errors_joint.shape[0] ):
                file.write( '%d,' % frame )
                for joint in range( errors_joint.shape[1] - 1 ):
                    file.write( '%.5e,' % errors_joint[frame,joint] )
                file.write( '%.5e\n' % errors_joint[frame,-1] )
            file.close()
            # -------------------------------------------- #

        for f, e in zip(frame_targets, all_errors.mean(axis = 0)[frame_targets] ):
            test_file.write( '%s,average,%d,%.5e\n' % ( subject_test, (f+1)/25*1000, e) )

    test_file.close()


# RUN EVALUATION
if __name__ == '__main__':

    model = PoseNetworkEnsemble(prefix_length = 50)

    if torch.cuda.is_available():
        model.cuda()

    model.load_weights(short_term_weights_path)
    model.eval()
    run_evaluation(model)