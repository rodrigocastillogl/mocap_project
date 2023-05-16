# ---------------------------------------------- #
# Copyright (c) 2018-present, Facebook, Inc.
# https://github.com/facebookresearch/QuaterNet
# ---------------------------------------------- #

import torch
import numpy as np
from common.mocap_dataset import MocapDataset
from common.quaternion import qeuler_np
from short_term.pose_network_short_term import PoseNetworkShortTerm
from short_term.dataset_h36m import dataset, subjects_test, short_term_weights_path

torch.manual_seed(1234)

def find_inidices_srnn( data, action, subject, num_seeds, prefix_length, target_length ):
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
    indices = find_inidices_srnn(seq_map, action, subject, num_seeds, prefix_length, target_length)

    # since actions were downsampled by a factor of 2, keeping all strides,
    # there are two sequences for every action 
    seeds = [ ( action, (i%2)+1, indices[i] ) for i in range(num_seeds) ]

    out = []
    for i in range(num_seeds):
        _, subsequence, idx = seeds[i]
        idx = idx + 50
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
    for d in test_data:
        source = np.concatenate( (d[0], d[1][:1]), axis = 0).reshape(-1, 32*4)
        target = d[2].reshape(-1, 32*4)

        if model is None:
            target_predicted = np.tile( source[-1], target.shape[0] ).reshape(-1, 32*4)
        else:
            target_predicted = model.predict(
                np.expand_dims(source, 0), target_length = np.max(frame_targets) + 1
            ).reshape(-1, 32*4)
        
        target = qeuler_np( target[:target_predicted.shape[0]].reshape(-1,4), 'zyx' ).reshape(-1, 96)
        print(target[4][10:13])
        target_predicted = qeuler_np( target_predicted.reshape(-1,4), 'zyx').reshape(-1,96)
        e = np.sqrt( np.sum( (target_predicted[:,3:] - target[:,3:] )**2, axis = 1 ) )
        errors.append(e)
    errors = np.mean( np.array(errors), axis = 0 )
    
    return errors


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


def run_evaluation( model = None ):
    """
    Evaluate model and display results.
    Input
    -----
        * model : model to evaluate
    Output
    ------
        None
    """

    actions = [ 'walking', 'eating', 'smoking', 'discussion', 'directions', 'greeting',
               'phoning', 'posing', 'purchases', 'sitting', 'sittingdown', 'takingphoto',
               'waiting', 'walkingdog', 'walkingtogether']
    
    for subject_test in subjects_test:
        print( 'Testing on subject ' + subject_test )
        print()
        for idx, action in enumerate( actions ):
            test_data = get_test_data( dataset, action, int(subject_test[1:]) )
            errors = evaluate(model, test_data)
            all_errors[idx] = errors
            print_results(action, errors)
        print_results('average', all_errors.mean(axis = 0) )


# RUN EVALUATION
if __name__ == '__main__':

    model = PoseNetworkShortTerm(prefix_length = 50)

    if torch.cuda.is_available():
        model.cuda()

    model.load_weights(short_term_weights_path)
    model.eval()
    run_evaluation(model)