import os
import errno
import zipfile
import numpy as np
import csv
import sys
import re
from urllib.request import urlretrieve
from glob import glob
from shutil import rmtree

from common.quaternion import expmap_to_quaternion, qfix

if __name__ == '__main__':

    output_directory = 'datasets'
    output_filename  = 'dataset_h36m'
    h36m_dataset_url = 'http://www.cs.stanford.edu/people/ashesh/h3.6m.zip'

    # create output directory
    try:
        os.makedirs(output_directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    
    output_file_path = output_directory + '/' + output_filename

    if os.path.exists(output_file_path + '.npz'):
        # if datasets already exists, then show a message and do nothing
        print( 'The dataset already exists at ' + output_file_path + '.npz' )
    else:
        # if datasset does not exist

        # download the Human3.6M dataset zip file
        print('Downloading Human3.6M dataset (it may take a while)...')
        h36m_path = output_directory + '/h3.6m.zip'
        urlretrieve(h36m_dataset_url, h36m_path)

        # extract fata from zip file
        print('Extracting Human3.6M dataset ...')
        with zipfile.ZipFile(h36m_path, 'r') as archive:
            archive.extractall(output_directory)
        
        # remove zip file
        os.remove(h36m_path)

        def read_file(path):
            """
            Read MoCap CSV file in Exponential Map parameterization

            Input
            -----
                * path : csv file path
            Output
            ------
                * NumPy tensor with shape (L, J, 3)

            L ->  sequence length ,  J -> number of joints
            """

            # open and read csv file
            data = []
            with open(path) as csvfile:
                reader = csv.reader( csvfile, delimiter = ',' )
                for row in reader:
                    data.append(row)
            
            # numpy array
            data = np.array( data, dtype = 'float64' )
            
            return data.reshape( data.shape[0], -1, 3 )
        

        # convert data to quaternions
        
        out_pos = []
        out_rot = []
        out_subjects = []
        out_actions  = []

        print('Converting dataset ...')
        subjects = sorted( glob( output_directory + '/h3.6m/dataset/*' ) )
        for subject in subjects:
            # for every subjec in the directory, read actions
            actions = sorted( glob( subject + '/*' ) )
            result_ = {}

            # for every action in the subject
            for action_filename in actions:

                data = read_file(action_filename)
                data = data[:, 1:]                 # discard root translation

                # convert exponential map to quaternion
                quat = expmap_to_quaternion(-data)
                # quaternion continuity
                quat = qfix(quat)

                out_pos.append( np.zeros( (quat.shape[0], 3) ) ) # no trajectory
                out_rot.append( quat )

                tokens = re.split( '\/|\.', action_filename.replace('\\', '/') )
                subject_name = tokens[-3]
                out_subjects.append(subject_name)
                action_name = tokens[-2]
                out_actions.append(action_name)
        
        # save files
        print('Saving ...')
        np.savez_compressed(
            output_file_path        ,
            trajectories = out_pos  ,
            rotations = out_rot     ,
            subjects = out_subjects ,
            actions = out_actions
        )

        # delete extracted files
        rmtree( output_directory + 'h3.6m' )
        
        # Complete
        print('Done.')
