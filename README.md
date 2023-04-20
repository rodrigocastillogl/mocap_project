# Recurrent Model for Human Motion Prediction

This is the implementation of the short-term model described in the paper:

> Dario Pavllo, David Grangier, and Michael Auli. QuaterNet: A Quaternion-based Recurrent Model for Human Motion. In British Machine Vision Conference (BMVC), 2018.

It is a copy of the repository made by Dario Pavllo and Michael Auli ([QuaterNet](https://github.com/facebookresearch/QuaterNet))
where I implement some changes to try new features in the data representation and the network architecture.

## Dependencies

* Python 3+ distribution
* PyTorch >= 0.4.0
* NumPy and SciPy

Optional:

* matplotlib: to render and display interactive animations.
* fmpeg: to export MP4 videos.
* imagemagick: to export GIFs.

A GPU is recommended if you want to train the models in reasonable time. If you plan on testing the pretrained models, the CPU is fine.

## Quickstart

To download and prepare dataset:

    python prepare_data_short_term.py

To train and test the model:

    python train_short_term.py

To test the model:

    python test_short_term.py

## Files description

* common/

    * **quaternion.py**
        
        Quaternion methods implementation for *Pytorch* tensors and *NumPy* arrays:
        qmul, qrot, qeuler, qfix, expmap_to_quaternion, euler_to_quaternion.

    * **skeleton.py**

    * **mocap_dataset.py**
        
        * `class MocapDataset`: Dataset object for Motion Capture
        
            You can load MoCap data using this class. It is stored in a dictironary with the following structure:

            `self._data` -> `subject` -> `action` -> `rotations`, `trajectory`, `positions`, `euler_angles`.

            It has a `Skeleton` object as attribute where offsets and hierarchy are defined.
            Then , data from all subjects in the dataset are fitted to this skeleton.

            `MocapSatset['subject']` returns data of `'subject'`, which is a dictionary of actions, etc.

    * **quaternet.py**

    * **pose_network.py**
    
    * **visualization.py**
    
* short_term/

    * **dataset_h36m.py**
    * **pose_network_short_term.py**

* **prepare_data_short_term.py**

* **train_short_term.py**

* **test_short_term.py**