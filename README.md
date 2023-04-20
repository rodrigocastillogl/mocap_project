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
        
        * Implementation of quaternion methods for *Pytorch* tensors and *NumPy* arrays:
        `qmul`, `qrot`, `qeuler`, `qfix`, `expmap_to_quaternion`, `euler_to_quaternion`.

    * **skeleton.py**

        * `class Skeleton`
        
            Defines a parameterized skeleton for MoCap data, by means of  the following attributes:
            `offsets`, `num_joints`, `parents`, `children`, `joints_left` and `joints_right`.

            You can get joint positions using the `forward_kinematics` method.

    * **mocap_dataset.py**
        
        * `class MocapDataset`
        
            Dataset object for Motion Capture. You can load and store MoCap data using this class.
            It is stored in a dictironary with the following structure:

            `self._data` -> `subject` -> `action` -> `rotations`, `trajectory`, `positions`, `euler_angles`.

            It has a `Skeleton` object as attribute where offsets and hierarchy are defined.
            Then , data from all subjects in the dataset are fitted to this skeleton.

            `MocapDatset['subject']` returns data of `'subject'`, which is a dictionary of actions, etc.

    * **quaternet.py**

        * `class QuaterNet`
        
            Defines the Network architechture proposed by Pavllo, Grangier and Auli, given by the following Figure:
        
            [instert model figure]

            Implementation of the forward propagation method to evaluate the model.

            Does not implement: load batch method, loss function, train method.

    * **pose_network.py**

        * `class PoseNetwork`

            Derived from `QuaterNet`.

            Declaration of: `_prepare_next_batch_impl` (load batch method) and `_loss_impl` (loss function).
            
            Definition of `train`, `load_weights` and `save_weights` methods.

    
    * **visualization.py**

        * *Matplotlib* based methods to display/save skeleton motion animations.
    
* short_term/

    * **dataset_h36m.py**

        * Defines the training, validation and test sets for the H3.6M dataset.
        * Defines the `Skeleton` object used with the H3.6M dataset.
        * Defines the `MocapDataset` object used for the H3.6M dataset, and load data.

    * **pose_network_short_term.py**

        * `class PoseNetworkShortTerm`
        
            Derived from `PoseNetwork`.

            Definition of: `_prepare_next_batch_impl` (load batch method) and `_loss_impl` (loss function).

            Defines the `predict` method: given an input sequence, predicts the next sequence of poses (evaluates the model with `no_grad`).

* **prepare_data_short_term.py**

    * Downloads [H3.6M Dataset](http://www.cs.stanford.edu/people/ashesh/h3.6m.zip) in Exponential map parameterization (as used in [Martínez, et. al. 2017](https://arxiv.org/abs/1705.02445)).
    
    * Transforms data to quaternion parameterization and saves it in `datasets/h3.6m`.


* **train_short_term.py**

* **test_short_term.py**