# ---------------------------------------------- #
# Copyright (c) 2018-present, Facebook, Inc.
# https://github.com/facebookresearch/QuaterNet
# ---------------------------------------------- #

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
from mpl_toolkits import Axes3D
import numpy as np
import torch

def render_animation(data, skeleton, fps, output = 'interactive', bitrate = 10000):
    """
    Render or show an animation.
    Input
    -----
        * data : skeleton configurations
                dimension: 
        * skeleton : skeleton object.
        * fps : sampling rate (frames per second).
        * output: mode
            - interactive : display an interactive figure.
            - html : render the animation as a HMTL5 video.
            - filename.mp4 : render and export the animation as an h264 video (requires ffmpeg).
            - filename.gif : render and export the animation as a git file (requires imagemagick).
    Output
    ------
        None
    """

    x = 0
    y = 1
    z = 2

    radius = torch.max( skeleton.offsets() ).item() * 5

    skeleton_parents = skeleton.parents()

    # inactive mode
    plt.ioff()
    fig = plt.figure( fig_size = (4,4) )
    ax = fig.add_subplot( 1, 1, 1, projection = '3d' )
    ax.view_init( elev = 20, azim = 30 )

    # axes ranges
    ax.set_xlim3d( [-radius/2, radius/2] )
    ax.set_zlim3d( [0, radius] )
    ax.set_limy3d( [-radius/2, radius] )
    ax.set_aspect('equal')
    
    # no ticks
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    # view distance
    ax.dist = 7.5       # default: 10, 7.5 makes it look closer

    lines = []
    initilized = False

    trajectory = data[:, 0, [0,2]]

