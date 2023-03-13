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
    avg_segment_length = np.mean(
        np.linalg.norm( np.diff(trajectory, axis = 0), axis = 1) + 1e-3
    )
    draw_offset = int( 25 / avg_segment_length)
    spline_line, = ax.plot(*trajectory.T)
    camera_pos = trajectory
    height_offset = np.min(data[:, :, 1])
    data = data.copy()
    data[:,:, 1] -= height_offset

    def update(frame):
        """
        Description
        Input
        -----
            * frame :
        Output
        ------
            None
        """

        nonlocal initilized

        # set axes limitis according to trajectory (to camera position)
        ax.set_xlim3d(
            [ -radius/2 + camera_pos[frame, 0], radius/2 + camera_pos[frame, 0] ]
        )
        ax.set_ylim3d(
            [ -radius/2 + camera_pos[frame, 1], radius/2 + camera_pos[frame, 1] ]
        )

        positions_world = data[frame]
        for i in range(positions_world.shape[0]):
            if skeleton_parents[i] == -1:
                continue
            if not initilized:
                # to distinguish between right and left joints 
                col = 'red' if i in skeleton.joints_right() else 'black'
                # draw skeleton
                lines.append(
                    ax.plot(
                    [positions_world[i, x], positions_world[skeleton_parents[i], x]],
                    [positions_world[i, y], positions_world[skeleton_parents[i], y]],
                    [positions_world[i, z], positions_world[skeleton_parents[i], z]],
                    zdir = 'y', c = col
                    )
                )
            else:
                lines[i-1][0].set_xdata(
                    [ positions_world[i, x], positions_world[skeleton_parents[i], x] ]
                )
                lines[i-1][0].set_ydata(
                    [ positions_world[i, y], positions_world[skeleton_parents[i], y] ]
                )
                lines[i-1][0].set_3d_properties(
                    [ positions_world[i, z], positions_world[skeleton_parents[i], z] ],
                    zdir = 'y'
                )
            
            l = max( frame - draw_offset, 0)
            r = min( frame+draw_offset, trajectory.shape[0] )

            spline_line.set_xdata( trajectory[l:r, 0] )
            spline_line.set_ydata( np.zeros_like(trajectory[l:r, 0]) )
            spline_line.set_3d_properties( trajectory[l:r, 1], zdir = 'y')

            initilized = True

            # for interactive mode: if we get to the final frame, then close all the figures
            if output == 'interactive' and frame == data.shape[0] - 1:
                plt.close('all')
            
            fig.tight_layout()
            anim = FuncAnimation(
                fig    ,
                update ,
                frames = np.arange(0, data.shape[0]),
                interval = 1000/fps ,
                repeat = False
            )

            if output == 'interactive':
                plt.show()
                return anim
            elif output == 'html':
                return anim.to_html5_video()
            elif output.endswith('.mp4'):
                Writer = writers['ffmpeg']
                writer = Writer(fps = fps, metadata={}, bitrate=bitrate)
                anim.save( output, writer = writer )
            elif output.endswith('.gif'):
                anim.save(output, dpi=80, writer='imagemagick')
            else:
                raise ValueError(
                    'Unsupported output format (only interactive,  html, .mp4 and .gif)'
                )
            
            plt.close()