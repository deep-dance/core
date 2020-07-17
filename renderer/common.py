import sys
import matplotlib.pyplot as plt

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def progress(message, value):
    sys.stdout.write(bcolors.HEADER + "\r" + message + " %03i..." % value + bcolors.ENDC)
    sys.stdout.flush()

def notice(notice, log=True):
    notice = bcolors.HEADER + notice + bcolors.ENDC
    if log:
        print(notice)
    return notice

bones = [
    # Right leg
    (0, 1),
    (1, 2),
    (2, 3),
    # Left leg
    (0, 4),
    (4, 5),
    (5, 6),
    # Torso
    (0, 7),
    (7, 8),
    # Head
    (8, 9),
    (9, 10),
    # Right arm
    (8, 14),
    (14, 15),
    (15, 16),
    # Left arm
    (8, 11),
    (11, 12),
    (12, 13),
]

def init_plot():
    fig = plt.figure()
    fig.set_size_inches(18.5, 10.5)

    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(0, 2)

    return fig, ax

def plot_pose(ax, frame):
    ax.clear()
    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(0, 2)

    for kp in frame:
        ax.scatter(kp[0], kp[1], kp[2])

    for (j_from, j_to) in bones:
        ax.plot3D(
            [frame[j_from][0], frame[j_to][0]],
            [frame[j_from][1], frame[j_to][1]],
            [frame[j_from][2], frame[j_to][2]],
            'gray')