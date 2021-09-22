import numpy as np
import h5py
import sys
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
"""
To run this code, make sure 'sol.h5' and 'Length_1.out' files are in the same folder. The run as follows:

python droplet.py $(animation_name) $(figure_name)
"""
def h5printR(item, leading = ''):
    for key in item:
        if isinstance(item[key], h5py.Dataset):
            print(leading + key + ': ' + str(item[key].shape))
        else:
            print(leading + key)
            h5printR(item[key], leading + '  ')
#
def h5print(filename):
    with h5py.File(filename, 'r') as h:
        print(filename)
        h5printR(h, '  ')
###############################################################
n = len(sys.argv)
if(n<3):
    print("We need both animation name and figure name. Please provide both names.")
else:
    skip = 5
    N = 5
    filename = "sol.h5"
    fname = sys.argv[1]
    # h5print(filename)
    ################################################################
    radius = {}
    length = {}
    X = {}
    with h5py.File(filename, "r") as f:
        # List all groups
        print("Keys: %s" % f.keys())
        A = list(f.keys())[5]
        B = list(f.keys())[0]
        T = list(f.keys())[3]
        C = list(f.keys())[1]
        # Get the data
        a = f[A]
        b = f[B]
        c = f[C]
        count = len(b.keys())
        for i in np.arange(1, count+1, 1, dtype=int):
            radius["h"+str(i)] = np.array(a.get("Numerical_Solution_"+str(i)+"_radius"))
            if (i>1):
                radius["h"+str(i)] = radius["h"+str(i)][radius["h"+str(i-1)].shape[0]:]
            length["l"+str(i)] = np.genfromtxt("Length_"+str(i)+".out", dtype='float', delimiter=",")
            length["l"+str(i)] = length["l"+str(i)][~np.isnan(length["l"+str(i)])]
            length["l"+str(i)] = length["l"+str(i)].reshape((radius["h"+str(i)].shape[0],radius["h"+str(i)].shape[1]))

            radius["h"+str(i)] = radius["h"+str(i)][:,np.argsort(length["l"+str(i)][0])]
            length["l"+str(i)] = length["l"+str(i)][:,np.argsort(length["l"+str(i)][0])]
            X["x"+str(i)] = np.zeros_like(length["l"+str(i)])
        t = np.transpose(np.array(f[T]))[0]
        steps = t.shape[0]
        step = np.arange(0,steps,skip)
    if(skip>1):
        step = np.concatenate((step[:-N], np.linspace(step[-N]+1, step[-1], (N-1)*skip, dtype='int')) )
    ##################################################################
    lmax = max((np.max(length[key]) for key in length))
    hmax = max((np.max(radius[key]) for key in radius))
    c = lmax/radius['h1'][0][0]
    figname = sys.argv[2]
    fig, ax = plt.subplots(figsize=(10,5.05*c))
    ax.set_xlabel('radius', fontsize=42, fontweight='bold')
    ax.set_ylabel('length', fontsize=42, fontweight='bold')
    ax.set_xlim(-1.1*hmax,1.1*hmax)
    # ax.set_ylim(0,3.675*hmax)
    ax.set_ylim(0,lmax)
    ax.tick_params(axis='both', labelsize=34)
    plt.gca().invert_yaxis()
    ax.fill_betweenx(length['l1'][step[-1]], -radius['h1'][step[-1]], radius['h1'][step[-1]], color='dodgerblue', alpha=1)
    for s in np.linspace(int(step[-1]/2),step[-1],6, dtype=int):
        ax.plot(radius['h1'][s],length['l1'][s], color='b', linewidth=3)
        ax.plot(-radius['h1'][s],length['l1'][s], color='b', linewidth=3)
    plt.savefig(figname+".png", dpi=300)
    ##################################################################
    fig, ax = plt.subplots(figsize=(10,5.05*c))
    line1, = ax.plot([],[], color='b')
    line2, = ax.plot([],[], color='b')
    ax.set_xlabel('radius', fontsize=28)
    ax.set_ylabel('length', fontsize=28)
    ax.set_xlim(-1.1*hmax,1.1*hmax)
    ax.set_ylim(0,lmax)
    plt.gca().invert_yaxis()
    h1_shape = radius['h1'].shape[0]

    ax.spines['right'].set_color('none')
    ax.spines['left'].set_linewidth(5)
    ax.spines['bottom'].set_linewidth(5)
    ax.xaxis.grid(True, linestyle='--', linewidth=3)

    def animate(step):

        plotTitle = ax.set_title("Time={:.6f}".format(t[step]), fontsize=28)
        line2.set_data(radius['h1'][step],length['l1'][step])
        line1.set_data(-radius['h1'][step],length['l1'][step])
        ax.collections.clear()
        ax.fill_betweenx(length['l1'][step], -radius['h1'][step], radius['h1'][step], color='dodgerblue')
        ax.scatter(X['x1'][step],length['l1'][step], marker="_", s=200)

        return line1, line2,

    ani = FuncAnimation(fig, func=animate, frames=step, save_count=1, interval=1, blit=False)
    ani.save(fname+".mp4",writer='ffmpeg',fps=60, dpi=100)
    ################################################################
