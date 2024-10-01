#!/usr/bin/env python3
#
#                0   1   2   3 4                 5
# ./plot_1x1v.py out_160_200_1_Landau-damping-C0_8procs.txt
#
import sys, os, math, glob
import matplotlib.pyplot as plt
import numpy as npy
import matplotlib.ticker as ticker
import random as rand
from numpy import array
import locale
import pandas as pd
#import seaborn as sns

locale.setlocale(locale.LC_ALL, '')
#plt.rcParams["font.size"] = 11
#plt.rcParams["font.weight"] = "bold"
#plt.rcParams["axes.labelweight"] = "bold"
#plt.rcParams.update({'font.size': 11})
#plt.rcParams.update({'font.weight': "bold"})
ncells_str = "0"
npart_strLoc = "0"
npart_str_global = "0"
nfiles = 0
max_steps=1000
max_files = 1
Emax = npy.zeros( max_steps ) # Emax data, series idx
num_glab_p = []
Times = npy.zeros(max_steps)
series_name_AMR_GP = []
for filename in sys.argv[1:]:
    base = filename.split('.')
    parts = base[0].split('_')
    print('file = ',parts)
    ncells_str = parts[1]
    npart_loc = parts[2]
    #np = int(npart_str_loc)
    name_str = parts[4] # should all be the same
    #print(name_str)
    name_str = name_str.replace("-", " ")
    stepi = 0
    for text in open(filename,"r"):
        words = text.split()
        n = len(words)
        # Global Np = 128000 Global cells = 64
        if n > 1 and words[0] == 'Global':
            npart_str_global = words[3]
            npart_global = int(npart_str_global)
            npart_str_global = format(int(npart_str_global), ',')
            index = None
            try:
                index = num_glab_p.index(npart_str_global)
            except ValueError:
                num_glab_p.append(npart_str_global)
                series_name_AMR_GP.append('No AMR, ' + npart_str_global + ' particles')
                print ('-------- add ', npart_str_global);
        elif n > 1 and words[0] == 'E:' and float(words[1]) > 0.0 :
            Times[stepi] = float(words[1])
            Emax[stepi] = float(words[5])
            stepi = stepi + 1
    nfiles = nfiles + 1
    print ('num particles: ', npart_loc, npart_str_global, ', num cells: ', ncells_str)
#n_np = len(NPs_loc)
#print(num_glab_p[:,0:n_np])
#marks = ['s','o', 'D', 'P']
styles = ['kx:','bD:', 'go--', 'mD-.', 'c+:']
#
# plot
#
plt.rcParams["figure.dpi"] = 300
#plt.rcParams["figure.figsize"] = [5.00, 4.0]
#plt.rcParams["figure.autolayout"] = True
series_name_AMR = [r'$E_{max}$'] #,'1 AMR level','2 AMR level','3 AMR level']
print(series_name_AMR)
#DEseries_name_Np = [[str(ele) for ele in sub] for sub in NPs_loc]
# printing result
#print("The data type converted Matrix : " + str(series_name_Np))
ylabel = r'$E_{max}$'
#
# plot Emax
#
titles = [name_str + ': ' + ncells_str + ' cells', name_str + ': ' + ncells_str + ' cells, ~160K particles', name_str + ': ' + ncells_str + ' cells, ' + f'{int(npart_loc) * int(ncells_str):,}' + ' particles']
if sum(Emax[:stepi]) > 0:
    df = pd.DataFrame(data=Emax[:stepi], index=Times[:stepi], columns=series_name_AMR)
    ax = df.plot(lw=1, style=styles, markersize=1, logx=False, logy=True, grid=True, legend=False)
    title = titles[2] #title = name_str + ': ' + ncells_str + ' cells, ~160K particles'
    ax.set_title(title, pad=20, fontdict={'fontsize':14}) # , fontdict={'fontsize':16}
    patches, labels = ax.get_legend_handles_labels()
    ax.legend(patches, labels, loc='best', fontsize=10) #, fontsize=14
    plt.legend(npy.unique(labels), fontsize=10)
    plt.tight_layout()
    xmin, xmax, ymin, ymax = plt.axis()
    xmin, xmax, ymin, ymax = plt.axis([xmin, xmax, ymin, ymax])
    #xmin, xmax, ymin, ymax = plt.axis([xmin, xmax, 350, 375])
    #ax.set_xlabel(xlabel + normal_species + r', ($t_0=$' + t_0_arr[nfiles] + ')') #, fontdict={'fontsize':16})
    ax.set_xlabel('Time () ', fontdict={'fontsize':16}) #, fontdict={'fontsize':16})
    ax.set_ylabel(ylabel, fontdict={'fontsize':16}) #, fontdict={'fontsize':16})
    plt.savefig( name_str.split(' ')[0] + '-E-max-' + ncells_str + '-cells.png', bbox_inches='tight')
    plt.show()
