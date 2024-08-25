#!/usr/bin/env python3
#
# ./plot.py out
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
ncells_str2 = "0"
npart_str_global = "0"
nfiles = 0
max_steps=500
max_files = 1
Emax = npy.zeros( (max_steps, 5, 3) ) # Emax data, AMR ID, series idx
num_glab_p = []
Times = npy.zeros(max_steps)
amr_max = 0
amr_min = 10
series_name_AMR_GP = []
for filename in sys.argv[1:]:
    base = filename.split('.')
    parts = base[0].split('_')
    print('file = ',parts)
    ncells_str = parts[1]
    npart_loc = int(parts[2])
    #np = int(npart_str_loc)
    amr_str = parts[3]
    amr_id = int(amr_str) # number of AMR levels same as ID (0,1,2,3)
    if amr_id > 0: amr_id = amr_id - 1 # 0 and 1 are both full grid
    if amr_id + 1 > amr_max: amr_max = amr_id + 1
    amr_str = str(amr_id)
    if amr_id < amr_min: amr_min = amr_id
    print ('amr: ',amr_id, amr_str, 'amr_max = ', amr_max, 'amr_min = ', amr_min)
    name_str = parts[4] # should all be the same
    #print(name_str)
    name_str = name_str.replace("-", " ")
    stepi = 0
    for text in open(filename,"r"):
        words = text.split()
        n = len(words)
        # Global Np = 128000 Global cells = 64
        if n > 1 and words[0] == 'Global':
            ncells_str2 = words[7] # not needed
            npart_str_global = words[3]
            npart_global = int(npart_str_global)
            npart_str_global = format(int(npart_str_global), ',')
            index = None
            try:
                index = num_glab_p.index(npart_str_global)
            except ValueError:
                num_glab_p.append(npart_str_global)
                if amr_id == 0: series_name_AMR_GP.append('No AMR, ' + npart_str_global + ' particles')
                else: series_name_AMR_GP.append('AMR ' + str(amr_id) + ', ' + npart_str_global + ' particles')
                print ('-------- add ', npart_str_global);
        elif n > 1 and words[0] == 'E:' and float(words[1]) > 0.0 :
            hit = 0
            Times[stepi] = float(words[1])
            if npart_loc == 1024:
                Emax[stepi,amr_id,0] = float(words[5])
                hit = hit + 1
            if npart_global == 307200:
                Emax[stepi,amr_id, 1] = float(words[5])
                hit = hit + 1
            if hit == 0:
                Emax[stepi,amr_id, 2] = float(words[5])
            stepi = stepi + 1
    nfiles = nfiles + 1
    print ('num particles: ', npart_loc, npart_str_global, ', num cells: ', ncells_str, ncells_str2)
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
series_name_AMR = ['No AMR','1 AMR level','2 AMR level','3 AMR level']
#DEseries_name_Np = [[str(ele) for ele in sub] for sub in NPs_loc]
# printing result 
#print("The data type converted Matrix : " + str(series_name_Np))
ylabel = r'$E_{max}$'
print (Emax[0:3,amr_min:amr_max,0])
print(num_glab_p,'amr_max',amr_max)
print (Emax[0:3,amr_min:amr_max,1])
print (Emax[0:3,amr_min:amr_max,2])
#
# plot AMR vs NPART = 1024
#
titles = [name_str + ': ' + ncells_str2 + ' cells', name_str + ': ' + ncells_str2 + ' cells, ~160K particles', name_str + ': ' + ncells_str2 + ' cells (extra)']
if sum(sum(Emax[:stepi,amr_min:amr_max,0])) > 0:
    print ('series_name_AMR_GP = ',series_name_AMR_GP)
    df = pd.DataFrame(data=Emax[:stepi,amr_min:amr_max,0], index=Times[:stepi], columns=series_name_AMR_GP)
    ax = df.plot(lw=1, style=styles, markersize=1, logx=False, logy=True, grid=True, legend=False)
    title = titles[0] #name_str + ': ' + ncells_str2 + ' cells'
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
    plt.savefig( name_str.split(' ')[0] + '-E-max-AMR-' + ncells_str + '-cells.png', bbox_inches='tight')
    plt.show()
#
# plot AMR vs ~160K particles
#
if sum(sum(Emax[:stepi,amr_min:amr_max,1])) > 0:
    print(series_name_AMR[amr_min:amr_max])
    df = pd.DataFrame(data=Emax[:stepi,amr_min:amr_max,1], index=Times[:stepi], columns=series_name_AMR[amr_min:amr_max])
    ax = df.plot(lw=1, style=styles, markersize=1, logx=False, logy=True, grid=True, legend=False)
    title = titles[1] #title = name_str + ': ' + ncells_str2 + ' cells, ~160K particles'
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
    plt.savefig( name_str.split(' ')[0] + '-E-max-160kp-' + ncells_str + '-cells.png', bbox_inches='tight')
    plt.show()
#
# extra
#
if sum(sum(Emax[:stepi,amr_min:amr_max,2])) > 0:
    print(series_name_AMR[amr_min:amr_max])
    df = pd.DataFrame(data=Emax[:stepi,amr_min:amr_max,2], index=Times[:stepi], columns=series_name_AMR[amr_min:amr_max])
    ax = df.plot(lw=1, style=styles, markersize=1, logx=False, logy=True, grid=True, legend=False)
    title = titles[2] #title = name_str + ': ' + ncells_str2 + ' cells, ~160K particles'
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
