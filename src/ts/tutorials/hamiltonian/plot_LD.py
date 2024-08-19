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
npart_str = "0"
ncells_str2 = "0"
npart_str2 = "0"
nfiles = 0
max_steps=20000
max_files = 1
Emax = npy.zeros(max_steps)
Times = npy.zeros(max_steps)
for filename in sys.argv[1:]: # just one
    base = filename.split('.')
    parts = base[0].split('_')
    print('file = ',parts)
    ncells_str = parts[1]
    npart_str = parts[2]
    stepi = 0
    for text in open(filename,"r"):
        words = text.split()
        n = len(words)
        # Global Np = 128000 Global cells = 64
        if n > 1 and words[0] == 'Global':
            npart_str2 = words[3]
            ncells_str2 = words[7]
        elif n > 1 and words[0] == 'E:' and float(words[1]) > 0 :
            Times[stepi] = float(words[1])
            Emax[stepi] = float(words[5])
            stepi = stepi + 1
    nfiles = nfiles + 1
print (ncells_str,npart_str,ncells_str2,npart_str2)
#marks = ['s','o', 'D', 'P']
styles = ['k:','bD:', 'go-', 'go:']
styles2 = ['k:']
styles3 = ['r:']
#
# plot
#
plt.rcParams["figure.dpi"] = 300
#plt.rcParams["figure.figsize"] = [5.00, 4.0]
#plt.rcParams["figure.autolayout"] = True
series_name = [r'$E_{max}$']
sig_strs = [r'\sigma = 1']
ylabel = r'$E_{max}$'
df = pd.DataFrame(data=Emax[:stepi], index=Times[:stepi], columns=series_name)
ax = df.plot(lw=1, style=styles, markersize=1, logx=False, logy=True, grid=True, legend=False)
pstr = format(int(npart_str2), ',')
title=r'$E_{max}$ ' + ncells_str2 + ' cells; ' + pstr  + ' particles'
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
plt.savefig('E-max' + ncells_str + '-'+ npart_str2 + '.png', bbox_inches='tight')
plt.show()

#cm = sns.light_palette("green_r", as_cmap=True)
#styler = df.style.hide(axis="index")
#styler = df.style.background_gradient(axis=0, low=0.75, high=1.0, cmap='YlOrRd').hide(axis="index")
#df.style.background_gradient(axis=0, gmap=df['Solve'], cmap='YlOrRd')
#styler = df.style.hide(axis="index").background_gradient(subset=(df['Solve']>11, "Solve"))
#styler = df.style.hide(axis="index").background_gradient(cmap='RdYlGn_r', subset=['Jacobian', 'Solve', 'Total time'], low=.1, high=.1).background_gradient(cmap='RdYlGn_r', subset=['error (\%)'], low=.1, high=.1).format(precision=0, thousands=",")
#print(styler.to_latex(convert_css=True,column_format='cccrrrr', position='h!', hrules=True, caption = 'Anisotropic thermalization timings (seconds). Number of cells, number of integration points (IPs), order of elements, Jacobian matrix construction, linear solve time, total run time, and percent error from perfect thermalization on ' + machine_full, label='tab:timings'+mach_short))

