#!/usr/bin/env python
#
# run with files to be processed for one plot from run.bsub output.
#  eg, ./plot.py out_*kokkos_tpl*
#
import sys, os, math, glob
import matplotlib.pyplot as plt
import numpy as npy
import matplotlib.ticker as ticker
import random as rand
from numpy import array
import locale
import pandas as pd
locale.setlocale(locale.LC_ALL, '')
plt.rcParams["font.size"] = 11
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams.update({'font.size': 11})
plt.rcParams.update({'font.weight': "bold"})
idx = 0
nprocs = npy.empty(6, dtype=int)
solve_timeT = npy.empty([6,4])
run_type = ''
run_sub_type = ''
run_params = ''
max_proc_id = -1
max_ref_id = -1
for filename in sys.argv[1:]:
    words = filename.split('_')
    nps = int(words[1])
    proc_id = int(math.log(nps,8))
    if proc_id > max_proc_id: max_proc_id = proc_id
    run_type = words[2]
    run_sub_type = words[3]
    refine_id = int(words[4]) - 2
    if refine_id > max_ref_id: max_ref_id = refine_id
    run_params = words[5]
    run_params = run_params.split('.')[0]
    nprocs[proc_id] = nps
    if   run_sub_type == 'def':        sub_type_str = 'GPU aware MPI'
    elif run_sub_type == 'nogpuaware': sub_type_str = 'CPU MPI'
    elif run_sub_type == 'tpl':        sub_type_str = 'cuSparse'
    elif run_sub_type == 'notpl':      sub_type_str = 'Kokkos'
    elif run_sub_type == 'devreduce':  sub_type_str = 'reduce, 1 RS'
    elif run_sub_type == 'nodevreduce':sub_type_str = 'No reduce, 1 RS'
    else :                             sub_type_str = 'xxx'
    if run_type == 'cuda': run_type_str = 'cuSparse'
    else : run_type_str = 'Kokkos kernels'
    short_run_str = run_type_str
    if short_run_str == 'Kokkos kernels': short_run_str = 'Kokkos'
    short_sub_run_str = ''
    if run_type == 'kokkos' and sub_type_str == 'cuSparse': short_sub_run_str = 'cuSp'
    elif run_sub_type == 'Kokkos' : short_sub_run_str = 'native'
    #print ('numprocs=', nps, run_type, run_type_str, run_sub_type,sub_type_str
    idx2 = 0
    for text in open(filename,"r"): 
        words = text.split()
        n = len(words)
        if n > 1 and words[0] == 'KSPSolve':
            idx2 = idx2 + 1
            if idx2 == 2:
                stime = float(words[3])
                solve_timeT[proc_id,refine_id] = stime
    idx = idx + 1
if max_ref_id == -1: print ('no DATA -- need files as arguments !!!!!!')
#print (run_type,run_sub_type,sub_type_str,run_type_str)
solve_time = solve_timeT[0:max_proc_id+1,0:max_ref_id+1]
marks2 = [['ro-','bo-.','mo--','go:'],['r^-', 'b^-.', 'm^--', 'g^:'],['r*-', 'b*-.', 'm*--', 'g*:'],['r+-', 'b+-.', 'm+--', 'g+:'],['rx-', 'bx-.', 'mx--', 'gx:'],['rv-', 'bv-.', 'mv--', 'gv:'],['rD-', 'bD-.', 'mD--', 'gD:'],['ro-', 'bo-.', 'mo--', 'go:'],['r|-', 'b|-.', 'm|--', 'g|:'],['rp-', 'bp-.', 'mp--', 'gp:'],['r*-', 'b*-.', 'm*--', 'g*:']]
stri1 = ''
stri2 = ''
stri1 = run_type_str + ' (' + sub_type_str + ')'
stri2 = '_' + run_type_str + '_' + sub_type_str + '_'
#
# Weak scaling - GPU
#
series_name_full = ['24$\cdot 4^3$ cells/node', '24$\cdot 8^3$ cells/node', '24$\cdot 16^3$ cells/node', '24$\cdot 32^3$ cells/node']
series_name_short = ['24$\cdot 4^3$', '24$\cdot 8^3$', '24$\cdot 16^3$', '24$\cdot 32^3$']
nprocs = nprocs[0:max_proc_id+1];

df = pd.DataFrame(data=solve_time, index=nprocs, columns=series_name_full[0:max_ref_id+1])
df2 = pd.DataFrame(data=solve_time, index=nprocs, columns=series_name_short[0:max_ref_id+1])
df2.index.name = 'Nodes'
df2.columns.name = short_run_str + ' ' + short_sub_run_str + ' - Cells/node:'
#print df
ax = df.plot(lw=2, colormap='jet', marker='s', markersize=10, title='3D Q2 Laplacian AMG solve, ' + run_type_str + ' - ' + sub_type_str, logx=True, grid=True, legend=False)
patches, labels = ax.get_legend_handles_labels()
ax.legend(patches, labels, loc='upper left')
xmin, xmax, ymin, ymax = plt.axis()
xmin, xmax, ymin, ymax = plt.axis([xmin*.9, xmax*1.1, 0, 8])
ax.set_xlabel('# SUMMIT nodes ('+run_params+')')
ax.set_ylabel('AMG Solve Time (rtol=$\mathbf{10^{-12}}$)')
plt.savefig('weak_scaling_' + run_type + '_' +  run_sub_type + '_' + run_params + '.png')
#latex table

print(df2.to_latex(longtable=False,escape=False,float_format="{:0.2f}".format))
# caption = '3D Q2 Laplacian solve times: ' + run_type_str + ' - ' + sub_type_str,  , label='tab:GAMGtimes-'+run_type_str+'-'+sub_type_str
