#!/usr/bin/env python3
"""
# Created: Sun Jul 17 08:25:36 2022 (-0400)
# @author: Jacob Faibussowitsch
"""
import os
import shutil
import pathlib
import subprocess
import pickle

def subprocess_run(*args,**kwargs):
  kwargs.setdefault('capture_output',True)
  kwargs.setdefault('universal_newlines',True)
  args = list(map(str,args))
  ret = subprocess.run(args,**kwargs)
  if ret.returncode != 0:
    mess = f'Subproccess error! Command {args} returned non-zero exit status {ret.returncode}\nstdout:\n{ret.stdout}\nstderr:\n{ret.stderr}'
    raise RuntimeError(mess)
  if ret.stderr:
    print('')
    print(ret.stderr)
  return ret.stdout

def run_exec(exec_path,nit,nwarmup,mat_size,ksp_type,profile,log_view,show,*args):
  name = exec_path.stem.split('_')[-1]+'_'+ksp_type
  arg_list = [
    exec_path,'-n_it',nit,'-n_warmup',nwarmup,'-n',mat_size,'-vec_type','cuda',
    '-use_gpu_aware_mpi',0,'-ksp_type',ksp_type,'-log_view'
  ]
  if not str(exec_path).endswith('_main'):
    arg_list.extend(['-root_device_context_stream_type','default_blocking'])
  if log_view:
    arg_list.append('-log_view_gpu_time')
  if profile:
    arg_list = [
      'nsys','profile','--force-overwrite=true','--cudabacktrace=all','--samples-per-backtrace=2',
      '--trace=cuda,osrt,cublas-verbose,cusparse-verbose,nvtx','--capture-range=cudaProfilerApi',
      '--capture-range-end=stop','-o',name
    ]+arg_list
  arg_list.extend(list(args))
  ret = subprocess_run(*arg_list).splitlines()
  rest_lines = []
  log_lines  = []
  log_output = []
  full_log_capture = 0
  log_capture = 0
  for line in ret:
    if 'Norm of error' in line:
      answer = line
    elif 'KSP type' in line and 'total time' in line:
      timings = line
    else:
      if full_log_capture or line.startswith('*'*160):
        full_log_capture = 1
        log_lines.append(line)
        if line.startswith('Using libraries:'):
          full_log_capture = 0
      else:
        rest_lines.append(line)
    if line.lstrip().startswith('--- Event Stage') and 'Timing' in line:
      log_capture += 1
    elif log_capture == 1:
      line = line.strip()
      if line.startswith('---'):
        log_capture += 1
      elif line:
        log_output.append(line)
  if len(rest_lines) and show:
    print('\n'.join(rest_lines))
  if len(log_lines) and log_view and show:
    print('\n'.join(log_lines))
  if not show:
    print('-'*40)
  e_norm,iterations = answer.split(',')

  def extract_time(name):
    return float(timings.partition(name)[2].split(',')[0].strip('. s'))

  data = {
    'name'    : name,
    'e_norm'  : float(e_norm.split()[-1]),
    'iter'    : int(iterations.split()[-1]),
    't_total' : extract_time('total time'),
    't_avg'   : extract_time('avg'),
    't_min'   : extract_time('min'),
    'stdout'  : ret,
    'log_view': log_output,
  }
  return data

def run_exp(mat_size,nit,nwarmup,profile,petsc_arch,log_view,show,*args):
  try:
    petsc_dir = pathlib.Path(os.environ['PETSC_DIR'])
  except KeyError:
    petsc_dir = pathlib.Path('~/petsc')

  petsc_dir = petsc_dir.resolve()
  os.environ['PETSC_DIR']  = str(petsc_dir)
  os.environ['PETSC_ARCH'] = petsc_arch

  exec_base  = petsc_dir/'src'/'ksp'/'ksp'/'tutorials'/'ex500'
  async_exec = exec_base.with_stem('ex500_async')
  no_cu_exec = exec_base.with_stem('ex500_no_cuda')
  main_exec  = exec_base.with_stem('ex500_main')

  print('building PETSc... ',end='',flush=True)
  subprocess_run('make','libs','-j','10')
  print('done')
  print(f'building {exec_base.stem}... ',end='',flush=True)
  subprocess_run('make',exec_base.stem)
  print('done')
  print(f'copying {exec_base.stem} to {async_exec.stem}... ',end='',flush=True)
  shutil.copy2(exec_base,async_exec)
  print('done')
  # os.environ['PETSC_ARCH'] = 'test-arch'
  # print(f'building {no_cu_exec.stem}... ',end='',flush=True)
  # subprocess_run('make',no_cu_exec.stem)
  # os.environ['PETSC_ARCH'] = petsc_arch
  print('done')
  print('='*50)
  print('mat_size',mat_size,'n iter',nit,'n warmup',nwarmup)
  data_list = []
  work_list = [(main_exec,'pipefgmres'),(async_exec,'pipefgmresasync'),(async_exec,'pipefgmres')]
  for idx,(exec_name,ksp_type) in enumerate(work_list):
    latest = run_exec(exec_name,nit,nwarmup,mat_size,ksp_type,profile,log_view,show,*args)
    print(latest['name'],f'{latest["t_total"]}s')
    if len(data_list):
      last = data_list[-1]
      for key in ('e_norm','iter'):
        assert last[key] == latest[key], f'{last["name"]}[{key}] {last[key]} != {latest["name"]}[{key}] {latest[key]}'
    data_list.append(latest)
  data_list.sort(key=lambda x: x['t_total'])
  max_width = max(len(x['name']) for x in data_list)+1

  def extract(data_dict):
    return {l.split()[0] : l.split()[1:] for l in data_dict['log_view'] if not l.startswith('DCtx')}

  def extract_xfer_stats(log_entries):
    total_cpu2gpu = [0,0]
    total_gpu2cpu = [0,0]
    for event,event_data in log_entries.items():
      total_cpu2gpu[0] += int(event_data[-5])   # count
      total_cpu2gpu[1] += float(event_data[-4]) # size
      total_gpu2cpu[0] += int(event_data[-3])   # count
      total_gpu2cpu[1] += float(event_data[-2]) # size
    print('total cpu2gpu transfers',total_cpu2gpu[0],'size',total_cpu2gpu[1],'(MB)')
    print('total gpu2cpu transfers',total_gpu2cpu[0],'size',total_gpu2cpu[1],'(MB)')
    return total_cpu2gpu,total_gpu2cpu

  fastest_log = extract(data_list[0])
  data_list[0]['log_events'] = fastest_log
  if len(fastest_log.keys()):
    max_event_len = max(map(len,fastest_log.keys()))

    print('<---',data_list[0]['name'],'--->')
    h2d,d2h = extract_xfer_stats(fastest_log)
    data_list[0]['h2d'] = h2d
    data_list[0]['d2h'] = d2h
    for rank,entry in enumerate(data_list[1:]):
      log_entries = extract(entry)
      print('<---',entry['name'],'--->')
      h2d,d2h = extract_xfer_stats(log_entries)
      data_list[rank+1]['h2d'] = h2d
      data_list[rank+1]['d2h'] = d2h
      data_list[rank+1]['log_events'] = log_entries
      if not log_view:
        continue
      event_list = []
      for event,event_data in log_entries.items():
        event_list.append((event,float(fastest_log[event][2]),float(event_data[2])))
      event_list.sort(key=lambda x: abs(x[1]-x[2])/min(x[1],x[2]))
      out_list = []
      for name,ftime,stime in event_list:
        if ftime == stime:
          cmp_op = '='
          descr  = 'equal'
        else:
          pc_slower = int((abs(stime-ftime)/min(ftime,stime))*100)
          if ftime < stime:
            cmp_op = '<'
            descr  = 'slower *'
          else:
            cmp_op = '>'
            descr  = 'faster'

        out_list.append(f'{name:<{max_event_len}}: {ftime:<10} ({fastest_log[event][0]} calls) {cmp_op} {stime:<10} ({log_entries[event][0]} calls) -> {pc_slower:>4}% {descr}')

      try:
        spacer = out_list[0].index('<')
      except ValueError:
        spacer = out_list[0].index('>')
      else:
        spacer = min(spacer,out_list[0].index('>'))
      header = '<'+('-'*(max_event_len-1))+' '+data_list[0]['name']
      header += (' '*(spacer-len(header)))+'| '+entry['name']+' '
      header += ('-'*(len(out_list[0])-len(header)-1))+'>'
      print(header,'\n'.join(out_list),sep='\n')
  entry_name = 't_min'
  print('<--- timings --->')
  print('Ranked by:',entry_name)
  for rank,entry in enumerate(sorted(data_list,key=lambda x: x[entry_name])):
    time = entry[entry_name]
    if rank == 0:
      fastest    = time
      out_phrase = 'fastest'
    else:
      pc_slower  = int(((time-fastest)/fastest)*100)
      out_phrase = f'{pc_slower}% slower'
    print(f'{rank+1} {entry["name"]+":":<{max_width}} {time}s ({out_phrase})')
  return data_list

def main(timeall,mat_max_size,nit,*args,**kwargs):
  if timeall:
    data_dict = {}
    mat_size  = 10
    nit       = 1000
    while mat_size <= mat_max_size:
      data_dict[mat_size] = run_exp(mat_size,nit,*args,**kwargs)
      nit       = int(max(nit/10,5))
      mat_size *= 10
    output_file = 'async_pipefgmres_data.pkl'
    with open(output_file,'wb') as fd:
      pickle.dump(data_dict,fd)
    print('wrote to',output_file)
  else:
    run_exp(mat_max_size,nit,*args,**kwargs)
  return


if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument('-n','--num-iterations',type=int,default=1000,help='number of timing iterations')
  parser.add_argument('-nw','--num-warmup',type=int,default=2,help='number of warmup loops')
  parser.add_argument('-nm','--matrix-max-size',type=int,default=10,help='size of matrix')
  parser.add_argument('--profile',action='store_true',help='profile using cuda')
  parser.add_argument('-pa','--petsc-arch',default='arch-cuda-opt',help='override PETSC_ARCH')
  parser.add_argument('--log-view',action='store_true',help='parse log_view')
  parser.add_argument('-s','--show',action='store_true',help='verbose stdout')
  parser.add_argument('-t','--timeit',action='store_true',help='create a timing run of multiple matrix sizes')
  args,rest = parser.parse_known_args()
  main(args.timeit,args.matrix_max_size,args.num_iterations,args.num_warmup,bool(args.profile),args.petsc_arch,args.log_view,args.show,*rest)
