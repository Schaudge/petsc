#!/usr/bin/env python

import os
import hashlib

def hash(args):
  '''Generates a hash of the PATH, configure args, and config directory'''
  hash = None
  if args == None:
    try:
      with open(os.path.join(os.environ.get('PETSC_DIR'),os.environ.get('PETSC_ARCH'),'lib','petsc','conf','petscvariables'),'rb') as f:
        while 1:
          l = f.readline()
          if not l: return hash
          s = l.find(b'CONFIGURE_OPTIONS =')
          if s > -1:
            l = l.decode(encoding='UTF-8',errors='replace')
            args = str(l[s+19:-1])
            args = args.split()
            break
    except:
      pass
  hash = 'args:\n' + '\n'.join('    '+a for a in sorted(args)) + '\n'
  hash += 'PATH=' + os.environ.get('PATH', '') + '\n'
  try:
    for root, dirs, files in os.walk(os.path.join(os.environ.get('PETSC_DIR'),'config')):
      if root == 'config':
        dirs.remove('examples')
      for f in files:
        if not f.endswith('.py') or f.startswith('.') or f.startswith('#'):
          continue
        fname = os.path.join(root, f)
        with open(fname,'rb') as f:
          hash += hashlib.sha256(f.read()).hexdigest() + '  ' + fname + '\n'
  except:
    pass
  return hash

def checkhash(hash,hashfile):
  '''Compares the hash to a hash in a file, returns 1 if they match, else 0'''
  if not hash: return 0
  try:
    with open(hashfile, 'r') as f:
      b = f.read()
  except:
    return 0
  return hash == b

if __name__ == '__main__':
  import sys
  hashfile = os.path.join(os.environ.get('PETSC_DIR'),os.environ.get('PETSC_ARCH'),'lib','petsc','conf','configure-hash')
  if not checkhash(hash(None),hashfile):
    print('*******************************WARNING******************************')
    print('The configuration for '+os.environ.get('PETSC_ARCH')+' has changed since the last time you ran ./configure')
    print('You may want to run '+  os.path.join(os.environ.get('PETSC_DIR'),os.environ.get('PETSC_ARCH'),'lib','petsc','conf','reconfigure-'+os.environ.get('PETSC_ARCH')+'.py'))
    print('*******************************WARNING******************************')
