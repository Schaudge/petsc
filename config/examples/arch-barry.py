#!/usr/bin/env python
import os
import prefixconverter
#
#  This is a demonstration file, it will not go into the repository
#
prefix = '/Users/barrysmith/packages/'+os.path.basename(__file__).replace('.py','')
configure_options = [
  '--prefix='+prefix,
  '--with-prefix-replace',
  '--download-sowing',
  '--download-sowing-public',
  '--with-cc=mpicc',
  '--with-fc=mpif90',
  '--with-cxx=mpicxx',
  'COPTFLAGS=-g -O',
  'FOPTFLAGS=-g -O',
  'CXXOPTFLAGS=-g -O',
  '--download-hypre=1',
  ]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure

  try:
    # Run the --prefix based configure to build the external packages
    # Will only run again if a change is made in the configuration
    configure.petsc_configure(configure_options)
    ok = 1
  except SystemExit as e:
    if e.code is None or e.code == 0: ok = 1
    else: ok = 0
  if ok:
    trueconfigure_options = prefixconverter.ConvertFromDownLoadToPrefix(prefix,configure_options)
    # Run the true configure
    configure.petsc_configure(trueconfigure_options)
