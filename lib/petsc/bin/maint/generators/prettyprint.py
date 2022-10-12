#!/usr/bin/env python
#!/bin/env python
# $Id: adprocess.py,v 1.12 2001/08/24 18:26:15 bsmith Exp $
#
# change python to whatever is needed on your system to invoke python
#
#  Reads classes.data and prints the information out nicely
#
#  Crude as all hack!
#
#  Calling sequence:
#      prettyprint.py
##
from __future__ import print_function
import os
import re
import sys
from string import *
import pickle

def main(args):
  file = open('classes.data','rb')
  enums   = pickle.load(file)
  types  = pickle.load(file)
  structs = pickle.load(file)
  aliases = pickle.load(file)
  fclasses = pickle.load(file)
  cclasses = pickle.load(file)
  vclasses = pickle.load(file)

  # strip Type name and common prefix from values
  ntypes = types
  types = {}
  for i in ntypes:
    i4 = i[0:-4]
    l = len(i4)
    types[i4] = {}
    for j in ntypes[i]:
      if not j.lower().startswith(i4.lower()):
        print("Wrong name prefix: Type "+i+" name "+j+" value "+ntypes[i][j])
      if not j.lower().endswith(ntypes[i][j]):
        print("Wrong name suffix: Type "+i+" name "+j+" value "+ntypes[i][j])
      jn = j[l:]
      types[i4][jn] = ntypes[i][j]

  print("----- Aliases --------")
  for i in aliases:
    print(i+" = "+aliases[i])
  print(" ")
  print("----- Enums --------")
  for i in enums:
    print(i)
    for j in enums[i]:
      print("  "+j)
  #print(" ")
  #print("----- Types; Implementations of Classes  --------")
  #for i in types:
  #  print(i)
  #  for j in types[i]:
  #    print("  "+j+" = "+types[i][j])
  print(" ")
  print("----- Structs --------")
  for i in structs:
    print(i)
    for j in structs[i]:
      print("  "+j)
  print(" ")
  print("----- Function Classes (function families without data) --------")
  for i in fclasses:
    print(i)
    for j in fclasses[i]:
      print("  "+j+"()")
      for k in fclasses[i][j]:
        print("    "+k)
  print(" ")
  print("----- Concrete Classes --------")
  std_methods = ['Create', 'Destroy', 'SetFromOptions', 'View','ViewFromOptions', 'SetOptionsPrefix', 'AppendOptionsPrefix']
  for i in cclasses:
    print(i)
    for j in std_methods:
      if not j in cclasses[i]:
        print("  "+j+"() Missing")
    for j in cclasses[i]:
      if not j in std_methods:
        print("  "+j+"()")
        for k in cclasses[i][j]:
          print("    "+k)
  print(" ")
  print("----- Virtual Classes --------")
  std_methods.extend(['SetType', 'GetType'])
  fnd = {}
  for i in vclasses:
    print(i)
    for j in std_methods:
      if not j in vclasses[i]:
        print("  "+j+"() Missing")

    # print each type and type sprecific functions
    for j in types[i]:
      print("  "+j)
      for k in vclasses[i]:
        if k.startswith('Create') and k.lower().endswith(j.lower()):
          print("    "+k+"()")
          fnd[k] = True
          for l in vclasses[i][k]:
            print("    "+l)
        if k.lower().startswith(j.lower()):
          print("    "+k+"()")
          fnd[k] = True
          for l in vclasses[i][k]:
            print("      "+l)

    print("")
    # print other create functions
    for j in sorted(vclasses[i]):
      if not j in std_methods and not j in fnd and j.startswith('Create'):
          print("  "+j+"()")
          for k in vclasses[i][j]:
            print("    "+k)

    print("")
    # print general functions
    for j in sorted(vclasses[i]):
      if not j in std_methods and not j in fnd and not j.startswith('Create'):
          print("  "+j+"()")
          for k in vclasses[i][j]:
            print("    "+k)

#
# The classes in this file can also be used in other python-programs by using 'import'
#
if __name__ ==  '__main__':
  main(sys.argv[1:])

