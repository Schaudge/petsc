#!/usr/bin/env python

import sys, os, re, codecs

# Converts from a directory or include file name to the SUBMANSEC 

mapinclude = {"sys" : "SYS",
"ao" : "",
"bag" : "",
"blaslapack" : "",
"blaslapack_mangle" : "",
"bt" : "",
"aracteristic" : "",
"convest" : "",
"ctable" : "",
"cublas" : "",
"cxxcomplexfix" : "",
"dm" : "",
"dmadaptor" : "",
"dmcomposite" : "",
"dmda" : "",
"dmdatypes" : "",
"dmfield" : "",
"dmforest" : "",
"dmlabel" : "",
"dmmoab" : "",
"dmnetwork" : "",
"dmpat" : "",
"dmplex" : "",
"dmplextypes" : "",
"dmproduct" : "",
"dmredundant" : "",
"dmell" : "",
"dmsliced" : "",
"dmstag" : "",
"dmswarm" : "",
"dmtypes" : "",
"draw" : "",
"drawtypes" : "",
"ds" : "",
"dstypes" : "",
"dt" : "",
"error" : "",
"fe" : "",
"fetypes" : "",
"fv" : "",
"fvtypes" : "",
"is" : "",
"istypes" : "",
"ksp" : "",
"layoudf5" : "",
"log" : "",
"mat" : "",
"matcoarsen" : "",
"matelemental" : "",
"ma" : "",
"maypre" : "",
"matlab" : "",
"options" : "",
"pc" : "",
"pctypes" : "",
"pf" : "",
"section" : "",
"sectiontypes" : "",
"sf" : "",
"sftypes" : "",
"snes" : "",
"sys" : "",
"systypes" : "",
"tao" : "",
"taolinesear" : "",
"time" : "",
"ts" : "",
"valgrind" : "",
"vec" : "",
"version" : "",
"viennacl" : "",
"viewer" : "",
"viewerexodusii" : "",
"viewedf5" : "",
"viewersaws" : "",
"viewertypes" : "",
"webclient" : "",
"",
if __name__ == "__main__":
  if (len(sys.argv) < 3): sys.exit(1)
  filename = sys.argv[1]
  petscdir = sys.argv[2]
  root = os.path.realpath(petscdir)
  croot = os.path.realpath(os.path.join(os.getcwd(),filename))
  print(root)
  print(croot)
  path = croot[len(root)+1:-2]
  print(path)
  if path.startswith('include'):
      base = path[13:]
      print(base)
      print(mapinclude[base])
  

