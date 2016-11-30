#!/usr/bin/env python
import glob
import sys
import re
import os
import stat
import types
import optparse
import string
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0,currentdir) 
from examplesMkParse import *

"""
Quick start
-----------

Architecture independent (just analyze makefiles):

  bin/maint/examplesWalker.py -f examplesAnalyze src

     Show tests source files and tests as organized by type (TESTEXAMPLES*), 
     Output: 
        MakefileAnalysis-tests.txt      MakefileSummary-tests.txt
        MakefileAnalysis-tutorials.txt  MakefileSummary-tutorials.txt

  bin/maint/examplesWalker.py -f examplesConsistency src
     Show consistency between the EXAMPLES* variables and TESTEXAMPLES* variables
     Output: 
        ConsistencyAnalysis-tests.txt      ConsistencySummary-tests.txt
        ConsistencyAnalysis-tutorials.txt  ConsistencySummary-tutorials.txt

Architecture dependent (need to modify files and run tests - see below):

  bin/maint/examplesWalker.py -f getFileSizes src
     Give the example file sizes for the tests (see below)
     Output:  FileSizes.txt

  bin/maint/examplesWalker.py -m <output of make alltests>
     Show examples that are run
     Output:  
        RunArchAnalysis-tests.txt      RunArchSummary-tests.txt
        RunArchAnalysis-tutorials.txt  RunArchSummary-tutorials.txt

Description
------------------

This is a simple os.walk through the examples with various 
"actions" that can be performed on them.  Each action 
is to fill a dictionary as its output.  The signature of
the action is::

   action(root,dirs,files,dict)

For every action, there is an equivalent <action>_summarize
that then takes the dictionary and either prints or 
writes to a file.  The signature is::

   action_summarize(dict)

Currently the actions are: 
    printFiles
      -- Just a simple routine to demonstrate the walker
    examplesAnalyze
      -- Go through summarize the state of the examples
      -- Does so in an architecture independent way
    examplesConsistency
      -- Go through makefiles and see if the documentation
         which uses EXAMPLES* is consistent with what is tested
         by TESTEXAMPLES*
    getFileSizes
      -- After editing lib/petsc/conf/rules to redefine RM (see below),
         you can keep the executables around.  Then specify this
         action to calculate the file sizes of those executables

One other mode:
  bin/maint/examplesWalker.py -m <output of make alltests>
      -- If lib/petsc/conf/tests is modified to turn on output
         the tests run, then one can see the tests are written
         out to the summary.  We then process that output and 
         summarize the results (using examplesAnalyze_summarize)
"""

class PETScExamples(makeParse):
  def __init__(self,petsc_dir,replaceSource,verbosity=1):
    super(PETScExamples, self).__init__(petsc_dir,replaceSource,verbosity)
    return

  def getCorrespondingKeys(self,extype):
    """
    Which TESTEX* vars should we look for given an EXAMPLE* variable
    """
    if extype=='EXAMPLESC':
      return ['TESTEXAMPLES_C','TESTEXAMPLES_C_X']
    elif extype=='EXAMPLESF':
      return ['TESTEXAMPLES_F','TESTEXAMPLES_FORTRAN']
    else:
      raise "Error: Do not know extype "+extype
    return

  def getMatchingKeys(self,extype,mkDict):
    """
     Given EXAMPLES*, see what matching keys are in the dict
    """
    mtchList=[]
    for ckey in self.getCorrespondingKeys(extype):
      if mkDict.has_key(ckey): mtchList.append(ckey)
    return mtchList

  def examplesConsistency(self,root,dirs,files,dataDict):
    """
     Documentation for examples is generated by what is in
     EXAMPLESC and EXAMPLESF variables (see lib/petsc/conf/rules)
     This goes through and compares what is in those variables
     with what is in corresponding TESTEXAMPLES_* variables
    """
    debug=False
    # Go through and parse the makefiles
    fullmake=os.path.join(root,"makefile")
    fh=open(fullmake,"r")
    dataDict[fullmake]={}
    i=0
    searchTypes=[]
    for stype in ['EXAMPLESC','EXAMPLESF']:
      searchTypes.append(stype)
      searchTypes=searchTypes+self.getCorrespondingKeys(stype)
    if debug: print fullmake
    while 1:
      line=fh.readline()
      if not line: break
      if not "=" in line:  continue  # Just looking at variables
      var=line.split("=")[0].strip()
      if " " in var: continue        # eliminate bash commands that appear as variables
      if debug: print "  "+var
      if var in searchTypes:
        parseDict=self.parseline(fh,line,root)
        if debug: print "  "+var, sfiles
        if len(parseDict['srcs'])>0: dataDict[fullmake][var]=parseDict
        continue
    fh.close()
    #print root,files
    return

  def printFiles_summarize(self,dataDict):
    """
     Simple example of an action
    """
    for root in dataDict:
      print root+": "+" ".join(dataDict[root])
    return

  def printFiles(self,root,dirs,files,dataDict):
    """
     Simple example of an action
    """
    dataDict[root]=files
    return

  def getFileSizes_summarize(self,dataDict):
    """
     Summarize the file sizes
    """
    fh=open("FileSizes.txt","w")
    totalSize=0
    nfiles=0
    toMBorKB=1./1024.0
    for root in dataDict:
      for f in dataDict[root]:
        size=dataDict[root][f]*toMBorKB
        fh.write(f+": "+ "%.1f" % size +" KB\n")
        totalSize=totalSize+size
        nfiles=nfiles+1
    totalSizeMB=totalSize*toMBorKB
    fh.write("----------------------------------------\n")
    fh.write("totalSize = "+ "%.1f" % size +" MB\n")
    fh.write("Number of execuables = "+str(nfiles)+"\n")
    print "See: FileSizes.txt"
    return

  def getFileSizes(self,root,dirs,files,dataDict):
    """
     If you edit this file:
       lib/petsc/conf/rules
      and add at the bottom:
       RM=echo
     Then you will leave the executables in place.
     Once they are in place, run this script and you will get a summary of
     the file sizes
    """
    # Find executables
    xFiles={}
    for fname in files:
      f=os.path.join(root,fname)
      if os.access(f,os.X_OK):
        xFiles[f]=os.path.getsize(f)
    if len(xFiles.keys())>0: dataDict[root]=xFiles
    return

  def examplesConsistency_summarize(self,dataDict):
    """
     Go through makefile and see where examples 
    """
    indent="  "
    for type in ["tutorials","tests"]:
      fh=open("ConsistencyAnalysis-"+type+".txt","w")
      gh=open("ConsistencySummary-"+type+".txt","w")
      nallsrcs=0; nalltsts=0
      for mkfile in dataDict:
        if not type in mkfile: continue
        fh.write(mkfile+"\n")
        gh.write(mkfile+"\n")
        for extype in ['EXAMPLESC','EXAMPLESF']:
          matchKeys=self.getMatchingKeys(extype,dataDict[mkfile])
          # Check to see if this mkfile even has types
          if not dataDict[mkfile].has_key(extype):
            if len(matchKeys)>0:
              foundKeys=" ".join(matchKeys)
              fh.write(indent*2+foundKeys+" found BUT "+extype+"not found\n")
              gh.write(indent*2+extype+"should be documented here\n")
            else:
              continue # Moving right along
          # We have the key
          else:
            if len(matchKeys)==0:
              fh.write(indent*2+extype+" found BUT no corresponding types found\n")
              gh.write(indent*2+extype+" is documented without testing under any PETSC_ARCH\n")
              continue # Moving right along
            matchList=[]; allTests=[]
            for mkey in matchKeys:
              matchList=matchList+dataDict[mkfile][mkey]['srcs']
              allTests=allTests  +dataDict[mkfile][mkey]['tsts']
            fh.write(indent+extype+"\n")
            nsrcs=0
            for exfile in dataDict[mkfile][extype]['srcs']:
               if exfile in matchList: 
                 matchList.remove(exfile)
                 testList=self.findTests(exfile,allTests)
                 ntests=len(testList)
                 if ntests==0:
                   fh.write(indent*2+exfile+" found BUT no tests found\n")
                 else:
                   tests=" ".join(testList)
                   fh.write(indent*2+exfile+" found with these tests: "+tests+"\n")
               else:
                 nsrcs=nsrcs+1
                 fh.write(indent*2+"NOT found in TEST*: "+exfile+"\n")
            lstr=" files are documented without testing under any PETSC_ARCH\n"
            if nsrcs>0: gh.write(indent*2+extype+": "+str(nsrcs)+lstr)
            nsrcs=0
            for mtch in matchList:
              fh.write(indent*2+"In TEST* but not EXAMPLE*: "+mtch+"\n")
              nsrcs=nsrcs+1
            lstr=" files have undocumented tests\n"
            if nsrcs>0: gh.write(indent*2+extype+": "+str(nsrcs)+lstr)
            fh.write("\n")
        fh.write("\n"); gh.write("\n")
      fh.close()
      gh.close()
    #print dataDict
    for type in ["tutorials","tests"]:
      print "ConsistencyAnalysis-"+type+".txt"
      print "ConsistencySummary-"+type+".txt"
    return

  def examplesAnalyze_summarize(self,dataDict):
    """
     Write out files that are from the result of either the makefiles
     analysis (default with walker) or from run output.
     The run output has a special dictionary key to differentiate the
     output
    """
    indent="  "
    if dataDict.has_key("outputname"):
      baseName=dataDict["outputname"]
    else:
      baseName="Makefile"
    for type in ["tutorials","tests"]:
      fh=open(baseName+"Analysis-"+type+".txt","w")
      gh=open(baseName+"Summary-"+type+".txt","w")
      nallsrcs=0; nalltsts=0
      for mkfile in dataDict:
        if not type in mkfile: continue
        nsrcs=0; ntsts=0
        fh.write(mkfile+"\n")
        runexFiles=dataDict[mkfile]['runexFiles']
        for extype in dataDict[mkfile]:
          if extype == "runexFiles": continue
          fh.write(indent+extype+"\n")
          allTests=dataDict[mkfile][extype]['tsts']
          for exfile in dataDict[mkfile][extype]['srcs']:
             nsrcs=nsrcs+1
             testList=self.findTests(exfile,allTests)
             ntests=len(testList)
             ntsts=ntsts+ntests
             if ntests==0:
               fh.write(indent*2+exfile+": No tests found\n")
             else:
               tests=" ".join(testList)
               fh.write(indent*2+exfile+": "+tests+"\n")
               for t in testList: 
                 if t in runexFiles: runexFiles.remove(t)
          fh.write("\n")
        nrunex=len(runexFiles)
        if nrunex>0:
         runexStr=" ".join(runexFiles)
         fh.write(indent+"RUNEX SCRIPTS NOT USED: "+runexStr+"\n")
        fh.write("\n")
        sumStr=str(nsrcs)+" srcfiles; "+str(ntsts)+" tests"
        if nrunex>0: sumStr=sumStr+"; "+str(nrunex)+" tests not used"
        gh.write(mkfile+": "+sumStr+"\n")
        nallsrcs=nallsrcs+nsrcs; nalltsts=nalltsts+ntsts
      fh.close()
      gh.write("-----------------------------------\n")
      gh.write("Total number of sources: "+str(nallsrcs)+"\n")
      gh.write("Total number of tests:   "+str(nalltsts)+"\n")
      gh.close()
    #print dataDict
    for type in ["tutorials","tests"]:
      print "See: "+baseName+"Analysis-"+type+".txt"
      print "See: "+baseName+"Summary-"+type+".txt"
    return

  def examplesAnalyze(self,root,dirs,files,dataDict):
    """
     Go through makefile and see what examples and tests are listed
     Dictionary structure is of the form:
       dataDict[makefile]['srcs']=sourcesList
       dataDict[makefile]['tsts']=testsList
    """
    debug=False
    # Go through and parse the makefiles
    fullmake=os.path.join(root,"makefile")
    fh=open(fullmake,"r")
    dataDict[fullmake]={}
    i=0
    allRunex=[]
    if debug: print fullmake
    while 1:
      line=fh.readline()
      if not line: break
      if line.startswith("runex"): allRunex.append(line.split(":")[0])
      if not "=" in line:  continue  # Just looking at variables
      var=line.split("=")[0].strip()
      if " " in var: continue        # eliminate bash commands that appear as variables
      if var.startswith("TESTEX"):
        if debug: print "  >", var
        parseDict=self.parseline(fh,line,root)
        if len(parseDict['srcs'])>0: dataDict[fullmake][var]=parseDict
        continue
    dataDict[fullmake]['runexFiles']=allRunex
    fh.close()
    #if "pde_constrained" in root: raise ValueError('Testing')
    #print root,files
    return

  def walktree(self,top,action="printFiles"):
    """
    Walk a directory tree, starting from 'top'
    """
    #print "action", action
    # Goal of action is to fill this dictionary
    dataDict={}
    for root, dirs, files in os.walk(top, topdown=False):
      if not "examples" in root: continue
      if not os.path.isfile(os.path.join(root,"makefile")): continue
      if root.endswith("tests") or root.endswith("tutorials"):
        eval("self."+action+"(root,dirs,files,dataDict)")
      if type(top) != types.StringType:
          raise TypeError("top must be a string")
    # Now summarize this dictionary
    eval("self."+action+"_summarize(dataDict)")
    return dataDict

  def archTestAnalyze(self,makeoutput):
    """
    To use:
      In file: lib/petsc/conf/test

      Change this line:
        ALLTESTS_PRINT_PROGRESS = no
      to
        ALLTESTS_PRINT_PROGRESS = debugtest

      And run:: 
        make PETSC_DIR=$PETSC_DIR PETSC_ARCH=$PETSC_ARCH alltests

      Save the output, and run this script on it (-m output).
      Output will be in 
    """
    fh=open(makeoutput)
    # Dictionary datastructure of same form used in examplesAnalyze
    testDict={}
    while 1:
      line=fh.readline()
      if not line: break
      if "Testing:" in line:
        var=line.split("Testing: ")[1].split()[0].strip().upper()
      if "Running examples in" in line:
        exdir=line.split("Running examples in")[1].strip().rstrip(":")
        # This is the full path which is not import so simplify this:
        parentdir=os.path.basename(exdir.split("/src/")[0])
        exdir=exdir.split(parentdir+"/")[1]
        # foo is because parseline expects VAR=stuff pattern
        line="FOO = "+fh.readline()
        parseDict=self.parseline(fh,line,exdir)
        if not testDict.has_key(exdir): testDict[exdir]={}
        testDict[exdir][var]={}
        testDict[exdir][var]['srcs']=parseDict['srcs']
        testDict[exdir][var]['tsts']=parseDict['tsts']
    # Now that we have our dictionary loaded up, pretty print it
    testDict["outputname"]="RunArch"
    examplesAnalyze_summarize(testDict)
    return

def main():
    parser = optparse.OptionParser(usage="%prog [options] startdir")
    parser.add_option('-s', '--startdir', dest='startdir',
                      help='Where to start the recursion',
                      default='')
    parser.add_option('-f', '--functioneval', dest='functioneval',
                      help='Function to evaluate while traversing example dirs: printFiles default), examplesConsistencyEval', 
                      default='')
    parser.add_option('-m', '--makeoutput', dest='makeoutput',
                      help='Name of make alttests output file',
                      default='')
    options, args = parser.parse_args()

    # Process arguments
    # The makeoutput option is not a walker so just get it over with

    petsc_dir=None
    if options.petsc_dir: petsc_dir=options.petsc_dir
    if petsc_dir is None: petsc_dir=os.path.dirname(os.path.dirname(currentdir))

    pEx=PETScExamples(petsc_dir)

    if not options.makeoutput=='':
      pEx.archTestAnalyze(options.makeoutput)
      return
    # Do the walker processing
    startdir=''
    if len(args) > 1:
      parser.print_usage()
      return
    elif len(args) == 1:
      startdir=args[0]
    else:
      if not options.startdir == '':
        startdir=options.startdir
    if not startdir:
      parser.print_usage()
      return
    if not options.functioneval=='':
      pEx.walktree(startdir,action=options.functioneval)
    else:
      pEx.walktree(startdir)

if __name__ == "__main__":
        main()
