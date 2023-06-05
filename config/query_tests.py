#!/usr/bin/env python3
import fnmatch
import glob
import inspect
import os
import optparse
import pickle
import re
import sys

thisfile = os.path.abspath(inspect.getfile(inspect.currentframe()))
pdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(thisfile)))))
sys.path.insert(0, os.path.join(pdir, 'config'))

import testparse
from gmakegentest import nameSpace


"""
  Tool for querying the tests.

  Which tests to query?  Two options:
      1. Query only the tests that are run for a given configuration.
      2. Query all of the test files in the source directory
  For #1:
     Use dataDict as written out by gmakegentest.py in $PETSC_ARCH/$TESTBASE
  For #2:
     Walk the entire tree parsing the files as we go along using testparse.
     The tree walker is simpler than what is in gmakegentest.py

  The dataDict follows that generated by testparse.  gmakegentest.py does
  further manipulations of the dataDict to handle things like for loops
  so if using #2, those modifications are not included.

  Querying:
      The dataDict dictionary is then "inverted" to create a dictionary with the
      range of field values as keys and list test names as the values.  This
      allows fast searching

"""

def isFile(maybeFile):
  ext=os.path.splitext(maybeFile)[1]
  if not ext: return False
  if ext not in ['.c','.cxx','.cpp','F90','F','cu']: return False
  return True

def pathToLabel(path):
  """
  Because the scripts have a non-unique naming, the pretty-printing
  needs to convey the srcdir and srcfile.  There are two ways of doing this.
  """
  # Strip off any top-level directories or spaces
  path=path.strip().replace(pdir,'')
  path=path.replace('src/','')
  if isFile(path):
    prefix=os.path.dirname(path).replace("/","_")
    suffix=os.path.splitext(os.path.basename(path))[0]
    label=prefix+"-"+suffix+'_*'
  else:
    path=path.rstrip('/')
    label=path.replace("/","_").replace('tests_','tests-').replace('tutorials_','tutorials-')
  return label

def get_value(varset):
  """
  Searching args is a bit funky:
  Consider
      args:  -ksp_monitor_short -pc_type ml -ksp_max_it 3
  Search terms are:
    ksp_monitor, 'pc_type ml', ksp_max_it
  Also ignore all loops
    -pc_fieldsplit_diag_use_amat {{0 1}}
  Gives: pc_fieldsplit_diag_use_amat as the search term
  Also ignore -f ...  (use matrices from file) because I'll assume
   that this kind of information isn't needed for testing.  If it's
   a separate search than just grep it
  """
  if varset.startswith('-f '): return None

  # First  remove loops
  value=re.sub('{{.*}}','',varset)
  # Next remove -
  value=varset.lstrip("-")
  # Get rid of numbers
  value=re.sub(r"[+-]? *(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?",'',value)
  # return without spaces
  return value.strip()

def query(invDict,fields,labels):
    """
    Search the keys using fnmatch to find matching names and return list with
    the results 
    """
    setlist=[]  # setlist is a list of lists that set operations will operate on
    llist=labels.replace('|',',').split(',')
    i=-1
    for field in fields.replace('|',',').split(','):
        i+=1
        label=llist[i]
        if field == 'name':
            if '/' in label: 
              label=pathToLabel(label)
            elif label.startswith('src'):
                  label=label.lstrip('src').lstrip('*')
            setlist.append(fnmatch.filter(invDict['name'],label))
            continue

        foundLabel=False   # easy to do if you misspell argument search
        label=label.lower()
        for key in invDict[field]:
            if fnmatch.filter([key.lower()],label):
              foundLabel=True
              # Do not return values with not unless label itself has not
              if label.startswith('!') and not key.startswith('!'): continue
              if not label.startswith('!') and key.startswith('!'): continue
              setlist.append(invDict[field][key])
        if not foundLabel:
          setlist.append([])

    # Now process the union and intersection operators based on setlist
    allresults=[]
    # Union
    i=-1
    for ufield in fields.split(','):
       i+=1
       if '|' in ufield:
         # Intersection
         label=llist[i]
         results=set(setlist[i])
         for field in ufield.split('|')[1:]:
             i+=1
             label=llist[i]
             results=results.intersection(set(setlist[i]))
         allresults+=list(results)
       else:
         allresults+=setlist[i]

    # remove duplicate entries and sort to give consistent results
    uniqlist=list(set(allresults))
    uniqlist.sort()
    return  uniqlist

def get_inverse_dictionary(dataDict,fields,srcdir):
    """
    Create a dictionary with the values of field as the keys, and the name of
    the tests as the results.
    """
    invDict={}
    # Comma-delimited lists denote union
    for field in fields.replace('|',',').split(','):
        if field not in invDict:
            if field == 'name':
                 invDict[field]=[]   # List for ease
            else:
                 invDict[field]={}
        for root in dataDict:
          for exfile in dataDict[root]:
            for test in dataDict[root][exfile]:
              if test in testparse.buildkeys: continue
              defroot = testparse.getDefaultOutputFileRoot(test)
              fname=nameSpace(defroot,os.path.relpath(root,srcdir))
              if field == 'name':
                  invDict['name'].append(fname)
                  continue
              if field not in dataDict[root][exfile][test]: continue
              values=dataDict[root][exfile][test][field]

              if not field == 'args' and not field == 'diff_args':
                for val in values.split():
                    if val in invDict[field]:
                        invDict[field][val].append(fname)
                    else:
                        invDict[field][val] = [fname]
              else:
                # Args are funky.  
                for varset in re.split(r'(^|\W)-(?=[a-zA-Z])',values):
                  val=get_value(varset)
                  if not val: continue
                  if val in invDict[field]:
                    invDict[field][val].append(fname)
                  else:
                    invDict[field][val] = [fname]
        # remove duplicate entries (multiple test/file)
        if not field == 'name':
          for val in invDict[field]:
            invDict[field][val]=list(set(invDict[field][val]))

    return invDict

def get_gmakegentest_data(testdir,petsc_dir,petsc_arch):
    """
     Write out the dataDict into a pickle file
    """
    # This needs to be consistent with gmakegentest.py of course
    pkl_file=os.path.join(testdir,'datatest.pkl')
    # If it doesn't exist, then we need to regenerate
    if not os.path.exists(pkl_file):
      startdir=os.path.abspath(os.curdir)
      os.chdir(petsc_dir)
      args='--petsc-dir='+petsc_dir+' --petsc-arch='+petsc_arch+' --testdir='+testdir
      buf = os.popen('config/gmakegentest.py '+args).read()
      os.chdir(startdir)

    fd = open(pkl_file, 'rb')
    dataDict=pickle.load(fd)
    fd.close()
    return dataDict

def walktree(top):
    """
    Walk a directory tree, starting from 'top'
    """
    verbose = False
    dataDict = {}
    alldatafiles = []
    for root, dirs, files in os.walk(top, topdown=False):
        if root == 'output': continue
        if '.dSYM' in root: continue
        if verbose: print(root)

        dataDict[root] = {}

        for exfile in files:
            # Ignore emacs files
            if exfile.startswith("#") or exfile.startswith(".#"): continue
            ext=os.path.splitext(exfile)[1]
            if ext[1:] not in ['c','cxx','cpp','cu','F90','F']: continue

            # Convenience
            fullex = os.path.join(root, exfile)
            if verbose: print('   --> '+fullex)
            dataDict[root].update(testparse.parseTestFile(fullex, 0))

    return dataDict

def do_query(use_source, startdir, srcdir, testdir, petsc_dir, petsc_arch,
             fields, labels, searchin):
    """
    Do the actual query
    This part of the code is placed here instead of main()
    to show how one could translate this into ipython/jupyer notebook
    commands for more advanced queries
    """
    # Get dictionary
    if use_source:
        dataDict=walktree(startdir)
    else:
        dataDict=get_gmakegentest_data(testdir, petsc_dir, petsc_arch)

    # Get inverse dictionary for searching
    invDict=get_inverse_dictionary(dataDict, fields, srcdir)

    # Now do query
    resList=query(invDict, fields, labels)

    # Filter results using searchin
    newresList=[]
    if searchin.strip():
        if not searchin.startswith('!'):
            for key in resList:
                if fnmatch.filter([key],searchin):
                  newresList.append(key)
        else:
            for key in resList:
                if not fnmatch.filter([key],searchin[1:]):
                  newresList.append(key)
        resList=newresList

    # Print in flat list suitable for use by gmakefile.test
    print(' '.join(resList))

    return

def expand_path_like(petscdir,petscarch,pathlike):
    def remove_prefix(text,prefix):
        return text[text.startswith(prefix) and len(prefix):]

    # expand user second, as expandvars may insert a '~'
    string = os.path.expanduser(os.path.expandvars(pathlike))
    # if the dirname check succeeds then likely we have a glob expression
    pardir = os.path.dirname(string)
    if os.path.exists(pardir):
        suffix   = string.replace(pardir,'') # get whatever is left over
        pathlike = remove_prefix(os.path.relpath(os.path.abspath(pardir),petscdir),'.'+os.path.sep)
        if petscarch == '':
            pathlike = pathlike.replace(os.path.sep.join(('share','petsc','examples'))+'/','')
        pathlike += suffix
    return pathlike

def main():
    parser = optparse.OptionParser(usage="%prog [options] field match_pattern")
    parser.add_option('-s', '--startdir', dest='startdir',
                      help='Where to start the recursion if not srcdir',
                      default='')
    parser.add_option('-p', '--petsc-dir', dest='petsc_dir',
                      help='Set PETSC_DIR different from environment',
                      default=os.environ.get('PETSC_DIR'))
    parser.add_option('-a', '--petsc-arch', dest='petsc_arch',
                      help='Set PETSC_ARCH different from environment',
                      default=os.environ.get('PETSC_ARCH'))
    parser.add_option('--srcdir', dest='srcdir',
                      help='Set location of sources different from PETSC_DIR/src.  Must be full path.',
                      default='src')
    parser.add_option('-t', '--testdir', dest='testdir',  
                      help='Test directory if not PETSC_ARCH/tests.  Must be full path',
                      default='tests')
    parser.add_option('-u', '--use-source', action="store_false",
                      dest='use_source',
                      help='Query all sources rather than those configured in PETSC_ARCH')
    parser.add_option('-i', '--searchin', dest='searchin',
                      help='Filter results from the arguments',
                      default='')

    opts, args = parser.parse_args()

    # Argument Sanity checks
    if len(args) != 2:
        parser.print_usage()
        print('Arguments: ')
        print('  field:          Field to search for; e.g., requires')
        print('                  To just match names, use "name"')
        print('  match_pattern:  Matching pattern for field; e.g., cuda')
        return

    def shell_unquote(string):
      """
      Remove quotes from STRING. Useful in the case where you need to bury escaped quotes in a query
      string in order to escape shell characters. For example:

      $ make test query='foo,bar' queryval='requires|name'
      /usr/bin/bash: line 1: name: command not found

      While the original shell does not see the pipe character, the actual query is done via a second
      shell, which is (literally) passed '$(queryval)', i.e. 'queryval='requires|name'' when expanded.
      Note the fact that the expansion cancels out the quoting!!!

      You can fix this by doing:

      $ make test query='foo,bar' queryval='"requires|name"'

      However this then shows up here as labels = 'queryval="requires|name"'. So we need to remove the
      '"'. Applying shlex.split() on this returns:

      >>> shlex.split('queryval="requires|name"')
      ['queryval=requires|name']

      And voila. Note also that:

      >>> shlex.split('queryval=requires|name')
      ['queryval=requires|name']
      """
      import shlex

      if string:
        ret = shlex.split(string)
        assert len(ret) == 1, "Dont know what to do if shlex.split() produces more than 1 value?"
        string = ret[0]
      return string

    def alternate_command_preprocess(string):
      """
      Replace the alternate versions in STRING with the regular variants
      """
      return string.replace('%OR%', '|').replace('%AND%', ',').replace('%NEG%', '!')

    # Process arguments and options -- mostly just paths here
    field=alternate_command_preprocess(shell_unquote(args[0]))
    match=alternate_command_preprocess(shell_unquote(args[1]))
    searchin=opts.searchin

    petsc_dir = opts.petsc_dir
    petsc_arch = opts.petsc_arch
    petsc_full_arch = os.path.join(petsc_dir, petsc_arch)

    if petsc_arch == '':
        petsc_full_src = os.path.join(petsc_dir, 'share', 'petsc', 'examples', 'src')
    else:
      if opts.srcdir == 'src':
        petsc_full_src = os.path.join(petsc_dir, 'src')
      else:
        petsc_full_src = opts.srcdir
    if opts.testdir == 'tests':
      petsc_full_test = os.path.join(petsc_full_arch, 'tests')
    else:
      petsc_full_test = opts.testdir
    if opts.startdir:
      startdir=opts.startdir=petsc_full_src
    else:
      startdir=petsc_full_src

    # Options Sanity checks
    if not os.path.isdir(petsc_dir):
        print("PETSC_DIR must be a directory")
        return

    if not opts.use_source:
        if not os.path.isdir(petsc_full_arch):
            print("PETSC_DIR/PETSC_ARCH must be a directory")
            return
        elif not os.path.isdir(petsc_full_test):
            print("Testdir must be a directory"+petsc_full_test)
            return
    else:
        if not os.path.isdir(petsc_full_src):
            print("Source directory must be a directory"+petsc_full_src)
            return

    match = expand_path_like(petsc_dir,petsc_arch,match)

    # Do the actual query
    do_query(opts.use_source, startdir, petsc_full_src, petsc_full_test,
             petsc_dir, petsc_arch, field, match, searchin)

    return


if __name__ == "__main__":
        main()
