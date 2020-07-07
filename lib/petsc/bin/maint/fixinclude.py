#!/usr/bin/env python

import sys, os, re, codecs

# Used by the makefile lib/petsc/conf/rules html to replace #includes in the source files with links when possible

if __name__ == "__main__":
  if (len(sys.argv) < 3): sys.exit(1)
  filename = sys.argv[1]
  petscdir = sys.argv[2]
  root = os.path.relpath(os.path.realpath(petscdir),os.path.realpath(os.getcwd()))
  froot = os.path.relpath(os.path.realpath(petscdir),os.path.dirname(os.path.realpath(filename)))

  # using sys.stdin produced UnicodeDecodeError: 'utf-8' codec can't decode byte 0x88 in position 7892: invalid start byte on Barry's Mac
  with codecs.open('/dev/stdin','r',encoding='utf-8',errors='replace') as fd:
    for line in fd:
      if 'include' in line:
        ins = re.search('#include [ ]*&lt;',line)
        if ins:
          includename = line[ins.end():re.search('&gt;[a-zA-Z0-9/<>#*"=. ]*',line).start()]
        else:
          ins = re.search('#include [ ]*"',line)
          if ins:
            includename = line[ins.end():]
            ins = re.search('"',includename)
            includename = includename[:ins.end()-1]
          else:
            includename = ''
        if includename:
          ln = re.search('<a name="line[0-9]*">[ 0-9]*: </a>',line)
          if ln:
            linenumber = line[ln.start():ln.end()]
          else:
            linenumber = ''
          if os.path.isfile(includename):
            sys.stdout.write(linenumber+'#include <A href="'+includename+'.html">&lt;'+includename+'&gt;</A>\n')
          elif os.path.isfile(os.path.join(root,'include',includename)):
            sys.stdout.write(linenumber+'#include <A href="'+os.path.relpath(os.path.join(froot,'include',includename))+'.html">&lt;'+includename+'&gt;</A>\n')
          elif os.path.isfile(os.path.join(root,includename)):
            sys.stdout.write(linenumber+'#include <A href="'+os.path.relpath(os.path.join(froot,includename))+'.html">&lt;'+includename+'&gt;</A>\n')
          else:
            sys.stdout.write(line)
        else:
          sys.stdout.write(line)
      else:
        sys.stdout.write(line)

