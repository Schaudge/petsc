import config.package
import sys

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.functions        = ['clGetPlatformIDs']
    if sys.platform.startswith('darwin'): # Apple requires special care (OpenCL/cl.h)
      self.includes         = ['OpenCL/cl.h']
    else:
      self.includes         = ['CL/cl.h']
    self.liblist          = [['libOpenCL.a'], ['-framework opencl'], ['libOpenCL.lib']]

  def getSearchDirectories(self):
    import os
    return [os.path.join('/usr','local','cuda')]

  def checkVersion(self):
    '''OpenCL lists all versions in the cl.h include file so select the last one'''
    import re
    flagsArg = self.getPreprocessorFlagsArg()
    oldFlags = getattr(self.compilers, flagsArg)
    setattr(self.compilers, flagsArg, oldFlags+' '+self.headers.toString(self.include))
    try:
      output = self.outputPreprocess('#include "CL/cl.h"')
    except:
      self.log.write('For '+self.package+' unable to run preprocessor to obtain version information, skipping version check\n')
      setattr(self.compilers, flagsArg,oldFlags)
      return
    setattr(self.compilers, flagsArg,oldFlags)
    for i in output.split('\n'):
      found = re.match('#define CL_VERSION_[_0-9]* *',i)
      if found:
        found = found.group(0)
        found = re.sub('#define CL_VERSION_','',found)
        self.version = re.sub('_','.',found.split(' ')[0])
