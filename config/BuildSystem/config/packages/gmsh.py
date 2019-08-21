import config.package
import os

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    config.package.CMakePackage.__init__(self, framework)
    self.gitcommit         = 'master'
    self.download          = ['git://https://gitlab.onelab.info/gmsh/gmsh.git']
    self.functions         = []
    self.includes          = []
    self.liblist           = [[]]
    self.linkedbypetsc     = 0
    self.useddirectly      = 0
    return

  def setupDependencies(self, framework):
    config.package.CMakePackage.setupDependencies(self, framework)
    self.blasLapack    = framework.require('config.packages.BlasLapack',self)
    self.med           = framework.require('config.packages.med',self)
    self.hdf5          = framework.require('config.packages.hdf5',self)
    self.compilerFlags = framework.require('config.compilerFlags', self)
    self.mathlib       = framework.require('config.packages.mathlib',self)
    self.zlib          = framework.require('config.packages.zlib',self)
    self.szlib         = framework.require('config.packages.szlib',self)
    self.gmp           = framework.require('config.packages.gmp',self)
    self.deps          = [ self.zlib, self.szlib, self.mathlib]
    return

  def formCMakeConfigureArgs(self):
    if self.cmake.version_tuple < (2,8,0): raise RuntimeError('Gmsh requires cmake version 2.8 or higher')
    args = config.package.CMakePackage.formCMakeConfigureArgs(self)
    args.append('-DBLAS_LAPACK_LIBRARIES="'+self.libraries.toString(self.blasLapack.dlib)+'"')
    if self.hdf5.found:
      args.append('-DHDF5_ROOT="'+self.hdf5.directory+'"')
    if self.med.found:
      if not self.hdf5.found: raise RuntimeError('GMSH using med requires also HDF5, perhaps you need --download-hdf5')
      args.append('-DENABLE_MED:BOOL=ON')
      args.append('-DMED_LIB='+os.path.join(self.med.directory,'lib','libmedC.a'))
    args.append('-DSZ_LIB='+os.path.join(self.szlib.directory,'lib','libsz.a'))
    args.append('-DZLIB_ROOT="'+self.zlib.directory+'"')
    args.append('-DGMP_LIB='+os.path.join(self.gmp.directory,'lib','libgmp.a'))
    args.append('-DGMP_INCLUDE='+self.gmp.directory)
    return args

  def checkVersion(self):
    self.getExecutable('gmsh', os.path.join(self.directory,'bin'), getFullPath = 1)
    try:
      (output, error, status) = config.base.Configure.executeShellCommand(self.gmsh+' -version', log = self.log)
      if status:
        self.log.write('gmsh --version failed: '+str(e)+'\n')
        return
    except:
      self.log.write('gmsh --version failed: '+str(e)+'\n')
      return
    gver = None
    try:
      import re
      gver = re.compile('([0-9]+).([0-9]+).([0-9]+)').match(error)
    except:
      self.log.write('gmsh search for version information failed')
      return
    if gver:
      try:
         self.foundversion = ".".join(gver.groups())
         self.version_tuple = self.versionToTuple(self.foundversion)
         self.log.write('gmsh version found '+self.foundversion+'\n')
         return
      except:
        self.log.write('gmsh cannot find version information\n')