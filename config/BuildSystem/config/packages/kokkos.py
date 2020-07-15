import config.package
import os

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    config.package.CMakePackage.__init__(self, framework)
    self.gitcommit        = '3.1.01'
    self.versionname      = 'KOKKOS_VERSION'
    self.download         = ['git://https://github.com/kokkos/kokkos.git']
    self.downloaddirnames = ['kokkos']
    self.includes         = ['Kokkos_Macros.hpp']
    self.liblist          = [['libkokkoscontainers.a','libkokkoscore.a']]
    #self.functions        = ['']
    self.cxx              = 1
    self.requirescxx11    = 1
    self.downloadonWindows= 0
    self.hastests         = 1
    self.requiresrpath    = 1
    self.precisions       = ['double']
    return

  def setupDependencies(self, framework):
    config.package.CMakePackage.setupDependencies(self, framework)
    self.compilerFlags   = framework.require('config.compilerFlags', self)
    self.blasLapack      = framework.require('config.packages.BlasLapack',self)
    self.mpi             = framework.require('config.packages.MPI',self)
    self.flibs           = framework.require('config.packages.flibs',self)
    self.cxxlibs         = framework.require('config.packages.cxxlibs',self)
    self.mathlib         = framework.require('config.packages.mathlib',self)
    self.deps            = [self.mpi,self.blasLapack,self.flibs,self.cxxlibs,self.mathlib]
    self.openmp          = framework.require('config.packages.openmp',self)
    self.hwloc           = framework.require('config.packages.hwloc',self)
    self.pthread         = framework.require('config.packages.pthread',self)
    self.mpi             = framework.require('config.packages.MPI',self)
    self.odeps           = [self.openmp,self.hwloc]
    return

  def versionToStandardForm(self,ver):
    '''Converts from kokkos 30101 notation to standard notation 3.1.01'''
    return ".".join(map(str,[int(ver)//10000, int(ver)//100%100, int(ver)%100]))

  # duplicate from Trilinos.py
  def toString(self,string):
    string    = self.libraries.toString(string)
    if self.requiresrpath: return string
    newstring = ''
    for i in string.split(' '):
      if i.find('-rpath') == -1:
        newstring = newstring+' '+i
    return newstring.strip()

  def formCMakeConfigureArgs(self):
    args = config.package.CMakePackage.formCMakeConfigureArgs(self)
    args.append('-DUSE_XSDK_DEFAULTS=YES')
    if self.compilerFlags.debugging:
      args.append('-DCMAKE_BUILD_TYPE=DEBUG')
    else:
      args.append('-DCMAKE_BUILD_TYPE=RELEASE')
      args.append('-DXSDK_ENABLE_DEBUG=NO')

    # Trilinos cmake does not set this variable (as it should) so cmake install does not properly reset the -id and rpath of --prefix installed Trilinos libraries
    args.append('-DCMAKE_INSTALL_NAME_DIR:STRING="'+os.path.join(self.installDir,self.libdir)+'"')

    if self.mpi.found:
      args.append('-DKokkos_ENABLE_MPI=ON')

    if self.hwloc.found:
      args.append('-DKokkos_ENABLE_HWLOC=ON')
      args.append('-DKokkos_HWLOC_DIR='+self.hwloc.directory)

    # looks for pthread by default so need to turn it off unless specifically requested
    pthreadfound = self.pthread.found
    if not 'with-pthread' in self.framework.clArgDB:
      pthreadfound = 0

    if self.openmp.found and pthreadfound:
      raise RuntimeError("Kokkos only supports a single parallel system during its configuration")

    args.append('-DKokkos_ENABLE_SERIAL=ON')
    if self.openmp.found:
      args.append('-DKokkos_ENABLE_OPENMP=ON')
    if pthreadfound:
      args.append('-DKokkos_ENABLE_PTHREAD=ON')

    return args


