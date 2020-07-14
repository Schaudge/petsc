import config.package
import os

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    config.package.CMakePackage.__init__(self, framework)
    self.gitcommit         = 'master'
    self.download          = ['git://https://github.com/libaxb/libaxb-dev']
    self.includes          = ['libaxb.h']
    self.liblist           = [['libaxb.a']]
    self.precisions        = ['double']
    self.functions         = ['axbInit']
    self.downloadonWindows = 1
    self.complex           = 0
    return

  def setupDependencies(self, framework):
    config.package.CMakePackage.setupDependencies(self, framework)
    self.compilerFlags   = framework.require('config.compilerFlags', self)
    self.sharedLibraries = framework.require('PETSc.options.sharedLibraries', self)
    self.scalartypes     = framework.require('PETSc.options.scalarTypes',self)
    self.indexTypes      = framework.require('PETSc.options.indexTypes', self)
    self.setCompilers    = framework.require('config.setCompilers',self)
    self.installdir      = framework.require('PETSc.options.installDir',self)
    self.cuda            = framework.require('config.packages.cuda',self)
    self.opencl          = framework.require('config.packages.opencl',self)
    self.openmp          = framework.require('config.packages.openmp',self)
    self.deps            = []
    self.odeps           = [self.cuda,self.opencl,self.openmp]
    return

  def formCMakeConfigureArgs(self):
    if not self.cmake.found:
      raise RuntimeError('CMake > 2.5 is needed to build libaxb')

    args = config.package.CMakePackage.formCMakeConfigureArgs(self)
    if not self.compilerFlags.debugging:
      args.append('-DCMAKE_BUILD_TYPE=Release')
    if self.checkSharedLibrariesEnabled():
      args.append('-DCMAKE_INSTALL_RPATH_USE_LINK_PATH:BOOL=ON')


    if self.cuda.found:
      args.append('-DENABLE_CUDA=1')
    if self.opencl.found:
      args.append('-DENABLE_OPENCL=1')
    if self.openmp.found:
      args.append('-DENABLE_OPENMP=1')

    return args

  def consistencyChecks(self):
    config.package.Package.consistencyChecks(self)
    return