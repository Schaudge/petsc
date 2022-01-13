import config.package
import os

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    config.package.CMakePackage.__init__(self, framework)
    self.minversion       = '1.3.0'  # Unclear
    self.version          = '1.4.0'
    self.versionname      = 'GINKGO_MAJOR_VERSION.GINKGO_MINOR_VERSION.GINKGO_PATCH_VERSION'
    self.gitcommit        = 'f811917c1def4d0fcd8db3fe5c948ce13409e28e'  # v1.4.0

    self.download         = ['git://https://github.com/ginkgo-project/ginkgo','https://github.com/ginkgo-project/archive/'+self.gitcommit+'.tar.gz']
    self.functionsCxx     = [1,'auto execCPU = gko::OmpExecutor::create();','']
    self.includes         = ['ginkgo.hpp']
    self.includedir       = os.path.join('include','ginkgo')
    self.buildLanguages    = ['Cxx']
    self.minCxxVersion     = 'c++11'
    self.liblist          = [['libginkgo.a']]
    self.downloadonWindows= 1
    self.hastests         = 1
    self.hastestsdatafiles= 1
    self.precisions       = ['double']
    self.cxx              = 1
    self.minCxxVersion    = 'c++14'
    return

  def setupDependencies(self, framework):
    config.package.CMakePackage.setupDependencies(self, framework)
    self.cxxlibs        = framework.require('config.packages.cxxlibs',self)
    self.blasLapack     = framework.require('config.packages.BlasLapack',self)
    self.mathlib        = framework.require('config.packages.mathlib',self)
    self.mpi            = framework.require('config.packages.MPI',self)
    self.cuda           = framework.require('config.packages.cuda',self)
    self.hip            = framework.require('config.packages.hip',self)
    self.compilerFlags  = framework.require('config.compilerFlags', self)
    self.openmp         = framework.require('config.packages.openmp',self)
    self.odeps          = [self.cuda,self.hip,self.openmp]
    self.deps           = [self.mpi,self.blasLapack,self.cxxlibs,self.mathlib]
    return

  def formCMakeConfigureArgs(self):
    args = config.package.CMakePackage.formCMakeConfigureArgs(self)
    # Need to explicitly turn things off so ginkgo doesn't override it
    #This option may be needed BUILD_SHARED_LIBS
    if self.cuda.found:
      args.append('-DGINKGO_BUILD_CUDA=ON')
    else:
      args.append('-DGINKGO_BUILD_CUDA=OFF')
    if self.hip.found:
      args.append('-DGINKGO_BUILD_HIP=OFF')
      # Ginkgo needs hiprand/rocrand which we aren't requiring yet
      #args.append('-DGINKGO_BUILD_HIP=ON')
    else:
      args.append('-DGINKGO_BUILD_HIP=OFF')
    if self.openmp.found:
      args.append('-DGINKGO_BUILD_OMP=ON')
    else:
      args.append('-DGINKGO_BUILD_OMP=OFF')

    args.append('-DGINKGO_BUILD_TESTS=ON')
    args.append('-DGINKGO_BUILD_EXAMPLES=OFF')
    args.append('-DGINKGO_BUILD_BENCHMARKS=OFF')
    args.append('-DGINKGO_BUILD_HWLOC=OFF')         # petsc controls
    args.append('-DMPI_C_COMPILE_FLAGS:STRING=""')
    args.append('-DMPI_C_INCLUDE_PATH:STRING=""')
    args.append('-DMPI_C_HEADER_DIR:STRING=""')
    args.append('-DMPI_C_LIBRARIES:STRING=""')
    return args

  def configureLibrary(self):
    config.package.Package.configureLibrary(self)
    #if self.defaultIndexSize == 64:
    #  if not self.checkCompile('#include "ginkgo.h"','#if !defined(_LONGINT)\n#error "No longint"\n#endif\n'):
    #    raise RuntimeError('PETSc is being configured using --with-64-bit-indices but ginkgo library is built for 32 bit integers.\n\
    #Suggest using --download-ginkgo')
    #else:
    #  if not self.checkCompile('#include "ginkgo.h"','#if defined(_LONGINT)\n#error "longint is defined"\n#endif\n'):
    #    raise RuntimeError('PETSc is being configured without using --with-64-bit-indices but ginkgo library is built for 64 bit integers.\n\
    #Suggest using --download-ginkgo')
