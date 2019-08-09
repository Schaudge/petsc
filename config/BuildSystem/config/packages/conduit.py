import config.package

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    import os
    config.package.CMakePackage.__init__(self, framework, cmakesrcdir='src')
    self.download          = ['git://https://github.com/llnl/conduit.git']
    self.gitcommit         = 'origin/master'
    self.cxx               = 1
    self.includes          = ['conduit/conduit.hpp']
    self.liblist           = [['libconduit_blueprint.a', 'libconduit_relay.a', 'libconduit.a']]
    return

  def setupDependencies(self, framework):
    config.package.CMakePackage.setupDependencies(self, framework)
    self.compilerFlags   = framework.require('config.compilerFlags', self)
    self.sharedLibraries = framework.require('PETSc.options.sharedLibraries', self)
    self.hdf5            = framework.require('config.packages.hdf5', self)
    self.adios           = framework.require('config.packages.adios', self)
    self.odeps           = [self.hdf5]
    return

  def updateGitDir(self):
    import os
    config.package.GNUPackage.updateGitDir(self)
    if not hasattr(self.sourceControl, 'git') or (self.packageDir != os.path.join(self.externalPackagesDir,'git.'+self.package)):
      return
    Dir = self.getDir()
    try:
      blt = self.blt
    except AttributeError:
      try:
        self.executeShellCommand([self.sourceControl.git, 'submodule', 'update', '--init'], cwd=Dir, log=self.log)
        import os
        if os.path.isfile(os.path.join(Dir, 'src', 'blt', 'SetupBLT.cmake')):
          self.mfem = os.path.join(Dir, 'src', 'blt')
        else:
          raise RuntimeError
      except RuntimeError:
        raise RuntimeError('Could not initialize BLT submodule needed by Conduit')
    return

  def formCMakeConfigureArgs(self):
    if not self.cmake.found:
      raise RuntimeError('CMake >= 3.0 is needed to build Conduit')
    args = config.package.CMakePackage.formCMakeConfigureArgs(self)
    args.append('-DENABLE_DOCS:BOOL=OFF')
    args.append('-DENABLE_TESTS:BOOL=OFF')
    args.append('-DENABLE_EXAMPLES:BOOL=OFF')
    if self.hdf5.found:
      args.append('-DHDF5_DIR='+self.hdf5.getInstallDir())
    if self.adios.found:
      args.append('-DADIOS_DIR='+self.adios.getInstallDir())
    if not self.mpi.usingMPIUni:
      args.append('-DENABLE_MPI=ON')
    args.append('-DENABLE_PYTHON=OFF')
    if not self.compilerFlags.debugging:
      args.append('-DCMAKE_BUILD_TYPE=Release')
    if self.checkSharedLibrariesEnabled():
      args.append('-DCMAKE_INSTALL_RPATH_USE_LINK_PATH:BOOL=ON')
    return args
