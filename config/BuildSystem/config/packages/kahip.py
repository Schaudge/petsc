import config.package

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    config.package.CMakePackage.__init__(self, framework)
    self.gitcommit         = '88043fb'
    self.download          = ['git://https://github.com/fdkong/KaHIP.git']
    self.downloaddirnames  = ['petsc-kahip']
    self.functions         = []
    self.includes          = ['kaHIP_interface.h', 'parhip_interface.h']
    self.liblist           = [['libkahip.a'],['libparhip.a']]
    self.cxx       = 1
    return

  def setupDependencies(self, framework):
    config.package.CMakePackage.setupDependencies(self, framework)
    config.package.CMakePackage.setupDependencies(self, framework)
    self.compilerFlags = framework.require('config.compilerFlags', self)
    self.mpi           = framework.require('config.packages.MPI',self)
    self.deps          = [self.mpi]
    return

  def configureLibrary(self):
    config.package.Package.configureLibrary(self)
    oldFlags = self.compilers.CPPFLAGS
    self.compilers.CPPFLAGS += ' '+self.headers.toString(self.include)
    self.compilers.CPPFLAGS = oldFlags
    return

  def Install(self):
     try:
       self.logPrintBox('Configuring '+self.PACKAGE+' with scons, this may take several minutes')
       output1,err1,ret1  = config.package.Package.executeShellCommand('./complile.sh', cwd=folder, timeout=900, log = self.log)
     except RuntimeError as e:
       raise RuntimeError('Error configuring '+self.PACKAGE+' with scons '+str(e))

     return self.installDir
