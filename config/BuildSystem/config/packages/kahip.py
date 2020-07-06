import config.package

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    config.package.CMakePackage.__init__(self, framework)
    self.version           = 'v3.00'
    self.gitcommit         = '86a40ec71b8ddafd818982ca3950fc9c4974256a' # Mandatory bug fix
    self.download          = ['https://github.com/KaHIP/KaHIP/archive/'+self.gitcommit+'.tar.gz']
    self.functions         = ['ParHIPPartitionKWay']
    self.includes          = ['parhip_interface.h']
    self.liblist           = [['libparhip_interface.a']]

  def setupDependencies(self, framework):
    config.package.CMakePackage.setupDependencies(self, framework)
    self.compilerFlags = framework.require('config.compilerFlags', self)
    self.mpi           = framework.require('config.packages.MPI',self)
    self.mathlib       = framework.require('config.packages.mathlib',self)
    self.deps          = [self.mpi, self.mathlib]

  def formCMakeConfigureArgs(self):
    args = config.package.CMakePackage.formCMakeConfigureArgs(self)
    return args

