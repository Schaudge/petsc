import config.package

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.gitcommit         = 'origin/maint-4.0.0'
    self.download          = ['git://https://bitbucket.org/petsc/pkg-med.git','https://bitbucket.org/petsc/pkg-med/get/'+self.gitcommit+'.tar.gz']
    self.functions         = ['MEDfileOpen']
    self.includes          = ['med.h']
    self.liblist           = [['libmed.a']]
    self.needsMath         = 1
    self.precisions        = ['double'];
    return

  def setupDependencies(self, framework):
    config.package.GNUPackage.setupDependencies(self, framework)
    self.mpi            = framework.require('config.packages.MPI', self)
    self.hdf5           = framework.require('config.packages.hdf5', self)
    self.mathlib        = framework.require('config.packages.mathlib',self)
    self.deps           = [self.mpi, self.hdf5, self.mathlib]
    return

  def formGNUConfigureArgs(self):
    args = config.package.GNUPackage.formGNUConfigureArgs(self)
    args.append('--disable-python')
    args.append('--with-hdf5=%s' % self.hdf5.directory)
    return args

  def Install(self):
    '''bootstrap, then standard GNU configure; make; make install'''
    import os
    if not self.programs.libtoolize:
      raise RuntimeError('Could not bootstrap med using autotools: libtoolize not found')
    if not self.programs.autoreconf:
      raise RuntimeError('Could not bootstrap med using autotools: autoreconf not found')
    self.logPrintBox('Trying to bootstrap med using autotools; this may take several minutes')
    try:
      self.executeShellCommand('./bootstrap',cwd=self.packageDir,log=self.log)
    except RuntimeError as e:
      raise RuntimeError('Could not bootstrap med using autotools: maybe autotools (or recent enough autotools) could not be found?\nError: '+str(e))
    return config.package.GNUPackage.Install(self)
