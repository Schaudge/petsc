import config.package
import os

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.version        = '3.9.0'
    self.download       = ['Not publically avialable']
    self.functions      = []
    self.includes       = []
    self.liblist        = [[]]
    self.precisions     = ['double']
    self.complex        = 0

  def setupDependencies(self, framework):
    config.package.GNUPackage.setupDependencies(self, framework)
    self.blasLapack = framework.require('config.packages.BlasLapack',self)
    self.metis      = framework.require('config.packages.metis',self)
    self.deps       = [self.blasLapack]
    self.odeps      = [self.metis]

  def formGNUConfigureArgs(self):
    args = config.package.GNUPackage.formGNUConfigureArgs(self)
    if self.metis.found:
      args.append('--with-metis="'+self.metis.directory+'"')
    return args

