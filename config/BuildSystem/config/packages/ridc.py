import config.package
import os

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.download         = ['http://mathgeek.us/files/libridc-0.2.tar.gz',
                             '']
    self.downloaddirnames = ['libridc']
    self.includes         = ['ridc.h']
    self.liblist          = [['libridc.a']]
    self.functions        = []
    self.cxx              = 1
    self.precisions       = ['double']
    self.complex          = 0
    return

  def setupDependencies(self, framework):
    config.package.GNUPackage.setupDependencies(self, framework)
    self.compilerFlags  = framework.require('config.compilerFlags', self)
    return

  def formGNUConfigureArgs(self):
    args = config.package.GNUPackage.formGNUConfigureArgs(self)
    return args
