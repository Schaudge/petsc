import config.package
import os

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.version           = '6.1.2'
    self.download          = ['https://gmplib.org/download/gmp/gmp-6.1.2.tar.bz2']
    self.functions         = []
    self.includes          = []
    self.liblist           = [[]]
    self.precisions        = ['single','double']
    self.complex           = 0



