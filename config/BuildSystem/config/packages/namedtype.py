from __future__ import generators
import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.gitcommit         = 'v1.0.0'
    self.download          = ['https://github.com/joboccara/NamedType.git']
    self.includes          = ['named_type.hpp']
    self.liblist           = []
    self.downloadonWindows = 1
    return

  def Install(self):
    import shutil
    import os

    conffile = os.path.join(self.packageDir,self.package+'.petscconf')
    fd = open(conffile, 'w')
    fd.write(self.installDir)
    fd.close()
    if not self.installNeeded(conffile): return self.installDir

    IncludeDir = os.path.join(os.path.join(self.installDir, self.includedir))
    self.logPrintBox('Installing NamedType headers, this should not take long')
    output,err,ret  = config.base.Configure.executeShellCommand('cd '+self.packageDir+';' + 'cp *.hpp ' + IncludeDir, timeout=60, log = self.log)
    self.postInstall(output+err,conffile)
    return self.installDir
