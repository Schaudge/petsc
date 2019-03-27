import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.gitcommit = 'f66d4d92e75067376df198d5e13d703b3ba1c81f'
    self.download  = ['git://https://github.com/XBraid/xbraid.git']
    self.functions = ['braid_Init']
    self.includes  = ['braid.h']
    self.liblist   = [['libbraid.a']]
    self.cxx       = 1
    self.c         = 1
    self.hastests  = 0
    # xbraid include files are in the root directory
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    return

  def Install(self):
    import os

    g = open(os.path.join(self.packageDir,'makefile.user'),'w')

    # overwrite the existing settings
    self.framework.pushLanguage('Cxx')
    g.write('MPICXX           = '+self.setCompilers.getCompiler()+'\n')
    g.write('CXXFLAGS         = '+self.removeWarningFlags(self.setCompilers.getCompilerFlags())+'\n')
    self.framework.popLanguage()

    self.framework.pushLanguage('C')
    g.write('MPICC            = '+self.setCompilers.getCompiler()+'\n')
    g.write('CFLAGS           = '+self.removeWarningFlags(self.setCompilers.getCompilerFlags())+'\n')
    self.framework.popLanguage()

    self.framework.pushLanguage('FC')
    g.write('MPIF90           = '+self.setCompilers.getCompiler()+'\n')
    g.write('FORTFLAGS        = '+self.removeWarningFlags(self.setCompilers.getCompilerFlags())+'\n')
    self.framework.popLanguage()

    # the following is not used by XBraid
    g.write('AR               = '+self.setCompilers.AR+'\n')
    g.write('ARFLAGS          = '+self.setCompilers.AR_FLAGS+'\n')
    g.write('AR_LIB_SUFFIX    = '+self.setCompilers.AR_LIB_SUFFIX+'\n')
    g.write('RANLIB           = '+self.setCompilers.RANLIB+'\n')
    g.close()

    if self.installNeeded('makefile.user'):
      try:
        self.logPrintBox('Configuring, compiling and installing XBraid; this may take several seconds')
        libDir = os.path.join(self.installDir, self.libdir,'')
        incDir = os.path.join(self.installDir, self.includedir,'')
        if not os.path.isdir(libDir):
          os.mkdir(libDir)
        self.installDirProvider.printSudoPasswordMessage()
        output1,err1,ret1  = config.package.Package.executeShellCommand('cd '+self.packageDir+' && make braid && '+self.installSudo+'cp -f braid/*.h '+incDir +' && '+self.installSudo+'cp braid/*.a '+libDir, timeout=1000, log = self.log)
      except RuntimeError as e:
        raise RuntimeError('Error running make on XBraid: '+str(e))
      self.postInstall(output1+err1,'makefile.user')
    return self.installDir
