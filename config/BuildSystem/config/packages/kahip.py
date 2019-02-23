import config.package
import os, glob

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.gitcommit         = 'c7c8142'
    self.download          = ['git://https://github.com/fdkong/KaHIP.git']
    self.downloaddirnames  = ['petsc-kahip']
    self.functions         = []
    self.includes          = ['kaHIP_interface.h', 'parhip_interface.h']
    self.liblist           = [['libkahip.a'],['libparhip.a']]
    self.cxx               = 1
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.compilerFlags = framework.require('config.compilerFlags', self)
    self.mpi           = framework.require('config.packages.MPI',self)
    self.deps          = [self.mpi]
    return

  def Install(self):
    self.log.write('kahipDir = '+self.packageDir+' installDir '+self.installDir+'\n')

    # KaHIP uses Scons to build the whole system
    # We do not want to have one more unnecessary dependence
    # We automatically generate a makefile for GNU make
    mkfile = 'makefile'
    g = open(os.path.join(self.packageDir, mkfile), 'w')
    self.setCompilers.pushLanguage('CXX')
    g.write('CXX = '+self.setCompilers.getCompiler()+'\n')
    g.write('CXXFLAGS = '+self.setCompilers.getCompilerFlags()+'\n')
    self.setCompilers.popLanguage()
    g.write('\n')
    g.write('KaHIP_DIR       := '+self.packageDir+'/parallel'+'\n')
    g.write('kahip_srcfiles  := $(shell find $(KaHIP_DIR) -name "*.cpp")'+'\n')
    g.write('kahip_objects   := $(patsubst %.cpp, %.$(obj-suffix), $(kahip_srcfiles))'+'\n')
    g.write('all:$(kahip_objects)'+'\n')
    g.write('clean:'+'\n')
    g.write('     '+'rm -f $(kahip_objects)'+'\n')
    g.write('     '+'rm -f $(kahip_objects)'+'\n')
    g.close()

    if self.installNeeded(mkfile):
      try:
        self.logPrintBox('Compiling and installing KaHIP; this may take several minutes')
        self.installDirProvider.printSudoPasswordMessage()
        output,err,ret  = config.package.Package.executeShellCommandSeq(
          ['make clean',
           'make',
           self.setCompilers.AR+' '+self.setCompilers.AR_FLAGS+' '+'libparhip.'+
           self.setCompilers.AR_LIB_SUFFIX,
           self.setCompilers.RANLIB+' libparhip.'+self.setCompilers.AR_LIB_SUFFIX,
           [self.installSudo+'mkdir', '-p', os.path.join(self.installDir,self.libdir)],
           [self.installSudo+'cp', 'libparhip.'+self.setCompilers.AR_LIB_SUFFIX, os.path.join(self.installDir,self.libdir)]
          ], cwd=os.path.join(self.packageDir, 'code'), timeout=2500, log = self.log)

      except RuntimeError as e:
        raise RuntimeError('Error running make on KaHIP: '+str(e))
      self.postInstall(output+err, mkfile)
    return self.installDir
