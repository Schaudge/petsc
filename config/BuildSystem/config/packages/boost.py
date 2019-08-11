from __future__ import generators
import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.download          = ['https://dl.bintray.com/boostorg/release/1.70.0/source/boost_1_70_0.tar.bz2']
    self.includes          = ['boost/multi_index_container.hpp']
    self.liblist           = []
    self.cxx               = 1
    self.downloadonWindows = 1
    self.useddirectly      = 0
    return

  def setupHelp(self, help):
    import nargs
    config.package.Package.setupHelp(self, help)
    help.addArgument('BOOST', '-boost-headers-only=<bool>', nargs.ArgBool(None, 0, 'When true, do not build boost libraries, only install headers'))

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.openmp  = framework.require('config.packages.openmp',self)
    self.setCompilers  = framework.require('config.setCompilers',self)

  def Install(self):
    import shutil
    import os

    threading = ''
    if self.openmp.found: threading = ' threading=multi,single '

    #   try to build boost libraries with the current sequential compiler
    #    #
    #   Could not get this to work on Apple; instead used brew install -s  -cc=g++-9 boost
    #
    #    Assumes that clang is only used on Apple, needs to be fixed for clang on other systems
    #
    compiler = ''
    cxxcompiler = ''
    toolset = ''
    if self.setCompilers.usedMPICompilers:
      if config.setCompilers.Configure.isClang(self.getCompiler('C'),self.log):  # On Apple mpicc -show prints gcc when it should print clang
        compiler = 'clang'
        cxxcompiler = 'clang++'
      else:
        try:
          self.setCompilers.pushLanguage('C')
          pcompiler = self.setCompilers.getCompiler()
          self.setCompilers.popLanguage()
          self.framework.saveLog()
          output   = self.executeShellCommand(pcompiler + ' -show', log = self.log)[0]
          self.logWrite(self.framework.restoreLog())
          compiler = output.split(' ')[0]
          self.setCompilers.pushLanguage('Cxx')
          pcompiler = self.setCompilers.getCompiler()
          self.setCompilers.popLanguage()
          self.framework.saveLog()
          output   = self.executeShellCommand(pcompiler + ' -show', log = self.log)[0]
          self.logWrite(self.framework.restoreLog())
          cxxcompiler = output.split(' ')[0]
        except:
          raise RuntimeError("Unable to determined names of compilers needed for building Boost")

    else:
      self.setCompilers.pushLanguage('C')
      compiler = self.setCompilers.getCompiler()
      self.setCompilers.popLanguage()
      self.setCompilers.pushLanguage('CXX')
      cxxcompiler = self.setCompilers.getCompiler()
      self.setCompilers.popLanguage()
    if compiler.startswith('gcc'): toolset = 'gcc'
    elif compiler.startswith('clang'): toolset = 'darwin'
    else: toolset = compiler

    self.log.write('Using toolset '+toolset+' and compiler '+cxxcompiler+' to build boost libraries\n')
    fd = open(os.path.join(self.packageDir,'user-config.jam'),'w')
    fd.write('using darwin ;\n')
    fd.write('using '+toolset+' : 9.0 : '+cxxcompiler+' ;\n')
    fd.close()
    toolsetflag = 'toolset='+toolset+' cxxflags=-std=c++17' 

    conffile = os.path.join(self.packageDir,self.package+'.petscconf')
    fd = open(conffile, 'w')
    fd.write(self.installDir)
    fd.close()
    if not self.installNeeded(conffile): return self.installDir

    if self.framework.argDB['boost-headers-only']:
       boostIncludeDir = os.path.join(os.path.join(self.installDir, self.includedir), 'boost')
       self.logPrintBox('Configure option --boost-headers-only is ENABLED ... boost libraries will not be built')
       self.logPrintBox('Installing boost headers, this should not take long')
       try:
         if os.path.lexists(boostIncludeDir): os.remove(boostIncludeDir)
         output,err,ret  = config.base.Configure.executeShellCommand('cd '+self.packageDir+';' + 'ln -s $PWD/boost/ ' + boostIncludeDir, timeout=6000, log = self.log)
       except RuntimeError as e:
         raise RuntimeError('Error linking '+self.packageDir+' to '+ boostIncludeDir)
       return self.installDir
    else:
       if not self.checkCompile('#include <bzlib.h>', ''):
         raise RuntimeError('Boost requires bzlib.h. Please install it in default compiler search location.')

       self.log.write('boostDir = '+self.packageDir+' installDir '+self.installDir+'\n')
       self.logPrintBox('Building and installing boost, this may take many minutes')
       self.installDirProvider.printSudoPasswordMessage()
       try:
         output,err,ret  = config.base.Configure.executeShellCommand('cd '+self.packageDir+'; ./bootstrap.sh --prefix='+self.installDir+' ; ./b2 -j'+str(self.make.make_np)+' '+toolsetflag+' '+threading+' --debug-configuration --debug-building --user-config=user-config.jam ;'+self.installSudo+'./b2 install', timeout=6000, log = self.log)
       except RuntimeError as e:
         raise RuntimeError('Error building/install Boost files from '+os.path.join(self.packageDir, 'Boost')+' to '+self.packageDir)
       self.postInstall(output+err,conffile)
    return self.installDir
