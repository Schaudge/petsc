import config.package
import os

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.version           = '1.3.0'
    self.download          = ['https://github.com/ampl/mp/archive/'+self.version+'.tar.gz']
    self.functions         = []
    self.includes          = []
    self.liblist           = [[]]
    self.precisions        = ['double']
    self.complex           = 0
    self.hastests          = 1
    self.hastestsdatafiles = 1
    self.downloaddirnames  = 'mp-'+self.version

  def formGNUConfigureArgs(self):
    self.packageDir = os.path.join(self.packageDir,'src','asl','solvers')
    args = config.package.GNUPackage.formGNUConfigureArgs(self)
    return args

  def Install(self):
    '''Cannot use the common Install rule because it uses configurehere instead of configure as the command and has no install rule'''
    ##### getInstallDir calls this, and it sets up self.packageDir (source download), self.confDir and self.installDir
    args = self.formGNUConfigureArgs()  # allow package to change self.packageDir
    if self.download and self.argDB['download-'+self.downloadname.lower()+'-configure-arguments']:
       args.append(self.argDB['download-'+self.downloadname.lower()+'-configure-arguments'])
    args = ' '.join(args)
    conffile = os.path.join(self.packageDir,self.package+'.petscconf')
    fd = open(conffile, 'w')
    fd.write(args)
    fd.close()
    ### Use conffile to check whether a reconfigure/rebuild is required
    if not self.installNeeded(conffile):
      return self.installDir

    # Patch the source code
    try:
      patch = os.path.join(self.petscdir.dir,'config','BuildSystem','config','packages','asl-patches','asl.h.patch')
      output1,err1,ret1  = config.base.Configure.executeShellCommand('patch asl.h -p1 < '+patch, cwd=self.packageDir, timeout=20, log = self.log)
      patch = os.path.join(self.petscdir.dir,'config','BuildSystem','config','packages','asl-patches','dtoa.c.patch')
      output2,err2,ret1  = config.base.Configure.executeShellCommand('patch dtoa.c -p1 < '+patch, cwd=self.packageDir, timeout=20, log = self.log)
    except RuntimeError as e:
      pass

    ### Configure and Build package
    try:
      self.logPrintBox('Running configure on ' +self.PACKAGE+'; this may take several minutes')
      output1,err1,ret1  = config.base.Configure.executeShellCommand('./configurehere '+args, cwd=self.packageDir, timeout=200, log = self.log)
    except RuntimeError as e:
      raise RuntimeError('Error running configure on ' + self.PACKAGE+': '+str(e))
    try:
      self.logPrintBox('Running make on '+self.PACKAGE+'; this may take several minutes')
      if self.parallelMake: pmake = self.make.make_jnp+' '+self.makerulename+' '
      else: pmake = self.make.make+' '+self.makerulename+' '

      output2,err2,ret2  = config.base.Configure.executeShellCommand(self.make.make+' clean', cwd=self.packageDir, timeout=200, log = self.log)
      output3,err3,ret3  = config.base.Configure.executeShellCommand(pmake, cwd=self.packageDir, timeout=6000, log = self.log)
      self.logPrintBox('Running make install on '+self.PACKAGE+'; this may take several minutes')
      self.installDirProvider.printSudoPasswordMessage(self.installSudo)
      output4,err4,ret4 = config.package.Package.executeShellCommand(self.installSudo+' cp -f amplsolver.'+self.setCompilers.AR_LIB_SUFFIX+' '+os.path.join(self.installDir,self.libdir),cwd=self.packageDir)
      output5,err5,ret5 = config.package.Package.executeShellCommand(self.installSudo+' cp -f *.h '+os.path.join(self.installDir,self.includedir),cwd=self.packageDir)
    except RuntimeError as e:
      raise RuntimeError('Error running make; cp on '+self.PACKAGE+': '+str(e))
    self.postInstall(output1+err1+output2+err2+output3+err3+output4+err4+output5+err5, conffile)
    return self.installDir
