import config.package

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    import os
    config.package.GNUPackage.__init__(self, framework)
    self.gitcommit      = 'da593796123f99307b486350f8b2ef6ae7d2c375'
    self.download       = ['git://https://github.com/paboyle/Grid.git']
    self.buildLanguages = ['Cxx']
    self.maxCxxVersion  = 'c++17'
    self.includes       = ['Grid/Grid.h']
    self.liblist        = [['libGrid.a']]
    self.includedir     = os.path.join('include', 'Grid')
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.fftw  = framework.require('config.packages.fftw', self)
    self.fftwf3 = framework.require('config.packages.fftw-3', self)
    self.ssl   = framework.require('config.packages.ssl', self)
    self.gmp   = framework.require('config.packages.gmp', self)
    self.mpfr  = framework.require('config.packages.mpfr', self)
    self.eigen = framework.require('config.packages.eigen', self)
    self.deps  = [self.ssl, self.gmp, self.mpfr, self.eigen]
    return

  def formGNUConfigureArgs(self):
    import os
    args = config.package.GNUPackage.formGNUConfigureArgs(self)
    args.append('--exec-prefix='+os.path.join(self.installDir, 'Grid'))
    args.append('--with-openssl='+self.ssl.directory)
    args.append('--with-gmp='+self.gmp.getInstallDir())
    args.append('--with-mpfr='+self.mpfr.getInstallDir())
    # Check for --enable-simd=AVX
    args.append('--enable-comms=mpi-auto')
    return args

  def preInstall(self):
    import os

    # Link Eigen directories
    try:
      self.logPrintBox('Linking Eigen to ' +self.PACKAGE)
      eigenDir = os.path.join(self.installDir, 'include', 'eigen3', 'Eigen')
      gridDir  = os.path.join(self.packageDir, 'Grid', 'Eigen')
      gridInclude = os.path.join(self.installDir, 'include', 'Grid', 'Eigen')
      if not os.path.lexists(gridDir):
        os.symlink(eigenDir, gridDir)
      eigenDir = os.path.join(self.installDir, 'include', 'eigen3', 'unsupported', 'Eigen')
      eigenUnDir  = os.path.join(self.installDir, 'include', 'eigen3', 'Eigen', 'unsupported')
      if not os.path.lexists(eigenUnDir):
        os.symlink(eigenDir, eigenUnDir)
      if not os.path.lexists(self.includeDir):
        self.logPrintBox('Creating Eigen include link for ' +self.PACKAGE)
        os.makedirs(os.path.join(self.packageDir, 'Grid'))
        if not os.path.lexists(gridInclude):
          os.symlink(eigenDir, gridInclude)
    except OSError as e:
      raise RuntimeError('Error linking Eigen to ' + self.PACKAGE+': '+str(e))

    # Create Eigen.inc
    try:
      self.logPrintBox('Creating Eigen.inc in ' +self.PACKAGE)
      eigenDir = os.path.join(self.installDir, 'include', 'eigen3', 'Eigen')
      files    = []
      for root, directories, filenames in os.walk(eigenDir):
        for filename in filenames:
          files.append(os.path.join(root, filename))
      with open(os.path.join(self.packageDir, 'Grid', 'Eigen.inc'), 'w') as f:
        f.write('eigen_files =\\\n')
        for filename in files[:-1]:
          f.write('  ' + os.path.join(root, filename) + ' \\\n')
          print(os.path.join(root, filename))
        f.write('  ' + os.path.join(root, files[-1]) + '\n')

    except RuntimeError as e:
      raise RuntimeError('Error creating Eigen.inc in ' + self.PACKAGE+': '+str(e))

    try:
      self.logPrintBox('Generating Make.inc files for ' +self.PACKAGE+'; this may take several minutes')
      output,err,ret = config.base.Configure.executeShellCommand('./scripts/filelist', cwd=self.packageDir, timeout=100, log=self.log)
      if ret:
        raise RuntimeError('Error generating Make.inc: ' + output+err)
    except RuntimeError as e:
      raise RuntimeError('Error generating Make.inc in ' + self.PACKAGE+': '+str(e))
    config.package.GNUPackage.preInstall(self)
    return
  
  def postInstall(self, output, mkfile):
    import os
    self.logPrintBox('Generating Eigen include links for ' +self.PACKAGE)
    eigenDir = os.path.join(self.installDir, 'include', 'eigen3', 'Eigen')
    gridDir  = os.path.join(self.installDir, 'include', 'Grid', 'Eigen')
    if not os.path.lexists(gridDir):
        os.symlink(eigenDir, gridDir)
