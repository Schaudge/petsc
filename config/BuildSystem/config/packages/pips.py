import config.package

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    config.package.CMakePackage.__init__(self, framework)
    self.gitcommit         = 'barry/fixes-for-mac-allow-alternative-locations-of-libraries'
    self.download          = ['git://https://github.com/Argonne-National-Laboratory/PIPS.git']
    self.functions         = []
    self.includes          = []
    self.liblist           = [[]]
    self.hastests          = 0
    self.makeinstall       = 0

  # Builds only the PIPS-NLP portion of the package, the rest is next to impossible involving Boost
  # Does not install the PIPS package after it is built; simply leaves it in the ${PETSC_ARCH}/externalpackages/git.pips directory

  def setupDependencies(self, framework):
    config.package.CMakePackage.setupDependencies(self, framework)
    self.compilerFlags = framework.require('config.compilerFlags', self)
    self.mpi           = framework.require('config.packages.MPI',self)
    self.mathlib       = framework.require('config.packages.mathlib',self)
    self.blaslapack    = framework.require('config.packages.BlasLapack',self)
    self.cbc           = framework.require('config.packages.cbc', self)
    self.asl           = framework.require('config.packages.asl', self)
    self.ma57          = framework.require('config.packages.ma57', self)
    self.parmetis      = framework.require('config.packages.parmetis', self)
    self.metis         = framework.require('config.packages.metis', self)    
    self.mumps         = framework.require('config.packages.MUMPS', self)
    self.scalapack     = framework.require('config.packages.scalapack', self)
    self.openmp        = framework.require('config.packages.openmp',self)
    self.deps          = [self.parmetis, self.parmetis,self.mpi, self.cbc, self.asl, self.mumps, self.scalapack,self.blaslapack, self.openmp, self.mathlib]

  def formCMakeConfigureArgs(self):
    '''Add PIPS specific configure arguments'''
    args = config.package.CMakePackage.formCMakeConfigureArgs(self)
    args.append('-DBUILD_SHARED_LIBS=OFF')
    args.append('-DCMAKE_BUILD_TYPE=DEBUG')
    args.append('-DBUILD_ALL=OFF')
    args.append('-DBUILD_PIPS_NLP=ON')
    args.append('-B.')
    args.append('-H..')
    args.append('-DMA57_DIR="'+self.ma57.directory+'"')
    args.append('-DCOIN_DIR="'+self.cbc.directory+'"')
    args.append('-DAMPL_DIR="'+self.asl.directory+'"')
    args.append('-DMETIS_DIR="'+self.parmetis.directory+'"')
    args.append('-DMUMPS_DIR="'+self.mumps.directory+'"')
    args.append('-DSCALAPACK_LIBRARIES="'+self.libraries.toString(self.scalapack.dlib)+'"')
    args.append('-DMATH_LIBS="'+self.libraries.toString(self.blaslapack.dlib)+'"')

    # this is a crude attempt to determine what the Fortran MPI libraries are
    # it makes many assumptions about the possible form they take that may not be justified for all MPI wrappers
    # since we cannot pass in an empty value for -DMUMPS_FORT_LIB even when there are no seperate Fortran libraries we pass in -lmpi instead of the empty string
    try:
      outputcxx,err,status = self.executeShellCommand(self.compilers.CXX + ' -show', log = self.log)
      outputfc,err,status = self.executeShellCommand(self.compilers.FC + ' -show', log = self.log)
    except:
      #  cannot determine MPI fortran libraries so just pass in -lmpi and hope that it does no harm
      mpi_fortran_libs = '-lmpi'
    else:
      cxxargs = outputcxx.split()
      fcargs = outputfc.split()
      cxxlibs = []
      for i in cxxargs:
        if i.startswith('-l') and i.find('mpi') > -1:  # assume all MPI libraries are indicated with a -l and have mpi in their names
          cxxlibs.append(i)
      fclibs = []
      for i in fcargs:
        if i.startswith('-l') and i.find('mpi') > -1 and not i in cxxlibs:
          fclibs.append(i)
      if fclibs: mpi_fortran_libs = ' '.join(fclibs)
      else: mpi_fortran_libs = '-lmpi'
    args.append('-DMUMPS_FORT_LIB="'+mpi_fortran_libs+'"')
    return args
