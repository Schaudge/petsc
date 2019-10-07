from __future__ import generators
import config.base
import config.package
from sourceDatabase import SourceDB
import os

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.defaultPrecision    = 'double'
    self.f2c                 = 0  # indicates either the f2cblaslapack are used or there is no Fortran compiler (and system BLAS/LAPACK is used)
    self.has64bitindices     = 0
    self.mkl                 = 0  # indicates BLAS/LAPACK library used is Intel MKL
    self.separateBlas        = 1
    self.required            = 1
    self.lookforbydefault    = 1
    self.alternativedownload = 'f2cblaslapack'
    self.missingRoutines     = []
    self.usesopenmp          = 'unknown'
    self.known64             = 'unknown'

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.f2cblaslapack = framework.require('config.packages.f2cblaslapack', self)
    self.fblaslapack   = framework.require('config.packages.fblaslapack', self)
    self.openblas      = framework.require('config.packages.openblas', self)
    self.flibs         = framework.require('config.packages.flibs',self)
    self.mathlib       = framework.require('config.packages.mathlib',self)
    self.openmp        = framework.require('config.packages.openmp',self)
    self.deps          = [self.flibs,self.mathlib]
    return

  def __str__(self):
    output  = config.package.Package.__str__(self)
    if self.has64bitindices:
      output += '  uses 8 byte integers\n'
    else:
      output += '  uses 4 byte integers\n'
    return output

  def setupHelp(self, help):
    config.package.Package.setupHelp(self,help)
    import nargs
    help.addArgument('BLAS/LAPACK', '-with-blas-lib=<libraries: e.g. [/Users/..../libblas.a,...]>',    nargs.ArgLibrary(None, None, 'Indicate the library(s) containing BLAS (deprecated: use --with-blaslapack-lib)'))
    help.addArgument('BLAS/LAPACK', '-with-lapack-lib=<libraries: e.g. [/Users/..../liblapack.a,...]>',nargs.ArgLibrary(None, None, 'Indicate the library(s) containing LAPACK (deprecated: use --with-blaslapack-lib)'))
    help.addArgument('BLAS/LAPACK', '-with-blaslapack-suffix=<string>',nargs.ArgLibrary(None, None, 'Indicate a suffix for BLAS/LAPACK subroutine names.'))
    help.addArgument('BLAS/LAPACK', '-with-64-bit-blas-indices', nargs.ArgBool(None, 0, 'Try to use 64 bit integers for BLAS/LAPACK; will error if not available'))
    help.addArgument('BLAS/LAPACK', '-with-openmp-blas=<bool>', nargs.ArgBool(None, 0, 'Try to use OpenMP based BLAS; will error if not available'))
    help.addArgument('BLAS/LAPACK', '-known-64-bit-blas-indices=<bool>', nargs.ArgBool(None, None, 'Indicate if using 64 bit integer BLAS'))
    help.addArgument('BLAS/LAPACK', '-known-openmp-blas=<bool>', nargs.ArgBool(None, None, 'Indicate if using OpenMP based BLAS'))
    return

  def getPrefix(self):
    if self.compilers.fortranMangling == 'caps':
      if self.defaultPrecision == 'single': return 'S'
      if self.defaultPrecision == 'double': return 'D'
      if self.defaultPrecision == '__float128': return 'Q'
      if self.defaultPrecision == '__fp16': return 'H'
      return 'Unknown precision'
    else:
      if self.defaultPrecision == 'single': return 's'
      if self.defaultPrecision == 'double': return 'd'
      if self.defaultPrecision == '__float128': return 'q'
      if self.defaultPrecision == '__fp16': return 'h'
      return 'Unknown precision'

  def getType(self):
    if self.defaultPrecision == 'single': return 'float'
    return self.defaultPrecision

  def checkBlas(self, libDir, blasLibrary, otherLibs, fortranMangle, routineIn = 'dot'):
    '''Checking for BLAS symbols'''
    oldLibs = self.compilers.LIBS
    prototype = ''
    call      = ''
    routine   = self.mangleBlas(routineIn)
    if fortranMangle=='stdcall':
      if routine=='ddot'+self.suffix:
        prototype = 'double __stdcall DDOT(int*,double*,int*,double*,int*);'
        call      = 'DDOT(0,0,0,0,0);'
    self.libraries.saveLog()
    found   = self.libraries.check(blasLibrary, routine, otherLibs = otherLibs, fortranMangle = fortranMangle, prototype = prototype, call = call, libDir = libDir)
    self.logWrite(self.libraries.restoreLog())
    self.compilers.LIBS = oldLibs
    return found

  def checkLapack(self, libDir, lapackLibrary, otherLibs, fortranMangle, routinesIn = ['getrs','geev']):
    '''Checking for LAPACK symbols'''
    oldLibs = self.compilers.LIBS
    if not isinstance(routinesIn, list): routinesIn = [routinesIn]
    routines = list(routinesIn)
    found   = 1
    prototypes = ['']
    calls      = ['']
    routines   = map(self.mangleBlas, routines)
    if fortranMangle=='stdcall':
      if routines == ['dgetrs','dgeev']:
        prototypes = ['void __stdcall DGETRS(char*,int,int*,int*,double*,int*,int*,double*,int*,int*);',
                      'void __stdcall DGEEV(char*,int,char*,int,int*,double*,int*,double*,double*,double*,int*,double*,int*,double*,int*,int*);']
        calls      = ['DGETRS(0,0,0,0,0,0,0,0,0,0);',
                      'DGEEV(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);']
    for routine, prototype, call in zip(routines, prototypes, calls):
      self.libraries.saveLog()
      found = found and self.libraries.check(lapackLibrary, routine, otherLibs = otherLibs, fortranMangle = fortranMangle, prototype = prototype, call = call, libDir = libDir)
      self.logWrite(self.libraries.restoreLog())
      if not found: break
    self.compilers.LIBS = oldLibs
    return found

  def checkLib(self, libDir, lib):
    '''Checking for BLAS and LAPACK symbols'''
    #check for BLASLAPACK_STDCALL calling convention!!!!
    if not isinstance(lib, list): lib = [lib]
    foundBlas   = 0
    foundLapack = 0
    self.f2c    = 0
    # allow a user-specified suffix to be appended to BLAS/LAPACK symbols
    self.suffix = self.argDB.get('with-blaslapack-suffix', '')
    mangleFunc = self.compilers.fortranMangling
    self.logPrint('Checking for Fortran name mangling '+mangleFunc+' on BLAS/LAPACK')
    foundBlas = self.checkBlas(libDir,lib, self.dlib, mangleFunc,'dot')
    if foundBlas:
      foundLapack = self.checkLapack(libDir,lib, self.dlib, mangleFunc)
      if foundLapack:
        self.mangling = self.compilers.fortranMangling
        self.logPrint('Found Fortran mangling on BLAS/LAPACK which is '+self.compilers.fortranMangling)
        return 1
    if not self.compilers.fortranMangling == 'unchanged':
      self.logPrint('Checking for no name mangling on BLAS/LAPACK')
      self.mangling = 'unchanged'
      foundBlas = self.checkBlas(libDir,lib, self.dlib, 0, 'dot')
      if foundBlas:
        foundLapack = self.checkLapack(libDir,lib, self.dlib, 0, ['getrs','geev'])
        if foundLapack:
          self.logPrint('Found no name mangling on BLAS/LAPACK')
          return 1
    if not self.compilers.fortranMangling == 'underscore':
      save_f2c = self.f2c
      self.f2c = 1 # so that mangleBlas will do its job
      self.logPrint('Checking for underscore name mangling on BLAS/LAPACK')
      self.mangling = 'underscore'
      foundBlas = self.checkBlas(libDir,lib, self.dlib, 0, 'dot')
      if foundBlas:
        foundLapack = self.checkLapack(libDir,lib, self.dlib, 0, ['getrs','geev'])
        if foundLapack:
          self.logPrint('Found underscore name mangling on BLAS/LAPACK')
          return 1
      self.f2c = save_f2c
    self.logPrint('Unknown name mangling in BLAS/LAPACK')
    self.mangling = 'unknown'
    return 0

  def generateGuesses(self):
    if 'with-blas-lib' in self.argDB and not 'with-lapack-lib' in self.argDB:
      raise RuntimeError('If you use the --with-blas-lib=<lib> you must also use --with-lapack-lib=<lib> option')
    if not 'with-blas-lib' in self.argDB and 'with-lapack-lib' in self.argDB:
      raise RuntimeError('If you use the --with-lapack-lib=<lib> you must also use --with-blas-lib=<lib> option')
    if 'with-blas-lib' in self.argDB and 'with-blaslapack-dir' in self.argDB:
      raise RuntimeError('You cannot set both the library containing BLAS with --with-blas-lib=<lib>\nand the directory to search with --with-blaslapack-dir=<dir>')
    if 'with-blaslapack-lib' in self.argDB and 'with-blaslapack-dir' in self.argDB:
      raise RuntimeError('You cannot set both the library containing BLAS/LAPACK with --with-blaslapack-lib=<lib>\nand the directory to search with --with-blaslapack-dir=<dir>')

    if '-known-64-bit-blas-indices' in self.argDB: self.known64 = 64
    if '-known-usesopenmp-blas' in self.argDB: self.usesopen = 'yes'

    if 'with-blas-lib' in self.argDB and 'with-lapack-lib' in self.argDB:
      yield ('None, [self.argDB['with-lapack-lib'],self.argDB['with-blas-lib']],  'unknown', 'unknown')
        raise RuntimeError('You set a value for --with-blas-lib='+str(self.argDB['with-blas-lib'])+' and --with-lapack-lib='+str(self.argDB['with-lapack-lib'])+'but they cannot be used\n')
        if 'with-blaslapack-lib' in self.argDB:
      yield (None, self.argDB['with-blaslapack-lib'], 'unknown','unknown')
      raise RuntimeError('You set a value for --with-blaslapack-lib='+str(self.argDB['with-blaslapack-lib'])+' but they cannot be used\n')
    if 'with-blaslapack-dir' in self.argDB: libDir = self.argDB['with-blaslapack-dir']
    else: libDir = None
    for (libDir, lib, known64, usesopenmp) in self.generateGuessesLib(libDir):
      yield (libDir, lib, known64,usesopenmp)



  def generateGuesses3(self):
    # check that user has used the options properly
    if self.f2cblaslapack.found:
      self.f2c = 1
      # TODO: use self.f2cblaslapack.libDir directly
      libDir = os.path.join(self.f2cblaslapack.directory,'lib')
      yield ('f2cblaslapack',libDir,['libf2clapack.a','libf2cblas.a'],'32','no')
      yield ('f2cblaslapack', libDir, ['libf2clapack.a','libf2cblas.a','-lquadmath'], '32','no')
      raise RuntimeError('--download-f2cblaslapack libraries cannot be used')
    if self.fblaslapack.found:
      self.f2c = 0
      # TODO: use self.fblaslapack.libDir directly
      libDir = os.path.join(self.fblaslapack.directory,'lib')
      yield ('fblaslapack', libDir,[ 'libflapack.a','libfblas.a'], '32','no')
      raise RuntimeError('--download-fblaslapack libraries cannot be used')
    if self.openblas.found:
      self.f2c = 0
      yield ('OpenBLAS with full path', self.openblas.libDir, 'libopenblas.a',self.openblas.known64,self.openblas.usesopenmp)
      raise RuntimeError('--download-openblas libraries cannot be used')
    if 'with-blas-lib' in self.argDB and not 'with-lapack-lib' in self.argDB:
      raise RuntimeError('If you use the --with-blas-lib=<lib> you must also use --with-lapack-lib=<lib> option')
    if not 'with-blas-lib' in self.argDB and 'with-lapack-lib' in self.argDB:
      raise RuntimeError('If you use the --with-lapack-lib=<lib> you must also use --with-blas-lib=<lib> option')
    if 'with-blas-lib' in self.argDB and 'with-blaslapack-dir' in self.argDB:
      raise RuntimeError('You cannot set both the library containing BLAS with --with-blas-lib=<lib>\nand the directory to search with --with-blaslapack-dir=<dir>')
    if 'with-blaslapack-lib' in self.argDB and 'with-blaslapack-dir' in self.argDB:
      raise RuntimeError('You cannot set both the library containing BLAS/LAPACK with --with-blaslapack-lib=<lib>\nand the directory to search with --with-blaslapack-dir=<dir>')

    # Try specified BLASLAPACK library
    if 'with-blaslapack-lib' in self.argDB:
      yield ('User specified BLAS/LAPACK library', None, self.argDB['with-blaslapack-lib'], 'unknown','unknown')
      if self.defaultPrecision == '__float128':
        raise RuntimeError('__float128 precision requires f2c BLAS/LAPACK libraries; they are not available in '+str(self.argDB['with-blaslapack-lib'])+'; suggest --download-f2cblaslapack\n')
      else:
        raise RuntimeError('You set a value for --with-blaslapack-lib=<lib>, but '+str(self.argDB['with-blaslapack-lib'])+' cannot be used\n')
    # Try specified BLAS and LAPACK libraries
    if 'with-blas-lib' in self.argDB and 'with-lapack-lib' in self.argDB:
      yield ('User specified BLAS and LAPACK libraries',None, [self.argDB['with-lapack-lib'],self.argDB['with-blas-lib']],  'unknown', 'unknown')
      if self.defaultPrecision == '__float128':
        raise RuntimeError('__float128 precision requires f2c BLAS/LAPACK libraries; they are not available in '+str(self.argDB['with-blas-lib'])+' and '+str(self.argDB['with-lapack-lib'])+'; suggest --download-f2cblaslapack\n')
      else:
        raise RuntimeError('You set a value for --with-blas-lib=<lib> and --with-lapack-lib=<lib>, but '+str(self.argDB['with-blas-lib'])+' and '+str(self.argDB['with-lapack-lib'])+' cannot be used\n')

    if not 'with-blaslapack-dir' in self.argDB:
      mkl = os.getenv('MKLROOT')
      if mkl:
        # Since user did not select MKL specifically first try compiler defaults and only if they fail use the MKL
        # WARNING: duplicate code from below
        yield ('Default compiler libraries', None,'','unknown','unknown')
        yield ('Default compiler locations', None,['liblapack.a','libblas.a'], '32','no')
        yield ('Default compiler locations', None,['liblapack.a','libblis.a'], '32','no')
        yield ('User specified OpenBLAS', None, 'libopenblas.a','unknown','unkown')        
        yield ('Default compiler locations /usr/local/lib', os.path.join('/usr','local','lib'),['liblapack.a','libblas.a'], '32','no')
        yield ('Default compiler locations /usr/local/lib', os.path.join('/usr','local','lib'),['liblapack.a','libblis.a'], '32','no')
        yield ('Default compiler locations with gfortran', None, ['liblapack.a', 'libblas.a','libgfortran.a'],'32','no')
        self.logWrite('Did not detect default BLAS and LAPACK locations so using the value of MKLROOT to search as --with-blas-lapack-dir='+mkl)
        self.argDB['with-blaslapack-dir'] = mkl

    if self.argDB['with-64-bit-blas-indices']:
      ILP64 = '_ilp64'
      known = '64'
    else:
      ILP64 = '_lp64'
      known = '32'

    if self.openmp.found:
      ITHREAD='intel_thread'
      ITHREADGNU='gnu_thread'
      ompthread = 'yes'
    else:
      ITHREAD='sequential'
      ITHREADGNU='sequential'
      ompthread = 'no'

    # Looking for Multi-Threaded MKL for MKL_C/Pardiso
    useCPardiso=0
    usePardiso=0
    if self.argDB['with-mkl_cpardiso'] or 'with-mkl_cpardiso-dir' in self.argDB or 'with-mkl_cpardiso-lib' in self.argDB:
      useCPardiso=1
      mkl_blacs_64=[['mkl_blacs_intelmpi'+ILP64+''],['mkl_blacs_mpich'+ILP64+''],['mkl_blacs_sgimpt'+ILP64+''],['mkl_blacs_openmpi'+ILP64+'']]
      mkl_blacs_32=[['mkl_blacs_intelmpi'],['mkl_blacs_mpich'],['mkl_blacs_sgimpt'],['mkl_blacs_openmpi']]
    elif self.argDB['with-mkl_pardiso'] or 'with-mkl_pardiso-dir' in self.argDB or 'with-mkl_pardiso-lib' in self.argDB:
      usePardiso=1
      mkl_blacs_64=[[]]
      mkl_blacs_32=[[]]

    # Try specified installation root
    if 'with-blaslapack-dir' in self.argDB:
      dir = self.argDB['with-blaslapack-dir']
      # error if package-dir is in externalpackages
      if os.path.realpath(dir).find(os.path.realpath(self.externalPackagesDir)) >=0:
        fakeExternalPackagesDir = dir.replace(os.path.realpath(dir).replace(os.path.realpath(self.externalPackagesDir),''),'')
        raise RuntimeError('Bad option: '+'--with-blaslapack-dir='+self.argDB['with-blaslapack-dir']+'\n'+
                           fakeExternalPackagesDir+' is reserved for --download-package scratch space. \n'+
                           'Do not install software in this location nor use software in this directory.')
      if self.defaultPrecision == '__float128':
        yield ('User specified installation root (F2CBLASLAPACK)', dir,['libf2clapack.a','libf2cblas.a'], '32','no')
        raise RuntimeError('__float128 precision requires f2c libraries; they are not available in '+dir+'; suggest --download-f2cblaslapack\n')

      if not (len(dir) > 2 and dir[1] == ':') :
        dir = os.path.abspath(dir)
      self.log.write('Looking for BLAS/LAPACK in user specified directory: '+dir+'\n')

      if useCPardiso or usePardiso:
        self.logPrintBox('BLASLAPACK: Looking for Multithreaded MKL for C/Pardiso')
        for libdir in [os.path.join('lib','64'),os.path.join('lib','ia64'),os.path.join('lib','em64t'),os.path.join('lib','intel64'),'lib','64','ia64','em64t','intel64',os.path.join('lib','32'),os.path.join('lib','ia32'),'32','ia32','']:
          libDir = os.path.join(dir,libdir)
          #  iomp5 is provided by the Intel compilers on MacOS. Run source /opt/intel/bin/compilervars.sh intel64 to have it added to LIBRARY_PATH
          #  then locate libimp5.dylib in the LIBRARY_PATH and copy it to os.path.join(dir,libdir)
          for i in mkl_blacs_64:
            yield ('User specified MKL-C/Pardiso Intel-Linux64', libDir,['mkl_intel'+ILP64,'mkl_core','mkl_intel_thread']+i+['iomp5','dl','pthread'],known,'yes')
            yield ('User specified MKL-C/Pardiso GNU-Linux64', libDir,['mkl_intel'+ILP64,'mkl_core','mkl_gnu_thread']+i+['gomp','dl','pthread'],known,'yes')
            yield ('User specified MKL-Pardiso Intel-Windows64', libDir,['mkl_core.lib','mkl_intel'+ILP64+'.lib','mkl_intel_thread.lib']+i+['libiomp5md.lib'],known,'yes')
          for i in mkl_blacs_32:
            yield ('User specified MKL-C/Pardiso Intel-Linux32', libDir,['mkl_intel','mkl_core','mkl_intel_thread']+i+['iomp5','dl','pthread'],'32','yes')
            yield ('User specified MKL-C/Pardiso GNU-Linux32', libDir,['mkl_intel','mkl_core','mkl_gnu_thread']+i+['gomp','dl','pthread'],'32','yes')
            yield ('User specified MKL-Pardiso Intel-Windows32', libDir,['mkl_core.lib','mkl_intel_c.lib','mkl_intel_thread.lib']+i+['libiomp5md.lib'],'32','yes')
        raise RuntimeError('You set a value for --with-blaslapack-dir=<dir>, and a --with-mkl_cpardiso or --with-mkl_pardiso option but '+self.argDB['with-blaslapack-dir']+' cannot be used\n')

      yield ('User specified installation root (HPUX)', dir, ['lapack','veclib'],'32','unknown')
      yield ('User specified OpenBLAS', dir, 'libopenblas.a','unknown','unkown')
      yield ('User specified installation root (F2CBLASLAPACK)', dir,['f2clapack','f2cblas'],'32','no')
      yield ('User specified installation root(FBLASLAPACK)', dir, ['flapack','fblas'],'32','no')
      # Check MATLAB [ILP64] MKL
      yield ('User specified MATLAB [ILP64] MKL Linux lib dir', [os.path.join(dir,'bin','glnxa64'),os.path.join(dir,'sys','os','glnxa64')],['mkl.so' ,'iomp5', 'pthread'],'64','yes')
      oldFlags = self.setCompilers.LDFLAGS
      yield ('User specified MATLAB [ILP64] MKL MacOS lib dir', [os.path.join(dir,'bin','maci64'),os.path.join(dir,'sys','os','maci64')],['mkl.dylib','iomp5', 'pthread'],'64','yes')
      yield ('User specified MKL11/12 and later', dir,['mkl_intel'+ILP64,'mkl_core','mkl_'+ITHREAD,'pthread'],known,ompthread)
      # Some new MKL 11/12 variations
      for libdir in [os.path.join('lib','intel64'),os.path.join('lib','32'),os.path.join('lib','ia32'),'32','ia32','']:
        libDir = os.path.join(dir,libdir)
        yield ('User specified MKL11/12 Linux32', libDir,['mkl_intel'+ILP64,'mkl_core','mkl_'+ITHREAD,'pthread'],known,ompthread)
        yield ('User specified MKL11/12 Linux32 for static linking (Cray)', None, ['-Wl,--start-group',os.path.join(libDir,'libmkl_intel'+ILP64+'.a'),'mkl_core','mkl_'+ITHREAD,'-Wl,--end-group','pthread'],known,ompthread)
      for libdir in [os.path.join('lib','intel64'),os.path.join('lib','64'),os.path.join('lib','ia64'),os.path.join('lib','em64t'),os.path.join('lib','intel64'),'lib','64','ia64','em64t','intel64','']:
        libDir = os.path.join(dir,libdir)
        yield ('User specified MKL11+ Linux64', libDir,['mkl_intel'+ILP64,'mkl_core','mkl_'+ITHREAD,'mkl_def','pthread'],known,ompthread)
        yield ('User specified MKL11+ Linux64 + Gnu', libDir,['mkl_intel'+ILP64,'mkl_core','mkl_'+ITHREADGNU,'mkl_def','pthread'],known,ompthread)
        yield ('User specified MKL11+ Mac-64', libDir,['mkl_intel'+ILP64,'mkl_core','mkl_'+ITHREAD,'pthread'],known,ompthread)
      # Older Linux MKL checks
      yield ('User specified MKL Linux lib dir',dir, ['mkl_lapack', 'mkl', 'guide', 'pthread'],'32','no')
      for libdir in ['32','64','em64t']:
        libDir = os.path.join(dir,libdir)
        yield ('User specified MKL Linux installation root',libDir,['mkl_lapack','mkl', 'guide', 'pthread'],'32','no')
      yield ('User specified MKL Linux-x86 lib dir', dir,['mkl_lapack', 'mkl_def', 'guide', 'pthread'],'32','no')
      yield ('User specified MKL Linux-x86 lib dir', dir,['mkl_lapack', 'mkl_def', 'guide', 'vml','pthread'],'32','no')
      yield ('User specified MKL Linux-ia64 lib dir', dir,['mkl_lapack', 'mkl_ipf', 'guide', 'pthread'],'32','no')
      yield ('User specified MKL Linux-em64t lib dir', dir,['mkl_lapack', 'mkl_em64t', 'guide', 'pthread'],'32','no')
      yield ('User specified MKL Linux-x86 installation root', os.path.join(dir,'lib','32'),['mkl_lapack','mkl_def', 'guide', 'pthread'],'32','no')
      yield ('User specified MKL Linux-x86 installation root', os.path.join(dir,'lib','32'),['mkl_lapack','mkl_def', 'guide', 'vml','pthread'],'32','no')
      yield ('User specified MKL Linux-ia64 installation root', os.path.join(dir,'lib','64'),['mkl_lapack','mkl_ipf', 'guide', 'pthread'],'32','no')
      yield ('User specified MKL Linux-em64t installation root', os.path.join(dir,'lib','em64t'),['mkl_lapack','mkl_em64t', 'guide', 'pthread'],'32','no')
      # Mac MKL check
      yield ('User specified MKL Mac-x86 lib dir', dir,['mkl_lapack', 'mkl_ia32', 'guide'],'32','no')
      yield ('User specified MKL Max-x86 installation root', os.path.join(dir,'Libraries','32'),['mkl_lapack','mkl_ia32', 'guide'],'32','no')
      yield ('User specified MKL Max-x86 installation root', os.path.join(dir,'lib','32'),['mkl_lapack','mkl_ia32', 'guide'],'32','no')
      yield ('User specified MKL Mac-em64t lib dir', dir,['mkl_lapack', 'mkl_intel'+ILP64, 'guide'],known,'no')
      yield ('User specified MKL Max-em64t installation root', os.path.join(dir,'Libraries','32'),['mkl_lapack','mkl_intel'+ILP64, 'guide'],'32','no')
      yield ('User specified MKL Max-em64t installation root', os.path.join(dir,'lib','32'),['mkl_lapack','mkl_intel'+ILP64, 'guide'],'32','no')
      # Check MKL on windows
      yield ('User specified MKL Windows lib dir', dir, 'mkl_c_dll.lib','32','no')
      yield ('User specified stdcall MKL Windows lib dir', dir, 'mkl_s_dll.lib','32','no')
      yield ('User specified ia64/em64t MKL Windows lib dir', dir, 'mkl_dll.lib','32','no')
      yield ('User specified MKL10-32 Windows lib dir', None, dir, ['mkl_intel_c_dll.lib','mkl_'+ITHREAD+'_dll.lib','mkl_core_dll.lib','libiomp5md.lib'],'32',ompthread)
      yield ('User specified MKL10-32 Windows stdcall lib dir', dir, ['mkl_intel_s_dll.lib','mkl_'+ITHREAD+'_dll.lib','mkl_core_dll.lib','libiomp5md.lib'],'32',ompthread)
      yield ('User specified MKL10-64 Windows lib dir', dir, ['mkl_intel'+ILP64+'_dll.lib','mkl_'+ITHREAD+'_dll.lib','mkl_core_dll.lib','libiomp5md.lib'],known,ompthread)
      mkldir = os.path.join(dir, 'ia32', 'lib')
      yield ('User specified MKL Windows installation root', mkldir, 'mkl_c_dll.lib','32','no')
      yield ('User specified stdcall MKL Windows installation root', mkldir, 'mkl_s_dll.lib','32','no')
      yield ('User specified MKL10-32 Windows installation root', mkldir, ['mkl_intel_c_dll.lib','mkl_'+ITHREAD+'_dll.lib','mkl_core_dll.lib','libiomp5md.lib'],'32',ompthread)
      yield ('User specified MKL10-32 Windows stdcall installation root', mkldir, ['mkl_intel_s_dll.lib','mkl_'+ITHREAD+'_dll.lib','mkl_core_dll.lib','libiomp5md.lib'],'32',ompthread)
      mkldir = os.path.join(dir, 'em64t', 'lib')
      yield ('User specified MKL10-64 Windows installation root', mkldir, ['mkl_intel'+ILP64+'_dll.lib','mkl_'+ITHREAD+'_dll.lib','mkl_core_dll.lib','libiomp5md.lib'],known,ompthread)
      yield ('User specified em64t MKL Windows installation root', mkldir, 'mkl_dll.lib','32','no')
      mkldir = os.path.join(dir, 'ia64', 'lib')
      yield ('User specified ia64 MKL Windows installation root', mkldir, 'mkl_dll.lib','32','no')
      yield ('User specified MKL10-64 Windows installation root', mkldir, ['mkl_intel'+ILP64+'_dll.lib','mkl_'+ITHREAD+'_dll.lib','mkl_core_dll.lib','libiomp5md.lib'],known,ompthread)
      # Check AMD ACML libraries
      libDir = os.path.join(dir,'lib')
      yield ('User specified AMD ACML lib dir', libDir,'acml','32','unknown')
      yield ('User specified AMD ACML lib dir', libDir,['acml', 'acml_mv'],'32','unknown')
      yield ('User specified AMD ACML lib dir', libDir, 'libacml_mp','32','unknown')
      yield ('User specified AMD ACML lib dir', libDir,['acml_mp', 'acml_mv'],'32','unknown')
      # Search for atlas
      yield ('User specified ATLAS Linux installation root', dir, ['liblapack','cblas','f77blas','libatlas'],'32','no')
      yield ('User specified ATLAS Linux installation root', dir, ['liblapack','f77blas', 'atlas'],'32','no')
      # Search for liblapack.a and libblas.a after the implementations with more specific name to avoid
      # finding these in /usr/lib despite using -L<blaslapack-dir> while attempting to get a different library.
      yield ('User specified installation root', dir, ['lapack','blas'],'unknown','unknown')
      yield ('User specified installation root', dir, ['lapack','blis'],'unknown','unknown')
      raise RuntimeError('You set a value for --with-blaslapack-dir=<dir>, but '+self.argDB['with-blaslapack-dir']+' cannot be used\n')
    if self.defaultPrecision == '__float128':
      raise RuntimeError('__float128 precision requires f2c libraries; suggest --download-f2cblaslapack\n')

    if useCPardiso or usePardiso:
      # WARNING: code duplication from above
      self.logPrintBox('BLASLAPACK: Looking for Multithreaded MKL for C/Pardiso in default locations')
      #  iomp5 is provided by the Intel compilers on MacOS. Run source /opt/intel/bin/compilervars.sh intel64 to have it added to LIBRARY_PATH
      #  then locate libimp5.dylib in the LIBRARY_PATH and copy it to os.path.join(dir,libdir)
      for i in mkl_blacs_64:
        yield ('User specified MKL-C/Pardiso Intel-Linux64', None,['mkl_intel'+ILP64,'mkl_core','mkl_intel_thread']+i+['iomp5','dl','pthread'],known,'yes')
        yield ('User specified MKL-C/Pardiso GNU-Linux64', None,['mkl_intel'+ILP64,'mkl_core','mkl_gnu_thread']+i+['gomp','dl','pthread'],known,'yes')
        yield ('User specified MKL-Pardiso Intel-Windows64', None,['mkl_core.lib','mkl_intel'+ILP64+'.lib','mkl_intel_thread.lib']+i+['libiomp5md.lib'],known,'yes')
      for i in mkl_blacs_32:
        yield ('User specified MKL-C/Pardiso Intel-Linux32', None,['mkl_intel','mkl_core','mkl_intel_thread']+i+['iomp5','dl','pthread'],'32','yes')
        yield ('User specified MKL-C/Pardiso GNU-Linux32', None,['mkl_intel','mkl_core','mkl_gnu_thread']+i+['gomp','dl','pthread'],'32','yes')
        yield ('User specified MKL-Pardiso Intel-Windows32', None,['mkl_core.lib','mkl_intel_c.lib','mkl_intel_thread.lib']+i+['libiomp5md.lib'],'32','yes')
      raise RuntimeError('You set a --with-mkl_cpardiso or --with-mkl_pardiso option but MKL BLAS/LAPACK cannot be used\n')

    # Try compiler defaults
    yield ('Default compiler libraries', None,'', 'unknown','unknown')
    yield ('Default compiler locations', None,['lapack','blas'], 'unknown','unknown')
    yield ('Default OpenBLAS', None, 'libopenblas.a','unknown','unkown')
    # Intel on Mac
    yield ('User specified MKL Mac-64', os.path.join('/opt','intel','mkl','lib'),['mkl_intel'+ILP64,'mkl_'+ITHREAD,'mkl_core','pthread'],known,ompthread)
    # Try Microsoft Windows location
    for MKL_Version in [os.path.join('MKL','9.0'),os.path.join('MKL','8.1.1'),os.path.join('MKL','8.1'),os.path.join('MKL','8.0.1'),os.path.join('MKL','8.0'),'MKL72','MKL70','MKL61','MKL']:
      mklpath = os.path.join('/cygdrive', 'c', 'Program Files', 'Intel', MKL_Version)
      mkldir = os.path.join(mklpath, 'ia32', 'lib')
      yield ('Microsoft Windows, Intel MKL library', mkldir,'mkl_c_dll.lib','32','no')
      yield ('Microsoft Windows, Intel MKL stdcall library', mkldir,'mkl_s_dll.lib','32','no')
      mkldir = os.path.join(mklpath, 'em64t', 'lib')
      yield ('Microsoft Windows, em64t Intel MKL library', mkldir,'mkl_dll.lib','32','no')
      mkldir = os.path.join(mklpath, 'ia64', 'lib')
      yield ('Microsoft Windows, ia64 Intel MKL library', mkldir,'mkl_dll.lib','32','no')
    # IRIX locations
    yield ('IRIX Mathematics library', None, 'libcomplib.sgimath.a','32','unknown')
    yield ('Another IRIX Mathematics library', None, 'libscs.a','32','unknown')
    yield ('Compaq/Alpha Mathematics library', None, 'libcxml.a','32','unknown')
    # IBM ESSL locations
    yield ('IBM ESSL Mathematics library', None, 'libessl.a','32','unknown')
    yield ('IBM ESSL Mathematics library for Blue Gene', None, 'libesslbg.a','32','unknown')
    yield ('HPUX', None,['lapack''veclib'],'unknown','unknown')
    # /usr/local/lib
    libDir = os.path.join('/usr','local','lib')
    yield ('Default compiler locations /usr/local/lib', libDir,['lapack','blas'],'32','no')
    yield ('Default compiler locations /usr/local/lib', libDir,'openblas','32','no')
    yield ('Default compiler locations with gfortran', libDir, ['lapack', 'blas','gfortran'],'32','no')
    yield ('Default Atlas location',libDir,['lapack','cblas','f77blas','atlas'], '32','no')
    yield ('Default Atlas location',libDir,['lapack','f77blas','atlas'], '32','no')
    yield ('Default compiler locations with G77', libDir, ['lapack', 'blas','g2c'],'32','no')
    # Try MacOSX location
    libDir = os.path.join('/Library', 'Frameworks', 'Intel_MKL.framework','Libraries','32')
    yield ('MacOSX with Intel MKL', libDir,['mkl_lapack','mkl_ia32','guide'],'32','no')
    yield ('MacOSX BLAS/LAPACK library', None, os.path.join('/System', 'Library', 'Frameworks', 'vecLib.framework', 'vecLib'),'32','unknown')
    # Sun locations
    yield ('Sun sunperf BLAS/LAPACK library', None, ['sunperf','sunmath'],'32','no')
    yield ('Sun sunperf BLAS/LAPACK library', None, ['sunperf','F77','M77','sunmath'],'32','no')
    yield ('Sun sunperf BLAS/LAPACK library', None, ['sunperf','fui','fsu','sunmath'],'32','no')
    # Try Microsoft Windows location
    for MKL_Version in [os.path.join('MKL','9.0'),os.path.join('MKL','8.1.1'),os.path.join('MKL','8.1'),os.path.join('MKL','8.0.1'),os.path.join('MKL','8.0'),'MKL72','MKL70','MKL61','MKL']:
      mklpath = os.path.join('/cygdrive', 'c', 'Program Files', 'Intel', MKL_Version)
      mkldir = os.path.join(mklpath, 'ia32', 'lib')
      yield ('Microsoft Windows, Intel MKL library', mkldir,'mkl_c_dll.lib','32','no')
      yield ('Microsoft Windows, Intel MKL stdcall library', mkldir,'mkl_s_dll.lib','32','no')
      mkldir = os.path.join(mklpath, 'em64t', 'lib')
      yield ('Microsoft Windows, em64t Intel MKL library', mkldir,'mkl_dll.lib','32','no')
      mkldir = os.path.join(mklpath, 'ia64', 'lib')
      yield ('Microsoft Windows, ia64 Intel MKL library', mkldir,'mkl_dll.lib','32','no')
    return

  def configureLibrary(self):
    if hasattr(self.compilers, 'FC'):
      self.alternativedownload = 'fblaslapack'
    for (name, libDir, self.lib, self.known64, self.usesopenmp) in self.generateGuesses():
      foundstd = 0
      if not isinstance(libDir, list): libDir = [libDir]
      for i in libDir:
        if i:
          if not os.path.isdir(i): continue
          if i == os.path.join('/usr','lib'): foundstd = 1
          self.log.write('Files and directories in '+i+':\n'+str(os.listdir(i))+'\n')
      self.log.write('================================================================================\n')
      self.log.write('Checking for BLAS and LAPACK in '+name+'\n')
      self.found = self.executeTest(self.checkLib, [libDir,self.lib])
      if self.found and libDir and libDir[0] and not foundstd:
          self.log.write('-----------------------------------------------------------------------\n')
          self.log.write('Checking for BLAS and LAPACK in standard library location (/usr/lib) and current directory search path (without provided directory paths) in '+name+'\n')
          found = self.executeTest(self.checkLib, [None,self.lib])
          if found:
            text = '\n'
            if os.getenv('LIBRARY_PATH'):
              text = 'or current LIBRARY_PATH:\n'+os.getenv('LIBRARY_PATH')+'\n'
            self.logPrintBox('***** WARNING:Looking for BLAS/LAPACK libraries '+str(self.lib)+' in directories '+str(libDir)+'\n\
but these libraries found in standard library location (e.g. /usr/lib) '+text+'\
If you just want to use these libraries then you do not need to provide --with-blaslapack-dir\n\
If you desire libraries specifically in '+str(libDir)+' (not the ones being found and used you may need to change your LIBRARY_PATH')
      if self.found:
        if not isinstance(self.lib, list): self.lib = [self.lib]
        self.dlib = self.lib+self.dlib
        self.framework.packages.append(self)
        break

    if not self.found:
      # check for split blas/blas-dev packages in Linux
      import glob
      blib = glob.glob('/usr/lib/libblas.*')
      if blib != [] and not (os.path.isfile('/usr/lib/libblas.so') or os.path.isfile('/usr/lib/libblas.a')):
        raise RuntimeError('Incomplete system BLAS install detected. Perhaps you need to install blas-dev or blas-devel package - that contains /usr/lib/libblas.so using apt or yum or equivalent package manager?')
      llib = glob.glob('/usr/lib/liblapack.*')
      if llib != [] and not (os.path.isfile('/usr/lib/liblapack.so') or os.path.isfile('/usr/lib/liblapack.a')):
        raise RuntimeError('Incomplete system LAPACK install detected. Perhaps you need to install lapack-dev or lapack-devel package - that contains /usr/lib/liblapack.so using apt or yum or equivalent package manager?')

      if hasattr(self.compilers, 'FC') and (self.defaultPrecision != '__float128') and (self.defaultPrecision != '__fp16') : pkg = 'fblaslapack'
      else: pkg = 'f2cblaslapack'
      raise RuntimeError('Could not find a functional BLAS/LAPACK. Run with --with-blaslapack-lib=<lib> to indicate the library containing BLAS/LAPACK.\n Or --download-'+pkg+'=1 to have one downloaded and installed\n')

    #  allow user to dictate which blas/lapack mangling to use (some blas/lapack libraries, like on Apple, provide several)
    if 'known-blaslapack-mangling' in self.argDB:
      self.mangling = self.argDB['known-blaslapack-mangling']

    if self.mangling == 'underscore':
        self.addDefine('BLASLAPACK_UNDERSCORE', 1)
    elif self.mangling == 'caps':
        self.addDefine('BLASLAPACK_CAPS', 1)
    elif self.mangling == 'stdcall':
        self.addDefine('BLASLAPACK_STDCALL', 1)

    if self.suffix != '':
        self.addDefine('BLASLAPACK_SUFFIX', self.suffix)

    if not self.f2cblaslapack.found and not self.fblaslapack.found:
      self.executeTest(self.checkMKL)
      if not self.mkl:
        self.executeTest(self.checkESSL)
        self.executeTest(self.checkPESSL)
        self.executeTest(self.checkMissing)
    self.executeTest(self.checklsame)
    if self.argDB['with-shared-libraries']:
      symbol = 'dgeev'+self.suffix
      if self.f2c:
        if self.mangling == 'underscore': symbol = symbol+'_'
      elif hasattr(self.compilers, 'FC'):
        symbol = self.compilers.mangleFortranFunction(symbol)
      if not self.setCompilers.checkIntoShared(symbol,self.dlib):
        raise RuntimeError('The BLAS/LAPACK libraries '+self.libraries.toStringNoDupes(self.dlib)+'\ncannot be used with a shared library\nEither run ./configure with --with-shared-libraries=0 or use a different BLAS/LAPACK library');
    self.executeTest(self.checkRuntimeIssues)
    if self.mkl and self.has64bitindices:
      self.addDefine('HAVE_MKL_INTEL_ILP64',1)
    if self.argDB['with-64-bit-blas-indices'] and not self.has64bitindices:
      raise RuntimeError('You requested 64 bit integer BLAS/LAPACK using --with-64-bit-blas-indices but they are not available given your other BLAS/LAPACK options')

  def checkMKL(self):
    '''Check for Intel MKL library'''
    self.libraries.saveLog()
    if self.libraries.check(self.dlib, 'mkl_set_num_threads'):
      self.mkl = 1
      self.addDefine('HAVE_MKL',1)
      '''Set include directory for mkl.h and friends'''
      '''(the include directory is in CPATH if mklvars.sh has been sourced.'''
      ''' if the script hasn't been sourced, we still try to pick up the include dir)'''
      if 'with-blaslapack-include' in self.argDB:
        incl = self.argDB['with-blaslapack-include']
        if not isinstance(incl, list): incl = [incl]
        self.include = incl
      if not self.checkCompile('#include "mkl_spblas.h"',''):
        self.logPrint('MKL include path not automatically picked up by compiler. Trying to find mkl_spblas.h...')
        if 'with-blaslapack-dir' in self.argDB:
          pathlist = [os.path.join(self.argDB['with-blaslapack-dir'],'include'),
                      os.path.join(self.argDB['with-blaslapack-dir'],'..','include'),
                      os.path.join(self.argDB['with-blaslapack-dir'],'..','..','include')]
          found = 0
          for path in pathlist:
            if os.path.isdir(path) and self.checkInclude([path], ['mkl_spblas.h']):
              self.include = [path]
              found = 1
              break

          if not found:
            self.logPrint('Unable to find MKL include directory!')
          else:
            self.logPrint('MKL include path set to ' + str(self.include))
      self.versionname         = 'INTEL_MKL_VERSION'
      self.versioninclude      = 'mkl_version.h'
      self.versiontitle        = 'Intel MKL Version'
      self.checkVersion()
    self.logWrite(self.libraries.restoreLog())
    return


  def checkESSL(self):
    '''Check for the IBM ESSL library'''
    self.libraries.saveLog()
    if self.libraries.check(self.dlib, 'iessl'):
      self.addDefine('HAVE_ESSL',1)
    self.logWrite(self.libraries.restoreLog())
    return

  def checkPESSL(self):
    '''Check for the IBM PESSL library - and error out - if used instead of ESSL'''
    self.libraries.saveLog()
    if self.libraries.check(self.dlib, 'ipessl'):
      self.logWrite(self.libraries.restoreLog())
      raise RuntimeError('Cannot use PESSL instead of ESSL!')
    self.logWrite(self.libraries.restoreLog())
    return

  def mangleBlas(self, baseName):
    prefix = self.getPrefix()
    if self.f2c and self.mangling == 'underscore':
      return prefix+baseName+self.suffix+'_'
    else:
      return prefix+baseName+self.suffix

  def mangleBlasNoPrefix(self, baseName):
    if self.f2c:
      if self.mangling == 'underscore':
        return baseName+self.suffix+'_'
      else:
        return baseName+self.suffix
    else:
      return self.compilers.mangleFortranFunction(baseName+self.suffix)

  def checkMissing(self):
    '''Check for possibly missing LAPACK routines'''
    mangleFunc = hasattr(self.compilers, 'FC') and not self.f2c
    routines = ['gels','gelss','geqrf','gerfs','gesv','gesvd','getrf','getri','gges',
                'hgeqz','hseqr','ormqr','potrf','potri','potrs','pttrf','pttrs',
                'stebz','stein','steqr','syev','syevx','sygvx','sytrf','sytri','sytrs',
                'tgsen','trsen','trtrs','orgqr']  # skip these: 'hetrf','hetri','hetrs',
    oldLibs = self.compilers.LIBS
    self.libraries.saveLog()
    found, missing = self.libraries.checkClassify(self.lib, map(self.mangleBlas,routines), fortranMangle = mangleFunc)
    self.logWrite(self.libraries.restoreLog())
    for baseName in routines:
      if self.mangleBlas(baseName) in missing:
        self.missingRoutines.append(baseName)
        self.addDefine('MISSING_LAPACK_'+baseName.upper(), 1)
    self.compilers.LIBS = oldLibs


  def checklsame(self):
    ''' Do the BLAS/LAPACK libraries have a valid lsame() function with correct binding.'''
    routine = 'lsame';
    if self.f2c:
      if self.mangling == 'underscore':
        routine = routine + self.suffix + '_'
    else:
      routine = self.compilers.mangleFortranFunction(routine)
    self.libraries.saveLog()
    if not self.libraries.check(self.dlib,routine,fortranMangle = 0):
      self.addDefine('MISSING_LAPACK_'+routine, 1)
    self.logWrite(self.libraries.restoreLog())

  def checkForRoutine(self,routine):
    ''' used by other packages to see if a BLAS routine is available
        This is not really correct because other packages do not (usually) know about f2cblasLapack'''
    self.libraries.saveLog()
    if self.f2c:
      if self.mangling == 'underscore':
        ret = self.libraries.check(self.dlib,routine+self.suffix+'_')
      else:
        ret = self.libraries.check(self.dlib,routine+self.suffix)
    else:
      ret = self.libraries.check(self.dlib,routine,fortranMangle = hasattr(self.compilers, 'FC'))
    self.logWrite(self.libraries.restoreLog())
    return ret

  def runTimeTest(self,name,includes,body,lib = None,nobatch=0):
    '''Either runs a test or adds it to the batch of runtime tests'''
    if name in self.framework.clArgDB: return self.argDB[name]
    if self.argDB['with-batch']:
      if nobatch:
        raise RuntimeError('In batch mode you must provide the value for --'+name)
      else:
        self.framework.addBatchInclude(includes)
        self.framework.addBatchBody(body)
        if lib: self.framework.addBatchLib(lib)
        if self.include: self.framework.batchIncludeDirs.extend([self.headers.getIncludeArgument(inc) for inc in self.include])
        return None
    else:
      result = None
      self.pushLanguage('C')
      filename = 'runtimetestoutput'
      body = '''FILE *output = fopen("'''+filename+'''","w");\n'''+body
      if lib:
        if not isinstance(lib, list): lib = [lib]
        oldLibs  = self.compilers.LIBS
        self.compilers.LIBS = self.libraries.toString(lib)+' '+self.compilers.LIBS
      if self.checkRun(includes, body) and os.path.exists(filename):
        f    = open(filename)
        out  = f.read()
        f.close()
        os.remove(filename)
        result = out.split("=")[1].split("'")[0]
      self.popLanguage()
      if lib:
        self.compilers.LIBS = oldLibs
      return result

  def checkRuntimeIssues(self):
    '''Determines if BLAS/LAPACK routines use 32 or 64 bit integers'''
    if self.known64 == '64':
      self.addDefine('HAVE_64BIT_BLAS_INDICES', 1)
      self.has64bitindices = 1
      self.log.write('64 bit blas indices based on the BLAS/LAPACK library being used\n')
    elif self.known64 == '32':
      self.log.write('32 bit blas indices based on the BLAS/LAPACK library being used\n')
    elif 'known-64-bit-blas-indices' in self.argDB:
      if self.argDB['known-64-bit-blas-indices']:
        self.addDefine('HAVE_64BIT_BLAS_INDICES', 1)
        self.has64bitindices = 1
      else:
        self.has64bitindices = 0
    elif self.argDB['with-batch']:
      self.logPrintBox('***** WARNING: Cannot determine if BLAS/LAPACK uses 32 bit or 64 bit integers\n\
in batch-mode! Assuming 32 bit integers. Run with --known-64-bit-blas-indices\n\
if you know they are 64 bit. Run with --known-64-bit-blas-indices=0 to remove\n\
this warning message *****')
      self.has64bitindices = 0
      self.log.write('In batch mode with unknown size of BLAS/LAPACK defaulting to 32 bit\n')
    else:
      includes = '''#include <sys/types.h>\n#include <stdlib.h>\n#include <stdio.h>\n#include <stddef.h>\n\n'''
      t = self.getType()
      body     = '''extern '''+t+''' '''+self.getPrefix()+self.mangleBlasNoPrefix('dot')+'''(const int*,const '''+t+'''*,const int *,const '''+t+'''*,const int*);
                  '''+t+''' x1mkl[4] = {3.0,5.0,7.0,9.0};
                  int one1mkl = 1,nmkl = 2;
                  '''+t+''' dotresultmkl = 0;
                  dotresultmkl = '''+self.getPrefix()+self.mangleBlasNoPrefix('dot')+'''(&nmkl,x1mkl,&one1mkl,x1mkl,&one1mkl);
                  fprintf(output, "-known-64-bit-blas-indices=%d",dotresultmkl != 34);'''
      result = self.runTimeTest('known-64-bit-blas-indices',includes,body,self.dlib,nobatch=1)
      if result is not None:
        self.log.write('Checking for 64 bit blas indices: result ' +str(result)+'\n')
        result = int(result)
        if result:
          if self.defaultPrecision == 'single':
            self.log.write('Checking for 64 bit blas indices: special check for Apple single precision\n')
            # On Apple single precision sdot() returns a double so we need to test that case
            body     = '''extern double '''+self.getPrefix()+self.mangleBlasNoPrefix('dot')+'''(const int*,const '''+t+'''*,const int *,const '''+t+'''*,const int*);
                  '''+t+''' x1mkl[4] = {3.0,5.0,7.0,9.0};
                  int one1mkl = 1,nmkl = 2;
                  double dotresultmkl = 0;
                  dotresultmkl = '''+self.getPrefix()+self.mangleBlasNoPrefix('dot')+'''(&nmkl,x1mkl,&one1mkl,x1mkl,&one1mkl);
                  fprintf(output, "--known-64-bit-blas-indices=%d",dotresultmkl != 34);'''
            result = self.runTimeTest('known-64-bit-blas-indices',includes,body,self.dlib,nobatch=1)
            result = int(result)
        if result:
          self.addDefine('HAVE_64BIT_BLAS_INDICES', 1)
          self.has64bitindices = 1
          self.log.write('Checking for 64 bit blas indices: result not equal to 1 so assuming 64 bit blas indices\n')
      else:
        self.addDefine('HAVE_64BIT_BLAS_INDICES', 1)
        self.has64bitindices = 1
        self.log.write('Checking for 64 bit blas indices: program did not return therefore assuming 64 bit blas indices\n')
    if not self.defaultPrecision == 'single': return
    self.log.write('Checking if sdot() returns a float or a double\n')
    if 'known-sdot-returns-double' in self.argDB:
      if self.argDB['known-sdot-returns-double']:
        self.addDefine('BLASLAPACK_SDOT_RETURNS_DOUBLE', 1)
    elif self.argDB['with-batch']:
      self.logPrintBox('***** WARNING: Cannot determine if BLAS sdot() returns a float or a double\n\
in batch-mode! Assuming float. Run with --known-sdot-returns-double=1\n\
if you know it returns a double (very unlikely). Run with\n\
--known-sdor-returns-double=0 to remove this warning message *****')
    else:
      includes = '''#include <sys/types.h>\n#include <stdlib.h>\n#include <stdio.h>\n#include <stddef.h>\n'''
      body     = '''extern float '''+self.mangleBlasNoPrefix('sdot')+'''(const int*,const float*,const int *,const float*,const int*);
                  float x1[1] = {3.0};
                  int one1 = 1;
                  long long int ione1 = 1;
                  float sdotresult = 0;
                  int blasint64 = '''+str(self.has64bitindices)+''';\n
                  if (!blasint64) {
                       sdotresult = '''+self.mangleBlasNoPrefix('sdot')+'''(&one1,x1,&one1,x1,&one1);
                     } else {
                       sdotresult = '''+self.mangleBlasNoPrefix('sdot')+'''((const int*)&ione1,x1,(const int*)&ione1,x1,(const int*)&ione1);
                     }
                  fprintf(output, "--known-sdot-returns-double=%d",sdotresult != 9);\n'''
      result = self.runTimeTest('known-sdot-returns-double',includes,body,self.dlib,nobatch=1)
      if result:
        self.log.write('Checking for sdot() return double: result ' +str(result)+'\n')
        result = int(result)
        if result:
          self.addDefine('BLASLAPACK_SDOT_RETURNS_DOUBLE', 1)
          self.log.write('Checking sdot(): Program did return with not 1 for output so assume returns double\n')
      else:
        self.log.write('Checking sdot(): Program did not return with output so assume returns single\n')
    self.log.write('Checking if snrm() returns a float or a double\n')
    if 'known-snrm2-returns-double' in self.argDB:
      if self.argDB['known-snrm2-returns-double']:
        self.addDefine('BLASLAPACK_SNRM2_RETURNS_DOUBLE', 1)
    elif self.argDB['with-batch']:
      self.logPrintBox('***** WARNING: Cannot determine if BLAS snrm2() returns a float or a double\n\
in batch-mode! Assuming float. Run with --known-snrm2-returns-double=1\n\
if you know it returns a double (very unlikely). Run with\n\
--known-snrm2-returns-double=0 to remove this warning message *****')
    else:
      includes = '''#include <sys/types.h>\n#include <stdlib.h>\n#include <stdio.h>\n#include <stddef.h>\n'''
      body     = '''extern float '''+self.mangleBlasNoPrefix('snrm2')+'''(const int*,const float*,const int*);
                  float x2[1] = {3.0};
                  int one2 = 1;
                  long long int ione2 = 1;
                  float normresult = 0;
                  int blasint64 = '''+str(self.has64bitindices)+''';\n
                  if (!blasint64) {
                       normresult = '''+self.mangleBlasNoPrefix('snrm2')+'''(&one2,x2,&one2);
                     } else {
                       normresult = '''+self.mangleBlasNoPrefix('snrm2')+'''((const int*)&ione2,x2,(const int*)&ione2);
                     }
                  fprintf(output, "--known-snrm2-returns-double=%d",normresult != 3);\n'''
      result = self.runTimeTest('known-snrm2-returns-double',includes,body,self.dlib,nobatch=1)
      if result:
        self.log.write('Checking for snrm2() return double: result ' +str(result)+'\n')
        result = int(result)
        if result:
          self.log.write('Checking snrm2(): Program did return with 1 for output so assume returns double\n')
          self.addDefine('BLASLAPACK_SNRM2_RETURNS_DOUBLE', 1)
      else:
        self.log.write('Checking snrm2(): Program did not return with output so assume returns single\n')
