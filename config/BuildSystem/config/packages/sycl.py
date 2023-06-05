import config.package
import os

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.minversion       = '20200101'
    self.versionname      = '__SYCL_COMPILER_VERSION' # The build date of the SYCL library, presented in the format YYYYMMDD.
    self.versioninclude   = 'CL/sycl/version.hpp'
    # CL/sycl.h is dpcpp.  Other SYCL impls may use SYCL/sycl.hpp -- defer
    self.includes         = ['CL/sycl.hpp']
    self.includedir       = 'include/sycl'
    self.functionsCxx     = [1,'namespace sycl = cl;','sycl::device::get_devices()']
    # Unlike CUDA or HIP, the blas issues are just part of MKL and handled as such.
    self.liblist          = [['libsycl.a'],
                             ['sycl.lib'],]
    self.precisions       = ['single','double']
    self.buildLanguages   = ['SYCL']
    self.minCxxVersion    = 'c++17'

    return

  def setupHelp(self, help):
    import nargs
    config.package.Package.setupHelp(self, help)
    help.addArgument('SYCL', '-with-sycl-arch', nargs.ArgString(None, None, 'Intel GPU architecture for code generation, for example gen9, xehp (this may be used by external packages)'))
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.setCompilers = framework.require('config.setCompilers',       self)
    self.languages    = framework.require('PETSc.options.languages',   self.setCompilers)
    self.headers      = framework.require('config.headers',            self)
    return

  def getSearchDirectories(self):
    import os
    self.pushLanguage('SYCL')
    petscSycl = self.getCompiler()
    self.popLanguage()
    self.getExecutable(petscSycl,getFullPath=1,resultName='systemSyclc')
    if hasattr(self,'systemSyclc'):
      syclcDir = os.path.dirname(self.systemSyclc)
      syclDir = os.path.split(syclcDir)[0]
      yield syclDir
    return

  def checkSizeofVoidP(self):
    '''Checks if the SYCL compiler agrees with the C compiler on what size of void * should be'''
    self.log.write('Checking if sizeof(void*) in SYCL is the same as with regular compiler\n')
    size = self.types.checkSizeof('void *', (8, 4), lang='SYCL', save=False)
    if size != self.types.sizes['void-p']:
      raise RuntimeError('SYCL Error: sizeof(void*) with SYCL compiler is ' + str(size) + ' which differs from sizeof(void*) with C compiler')
    return

  def configureTypes(self):
    import config.setCompilers
    if not self.getDefaultPrecision() in ['double', 'single']:
      raise RuntimeError('Must use either single or double precision with SYCL')
    self.checkSizeofVoidP()
    return

  def checkSYCLCDoubleAlign(self):
    if 'known-sycl-align-double' in self.argDB:
      if not self.argDB['known-sycl-align-double']:
        raise RuntimeError('SYCL error: PETSC currently requires that SYCL double alignment match the C compiler')
    else:
      typedef = 'typedef struct {double a; int b;} teststruct;\n'
      sycl_size = self.types.checkSizeof('teststruct', (16, 12), lang='SYCL', codeBegin=typedef, save=False)
      c_size = self.types.checkSizeof('teststruct', (16, 12), lang='C', codeBegin=typedef, save=False)
      if c_size != sycl_size:
        raise RuntimeError('SYCL compiler error: memory alignment doesn\'t match C compiler (try adding -malign-double to compiler options)')
    return

  def configureLibrary(self):
    self.addDefine('HAVE_SYCL','1')
    with self.setCompilers.Language('SYCL'):
      flags = '-fsycl'
      ldflags = ''
      if 'with-sycl-arch' in self.framework.clArgDB:
        self.syclArch = self.argDB['with-sycl-arch'].lower()
        if self.syclArch == 'x86_64':
          flags += ' -fsycl-targets=spir64_x86_64 '
        elif self.syclArch in ['gen','gen9','gen11','gen12lp','dg1','xehp','pvc']:
          # https://en.wikipedia.org/wiki/List_of_Intel_graphics_processing_units
          if self.syclArch == 'gen':
            devArg = 'gen9-' # compile for all targets of gen9 and up
          elif self.syclArch == 'xehp':
            devArg = '12.50.4'
          elif self.syclArch == 'pvc':
            devArg = '12.60.7'
          else:
            devArg = self.syclArch
          flags += ' -fsycl-targets=spir64_gen'
          ldflags = '-Xsycl-target-backend "-device '+ devArg + '"' # If it's used at compile time, icpx will warn: argument unused during compilation: '-Xsycl-target-backend -device 12.60.7'
        else:
          raise RuntimeError('SYCL arch is not supported: ' + self.syclArch)
      ppFlagName = self.setCompilers.getPreprocessorFlagsArg()
      oldFlags = getattr(self.setCompilers, ppFlagName)
      setattr(self.setCompilers, ppFlagName, oldFlags+' '+flags) # -fsycl is needed for icpx to preprocess sycl source code; the flag is also used at compilation
      self.setCompilers.addLinkerFlag(flags+' '+ldflags) # -fsycl is also needed by linker
    # FIXME:
    # We need '-fsycl ...' to link libpetsc.so when some .o files are generated by SYCLC.
    # Otherwise, we meet "No kernel named .." or "invalid kernel name" errors when running
    # petsc/sycl tests. The problem is when petsc's linker (PCC_LINKER) is not a sycl compiler,
    # it might not accept the flags. So we do this check. We also add these flags to PCC_LINKER_FLAGS.
    compiler = self.setCompilers.getCompiler(self.languages.clanguage) # ex. 'mpicc'
    if not self.setCompilers.isSYCL(compiler,self.log):
      raise RuntimeError(compiler + ' is not a SYCL compiler. When --with-sycl is enabled, you also need to provide a SYCL compiler to build PETSc, e.g., by --with-cc=<sycl compiler>')
    config.package.Package.configureLibrary(self)
    #self.checkSYCLCDoubleAlign()
    self.configureTypes()
    return
