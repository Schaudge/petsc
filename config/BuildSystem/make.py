import maker
import project

import os

class Make(maker.Make):
  def __init__(self):
    super().__init__()
    self.project = project.Project('https://bitbucket.org/petsc/buildsystem', self.getRoot())
    self.project.setWebDirectory('petsc@login.mcs.anl.gov://mcs/www-unix/ase')
    return

  def setupDependencies(self, sourceDB):
    super().setupDependencies(self, sourceDB)
    sourceDB.addDependency(os.path.join('client-python', 'cygwinpath.c'), os.path.join('client-python', 'cygwinpath.h'))
    return

  def updateDependencies(self, sourceDB):
    sourceDB.updateSource(os.path.join('client-python', 'cygwinpath.h'))
    super().updateDependencies(sourceDB)
    return

  def setupConfigure(self, framework):
    doConfigure = super().setupConfigure(framework)
    framework.header = os.path.join('client-python', 'cygwinpath.h')
    return doConfigure

  def configure(self, builder):
    framework   = super().configure(builder)
    self.python = framework.require('config.python', None)
    return

  def buildCygwinPath(self, builder):
    '''Builds the Python module which translates Cygwin paths'''
    builder.pushConfiguration('Triangle Library')
    compiler = builder.getCompilerObject()
    linker   = builder.getLinkerObject()
    compiler.includeDirectories.update(self.python.include)
    linker.libraries.update(self.python.lib)
    source = os.path.join('client-python', 'cygwinpath.c')
    object = os.path.join('client-python', 'cygwinpath.o')
    self.builder.compile([source], object)
    self.builder.link([object], os.path.join('client-python', 'cygwinpath.so'), shared = 1)
    builder.popConfiguration()
    return

  def build(self, builder):
    self.buildCygwinPath(builder)
    return

if __name__ == '__main__':
  Make().run()
