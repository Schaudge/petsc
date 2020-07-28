==============
Building PETSc
==============

Quickstart
==========

This tutorial walks you through installing the PETSc library on a local, UNIX-style machine
for the first time using :xref:`MPICH` and :xref:`BLASLAPACK`.

.. topic:: Commands

   .. code:: bash

      ./configure --download-mpich --download-fblaslapack

   or if MPI and :xref:`BLASLAPACK` is installed and in ``$PATH``

   .. code:: bash

      ./configure

Prerequisites
=============
.. Important::

   This tutorial assumes basic knowledge on the part of the user on how to
   navigate your system using the Command-Line Interface (CLI), a.k.a. "from the
   terminal". Being a library-based solver suite, PETSc does not have a
   front-end Graphical User Interface, so all tutorial examples here will
   heavily use the CLI.

   While this tutorial will provide all commands necessary, it will not explain
   the usage or syntax of commands not directly written by PETSc. If you find
   the commands confusing, consider reviewing tutorials on basic UNIX and shell
   usage.


Before beginning, make sure you have the
following *necessary* pre-requisites installed and up to date:

- :xref:`git`

- :xref:`Bash`

- :xref:`Python`

- :xref:`C Compiler <gcc>`

- [OPTIONAL] :xref:`C++ Compiler <g++>`

- [OPTIONAL] :xref:`Fortran Compiler <gfortran>`

It is important to make sure that your compilers are correctly installed (i.e. functional
and in your ``$PATH``). To test the compilers, run the following commands:

.. code:: bash

   printf '#include<stdio.h>\nint main(){printf("cc OK!\\n");}' > t.c && cc t.c && ./a.out && rm -f t.c a.out

.. note::

   While it is recommended that you have functional C++ and Fortran compilers installed,
   they are not directly required to run PETSc in its default state. If they are
   functioning, PETSc will automatically find them during the configure stage, however it
   is always useful to test them on your own.

   .. code:: bash

      printf '#include<iostream>\nint main(){std::cout<<"c++ OK!"<<std::endl;}' > t.cpp && cc++ t.cpp && ./a.out && rm -f t.cpp a.out
      printf 'program t\nprint"(a)","gfortran OK!"\nend program' > t.f90 && gfortran t.f90 && ./a.out && rm -f t.f90 a.out


If compilers are working, each command should print out ``<compiler_name> OK!`` on the command
line.

Should you be missing any of these dependencies or would like to update them, either
download and install the latest versions from their respective websites, or use your
preferred package manager to update them. For example on macOS using homebrew to install :xref:`Python`:

.. code:: bash

   brew update
   brew list    # Show all packages installed through brew
   brew upgrade # If they are already installed through brew!
   brew install python

Downloading Source
==================

With all dependencies installed, navigate to a suitable directory on your machine and
pull the latest version of the PETSc library to your machine with :xref:`git`. This will create
a directory "petsc" inside the current directory and retrieve the latest master branch
of the repository.

.. code:: bash

   mkdir -p ~/my/petsc/dir/
   cd ~/my/petsc/dir/
   git clone -b maint https://gitlab.com/petsc/petsc
   cd petsc

.. Warning::

   It is IMPERATIVE to install PETSc in a directory whose path does not contain any of
   the following special characters:

   ~ ! @ # $ % ^ & * ( ) ` ; < > ? , [ ] { } ' " | (including spaces!)

   While PETSc is equipped to handle these errors, other installed dependencies may not be
   so well protected.

The download process may take a few minutes to complete. Successfully running this command
should yield a similar output:

.. code:: bash

   git clone -b maint https://gitlab.com/petsc/petsc.git petsc

   Cloning into 'petsc'...
   remote: Enumerating objects: 862597, done.
   remote: Counting objects: 100% (862597/862597), done.
   remote: Compressing objects: 100% (197622/197622), done.
   remote: Total 862597 (delta 660708), reused 862285 (delta 660444)
   Receiving objects: 100% (862597/862597), 205.11 MiB | 3.17 MiB/s, done.
   Resolving deltas: 100% (660708/660708), done.
   Updating files: 100% (7748/7748), done.

   cd petsc
   git pull

   Already up to date.

**At this stage we will refer to** ``~/my/petsc/dir/petsc`` **as** ``$PETSC_DIR`` **to
avoid clutter.**

Configuration
=============

Next, PETSc needs to be configured using ``./configure`` for your system with your
specific options. This is the stage where users can specify the exact parameters to
customize their PETSc installation. Common configuration options are:

- Specifying different compilers

- Specifying different MPI implementations

- Enabling CUDA/OpenCL/ViennaCL support

- Specifying options for :xref:`BLASLAPACK`

- Specifying external packages to use or download automatically. PETSc can automatically download and install a wide range of other software, such as direct solvers.

- Setting various known machine quantities for PETSc to use such as additional compiler flags

.. Important::
   You MUST specify all of your configuration options at this stage. In order to enable
   additional options or packages in the future, you will have to reconfigure your PETSc
   installation in a similar manner with these options enabled.

   For a full list of available options call ``./configure --help`` from ``$PETSC_DIR``

All PETSc options and flags follow the standard CLI formats
``--option-string=<value>``
or
``--option-string``,
where ``<value>`` is typically either ``1`` (for true) or ``0`` (for false) or a directory
path. Directory paths must be absolute (i.e. full path from the root directory of your
machine), but do accept environment variables as input.

From ``$PETSC_DIR`` call the following ``./configure`` command to configure
PETSc as well as download and install :xref:`MPICH` on your system.

.. code:: bash

   ./configure --download-mpich --download-fblaslapack

PETSc will begin configuring and printing its progress. A successful configure will have
the following general structure as its output:

.. code-block:: text

   ===============================================================================
             Configuring PETSc to compile on your system
   ===============================================================================
   TESTING: configureSomething from PETSc.something(config/PETSc/configurescript.py:lineNUM)
   ===============================================================================
             Trying to download MPICH_DOWNLOAD_URL for MPICH
   ===============================================================================
   ===============================================================================
             Running configure on MPICH; this may take several minutes
   ===============================================================================
   ===============================================================================
	     Running make on MPICH; this may take several minutes
   ===============================================================================
   ===============================================================================
             Running make install on MPICH; this may take several minutes
   ===============================================================================
   ===============================================================================
             Trying to download FBLASLAPACK_URL for FBLASLAPACK
   ===============================================================================
   ===============================================================================
             Compiling FBLASLAPACK; this may take several minutes
   ===============================================================================
   ===============================================================================
             Trying to download SOWING_DOWNLOAD_URL for SOWING
   ===============================================================================
   ===============================================================================
             Running configure on SOWING; this may take several minutes
   ===============================================================================
   ===============================================================================
             Running make on SOWING; this may take several minutes
   ===============================================================================
   ===============================================================================
             Running make install on SOWING; this may take several minutes
   ===============================================================================
   Compilers:
     C Compiler:   Location information and flags
   C++ Compiler: Location information and flags
   .
   .
   .
   MPI:
        Includes:     Include path
   Other Installed Packages:
   .
   .
   .
   PETSc:
        PETSC_ARCH: {YOUR_PETSC_ARCH}
   PETSC_DIR:  {YOUR_PETSC_DIR}
   .
   .
   .
   .

   xxx=========================================================================xxx
   Configure stage complete. Now build PETSc libraries with (gnumake build):
   make PETSC_DIR=/your/petsc/dir PETSC_ARCH=your-petsc-arch  all
   xxx=========================================================================xxx

.. Warning::
   At this stage it is useful to make a note of the ``$PETSC_DIR`` and ``$PETSC_ARCH``
   variables, and set them as environment variables. Copy the values directly from your
   configure output:

   .. code:: bash

      export PETSC_DIR=/your/petsc/dir
      export PETSC_ARCH=your-petsc-arch

   You should set them in a login file (e.g. `~/.bash_profile`) to avoid having to reset them every
   time you open a fresh terminal.

   .. code:: bash

      echo "export PETSC_DIR=/your/petsc/dir" >> ~/.bash_profile
      echo "export PETSC_ARCH=your-petsc-arch" >> ~/.bash_profile

Compilation
===========

After successfully configuring, build the binaries from source using the ``make``
command. This stage may take a few minutes, and will consume a great deal of system
resources as the binaries are compiled in parallel.

If ``$PETSC_DIR`` and ``$PETSC_ARCH`` are defined as environment variables:

.. code:: bash

   make all check

If ``$PETSC_DIR`` and ``$PETSC_ARCH`` are not defined as environment variables, or you have
another installation of PETSc on the machine:

.. code:: bash

   make PETSC_DIR=/your/petsc/dir PETSC_ARCH=your-petsc-arch all check

A successful ``make`` will provide an output of the following structure:

.. code-block:: text

   -----------------------------------------
   PETSC_VERSION_RELEASE
   .
   .
   .
   -----------------------------------------
   #define SOME_PETSC_VARIABLE
   .
   .
   .
   -----------------------------------------
   Installed Compiler, Package, and Library Information
   .
   .
   .
   =========================================
          FC arch-darwin-c-debug/obj/sys/f90-mod/petscsysmod.o
          FC arch-darwin-c-debug/obj/sys/fsrc/somefort.o
          FC arch-darwin-c-debug/obj/sys/f90-src/fsrc/f90_fwrap.o
          CC arch-darwin-c-debug/obj/sys/info/verboseinfo.o
          CC arch-darwin-c-debug/obj/sys/info/ftn-auto/verboseinfof.o
          CC arch-darwin-c-debug/obj/sys/info/ftn-custom/zverboseinfof.o
	  .
	  .
	  .
	  FC arch-darwin-c-debug/obj/snes/f90-mod/petscsnesmod.o
          FC arch-darwin-c-debug/obj/ts/f90-mod/petsctsmod.o
          FC arch-darwin-c-debug/obj/tao/f90-mod/petsctaomod.o
     CLINKER arch-darwin-c-debug/lib/libpetsc.3.11.3.dylib
    DSYMUTIL arch-darwin-c-debug/lib/libpetsc.3.11.3.dylib
   gmake[2]: Leaving directory '/your/petsc/dir'
   gmake[1]: Leaving directory '/your/petsc/dir'
   =========================================
   Running test examples to verify correct installation
   Using PETSC_DIR=/your/petsc/dir and PETSC_ARCH=your-petsc-arch
   C/C++ example src/snes/examples/tutorials/ex19 run successfully with 1 MPI process
   C/C++ example src/snes/examples/tutorials/ex19 run successfully with 2 MPI processes
   Fortran example src/snes/examples/tutorials/ex5f run successfully with 1 MPI process
   Completed test examples

Congratulations!
================

You now have a working PETSc installation and are ready to start using the library.
