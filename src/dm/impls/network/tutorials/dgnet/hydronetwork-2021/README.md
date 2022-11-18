# hydronetwork is an application code for simulating the US river network.
The problem formulation is as follows:

## 1. Installing dependencies
The code is dependent on PETSc.

### PETSc (see https://www.mcs.anl.gov/petsc/documentation/installation.html)

a) Downloading and installing PETSc master branch as a regular/non-root user in /home/username/soft:
```
   cd
   mkdir soft
   cd soft
   git clone -b master git@gitlab.com:petsc/petsc.git
```

b) Set environment variables PETSC_ARCH and PETSC_DIR.
PETSc requires two environment variables to be set to know the location (PETSC_DIR) and the configuration environment (PETSC_ARCH):
```
   cd /home/username/soft/petsc
   # bash shell:
   export PETSC_DIR=<petsc-location>
   export PETSC_ARCH=<arch-name>
```
   arch-name can be any name. This enable user to install multiple versions of PETSc.

   Suggest adding followings into the file .bashrc or .bash_login:
```
     export PETSC_DIR=<petsc-location>
     export PETSC_ARCH=<arch-name>
     alias mpiexec=$PETSC_DIR/$PETSC_ARCH/bin/mpiexec   
```

c) Installation:
```
  cd $PETSC_DIR
  ./configure --with-cc=gcc --with-cxx=g++ --with-fc=gfortran --download-fblaslapack --download-mpich --download-metis --download-parmetis --download-cmake
  make
  make check
```

## 2. Build and run the code
```
  cd ~hydronetwork/tests
  make river1
  ./river1
  mpiexec -n 2 ./river1
```
  More examples can be found at the top of river1.c and makefile (e.g., 'make runriver1').
  
