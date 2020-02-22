
CFLAGS	        =
FFLAGS	        =
CPPFLAGS        = -Wno-pointer-arith
FPPFLAGS        =
LOCDIR          = src/snes/examples/tests/
EXAMPLESC       = ex1.c  ex7.c ex17.c ex68.c ex69.c
EXAMPLESCXX     = ex241.cxx
EXAMPLESF       = ex1f.F90 ex12f.F ex18f90.F90
DIRS	        =
MANSEC          = SNES

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

%.cxx : %.pxx
	sed -e '/main/,$$d' $^ > $*.cxx
	fieldnames=`grep FieldValues  $^ | sed -e s"/FieldValues [ ]*\([a-zA-Z09,]*\)[ ]*;/\"\1\"/"g | sed s'/,/\",\"/'g` ; echo $$fieldnames ; grep PetscErrorCode $^ | grep "("  | grep -v extern | sed "s/^/extern /g" | sed "s/$$/;/g" >> $*.cxx ;         sed -n -e '/main/,$$p' $^ | sed s"/Petsc [ ]*\([A-Za-z09]*\);/Petsc \1({$${fieldnames}});/"g >> $*.cxx ; cp $*.cxx  joe

