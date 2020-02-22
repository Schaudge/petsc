
CLFAGS          =
FFLAGS	        =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/ts/examples/tutorials/advection-diffusion-reaction/
EXAMPLESC       = ex1.c ex2.c ex3.c ex4.c ex5.c ex5opt_ic.c
EXAMPLESF       = shashi.F90
EXAMPLESFH      =
MANSEC          = TS
DIRS            =
CLEANFILES      = TS-data-*

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

%.cxx : %.pxx
	sed -e '/main/,$$d' $^ > $*.cxx
	fieldnames=`grep FieldValues  $^ | sed -e s"/FieldValues [ ]*\([a-zA-Z09,]*\)[ ]*;/\"\1\"/"g | sed s'/,/\",\"/'g` ; echo $$fieldnames ; grep PetscErrorCode $^ | grep "("  | grep -v extern | sed "s/^/extern /g" | sed "s/$$/;/g" >> $*.cxx ;         sed -n -e '/main/,$$p' $^ | sed s"/Petsc [ ]*\([A-Za-z09]*\);/Petsc \1({$${fieldnames}});/"g >> $*.cxx ; cp $*.cxx  joe

