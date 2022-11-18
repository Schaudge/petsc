CFLAGS	         =
FFLAGS	         =
CPPFLAGS         =
FPPFLAGS         =
LOCDIR           =
DIRS             = network
EXAMPLESC        =
EXAMPLESF        =
MANSEC           =
CLEANFILES       =
NP               =

SOURCES_WASH = $(wildcard ../src/river/*.c ../src/*.c)
OBJECTS_RIVER = $(patsubst %.c,%.o,$(SOURCES_WASH)) 

#OBJECTS_RIVER = ../tests/riverInterface.o

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

tsex7: tsex7.o  chkopts
	-${CLINKER} -o tsex7 tsex7.o  ${PETSC_TS_LIB}
	${RM} tsex7.o

tsex15: tsex15.o  chkopts
	-${CLINKER} -o tsex15 tsex15.o  ${PETSC_TS_LIB}
	${RM} tsex15.o

ex8: ex8.o  chkopts
	-${CLINKER} -o ex8 ex8.o  ${PETSC_KSP_LIB}
	${RM} ex8.o

tsex9: tsex9.o  chkopts
	-${CLINKER} -o tsex9 tsex9.o  ${PETSC_TS_LIB}
	${RM} tsex9.o

tsMtlb: tsMtlb.o $(OBJECTS_RIVER) chkopts
	-${CLINKER} -o tsMtlb tsMtlb.o $(OBJECTS_RIVER) ${PETSC_TS_LIB}
	${RM} tsMtlb.o $(OBJECTS_RIVER)

include ${PETSC_DIR}/lib/petsc/conf/test
