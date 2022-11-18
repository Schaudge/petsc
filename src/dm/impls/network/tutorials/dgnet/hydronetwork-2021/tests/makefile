ALL:WASH
#CFLAGS	         = -pg
FFLAGS	         =
CPPFLAGS         =
FPPFLAGS         =
EXAMPLESC        = junction.c river.c wash.c washts.c washutil.c
LOCDIR           =

SOURCES_WASH = $(wildcard ../src/river/*.c ../src/*.c)
OBJECTS_RIVER = $(patsubst %.c,%.o,$(SOURCES_WASH))

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

river1: river1.o $(OBJECTS_RIVER) chkopts
	-${CLINKER} -o river1 river1.o $(OBJECTS_RIVER) ${PETSC_TS_LIB}
	${RM} river1.o $(OBJECTS_RIVER)

####################################
runriver1:
	-@${MPIEXEC} -n 1 ./river1 -ts_max_steps 3 -case 0
runriver1_2:
	-@${MPIEXEC} -n 1 ./river1 -ts_max_steps 3 -case 1

runriver1_3.1:
	-@${MPIEXEC} -n 1 ./river1 -ts_max_steps 4000 -ts_dt .001 -ts_max_time 7.0 -river_monitor 0 -case 3 -subcase 1
runriver1_3.2:
	-@${MPIEXEC} -n 1 ./river1 -ts_max_steps 4000 -ts_dt .001 -ts_max_time 7.0 -river_monitor 0 -case 3 -subcase 2
runriver1_3.3:
	-@${MPIEXEC} -n 1 ./river1 -ts_max_steps 4000 -ts_dt .001 -ts_max_time 7.0 -river_monitor 0 -case 3 -subcase 3
runriver1_3.4:
	-@${MPIEXEC} -n 1 ./river1 -ts_max_steps 4000 -ts_dt .001 -ts_max_time 7.0 -river_monitor 0 -case 3 -subcase 4
runriver1_3.5:
	-@${MPIEXEC} -n 1 ./river1 -ts_max_steps 4000 -ts_dt .001 -ts_max_time 7.0 -river_monitor 0 -case 3 -subcase 5
runriver1_3.6:
	-@${MPIEXEC} -n 1 ./river1 -river_monitor 0,1,2 -ts_max_steps 1000 -f ../cases/sample2.inp -qmin -1.0 -qmax 5.0 -hmin 0.0 -hmax 1.0
runriver1_3.7:
	-@${MPIEXEC} -n 1 ./river1 -river_monitor 2,3,4,5 -ts_max_steps 1400 -f ../cases/sample2.inp -qmin -1.0 -qmax 8.0 -hmin 0.0 -hmax 1.0 -river_monitor_q

runriver1_4:
	-@${MPIEXEC} -n 3 ./river1 -ts_max_steps 3 -case 0
runriver1_5:
	-@${MPIEXEC} -n 3 ./river1 -ts_max_steps 3 -case 1
runriver1_6:
	-@${MPIEXEC} -n 3 ./river1 -ts_max_steps 3 -case 2

runriver1_7:
	-@${MPIEXEC} -n 1 ./river1 -ts_max_steps 30 -nsubnet 2
runriver1_8:
	-@${MPIEXEC} -n 3 ./river1 -ts_max_steps 30 -nsubnet 3
runriver1_9:
	-@${MPIEXEC} -n 6 ./river1 -ts_max_steps 30 -nsubnet 3

runriver1_10:
	-@${MPIEXEC} -n 1 ./river1 -f ../cases/sample2.inp -ts_max_steps 10 -nsubnet 2
runriver1_11:
	-@${MPIEXEC} -n 8 ./river1 -f ../cases/sample2.inp -ts_max_steps 10 -nsubnet 2

runriver1_12:
	-@${MPIEXEC} -n 32 ./river1 -ts_max_steps 10 -f ../cases/illinoisRiver071300.inp
runriver1_14:
	-@${MPIEXEC} -n 64 ./river1 -ts_max_steps 4 -f ../cases/mississippiRiver.inp

runriver1_15:
	-@${MPIEXEC} -n 64 ./river1 -ts_max_steps 4 -f ../cases/ohioRiver.inp
runriver1_16:
	-@${MPIEXEC} -n 64 ./river1 -ts_max_steps 4 -f ../cases/lowerMissouriRiver.inp

include ${PETSC_DIR}/lib/petsc/conf/test
