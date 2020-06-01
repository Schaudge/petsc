-include ../../../../petscdir.mk
CFLAGS		 += -g 
FFLAGS		 = 
CPPFLAGS         =
FPPFLAGS         =
LOCDIR		 = src/tao/leastsquares/tutorials/
MANSEC		 =
EXAMPLESC        = chwirut1.c chwirut2.c cs1.c tomography.c
EXAMPLESCH       =
EXAMPLESF        = chwirut1f.F90 chwirut2f.F90
EXAMPLESFH       = chwirut2f.h
TUTORIALS        = chwirut1.c chwirut2.c chwirut1f.F90 chwirut2f.F90 cs1.c tomography.c
DOCS		 = 
DIRS		 = matlab
CLEANFILES       = chwirut1 chwirut2 chwirut1f chwirut2f cs1 tomography

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

sindy_sine : sindy_sine.o sindy.o
	$(LINK.F) -o $@ $^ $(LDLIBS)
	$(RM) -f $<

sindy_sine_cosine : sindy_sine_cosine.o sindy.o
	$(LINK.F) -o $@ $^ $(LDLIBS)
	$(RM) -f $<

lorentz : lorentz.o sindy.o
	$(LINK.F) -o $@ $^ $(LDLIBS)
	$(RM) -f $<
