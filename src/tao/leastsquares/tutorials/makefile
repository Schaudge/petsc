-include ../../../../petscdir.mk
CFLAGS		 += -g -Wall -Wextra
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
SINDY            = sindy.o sparse_reg.o variable.o sindyregis.o
SINDYH           = sindy.h sindy_impl.h variable.h sparse_reg.h

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

$(SINDY): $(SINDYH)

sindy_sine : sindy_sine.o $(SINDY) $(SINDYH)
	$(LINK.c) -o $@ $^ $(LDLIBS)
	$(RM) -f $<

sindy_sine_cosine : sindy_sine_cosine.o $(SINDY) $(SINDYH)
	$(LINK.c) -o $@ $^ $(LDLIBS)
	$(RM) -f $<

pendulum : pendulum.o $(SINDY) $(SINDYH)
	$(LINK.c) -o $@ $^ $(LDLIBS)
	$(RM) -f $<

lorenz96 : lorenz96.o $(SINDY) $(SINDYH)
	$(LINK.c) -o $@ $^ $(LDLIBS)
	$(RM) -f $<

lorenz : lorenz.o $(SINDY) $(SINDYH)
	$(LINK.c) -o $@ $^ $(LDLIBS)
	$(RM) -f $<

sindy_sine_cosine_grid : sindy_sine_cosine_grid.o $(SINDY) $(SINDYH)
	$(LINK.c) -o $@ $^ $(LDLIBS)
	$(RM) -f $<

pde_power_grid : pde_power_grid.o sindy_pde.o $(SINDY) $(SINDYH)
	$(LINK.c) -o $@ $^ $(LDLIBS)
	$(RM) -f $<
