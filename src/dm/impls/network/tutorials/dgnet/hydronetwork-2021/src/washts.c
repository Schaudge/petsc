/*
   Subroutines for WASH TS solver
*/
#include "wash.h"

PetscErrorCode WashTSSetUp(Wash wash, TS ts)
{
  PetscErrorCode ierr;
  MPI_Comm       comm;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)ts, &comm);
  CHKERRQ(ierr);
  ierr = TSSetDM(ts, (DM)wash->dm);
  CHKERRQ(ierr);

  ierr = TSSetRHSFunction(ts, NULL, WashRHSFunction, wash);
  CHKERRQ(ierr);
  ierr = TSSetType(ts, TSEULER);
  CHKERRQ(ierr);
  ierr = TSSetPostStep(ts, TSWashPostStep);
  CHKERRQ(ierr);

  ierr = TSSetApplicationContext(ts, wash);
  CHKERRQ(ierr);
  ierr = TSSetMaxSteps(ts, 3);
  CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP);
  CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts, 0.1);
  CHKERRQ(ierr);
  ierr = TSSetPreStep(ts, TSWashPreStep);
  CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);
  CHKERRQ(ierr);
  ierr = TSGetTimeStep(ts, &wash->dt);
  CHKERRQ(ierr);
  ierr = PetscPrintf(comm, "%-10s wash->dt %g\n", PETSC_FUNCTION_NAME, wash->dt);
  CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
