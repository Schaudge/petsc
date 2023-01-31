#include <petsc/private/localnetrpimpl.h>    /*I "petscnetrs.h"  I*/
#include <petsc/private/riemannsolverimpl.h> /* should not be here */
#include <petscdmnetwork.h>

/*
    Heuristic Outflow Boundary Condtion that I use. Should be removed as it has no 
    serious justification for its existance other than to get a code to work. 

    Replace with a boundary condition class ?
*/

static PetscErrorCode NetRPSolveFlux_Outflow(NetRP rp, PetscInt vdeg, PetscBool *edgein, Vec U, Vec Flux)
{
  PetscInt           i, numfields;
  const PetscScalar *u;
  PetscScalar       *flux;
  PetscReal         *fluxrs;

  PetscFunctionBeginUser;
  PetscCheck(vdeg == 1, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "The Outflow NetRP requires exactly one edge. %" PetscInt_FMT " Edges inputted", vdeg);
  PetscCall(VecGetArrayRead(U, &u));
  PetscCall(VecGetArray(Flux, &flux));
  PetscCall(RiemannSolverEvaluate(rp->flux, u, u, &fluxrs, NULL));
  PetscCall(NetRPGetNumFields(rp, &numfields));
  for (i = 0; i < numfields; i++) { flux[i] = fluxrs[i]; }
  PetscCall(VecRestoreArrayRead(U, &u));
  PetscCall(VecRestoreArray(Flux, &flux));
  PetscFunctionReturn(0);
}

static PetscErrorCode NetRPSetFromOptions_Outflow(PetscOptionItems *PetscOptionsObject, NetRP rp)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode NetRPView_Outflow(NetRP rp, PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}
/* ------------------------------------------------------------ */

PETSC_EXTERN PetscErrorCode NetRPCreate_Outflow(NetRP rp)
{
  PetscFunctionBegin;
  rp->data                = NULL;
  rp->ops->setfromoptions = NetRPSetFromOptions_Outflow;
  rp->ops->view           = NetRPView_Outflow;
  rp->ops->solveFlux      = NetRPSolveFlux_Outflow;

  rp->physicsgenerality = Generic;
  rp->solvetype         = Other;
  PetscFunctionReturn(0);
}
