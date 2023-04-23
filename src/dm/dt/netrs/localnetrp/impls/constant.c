#include "petscriemannsolver.h"
#include "petscstring.h"
#include "petscsys.h"
#include "petscsystypes.h"
#include <math.h>
#include <petsc/private/localnetrpimpl.h>    /*I "petscnetrs.h"  I*/
#include <petsc/private/riemannsolverimpl.h> /* should not be here */

/*
  Constant Flux, for boundary conditions 
*/
typedef struct {
  PetscScalar *flux; 
  PetscScalar *u; 
} ConstantCtx;

static PetscErrorCode NetRPSolveFlux_Constant(NetRP rp, PetscInt vdeg, PetscBool *edgein, Vec U, Vec Flux)
{
  PetscInt           numfields;
  PetscScalar       *flux;
  ConstantCtx       *ctx = (ConstantCtx*) rp->data; 

  PetscFunctionBegin;
  PetscCheck(vdeg == 1, PetscObjectComm((PetscObject)rp), PETSC_ERR_ARG_WRONG, "The Constant NetRP requires exactly one edge. %" PetscInt_FMT " Edges inputted", vdeg);
  PetscCall(VecGetArray(Flux, &flux));
  PetscCall(NetRPGetNumFields(rp, &numfields));
  PetscCall(PetscArraycpy(flux,ctx->flux,numfields));
  PetscCall(VecRestoreArray(Flux, &flux));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode NetRPSetConstantData(NetRP rp,PetscScalar *u)
{
  ConstantCtx       *ctx = (ConstantCtx*) rp->data; 
  PetscInt           numfields;
  RiemannSolver      rs; 
  PetscReal          *flux; 

  PetscFunctionBegin; 
  if (!rp->setupcalled) PetscCall(NetRPSetUp(rp));
  PetscCall(NetRPGetNumFields(rp, &numfields));
  PetscCall(PetscArraycpy(ctx->u,u,numfields));
  PetscCall(NetRPGetFlux(rp, &rs)); 
  PetscCall(RiemannSolverEvaluateFlux(rs, u,&flux));
  PetscCall(PetscArraycpy(ctx->flux,flux,numfields));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode NetRPSetFromOptions_Constant(PetscOptionItems *PetscOptionsObject, NetRP rp)
{
  PetscInt          num_in,numfields; 
  PetscBool        flg; 
  PetscScalar      *u; 

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "NetRP Constant Options");
  PetscCall(NetRPGetNumFields(rp, &numfields));

  PetscCall(PetscMalloc1(numfields, &u));
  PetscCall(PetscOptionsScalarArray("-netrp_constant_u", "Solution Value to Evaluate Constant Flux at", "NetRP", u, &num_in, &flg));
  PetscCheck(num_in == numfields, PetscObjectComm((PetscObject)rp), PETSC_ERR_USER_INPUT, "Input must have an entry for each field. Has %" PetscInt_FMT " number of fields. Inputted %"PetscInt_FMT" entries" , numfields,num_in);
  if (flg) PetscCall(NetRPSetConstantData(rp,u)); 
  PetscCall(PetscFree(u));

  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode NetRPView_Constant(NetRP rp, PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode NetRPDestroy_Constant(NetRP rp) {
  ConstantCtx       *ctx = (ConstantCtx*) rp->data; 

  PetscFunctionBegin; 
  if(ctx->u)  PetscCall(PetscFree2(ctx->u,ctx->flux));
  PetscCall(PetscFree(rp->data));
  PetscCall(PetscObjectComposeFunction((PetscObject)rp, "NetRPConstantSetData_C", NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode NetRPSetUp_Constant(NetRP rp) {
  PetscInt           numfields;  
  ConstantCtx       *ctx = (ConstantCtx*) rp->data; 


  PetscFunctionBegin; 
  PetscCall(NetRPGetNumFields(rp, &numfields));
  PetscCall(PetscMalloc2(numfields,&ctx->u,numfields,&ctx->flux)); 
  PetscFunctionReturn(PETSC_SUCCESS);
}
/* ------------------------------------------------------------ */

PETSC_EXTERN PetscErrorCode NetRPCreate_Constant(NetRP rp)
{
  ConstantCtx *ctx; 

  PetscFunctionBegin;
  PetscCall(PetscNew(&ctx)); 
  rp->data                = ctx;
  rp->ops->setfromoptions = NetRPSetFromOptions_Constant;
  rp->ops->view           = NetRPView_Constant;
  rp->ops->solveFlux      = NetRPSolveFlux_Constant;
  rp->ops->destroy        = NetRPDestroy_Constant;
  rp->ops->setup          = NetRPSetUp_Constant; 

  rp->physicsgenerality = Generic;
  rp->solvetype         = Other;

    PetscCall(PetscObjectComposeFunction((PetscObject)rp, "NetRPConstantSetData_C", NetRPSetConstantData));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/* External functions here */

PetscErrorCode NetRPConstantSetData(NetRP rp, PetscScalar *u)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(rp, NETRP_CLASSID, 1);
  PetscValidPointer(u, 2);
  PetscTryMethod(rp, "NetRPConstantSetData_C", (NetRP, PetscScalar *), (rp, u));
  PetscFunctionReturn(PETSC_SUCCESS);
}