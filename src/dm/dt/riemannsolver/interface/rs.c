#include <petsc/private/riemannsolverimpl.h> /*I "petscriemannsolver.h"  I*/
#include <petscviewer.h>
#include <petscdraw.h>
#include <petscmat.h>
#include <petscksp.h>

/*
   Internal Default Behavior to compute max wave speeds for convex flux functions. Add reference 
   Explaining why this makes sense. 
*/

PetscErrorCode RiemannSolverConvexMaxSpeed_internal(RiemannSolver rs, const PetscReal *uL, const PetscReal *uR, PetscReal *maxspeed)
{
  PetscInt     i;
  PetscScalar *eig;

  PetscFunctionBegin;
  /* Compute maximum eigenvalue in magnitude for left states */
  PetscCall(RiemannSolverComputeEig(rs, uL, &eig));
  *maxspeed = 0;
  for (i = 0; i < rs->numfields; i++) {
    /* This only handles real eigenvalues, needs to generalized to handle complex eigenvalues */
    /* Strictly speaking, a conservation law requires these eigenvalues to be real, but numerically 
      there may be complex parts. */
    *maxspeed = PetscMax(PetscAbs(eig[i]), *maxspeed);
  }
  /* Now maximize over the eigenvalues of the right states */
  PetscCall(RiemannSolverComputeEig(rs, uR, &eig));
  for (i = 0; i < rs->numfields; i++) {
    /* This only handles real eigenvalues, needs to generalized to handle complex eigenvalues */
    /* Strictly speaking, a conservation law requires these eigenvalues to be real, but numerically 
      there may be complex parts. */
    *maxspeed = PetscMax(PetscAbs(eig[i]), *maxspeed);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Developer Note : Perhaps the RiemannSolver implementations should responsible for calling these internal set up 
  routines. So if a RiemannSolver implementation requires access to roematrices, jacobian matrices or etc ... 
  they activate it themselves, otherwise they don't bother. Would save some memory and marginal setup time. Though 
  in the grand scheme this might not save much (relative to the other costs in a balance law simulation). 
*/

/*C
   RiemannSolverSetUpJacobian_internal - Internal Specification for how to setup the jacobian matrix and jacobian solver. 

   Collective on RiemannSolver

   Input Parameter:
.  rs - the RiemannSolver context obtained from RiemanSolverCreate()

   Level: developer

.seealso: RiemannSolverSetUp()
C*/
PetscErrorCode RiemannSolverSetUpJacobian_internal(RiemannSolver rs)
{
  PC pc;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rs, RIEMANNSOLVER_CLASSID, 1);
  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, rs->numfields, rs->numfields, PETSC_IGNORE, &rs->Df));
  /* Now set up the linear solver. */
  PetscCall(KSPCreate(PETSC_COMM_SELF, &rs->dfksp));
  PetscCall(KSPGetPC(rs->dfksp, &pc));
  PetscCall(PCSetType(pc, PCLU));
  PetscCall(KSPSetType(rs->dfksp, KSPPREONLY)); /* Set to direct solver only */
  PetscCall(KSPSetOperators(rs->dfksp, rs->Df, rs->Df));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*C
   RiemannSolverResetJacobian_internal - Internal Specification for how to reset the Jacobian matrices.

   Collective on RiemannSolver

   Input Parameter:
.  rs - the RiemannSolver context obtained from RiemanSolverCreate()

   Level: developer

.seealso: RiemannSolverSetUp(), RiemannSolverSetUpRoe_internal(), RiemannSolverReset()
C*/
PetscErrorCode RiemannSolverResetJacobian_internal(RiemannSolver rs)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(rs, RIEMANNSOLVER_CLASSID, 1);
  PetscCall(MatDestroy(&rs->Df));
  PetscCall(KSPDestroy(&rs->dfksp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*C
   RiemannSolverSetUpRoe_internal - Internal Specification for how to setup the Roe matrices and Roe matrix Linear Solvers. 
   Called in RiemannSolverSetUp(), seperated for convience of editing. 

   Collective on RiemannSolver

   Input Parameter:
.  rs - the RiemannSolver context obtained from RiemanSolverCreate()

   Level: developer

.seealso: RiemannSolverSetUp()
C*/
PetscErrorCode RiemannSolverSetUpRoe_internal(RiemannSolver rs)
{
  PC pc;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rs, RIEMANNSOLVER_CLASSID, 1);

  /* Allow these parameters to be adjusted by the user later? NEED TO LOOK AT HOW TS DOES THIS TO COPY */

  /* does not share the same communicator as the RiemannSolver, does this affect diagnostic printout behavior?
      Need to be careful with this */
  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, rs->numfields, rs->numfields, PETSC_IGNORE, &rs->roemat));
  /* Note that this eigenmatrix could potentially reuse the eigen matrix, as in many cases (SWE Euler, 
  the roe avg is simply A(uL,uR)= Df(u_roe(uL,uR)) and will have the same eigen decomposition as Df */
  PetscCall(MatDuplicate(rs->roemat, MAT_DO_NOT_COPY_VALUES, &rs->roeeigen));
  PetscCall(MatCreateVecs(rs->roeeigen, NULL, &rs->u_roebasis));
  /*
   TODO: Rewrite this as default behavior and expose the roeksp (and roemat, roe ksp) to the RiemannSolver user so 
   that they can configure them as they would any other Mat and KSP object. Definitely smart to have good default
   behavior for these solvers. 
   
   Also, looking towards the generalized network riemann solvers, having the ability 
   to specify riemann solver / physics specific mat structures and ksp solvers is essential. As the resulting 
   riemann solvers will have very special structures that can be exploited for efficiency. For example 
   the linearized riemann solver of Jingmei Qiu will have a special physics independent strucutre to the matrix
   that should be exploited by a specialized linear solve. This doesn't directly relate to Roe matrices but I expect 
   (and we shall see as I implement more riemann solvers) that there will be similar situations there. 
  */

  /* Now set up the linear solver. */
  PetscCall(KSPCreate(PETSC_COMM_SELF, &rs->roeksp));
  PetscCall(KSPGetPC(rs->roeksp, &pc));
  PetscCall(PCSetType(pc, PCLU));
  PetscCall(KSPSetType(rs->roeksp, KSPPREONLY));                      /* Set to direct solver only */
  PetscCall(KSPSetOperators(rs->roeksp, rs->roeeigen, rs->roeeigen)); /* used to project onto roe eigenbasis */

  PetscFunctionReturn(PETSC_SUCCESS);
}

/*C
   RiemannSolverResetRoe_internal - Internal Specification for how to reset the Roe matrices and Roe matrix Linear Solvers. 
   Called in RiemannSolverReset()), seperated for convience of editing. 

   Collective on RiemannSolver

   Input Parameter:
.  rs - the RiemannSolver context obtained from RiemanSolverCreate()

   Level: developer

.seealso: RiemannSolverSetUp(), RiemannSolverSetUpRoe_internal(), RiemannSolverReset()
C*/
PetscErrorCode RiemannSolverResetRoe_internal(RiemannSolver rs)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(rs, RIEMANNSOLVER_CLASSID, 1);
  PetscCall(MatDestroy(&rs->roemat));
  PetscCall(MatDestroy(&rs->roeeigen));
  PetscCall(KSPDestroy(&rs->roeksp));
  PetscCall(VecDestroy(&rs->u_roebasis));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*C
   RiemannSolverSetUpEig_internal - Internal Specification for how to setup the Eig Decomposition matrices. 

   Collective on RiemannSolver

   Input Parameter:
.  rs - the RiemannSolver context obtained from RiemanSolverCreate()

   Level: developer

.seealso: RiemannSolverSetUp()
C*/
PetscErrorCode RiemannSolverSetUpEig_internal(RiemannSolver rs)
{
  PC pc;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rs, RIEMANNSOLVER_CLASSID, 1);

  /* Allow these parameters to be adjusted by the user later? NEED TO LOOK AT HOW TS DOES THIS TO COPY */

  /* does not share the same communicator as the RiemannSolver, does this affect diagnostic printout behavior?
      Need to be careful with this */
  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, rs->numfields, rs->numfields, PETSC_IGNORE, &rs->eigen));
  /* Now set up the linear solver. */
  PetscCall(KSPCreate(PETSC_COMM_SELF, &rs->eigenksp));
  PetscCall(KSPGetPC(rs->eigenksp, &pc));
  PetscCall(PCSetType(pc, PCLU));
  PetscCall(KSPSetType(rs->eigenksp, KSPPREONLY)); /* Set to direct solver only */
  PetscCall(KSPSetOperators(rs->eigenksp, rs->eigen, rs->eigen));

  /* Set the PetscVectors used for the kspsolve operation for the change of basis */
  /* Maybe should do manual solves? Well I guess I'll see as I start profiling? */
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, rs->numfields, &rs->u));
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, rs->numfields, &rs->ueig));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*C
   RiemannSolverResetEig_internal - TODO

   Collective on RiemannSolver

   Input Parameter:
.  rs - the RiemannSolver context obtained from RiemanSolverCreate()

   Level: developer

.seealso: RiemannSolverSetUp(), RiemannSolverSetUpRoe_internal(), RiemannSolverReset()
C*/
PetscErrorCode RiemannSolverResetEig_internal(RiemannSolver rs)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(rs, RIEMANNSOLVER_CLASSID, 1);
  PetscCall(MatDestroy(&rs->eigen));
  PetscCall(KSPDestroy(&rs->eigenksp));
  PetscCall(VecDestroy(&rs->u));
  PetscCall(VecDestroy(&rs->ueig));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   RiemannSolverSetUp - Sets up the internal data structures for the later use of a RiemannSolver. 

   Collective on RiemannSolver

   Input Parameter:
.  rs - the RiemannSolver context obtained from RiemanSolverCreate()

   Notes:
   Internally called when setting the flux function as internal data structures depend on the 
   dim and numfield parameters set there. Will not normally be called by users. 

   Level: advanced

.seealso: RiemannSolverCreate(), RiemannSolverSetFlux()
@*/
PetscErrorCode RiemannSolverSetUp(RiemannSolver rs)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(rs, RIEMANNSOLVER_CLASSID, 1);
  if (rs->setupcalled) PetscFunctionReturn(PETSC_SUCCESS);
  if (rs->numfields > -1) PetscCall(PetscMalloc2(rs->numfields, &rs->flux_wrk, rs->numfields, &rs->eig_wrk));
  if (rs->fluxfunconvex) { rs->ops->computemaxspeed = RiemannSolverConvexMaxSpeed_internal; } /* No current default behavior for nonconvex fluxs. Will error out currently */
  /* if we have a roe function allocate the structures to use it */
  if (rs->computeroemat) PetscCall(RiemannSolverSetUpRoe_internal(rs));
  if (rs->computeeigbasis) PetscCall(RiemannSolverSetUpEig_internal(rs));
  if (rs->fluxderfun) PetscCall(RiemannSolverSetUpJacobian_internal(rs));
  if (rs->ops->setup) { PetscCall((*rs->ops->setup)(rs)); }
  rs->setupcalled = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   RiemannSolverReset - Resets a RiemannSolver context and removes any allocated internal petsc objects

   Collective on RiemanSolver

   Input Parameter:
.  rs - the RiemanmSolver context obtained from RiemannSolverCreate()

   Level: beginner

.seealso: RiemannSolverCreate(), RiemannSolverSetUp(), RiemannSolverDestroy()
@*/
PetscErrorCode RiemannSolverReset(RiemannSolver rs)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(rs, RIEMANNSOLVER_CLASSID, 1);
  if (rs->ops->reset) { PetscCall((*rs->ops->reset)(rs)); }
  if (rs->snes) PetscCall(SNESReset(rs->snes));
  if (rs->ksp) PetscCall(KSPReset(rs->ksp));
  PetscCall(MatDestroy(&rs->mat));
  if (rs->flux_wrk) PetscCall(PetscFree2(rs->flux_wrk, rs->eig_wrk)); /* Not good code here */
  PetscCall(RiemannSolverResetRoe_internal(rs));
  PetscCall(RiemannSolverResetEig_internal(rs));
  PetscCall(RiemannSolverResetJacobian_internal(rs));
  /* Don't reset the physics portions of the riemannsolver (user inputted functions and dim, numfields) 
  as a user might want to swap the type of the riemann solver without having the reinput all of the physics of 
  the riemannsolver */
  rs->setupcalled = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   RiemannSolverDestroy - Destroys the RiemannSolver context that was created
   with RiemannSolverCreate().

   Collective on RiemannSolver

   Input Parameter:
.  rs - the RiemannSolver context obtained from RiemannSolverCreate()

   Level: beginner

.seealso: RiemannSolverCreate(), RiemannSolverSetUp()
@*/
PetscErrorCode RiemannSolverDestroy(RiemannSolver *rs)
{
  PetscFunctionBegin;
  if (!*rs) PetscFunctionReturn(PETSC_SUCCESS);
  PetscValidHeaderSpecific(*rs, RIEMANNSOLVER_CLASSID, 1);
  if (--((PetscObject)(*rs))->refct > 0) {
    *rs = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscCall(RiemannSolverReset(*rs));
  if ((*rs)->ops->destroy) PetscCall((*(*rs)->ops->destroy)((*rs)));
  PetscCall(SNESDestroy(&(*rs)->snes));
  PetscCall(KSPDestroy(&(*rs)->ksp));
  PetscCall(MatDestroy(&(*rs)->mat));
  PetscCall(PetscHeaderDestroy(rs));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   RiemannSolverEvaluate - Evaluate the Riemann Solver

   Not Collective on RiemannSolver

   Input Parameter:
.  rs  - The RiemannSolver context obtained from RiemannSolverCreate()
.  uL - An array with rs->numfield entries containing the left state of the riemann data
.  uR - An array with rs->numfield entries containing the right state of the riemann data

   Output Parameter: 
.  flux     -  location to put pointer to the array of length numfields*dim containing the numerical flux. This array is owned by the 
               RiemannSolver and should not be deallocated by the user.
   maxspeed -  Maximum wave speed computed by the RiemannSolver. Intended to be used for CFL computations. 

   Level: beginner

.seealso: RiemannSolverCreate(), RiemannSolverSetUp(), RiemannSolverSetFlux()
@*/
PetscErrorCode RiemannSolverEvaluate(RiemannSolver rs, const PetscReal *uL, const PetscReal *uR, PetscReal **flux, PetscReal *maxspeed)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(rs, RIEMANNSOLVER_CLASSID, 1);
  PetscCall(RiemannSolverSetUp(rs));
  PetscCall(rs->ops->evaluate(rs, uL, uR));
  *flux = rs->flux_wrk;
  if (maxspeed) { *maxspeed = rs->maxspeed; }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RiemannSolverEvaluateFlux(RiemannSolver rs, const PetscReal *u, PetscReal **flux)
{
  void *ctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rs, RIEMANNSOLVER_CLASSID, 1);
  PetscCall(RiemannSolverSetUp(rs));
  PetscCall(RiemannSolverGetApplicationContext(rs, &ctx));
  rs->fluxfun(ctx, u, rs->flux_wrk);
  *flux = rs->flux_wrk;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   RiemannSolverSetApplicationContext - Sets an optional user-defined context for
   the RiemannSolver.

   Logically Collective on TS

   Input Parameters:
+  rs - the TS context obtained from RiemannSolverCreate()
-  usrP - optional user context

   Level: intermediate

.seealso: RiemannSolverGetApplicationContext()
@*/
PetscErrorCode RiemannSolverSetApplicationContext(RiemannSolver rs, void *usrP)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(rs, RIEMANNSOLVER_CLASSID, 1);
  rs->user = usrP;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
    RiemannSolverGetApplicationContext - Gets the user-defined context for the
    RiemannSolver

    Not Collective

    Input Parameter:
.   rs - the RiemannSolver context obtained from RiemannSolverCreate()

    Output Parameter:
.   usrP - user context

    Level: intermediate

.seealso: RiemannSolverSetApplicationContext()
@*/
PetscErrorCode RiemannSolverGetApplicationContext(RiemannSolver rs, void *usrP)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(rs, RIEMANNSOLVER_CLASSID, 1);
  *(void **)usrP = rs->user;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
    RiemannSolverSetFluxDim - Sets the dimensions of the Flux Function to be used. This is called internally by 
    RiemannSolverSetFlux() and would not normally be called directly by users. It is convenient for testing purposes.  

    Collective

    Input Parameter:
.   rs  - The RiemannSolver context obtained from RiemannSolverCreate()
.   dim - The domain dimension of the flux function 
.   numfields  - The number of fields for the flux function, i.e range dimension

   Level: developer 

.seealso: RiemannSolverSetFlux() 
@*/
PetscErrorCode RiemannSolverSetFluxDim(RiemannSolver rs, PetscInt dim, PetscInt numfields)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(rs, RIEMANNSOLVER_CLASSID, 1);
  /* WIP : Only 1-dim Riemann Solvers are supported */
  if (dim != 1) { SETERRQ(PetscObjectComm((PetscObject)rs), PETSC_ERR_SUP, "%i dimension for flux functions are not supported. Only 1 dimensional flux function are supported. ", dim); }
  if (dim < 1) { SETERRQ(PetscObjectComm((PetscObject)rs), PETSC_ERR_SUP, "%i dimension not valid. Dimension must be non-negative ", dim); }
  if (numfields < 1) { SETERRQ(PetscObjectComm((PetscObject)rs), PETSC_ERR_SUP, "%i numfields not valid. numfields must be non-negative ", numfields); }

  rs->dim       = dim;
  rs->numfields = numfields;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
    RiemannSolverSetFlux - Sets the flux function used to compute the numerical flux. 

    Collective

    Input Parameter:
.   rs  - The RiemannSolver context obtained from RiemannSolverCreate()
.   dim - The domain dimension of the flux function 
.   numfields  - The number of fields for the flux function, i.e range dimension
.   flux - The flux function

   Level: beginner 

.seealso: RiemannSolverSetApplicationContext(), RiemannSolverEvaluate() 
@*/
PetscErrorCode RiemannSolverSetFlux(RiemannSolver rs, PetscInt dim, PetscInt numfields, PetscPointFlux flux)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(rs, RIEMANNSOLVER_CLASSID, 1);
  PetscCall(RiemannSolverSetFluxDim(rs, dim, numfields));
  rs->fluxfun = flux;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RiemannSolverGetNumFields(RiemannSolver rs, PetscInt *numfields)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(rs, RIEMANNSOLVER_CLASSID, 1);
  *numfields = rs->numfields;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
    RiemannSolverSetJacobian - Sets the jacobian flux function.

    Collective

    Input Parameter:
.   rs  - The RiemannSolver context obtained from RiemannSolverCreate()
.   flux - The flux function

   Level: beginner 

.seealso: RiemannSolverSetApplicationContext(), RiemannSolverSetFlux()
@*/
PetscErrorCode RiemannSolverSetJacobian(RiemannSolver rs, PetscPointFluxDer jacobian)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(rs, RIEMANNSOLVER_CLASSID, 1);
  rs->fluxderfun = jacobian;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
    RiemannSolverSetMaxSpeedFunct -  User specified function to compute the maximum wave speed for a riemann problem. This 
    has standard default behavior (for convex flux functions) for any riemann solver implementation that is specified 
    on setting type. Override this only if you know what you are doing and can expect improved behavior using some function specific to your particular 
    physics. Must be called after setting all other options. 

    Collective

    Input Parameter:
.   rs  - The RiemannSolver context obtained from RiemannSolverCreate()
.   maxspeedfunct - The function to compute the wave speed generated by the riemann problem. 

   Level: expert 

.seealso: RiemannSolverSetApplicationContext(), RiemannSolverSetFlux()) 
@*/

PetscErrorCode RiemannSolverSetMaxSpeedFunct(RiemannSolver rs, RiemannSolverMaxWaveSpeed maxspeedfunct)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(rs, RIEMANNSOLVER_CLASSID, 1);
  rs->ops->computemaxspeed = maxspeedfunct;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
    RiemannSolverSetFluxEig -  User specified function to compute the eigenvalues of the flux function derivative.
    This allows the RiemannSolver to use an analytical user specified function to compute the eigenvalues instead 
    of requiring a numerical eigenvalue computation. These eigenvalues are used to compute wave-speed estimates 
    for the RiemannSolver evaluation function. 

    Collective

    Input Parameter:
.   rs  - The RiemannSolver context obtained from RiemannSolverCreate()
.   fluxeig - The flux eigenvalue function

   Level: beginner 

.seealso: RiemannSolverSetApplicationContext(), RiemannSolverSetFlux()) 
@*/
PetscErrorCode RiemannSolverSetFluxEig(RiemannSolver rs, PetscPointFluxEig fluxeig)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(rs, RIEMANNSOLVER_CLASSID, 1);
  rs->fluxeigfun = fluxeig;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
    RiemannSolverComputeEig - Compute the Eigenvalues of the flux derivative at a given field point U. Usually only 
    called inside of a RiemannSolver implementation

    Collective

    Input Parameter:
.   rs  - The RiemannSolver context obtained from RiemannSolverCreate()
.   U   - The field point. A numfield sized array. 

   Output Parameter: 
.  eig  - Numfield sized array containing the computed eigenvalues. 

   Level: developer 

.seealso:  RiemannSolverSetFluxEig(), RiemannSolverSetFlux()
@*/

PetscErrorCode RiemannSolverComputeEig(RiemannSolver rs, const PetscReal *U, PetscScalar **eig)
{
  void *ctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rs, RIEMANNSOLVER_CLASSID, 1);
  PetscCall(RiemannSolverGetApplicationContext(rs, &ctx));
  if (rs->fluxeigfun) {
    rs->fluxeigfun(ctx, U, rs->eig_wrk);
  } else {
    SETERRQ(PetscObjectComm((PetscObject)rs), PETSC_ERR_SUP, "No function specified for computing the eigenvalues.");
  }
  *eig = rs->eig_wrk;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
    RiemannSolverComputeMaxSpeed - Compute the maximum wave speed for the given Riemann Problem. This is often 
    used within a RiemannSolver implementation. In most cases one would not call this function directly and 
    would instead call RiemannSolverEvaluate() to get the maximum wave speed along with the approximate Riemann 
    solution at the same time. 

    Collective
   Input Parameters: 

.  uL - An array with rs->numfield entries containing the left state of the riemann data
.  uR - An array with rs->numfield entries containing the right state of the riemann data

   Output Parameters: 
   maxspeed -  Maximum wave speed 

   Level: developer 

.seealso:  RiemannSolverSetFluxEig(), RiemannSolverSetFlux()
@*/

PetscErrorCode RiemannSolverComputeMaxSpeed(RiemannSolver rs, const PetscReal *uL, const PetscReal *uR, PetscReal *maxspeed)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(rs, RIEMANNSOLVER_CLASSID, 1);
  if (rs->ops->computemaxspeed) {
    PetscCall(rs->ops->computemaxspeed(rs, uR, uL, maxspeed));
  } else {
    SETERRQ(PetscObjectComm((PetscObject)rs), PETSC_ERR_SUP, "No function specified for computing the maximum wave speed. This shouldn't happen.");
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  RiemannSolverSetFromOptions - sets parameters in a RiemannSolver from the options database

  Collective on RiemannSolver

  Input Parameter:
. rs - the RiemannSolver object to set options for

  Options Database:

  Level: intermediate

.seealso 
@*/
PetscErrorCode RiemannSolverSetFromOptions(RiemannSolver rs)
{
  const char *defaultType;
  char        name[256];
  PetscBool   flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rs, RIEMANNSOLVER_CLASSID, 1);
  if (!((PetscObject)rs)->type_name) {
    defaultType = RIEMANNLAXFRIEDRICH;
  } else {
    defaultType = ((PetscObject)rs)->type_name;
  }
  if (!RiemannSolverRegisterAllCalled) PetscCall(RiemannSolverRegisterAll());

  PetscObjectOptionsBegin((PetscObject)rs);
  PetscCall(PetscOptionsFList("-riemann_type", "Riemann Solver", "RiemannSolverSetType", RiemannSolverList, defaultType, name, 256, &flg));
  if (flg) {
    PetscCall(RiemannSolverSetType(rs, name));
  } else if (!((PetscObject)rs)->type_name) {
    PetscCall(RiemannSolverSetType(rs, defaultType));
  }
  if (rs->ops->setfromoptions) { PetscCall((*rs->ops->setfromoptions)(PetscOptionsObject, rs)); }
  /* process any options handlers added with PetscObjectAddOptionsHandler() */
  PetscCall(PetscObjectProcessOptionsHandlers((PetscObject)rs, PetscOptionsObject));
  PetscOptionsEnd();
  /*
    TODO:  View from options here ? 
  */
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
    RiemannSolverView - Prints the RiemannSolver data structure.

    Collective on RiemannSovler

    Input Parameters:
+   rs - the RiemannSolver context obtained from RiemannSolverCreate()
-   viewer - visualization context

    Options Database Key:
   TODO: 
    Level: beginner

.seealso: PetscViewerASCIIOpen()
@*/
PetscErrorCode RiemannSolverView(RiemannSolver rs, PetscViewer viewer)
{
  PetscFunctionBegin;
  /*
   TODO 
  */
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
    RiemannSolverComputeRoeMatrix - Compute the Roe Matrix using the roe analytical roe matrix function 
    specified by the user. This is usually called by RiemannSolver implementations and should not be called 
    by the user in standard situations. 

    Collective
   Input Parameters: 

.  uL - An array with rs->numfield entries containing the left state of the riemann data
.  uR - An array with rs->numfield entries containing the right state of the riemann data

   Output Parameters: 
   Roe - A roe matrix for the states uL, uR. This is allocated by the RiemannSolver and should not be destroyed 
   by the user. 

   Level: advanced

.seealso:  RiemannSolverComputeEig()
@*/

PetscErrorCode RiemannSolverComputeRoeMatrix(RiemannSolver rs, const PetscReal *uL, const PetscReal *uR, Mat *Roe)
{
  void *ctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rs, RIEMANNSOLVER_CLASSID, 1);
  PetscCall(RiemannSolverGetApplicationContext(rs, &ctx));
  if (rs->computeroemat) {
    PetscCall(rs->computeroemat(ctx, uR, uL, rs->roemat));
    *Roe = rs->roemat;
  } else {
    SETERRQ(PetscObjectComm((PetscObject)rs), PETSC_ERR_SUP, "No Roe Matrix Specified. A function to construct a Roe Matrix must be specified by the User. ");
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
    RiemannSolverSetRoeMatrixFunct - Sets the function to compute the roe matrix for the given physics model. These 
    are required for any RiemanSolver that makes use of roe matrix internally (ADD A UTILITY FUNCTION FOR LISTING 
    THE RIEMANNSOLVERS THAT REQUIRE THIS)

    Collective

    Input Parameter:
.   rs  - The RiemannSolver context obtained from RiemannSolverCreate()
.   roematfunct - A RiemannSolverRoeMatrix function for computing the roe matrix. These are derived for each individual 
                  physics model. ADD REFERENCE HERE 

   Level: beginner 

.seealso: RiemannSolverSetFlux(), RiemannSolverComputeRoeMatrix()
@*/
PetscErrorCode RiemannSolverSetRoeMatrixFunct(RiemannSolver rs, RiemannSolverRoeMatrix roematfunct)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(rs, RIEMANNSOLVER_CLASSID, 1);
  rs->computeroemat = roematfunct;
  PetscFunctionReturn(PETSC_SUCCESS);
}
/* 
   TODO : Finish Documentations

   ALSO REDO ALL OF THIS STUFF. SO ugly 
*/
PetscErrorCode RiemannSolverSetLaxCurve(RiemannSolver rs, LaxCurve laxcurve)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(rs, RIEMANNSOLVER_CLASSID, 1);
  rs->evallaxcurve = laxcurve;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* 
   TODO        :  Finish Documentations
   ALSO REDO ALL OF THIS STUFF. SO ugly 
*/

PetscErrorCode RiemannSolverEvalLaxCurve(RiemannSolver rs, const PetscReal *u, PetscReal xi, PetscInt wavenumber, PetscReal *ubar)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(rs, RIEMANNSOLVER_CLASSID, 1);
  if (rs->evallaxcurve) {
    PetscCall(rs->evallaxcurve(rs, u, xi, wavenumber, ubar));
  } else {
    SETERRQ(PetscObjectComm((PetscObject)rs), PETSC_ERR_SUP, "No Lax Curve Function Specified");
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
/* 
   TODO : Finish Documentations

   ALSO REDO ALL OF THIS STUFF. SO ugly 
*/
PetscErrorCode RiemannSolverSetEigBasis(RiemannSolver rs, RiemannSolverEigBasis eigbasisfunct)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(rs, RIEMANNSOLVER_CLASSID, 1);
  rs->computeeigbasis = eigbasisfunct;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RiemannSolverComputeEigBasis(RiemannSolver rs, const PetscReal *u, Mat *EigBasis)
{
  void *ctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rs, RIEMANNSOLVER_CLASSID, 1);
  PetscCall(RiemannSolverGetApplicationContext(rs, &ctx));
  if (rs->computeeigbasis) {
    PetscCall(rs->computeeigbasis(ctx, u, rs->eigen));
    *EigBasis = rs->eigen;
  } else {
    SETERRQ(PetscObjectComm((PetscObject)rs), PETSC_ERR_SUP, "No Eigen Decomposition Specified. ");
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RiemannSolverChangetoEigBasis(RiemannSolver rs, const PetscReal *u, PetscReal *ueig)
{
  void *ctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rs, RIEMANNSOLVER_CLASSID, 1);
  PetscCall(RiemannSolverGetApplicationContext(rs, &ctx));
  PetscCall(VecPlaceArray(rs->u, u));
  PetscCall(VecPlaceArray(rs->ueig, ueig));
  PetscCall(KSPSolve(rs->eigenksp, rs->u, rs->ueig));
  PetscCall(VecResetArray(rs->u));
  PetscCall(VecResetArray(rs->ueig));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RiemannSolverComputeRoeAvg(RiemannSolver rs, const PetscReal *uL, const PetscReal *uR, PetscReal *uavg)
{
  void *ctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rs, RIEMANNSOLVER_CLASSID, 1);
  PetscCall(RiemannSolverGetApplicationContext(rs, &ctx));
  if (rs->roeavg) {
    PetscCall(rs->roeavg(ctx, uL, uR, uavg));
  } else {
    SETERRQ(PetscObjectComm((PetscObject)rs), PETSC_ERR_SUP, "No Roe Average Function Specified");
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
/* Compute the norm of the positive or negative eigenvector at u components of of a vector x */
PetscErrorCode RiemannSolverCharNorm(RiemannSolver rs, const PetscReal *u, const PetscReal *x, PetscInt sgn, PetscReal *norm)
{
  void        *ctx;
  PetscInt     field;
  PetscScalar *uchar, *eig;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rs, RIEMANNSOLVER_CLASSID, 1);
  PetscCall(RiemannSolverGetApplicationContext(rs, &ctx));
  PetscCall(RiemannSolverComputeEigBasis(rs, u, &rs->eigen));
  PetscCall(VecPlaceArray(rs->u, x));
  PetscCall(KSPSolve(rs->eigenksp, rs->u, rs->ueig));
  PetscCall(VecResetArray(rs->u));
  PetscCall(VecGetArray(rs->ueig, &uchar));
  /* Projection on the positive or negative characteristic basis */
  PetscCall(RiemannSolverComputeEig(rs, u, &eig));
  if (sgn < 0) {
    for (field = 0; field < rs->numfields; field++) {
      if (eig[field] > 0) {
        uchar[field] = 0.0;
      } else {
        uchar[field] *= eig[field];
      }
    }
  } else {
    for (field = 0; field < rs->numfields; field++) {
      if (eig[field] < 0) {
        uchar[field] = 0.0;
      } else {
        uchar[field] *= eig[field];
      }
    }
  }
  PetscCall(VecRestoreArray(rs->ueig, &uchar));
  PetscCall(MatMult(rs->eigen, rs->ueig, rs->u));
  PetscCall(VecNorm(rs->u, NORM_2, norm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
    RiemannSolverSetRoeAvgFunct - Sets Function to compute the roe average. 

    Collective

    Input Parameter:
.   rs  - The RiemannSolver context obtained from RiemannSolverCreate()
.   roeavgfunct - A RiemannSolverRoeMatrix function for computing the roe matrix. These are derived for each individual 
                  physics model. ADD REFERENCE HERE 

   Level: beginner 

.seealso: RiemannSolverSetFlux(), RiemannSolverComputeRoeMatrix()
@*/
PetscErrorCode RiemannSolverSetRoeAvgFunct(RiemannSolver rs, RiemannSolverRoeAvg roeavgfunct)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(rs, RIEMANNSOLVER_CLASSID, 1);
  rs->roeavg = roeavgfunct;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
    RiemannSolverTestEigDecomposition - Tests whether the provided eigenbasis and eigenvalue functions are actually 
    eigenvectors/eigenvalues of the provided fluxderivative function DF_u at the point u. Internally this simply 
    checks if  DF_u(u) R(u) = R(u)\lambda(u), where R(u) is the matrix of eigenvectors and \lambda(u) is the diagonal 
    matrix of eigenvalues. This is useful sanity check to test if user provided (or numerically computed) flux and
    eigen functions are correct. 

    Collective

    Input Parameter:
.   rs  -       The RiemannSolver context obtained from RiemannSolverCreate()
.   numvalues - The number of values that are being tested 
.   u   -       The array of values to test the eigendecomposition at
.   tol -       The tolerance to compute equality to. Input PETSC_DECIDE to let petsc decide the tolerance. 
    Output Parameter: 
.   isequal -  isequal[i] is true if \| DF(u) R(u) - R(u)\lambda(u)\|_\infty for the u[i] value. Allocated by caller.
.   norms (optional) - The \| DF(u) R(u) - R(u)\lambda(u)\|_\infty norms computed. Input null if not desired. Allocated
    by caller. 

   Level: intermediate

.seealso: RiemannSolverSetFlux(), RiemannSolverComputeRoeMatrix()
@*/
PetscErrorCode RiemannSolverTestEigDecomposition(RiemannSolver rs, PetscInt numvalues, const PetscReal **u, PetscReal tol, PetscBool *isequal, PetscReal *norms, PetscViewer viewer)
{
  Mat          DF, R, Eig, DFR, EigR, Diff;
  Vec          Eig_vec;
  PetscScalar *eig;
  PetscInt     i;
  PetscReal    norm;
  PetscBool    isascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rs, RIEMANNSOLVER_CLASSID, 1);
  /* If tol is PETSC_DECIDE set default. More complicated defaults could be implemented here */
  if (tol == PETSC_DECIDE) tol = 1e-10;
  /* Preallocate diagonal eigenvalue matrix for all values */
  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, rs->numfields, rs->numfields, PETSC_IGNORE, &Eig));
  /* Preallocate result of DFR-EigR */
  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, rs->numfields, rs->numfields, PETSC_IGNORE, &Diff));
  PetscCall(MatZeroEntries(Eig));
  PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF, 1, rs->numfields, NULL, &Eig_vec));
  for (i = 0; i < numvalues; i++) {
    PetscCall(RiemannSolverComputeEig(rs, u[i], &eig));
    PetscCall(VecPlaceArray(Eig_vec, eig));
    PetscCall(MatDiagonalSet(Eig, Eig_vec, INSERT_VALUES));
    PetscCall(VecResetArray(Eig_vec));
    PetscCall(RiemannSolverComputeJacobian(rs, u[i], &DF));
    PetscCall(RiemannSolverComputeEigBasis(rs, u[i], &R));
    /* In the first loop allocate the product matrices, reuse them throughout */
    if (i == 0) {
      PetscCall(MatMatMult(DF, R, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &DFR));
      PetscCall(MatMatMult(R, Eig, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &EigR));
    } else {
      PetscCall(MatMatMult(DF, R, MAT_REUSE_MATRIX, PETSC_DEFAULT, &DFR));
      PetscCall(MatMatMult(R, Eig, MAT_REUSE_MATRIX, PETSC_DEFAULT, &EigR));
    }
    PetscCall(MatCopy(DFR, Diff, SAME_NONZERO_PATTERN));
    PetscCall(MatAXPY(Diff, -1., EigR, SAME_NONZERO_PATTERN));
    PetscCall(MatNorm(Diff, NORM_INFINITY, &norm));
    if (norms) { norms[i] = norm; }
    if (isequal) { isequal[i] = (norm < tol); }

    if (norm > tol) {
      if (viewer) {
        PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
        if (isascii) {
          PetscCall(PetscViewerASCIIPushTab(viewer));
          PetscCall(PetscViewerASCIIPrintf(viewer, " The EigenDecomposition Failed for the Following State: \n"));
          PetscCall(PetscScalarView(rs->numfields, u[i], viewer));

          PetscCall(PetscViewerASCIIPushTab(viewer));

          PetscCall(PetscViewerASCIIPrintf(viewer, "\n\nEigenBasis (R): \n"));
          PetscCall(PetscViewerASCIIPushTab(viewer));
          PetscCall(MatView(R, viewer));
          PetscCall(PetscViewerASCIIPopTab(viewer));

          PetscCall(PetscViewerASCIIPrintf(viewer, "Jacobian (DF): \n"));
          PetscCall(PetscViewerASCIIPushTab(viewer));
          PetscCall(MatView(DF, viewer));
          PetscCall(PetscViewerASCIIPopTab(viewer));

          PetscCall(PetscViewerASCIIPrintf(viewer, "Eigenvalues (E): \n"));
          PetscCall(PetscViewerASCIIPushTab(viewer));
          PetscCall(MatView(Eig, viewer));
          PetscCall(PetscViewerASCIIPopTab(viewer));

          PetscCall(PetscViewerASCIIPrintf(viewer, "DF*R: \n"));
          PetscCall(PetscViewerASCIIPushTab(viewer));
          PetscCall(MatView(DFR, viewer));
          PetscCall(PetscViewerASCIIPopTab(viewer));

          PetscCall(PetscViewerASCIIPrintf(viewer, "R*E: \n"));
          PetscCall(PetscViewerASCIIPushTab(viewer));
          PetscCall(MatView(EigR, viewer));
          PetscCall(PetscViewerASCIIPopTab(viewer));

          PetscCall(PetscViewerASCIIPrintf(viewer, "R*E-DF*R: \n"));
          PetscCall(PetscViewerASCIIPushTab(viewer));
          PetscCall(MatView(Diff, viewer));
          PetscCall(PetscViewerASCIIPopTab(viewer));

          PetscCall(PetscViewerASCIIPrintf(viewer, " ||DF*R - R*E|| = %e\n\n\n", norm));

          PetscCall(PetscViewerASCIIPopTab(viewer));
          PetscCall(PetscViewerASCIIPopTab(viewer));
        }
      }
    }
  }
  PetscCall(MatDestroy(&DFR));
  PetscCall(MatDestroy(&EigR));
  PetscCall(MatDestroy(&Diff));
  PetscCall(MatDestroy(&Eig));
  PetscCall(VecDestroy(&Eig_vec));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RiemannSolverComputeJacobian(RiemannSolver rs, const PetscReal *u, Mat *jacobian)
{
  void *ctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rs, RIEMANNSOLVER_CLASSID, 1);
  PetscCall(RiemannSolverGetApplicationContext(rs, &ctx));
  if (rs->fluxderfun) {
    PetscCall(rs->fluxderfun(ctx, u, rs->Df));
    *jacobian = rs->Df;
  } else {
    SETERRQ(PetscObjectComm((PetscObject)rs), PETSC_ERR_SUP, "No Jacobian Function Specified");
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
    RiemannSolverTestRoeMat -  Test whether the roe average satisfies its defining properties. Namely 

    1. Hyperbolicity (NOT IMPLEMENTED CURRENTLY)
    2. Consistency A_roe(U,U) = Df(U) 
    3. Conservation A_roe(uR-uL) = f(uR) - f(uL) 
    Collective

    Input Parameter:
.   rs  -        The RiemannSolver context obtained from RiemannSolverCreate()
.   numvalues -  The number of values that are being tested 
.   ul   -       The array of left values to test the roematrix at. Also check consistency at these points.
.   ur   -       The array of right values to test the roematrix at.
.   tol  -        The tolerance to compute equality to. Input PETSC_DECIDE to let petsc decide the tolerance.  
    Output Parameter: 
.   isequal -   numvalues*3 array allocated by caller. Each isequal[3*j+i] true if test i passes for point j. 
.   norms   - numvalues*2 array with the norms computed for test 2 and 3. Set to NULL if not desired. 
   Level: intermediate

.seealso:  RiemannSolverComputeRoeMatrix()
@*/
PetscErrorCode RiemannSolverTestRoeMat(RiemannSolver rs, PetscInt numvalues, const PetscReal **uL, const PetscReal **uR, PetscReal tol, PetscBool *isequal, PetscReal *norms, PetscViewer viewer)
{
  Mat            DF, ARoe, Diff;
  Vec            Au, F_diff;
  PetscScalar   *u_diff, *f_diff;
  PetscInt       i, j;
  PetscErrorCode ierr;
  PetscReal      norm;
  PetscBool      isascii;
  void          *ctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rs, RIEMANNSOLVER_CLASSID, 1);
  PetscCall(RiemannSolverGetApplicationContext(rs, &ctx));
  /* If tol is PETSC_DECIDE set default. More complicated defaults could be implemented here */
  if (tol == PETSC_DECIDE) tol = 1e-10;
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, rs->numfields, &Au));
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, rs->numfields, &F_diff));
  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, rs->numfields, rs->numfields, NULL, &Diff));
  for (i = 0; i < numvalues; i++) {
    /* Check conservation */
    PetscCall(RiemannSolverComputeRoeMatrix(rs, uL[i], uR[i], &ARoe));
    PetscCall(VecGetArray(rs->u, &u_diff)); /* use work vector */
    for (j = 0; j < rs->numfields; j++) {   /* compute jump uR-uL */
      u_diff[j] = uR[i][j] - uL[i][j];
    }
    PetscCall(VecRestoreArray(rs->u, &u_diff));
    PetscCall(MatMult(ARoe, rs->u, Au));
    PetscCall(VecGetArray(F_diff, &f_diff));
    rs->fluxfun(ctx, uR[i], f_diff);
    CHKERRQ(ierr);
    rs->fluxfun(ctx, uL[i], rs->flux_wrk);
    CHKERRQ(ierr);
    for (j = 0; j < rs->numfields; j++) { /* compute jump f(uR)-f(uL) */
      f_diff[j] -= rs->flux_wrk[j];
    }
    PetscCall(VecRestoreArray(F_diff, &f_diff));
    PetscCall(VecAXPY(F_diff, -1., Au));
    PetscCall(VecNorm(F_diff, NORM_INFINITY, &norm));
    if (isequal) isequal[3 * i + 2] = (norm < tol);
    if (norms) { norms[2 * i + 1] = norm; }
    /* View if there are failures */
    if (norm > tol) {
      if (viewer) {
        PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
        if (isascii) {
          ierr = PetscViewerASCIIPushTab(viewer);
          PetscCall(PetscViewerASCIIPrintf(viewer, "The Roe Matrix Failed Conservation for the Following State: \n"));

          ierr = PetscViewerASCIIPushTab(viewer);
          PetscCall(PetscViewerASCIIPrintf(viewer, "uL:"));
          PetscCall(PetscScalarView(rs->numfields, uL[i], viewer));
          PetscCall(PetscViewerASCIIPrintf(viewer, "uR:"));
          PetscCall(PetscScalarView(rs->numfields, uR[i], viewer));

          PetscCall(PetscViewerASCIIPrintf(viewer, "The Roe Matrix: \n"));
          ierr = PetscViewerASCIIPushTab(viewer);
          PetscCall(MatView(ARoe, viewer));
          ierr = PetscViewerASCIIPopTab(viewer);

          PetscCall(PetscViewerASCIIPrintf(viewer, "F(uR) - F(uL) - ARoe(uR-uL) \n"));
          ierr = PetscViewerASCIIPushTab(viewer);
          PetscCall(VecView(F_diff, viewer));
          ierr = PetscViewerASCIIPopTab(viewer);

          PetscCall(PetscViewerASCIIPrintf(viewer, " ||F(uR) - F(uL) - ARoe(uR-uL)|| = %e\n \n \n", norm));

          ierr = PetscViewerASCIIPopTab(viewer);
          ierr = PetscViewerASCIIPopTab(viewer);
        }
      }
    }
    /* Check consistency */
    PetscCall(RiemannSolverComputeRoeMatrix(rs, uL[i], uL[i], &ARoe));
    PetscCall(RiemannSolverComputeJacobian(rs, uL[i], &DF));
    PetscCall(MatCopy(ARoe, Diff, SAME_NONZERO_PATTERN));
    PetscCall(MatAXPY(Diff, -1., DF, SAME_NONZERO_PATTERN));
    PetscCall(MatNorm(Diff, NORM_INFINITY, &norm));
    if (isequal) isequal[3 * i + 1] = (norm < tol);
    if (norms) { norms[2 * i] = norm; }
    if (norm > tol) {
      if (viewer) {
        PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
        if (isascii) {
          ierr = PetscViewerASCIIPushTab(viewer);
          PetscCall(PetscViewerASCIIPrintf(viewer, "The Roe Matrix Failed Consistency for the Following State: \n"));

          ierr = PetscViewerASCIIPushTab(viewer);
          PetscCall(PetscScalarView(rs->numfields, uL[i], viewer));

          PetscCall(PetscViewerASCIIPrintf(viewer, "The Roe Matrix: \n\n"));
          ierr = PetscViewerASCIIPushTab(viewer);
          PetscCall(MatView(ARoe, viewer)); /* see if I can print only entries without the type and ranks */
          ierr = PetscViewerASCIIPopTab(viewer);

          PetscCall(PetscViewerASCIIPrintf(viewer, "The Jacobian Matrix: \n"));
          ierr = PetscViewerASCIIPushTab(viewer);
          PetscCall(MatView(DF, viewer));
          ierr = PetscViewerASCIIPopTab(viewer);

          PetscCall(PetscViewerASCIIPrintf(viewer, "ARoe - DF: \n"));
          ierr = PetscViewerASCIIPushTab(viewer);
          PetscCall(MatView(Diff, viewer));
          ierr = PetscViewerASCIIPopTab(viewer);

          PetscCall(PetscViewerASCIIPrintf(viewer, " ||DF(u) - ARoe(u)|| = %e\n\n\n", norm));

          ierr = PetscViewerASCIIPopTab(viewer);
          ierr = PetscViewerASCIIPopTab(viewer);
        }
      }
    }
  }
  PetscCall(VecDestroy(&Au));
  PetscCall(VecDestroy(&F_diff));
  PetscCall(MatDestroy(&Diff));
  PetscFunctionReturn(PETSC_SUCCESS);
}
