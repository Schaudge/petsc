#include <petsc/private/riemannsolverimpl.h>        /*I "petscriemannsolver.h"  I*/
#include <petscviewer.h>
#include <petscdraw.h>

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
PetscErrorCode  RiemannSolverSetup(RiemannSolver rs)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rs,RIEMANNSOLVER_CLASSID,1);
  if (rs->setupcalled) PetscFunctionReturn(0); 
  ierr = PetscMalloc2(rs->numfields,&rs->flux_wrk,rs->numfields,&rs->eig_wrk);CHKERRQ(ierr);
  if (rs->ops->setup) {
    ierr = (*rs->ops->setup)(rs);CHKERRQ(ierr);
  }
  rs->setupcalled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@
   RiemannSolverReset - Resets a RiemannSolver context and removes any allocated internal petsc objects

   Collective on RiemanSolver

   Input Parameter:
.  rs - the RiemanmSolver context obtained from RiemannSolverCreate()

   Level: beginner

.seealso: RiemannSolverCreate(), RiemannSolverSetup(), RiemannSolverDestroy()
@*/
PetscErrorCode  RiemannSolverReset(RiemannSolver rs)
{
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rs,RIEMANNSOLVER_CLASSID,1);
  if (rs->ops->reset) {
    ierr = (*rs->ops->reset)(rs);CHKERRQ(ierr);
  }
  if (rs->snes) {ierr = SNESReset(rs->snes);CHKERRQ(ierr);}
  if (rs->ksp)  {ierr = KSPReset(rs->ksp);CHKERRQ(ierr);}
  ierr = MatDestroy(&rs->mat);CHKERRQ(ierr);
  if (rs->flux_wrk) {ierr = PetscFree2(rs->flux_wrk,rs->eig_wrk);CHKERRQ(ierr);} /* Not good code here */
  rs->dim = -1; 
  rs->numfields = -1; 
  rs->setupcalled = PETSC_FALSE;
  PetscFunctionReturn(0);
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
PetscErrorCode  RiemannSolverDestroy(RiemannSolver *rs)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*rs) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(*rs,RIEMANNSOLVER_CLASSID,1);
  if (--((PetscObject)(*rs))->refct > 0) {*rs = NULL; PetscFunctionReturn(0);}

  ierr = RiemannSolverReset(*rs);CHKERRQ(ierr);
  if ((*rs)->ops->destroy) {ierr = (*(*rs)->ops->destroy)((*rs));CHKERRQ(ierr);}
  ierr = SNESDestroy(&(*rs)->snes);CHKERRQ(ierr);
  ierr = KSPDestroy(&(*rs)->ksp);CHKERRQ(ierr);
  ierr = MatDestroy(&(*rs)->mat);CHKERRQ(ierr);
  ierr = PetscHeaderDestroy(rs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
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
PetscErrorCode  RiemannSolverEvaluate(RiemannSolver rs,const PetscReal *uL, const PetscReal *uR,PetscReal **flux, PetscReal *maxspeed)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rs,RIEMANNSOLVER_CLASSID,1);
  ierr = rs->ops->evaluate(rs,uL,uR);CHKERRQ(ierr);
  *flux = rs->flux_wrk;
  *maxspeed = rs->maxspeed;
  PetscFunctionReturn(0);
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
PetscErrorCode  RiemannSolverSetApplicationContext(RiemannSolver rs,void *usrP)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(rs,RIEMANNSOLVER_CLASSID,1);
  rs->user = usrP;
  PetscFunctionReturn(0);
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
PetscErrorCode  RiemannSolverGetApplicationContext(RiemannSolver rs,void *usrP)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(rs,RIEMANNSOLVER_CLASSID,1);
  *(void**)usrP = rs->user;
  PetscFunctionReturn(0);
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
PetscErrorCode RiemannSolverSetFlux(RiemannSolver rs,PetscInt dim, PetscInt numfields,PetscPointFlux flux)
{
   PetscErrorCode ierr;

   PetscFunctionBegin;
   PetscValidHeaderSpecific(rs,RIEMANNSOLVER_CLASSID,1);

   /* WIP : Only 1-dim Riemann Solvers are supported */ 
   if (dim != 1) {SETERRQ1(PetscObjectComm((PetscObject)rs),PETSC_ERR_SUP,"%i dimension for flux functions are not supported. Only 1 dimensional flux function are supported. ",dim);}
 
   if (dim < 1) {SETERRQ1(PetscObjectComm((PetscObject)rs),PETSC_ERR_SUP,"%i dimension not valid. Dimension must be non-negative ",dim);}
   if (numfields < 1){SETERRQ1(PetscObjectComm((PetscObject)rs),PETSC_ERR_SUP,"%i numfields not valid. numfields must be non-negative ",numfields);}

   if((numfields != rs->numfields || dim != rs->dim) &&  rs->setupcalled) /* Reset internal data structures */
   {
      ierr = RiemannSolverReset(rs);CHKERRQ(ierr);
   }
   rs->dim = dim; 
   rs->numfields = numfields; 
   rs->fluxfun = flux; 
   ierr = RiemannSolverSetup(rs);CHKERRQ(ierr);
   PetscFunctionReturn(0);
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

/* 
   TODO : This needs some more thought/work 
*/

PetscErrorCode RiemannSolverSetFluxEig(RiemannSolver rs,PetscPointFluxEig fluxeig) 
{
   PetscErrorCode ierr;

   PetscFunctionBegin;
   PetscValidHeaderSpecific(rs,RIEMANNSOLVER_CLASSID,1);

   rs->fluxeigfun = fluxeig; 
   ierr = RiemannSolverSetup(rs);CHKERRQ(ierr);
   PetscFunctionReturn(0);
}


/*@
    RiemannSolverComputeEig - Compute the Eigenvalues of the flux derivative at a given field point U. 

    Collective

    Input Parameter:
.   rs  - The RiemannSolver context obtained from RiemannSolverCreate()
.   U   - The field point. A numfield sized array. 

   Output Parameter: 
.  eig  - Numfield sized array containing the computed eigenvalues. 

   Level: beginner 

.seealso:  RiemannSolverSetFluxEig(), RiemannSolverSetFlux()
@*/

PetscErrorCode RiemannSolverComputeEig(RiemannSolver rs,const PetscReal *U,PetscScalar **eig) 
{
   PetscErrorCode ierr;
   void           *ctx;

   PetscFunctionBegin;
   PetscValidHeaderSpecific(rs,RIEMANNSOLVER_CLASSID,1);
   ierr = RiemannSolverGetApplicationContext(rs,&ctx);CHKERRQ(ierr);
   if(rs->fluxeigfun) 
   { 
      rs->fluxeigfun(ctx,U,rs->eig_wrk);
   } else {
      SETERRQ(PetscObjectComm((PetscObject)rs),PETSC_ERR_SUP,"No function specified for computing the eigenvalues.");
   }
   *eig= rs->eig_wrk;
   PetscFunctionReturn(0);
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
  const char    *defaultType;
  char           name[256];
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rs, RIEMANNSOLVER_CLASSID, 1);
  if (!((PetscObject) rs)->type_name) {
    defaultType = RIEMANNLAXFRIEDRICH;
  } else {
    defaultType = ((PetscObject) rs)->type_name;
  }
  if (!RiemannSolverRegisterAllCalled) {ierr = RiemannSolverRegisterAll();CHKERRQ(ierr);}

  ierr = PetscObjectOptionsBegin((PetscObject) rs);CHKERRQ(ierr);
  ierr = PetscOptionsFList("-riemann_type", "Riemann Solver", "RiemannSolverSetType", RiemannSolverList, defaultType, name, 256, &flg);CHKERRQ(ierr);
  if (flg) {
    ierr = RiemannSolverSetType(rs, name);CHKERRQ(ierr);
  } else if (!((PetscObject) rs)->type_name) {
    ierr = RiemannSolverSetType(rs, defaultType);CHKERRQ(ierr);
  }
  if (rs->ops->setfromoptions) {
    ierr = (*rs->ops->setfromoptions)(PetscOptionsObject,rs);CHKERRQ(ierr);
  }
  /* process any options handlers added with PetscObjectAddOptionsHandler() */
  ierr = PetscObjectProcessOptionsHandlers(PetscOptionsObject,(PetscObject) rs);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  /*
    TODO:  View from options here ? 
  */
  PetscFunctionReturn(0);
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
PetscErrorCode  RiemannSolverView(RiemannSolver rs,PetscViewer viewer)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /*
   TODO 
  */
  PetscFunctionReturn(0);
}
