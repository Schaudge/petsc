#include <petsc/private/riemannsolverimpl.h>        /*I "petscriemannsolver.h"  I*/
#include <petscviewer.h>
#include <petscdraw.h>
#include <petscmat.h>
#include <petscksp.h>

/*
   Internal Default Behavior to compute max wave speeds for convex flux functions. Add reference 
   Explaining why this makes sense. 
*/

PetscErrorCode RiemannSolverConvexMaxSpeed_internal(RiemannSolver rs,const PetscReal *uL,const PetscReal *uR,PetscReal *maxspeed) 
{
   PetscErrorCode ierr;
   PetscInt       i;
   PetscScalar    *eig;

  PetscFunctionBeginUser;
  /* Compute maximum eigenvalue in magnitude for left states */
  ierr = RiemannSolverComputeEig(rs,uL,&eig);CHKERRQ(ierr);
  *maxspeed = 0; 
  for(i=0;i<rs->numfields; i++) {
      /* This only handles real eigenvalues, needs to generalized to handle complex eigenvalues */
      /* Strictly speaking, a conservation law requires these eigenvalues to be real, but numerically 
      there may be complex parts. */ 
    *maxspeed = PetscMax(PetscAbs(eig[i]),*maxspeed); 
  }
   /* Now maximize over the eigenvalues of the right states */
  ierr = RiemannSolverComputeEig(rs,uR,&eig);CHKERRQ(ierr);
  for(i=0;i<rs->numfields; i++) {
      /* This only handles real eigenvalues, needs to generalized to handle complex eigenvalues */
      /* Strictly speaking, a conservation law requires these eigenvalues to be real, but numerically 
      there may be complex parts. */ 
    *maxspeed = PetscMax(PetscAbs(eig[i]),*maxspeed); 
  }
  PetscFunctionReturn(0);
}

/*@
   RiemannSolverSetUpRoe_internal - Internal Specification for how to setup the Roe matrices and Roe matrix Linear Solvers. 
   Called in RiemannSolverSetUp(), seperated for convience of editing. 

   Collective on RiemannSolver

   Input Parameter:
.  rs - the RiemannSolver context obtained from RiemanSolverCreate()

   Level: developer

.seealso: RiemannSolverSetUp()
@*/
PetscErrorCode RiemannSolverSetUpRoe_internal(RiemannSolver rs) 
{
  PetscErrorCode ierr;
  PC             pc;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rs,RIEMANNSOLVER_CLASSID,1);
  

  /* Allow these parameters to be adjusted by the user later? NEED TO LOOK AT HOW TS DOES THIS TO COPY */

   /* does not share the same communicator as the RiemannSolver, does this affect diagnostic printout behavior?
      Need to be careful with this */
  ierr = MatCreateSeqDense(PETSC_COMM_SELF,rs->numfields,rs->numfields,PETSC_NULL,&rs->roemat);CHKERRQ(ierr);
  

  /*
   TODO: Rewrite this as default behavior and expose the roeksp (and roemat) to the RiemannSolver user so 
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
  ierr = KSPCreate(PETSC_COMM_SELF,&rs->roeksp);CHKERRQ(ierr);
  ierr = KSPGetPC(rs->roeksp,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCLU);CHKERRQ(ierr);
  ierr = KSPSetType(rs->roeksp,KSPPREONLY);CHKERRQ(ierr); /* Set to direct solver only */
  ierr = KSPSetOperators(rs->roeksp,rs->roemat,rs->roemat);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/*@
   RiemannSolverResetRoe_internal - Internal Specification for how to reset the Roe matrices and Roe matrix Linear Solvers. 
   Called in RiemannSolverReset()), seperated for convience of editing. 

   Collective on RiemannSolver

   Input Parameter:
.  rs - the RiemannSolver context obtained from RiemanSolverCreate()

   Level: developer

.seealso: RiemannSolverSetUp(), RiemannSolverSetUpRoe_internal(), RiemannSolverReset()
@*/
PetscErrorCode RiemannSolverResetRoe_internal(RiemannSolver rs) 
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rs,RIEMANNSOLVER_CLASSID,1);
  ierr = MatDestroy(&rs->roemat);CHKERRQ(ierr);
  ierr = MatDestroy(&rs->roematinv);CHKERRQ(ierr);
  ierr = KSPDestroy(&rs->roeksp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   RiemannSolverSetUpEig_internal - Internal Specification for how to setup the Eig Decomposition matrices. 

   Collective on RiemannSolver

   Input Parameter:
.  rs - the RiemannSolver context obtained from RiemanSolverCreate()

   Level: developer

.seealso: RiemannSolverSetUp()
@*/
PetscErrorCode RiemannSolverSetUpEig_internal(RiemannSolver rs) 
{
  PetscErrorCode ierr;
  PC             pc;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rs,RIEMANNSOLVER_CLASSID,1);
  

  /* Allow these parameters to be adjusted by the user later? NEED TO LOOK AT HOW TS DOES THIS TO COPY */

   /* does not share the same communicator as the RiemannSolver, does this affect diagnostic printout behavior?
      Need to be careful with this */
  ierr = MatCreateSeqDense(PETSC_COMM_SELF,rs->numfields,rs->numfields,PETSC_NULL,&rs->eigen);CHKERRQ(ierr);
  /* Now set up the linear solver. */ 
  ierr = KSPCreate(PETSC_COMM_SELF,&rs->eigenksp);CHKERRQ(ierr);
  ierr = KSPGetPC(rs->eigenksp,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCLU);CHKERRQ(ierr);
  ierr = KSPSetType(rs->eigenksp,KSPPREONLY);CHKERRQ(ierr); /* Set to direct solver only */
  ierr = KSPSetOperators(rs->eigenksp,rs->eigen,rs->eigen);CHKERRQ(ierr);
  
  /* Set the PetscVectors used for the kspsolve operation for the change of basis */
  /* Maybe should do manual solves? Well I guess I'll see as I start profiling? */
  ierr = VecCreateSeq(PETSC_COMM_SELF,rs->numfields,&rs->u);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,rs->numfields,&rs->ueig);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   RiemannSolverResetEig_internal - TODO

   Collective on RiemannSolver

   Input Parameter:
.  rs - the RiemannSolver context obtained from RiemanSolverCreate()

   Level: developer

.seealso: RiemannSolverSetUp(), RiemannSolverSetUpRoe_internal(), RiemannSolverReset()
@*/
PetscErrorCode RiemannSolverResetEig_internal(RiemannSolver rs) 
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rs,RIEMANNSOLVER_CLASSID,1);
  ierr = MatDestroy(&rs->eigen);CHKERRQ(ierr);
  ierr = KSPDestroy(&rs->eigenksp);CHKERRQ(ierr);
  ierr = VecDestroy(&rs->u);CHKERRQ(ierr);
  ierr = VecDestroy(&rs->ueig);CHKERRQ(ierr);
  PetscFunctionReturn(0);
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
PetscErrorCode  RiemannSolverSetUp(RiemannSolver rs)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rs,RIEMANNSOLVER_CLASSID,1);
  if (rs->setupcalled) PetscFunctionReturn(0); 
  if (rs->numfields>-1) {ierr = PetscMalloc2(rs->numfields,&rs->flux_wrk,rs->numfields,&rs->eig_wrk);CHKERRQ(ierr);}
  if (rs->fluxfunconvex) {rs->ops->computemaxspeed = RiemannSolverConvexMaxSpeed_internal;} /* No current default behavior for nonconvex fluxs. Will error out currently */
  /* if we have a roe function allocate the structures to use it */
  if (rs->computeroemat) {ierr = RiemannSolverSetUpRoe_internal(rs);CHKERRQ(ierr);}
  if (rs->computeeigbasis){ierr = RiemannSolverSetUpEig_internal(rs);CHKERRQ(ierr);}
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

.seealso: RiemannSolverCreate(), RiemannSolverSetUp(), RiemannSolverDestroy()
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
  ierr = RiemannSolverResetRoe_internal(rs);CHKERRQ(ierr);
  ierr = RiemannSolverResetEig_internal(rs);CHKERRQ(ierr);
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
  ierr = RiemannSolverSetUp(rs);CHKERRQ(ierr);
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
   PetscFunctionBegin;
   PetscValidHeaderSpecific(rs,RIEMANNSOLVER_CLASSID,1);

   /* WIP : Only 1-dim Riemann Solvers are supported */ 
   if (dim != 1) {SETERRQ1(PetscObjectComm((PetscObject)rs),PETSC_ERR_SUP,"%i dimension for flux functions are not supported. Only 1 dimensional flux function are supported. ",dim);}
 
   if (dim < 1) {SETERRQ1(PetscObjectComm((PetscObject)rs),PETSC_ERR_SUP,"%i dimension not valid. Dimension must be non-negative ",dim);}
   if (numfields < 1){SETERRQ1(PetscObjectComm((PetscObject)rs),PETSC_ERR_SUP,"%i numfields not valid. numfields must be non-negative ",numfields);}

   rs->dim = dim;
   rs->numfields = numfields; 
   rs->fluxfun = flux; 
   PetscFunctionReturn(0);
}
/*@
    RiemannSolverSetFluxEig -  User specified function to compute the maximum wave speed for a riemann problem. This 
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

PetscErrorCode RiemannSolverSetMaxSpeedFunct(RiemannSolver rs ,RiemannSolverMaxWaveSpeed maxspeedfunct)
{
   PetscFunctionBegin;
   PetscValidHeaderSpecific(rs,RIEMANNSOLVER_CLASSID,1);
   rs->ops->computemaxspeed = maxspeedfunct; 
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
   PetscFunctionBegin;
   PetscValidHeaderSpecific(rs,RIEMANNSOLVER_CLASSID,1);
   rs->fluxeigfun = fluxeig; 
   PetscFunctionReturn(0);
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

PetscErrorCode RiemannSolverComputeMaxSpeed(RiemannSolver rs,const PetscReal *uL,const PetscReal *uR,PetscReal *maxspeed)
{
   PetscErrorCode ierr;

   PetscFunctionBegin;
   PetscValidHeaderSpecific(rs,RIEMANNSOLVER_CLASSID,1);
   if(rs->ops->computemaxspeed) 
   { 
      ierr = rs->ops->computemaxspeed(rs,uR,uL,maxspeed);CHKERRQ(ierr);
   } else {
      SETERRQ(PetscObjectComm((PetscObject)rs),PETSC_ERR_SUP,"No function specified for computing the maximum wave speed. This shouldn't happen.");
   }
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

  PetscFunctionBegin;
  /*
   TODO 
  */
  PetscFunctionReturn(0);
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

PetscErrorCode RiemannSolverComputeRoeMatrix(RiemannSolver rs,const PetscReal *uL,const PetscReal *uR ,Mat *Roe)
{
   PetscErrorCode ierr;
   void           *ctx;

   PetscFunctionBegin;
   PetscValidHeaderSpecific(rs,RIEMANNSOLVER_CLASSID,1);
   ierr = RiemannSolverGetApplicationContext(rs,&ctx);CHKERRQ(ierr);
   if(rs->computeroemat) 
   { 
      ierr = rs->computeroemat(ctx,uR,uL,&rs->roemat);CHKERRQ(ierr);
      *Roe  = rs->roemat;
   } else {
      SETERRQ(PetscObjectComm((PetscObject)rs),PETSC_ERR_SUP,"No Roe Matrix Specified. A function to construct a Roe Matrix must be specified by the User. ");
   }
   PetscFunctionReturn(0);
}

/*@
    RiemannSolverSetRoeMatrixFunct- Sets the function to compute the roe matrix for the given physics model. These 
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
PetscErrorCode RiemannSolverSetRoeMatrixFunct(RiemannSolver rs,RiemannSolverRoeMatrix roematfunct)
{
   PetscFunctionBegin;
   PetscValidHeaderSpecific(rs,RIEMANNSOLVER_CLASSID,1);
   rs->computeroemat = roematfunct; 
   PetscFunctionReturn(0);
}

/* 
   TODO : Finish Documentations

   ALSO REDO ALL OF THIS STUFF. SO ugly 
*/
PetscErrorCode RiemannSolverSetEigBasis(RiemannSolver rs,RiemannSolverEigBasis eigbasisfunct)
{
   PetscFunctionBegin;
   PetscValidHeaderSpecific(rs,RIEMANNSOLVER_CLASSID,1);
   rs->computeeigbasis = eigbasisfunct; 
   PetscFunctionReturn(0);
}

PetscErrorCode RiemannSolverComputeEigBasis(RiemannSolver rs,const PetscReal *u, Mat *EigBasis)
{
   PetscErrorCode ierr;
   void           *ctx;

   PetscFunctionBegin;
   PetscValidHeaderSpecific(rs,RIEMANNSOLVER_CLASSID,1);
   ierr = RiemannSolverGetApplicationContext(rs,&ctx);CHKERRQ(ierr);
   if(rs->computeeigbasis) 
   { 
      ierr = rs->computeeigbasis(ctx,u,rs->eigen);CHKERRQ(ierr);
      *EigBasis  = rs->eigen;
   } else {
      SETERRQ(PetscObjectComm((PetscObject)rs),PETSC_ERR_SUP,"No Eigen Decomposition Specified. ");
   }
   PetscFunctionReturn(0);
}

PetscErrorCode RiemannSolverChangetoEigBasis(RiemannSolver rs, const PetscReal *u,PetscReal *ueig)
{
   PetscErrorCode ierr;
   void           *ctx;

   PetscFunctionBegin;
   PetscValidHeaderSpecific(rs,RIEMANNSOLVER_CLASSID,1);
   ierr = RiemannSolverGetApplicationContext(rs,&ctx);CHKERRQ(ierr);
   ierr = VecPlaceArray(rs->u,u);CHKERRQ(ierr);
   ierr = VecPlaceArray(rs->ueig,ueig);CHKERRQ(ierr);
   ierr = KSPSolve(rs->eigenksp,rs->u,rs->ueig);CHKERRQ(ierr);
   ierr = VecResetArray(rs->u);CHKERRQ(ierr);
   ierr = VecResetArray(rs->ueig);CHKERRQ(ierr);
   PetscFunctionReturn(0);
}

PetscErrorCode RiemannSolverChangetoRiemEigBasis(RiemannSolver rs, const PetscReal *u,PetscReal *ueig)
{
   PetscErrorCode ierr;
   void           *ctx;

   PetscFunctionBegin;
   PetscValidHeaderSpecific(rs,RIEMANNSOLVER_CLASSID,1);
   ierr = RiemannSolverGetApplicationContext(rs,&ctx);CHKERRQ(ierr);
   ierr = VecPlaceArray(rs->u,u);CHKERRQ(ierr);
   ierr = VecPlaceArray(rs->ueig,ueig);CHKERRQ(ierr);
   ierr = KSPSolve(rs->eigenksp,rs->u,rs->ueig);CHKERRQ(ierr);
   ierr = VecResetArray(rs->u);CHKERRQ(ierr);
   ierr = VecResetArray(rs->ueig);CHKERRQ(ierr);
   PetscFunctionReturn(0);
}

PetscErrorCode RiemannSolverComputeRoeAvg(RiemannSolver rs,const PetscReal *uL,const PetscReal *uR,PetscReal *uavg)
{
   PetscErrorCode ierr;
   void           *ctx;

   PetscFunctionBegin;
   PetscValidHeaderSpecific(rs,RIEMANNSOLVER_CLASSID,1);
   ierr = RiemannSolverGetApplicationContext(rs,&ctx);CHKERRQ(ierr);
   if (rs->roeavg) {
      rs->roeavg(ctx,uL,uR,uavg);
   } else { 
      SETERRQ(PetscObjectComm((PetscObject)rs),PETSC_ERR_SUP,"No Roe Average Function Specified");
   }

   PetscFunctionReturn(0);
}

PetscErrorCode RiemannSolverCharNorm(RiemannSolver rs, const PetscReal *uroe, const PetscReal *u, PetscInt sgn,PetscReal *norm)
{
   PetscErrorCode ierr;
   void           *ctx;
   PetscInt       field;
   PetscScalar    *uchar,*eig; 

   PetscFunctionBegin;
   PetscValidHeaderSpecific(rs,RIEMANNSOLVER_CLASSID,1);
   ierr = RiemannSolverGetApplicationContext(rs,&ctx);CHKERRQ(ierr);
   ierr = RiemannSolverComputeEigBasis(rs,uroe,&rs->eigen);CHKERRQ(ierr);
   ierr = VecPlaceArray(rs->u,u);CHKERRQ(ierr);
   ierr = KSPSolve(rs->eigenksp,rs->u,rs->ueig);CHKERRQ(ierr);
   ierr = VecResetArray(rs->u);CHKERRQ(ierr);
   ierr = VecGetArray(rs->ueig,&uchar);CHKERRQ(ierr);
   /* Projection on the positive or negative characteristic basis */
   ierr = RiemannSolverComputeEig(rs,uroe,&eig);CHKERRQ(ierr);
   if (sgn<0) 
   {
      for(field=0; field<rs->numfields; field++)
      {
         if(eig[field]>0) {uchar[field] = 0.0;}
      }
   } else {
      for(field=0; field<rs->numfields; field++)
      {
         if(eig[field]<0) {uchar[field] = 0.0;}
      }
   }
   ierr = VecRestoreArray(rs->ueig,&uchar);CHKERRQ(ierr);
   ierr = MatMult(rs->eigen,rs->ueig,rs->u);CHKERRQ(ierr);
   ierr = VecNorm(rs->u,NORM_2,norm);CHKERRQ(ierr);
   PetscFunctionReturn(0);
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
PetscErrorCode RiemannSolverSetRoeAvgFunct(RiemannSolver rs,RiemannSolverRoeAvg roeavgfunct)
{
   PetscFunctionBegin;
   PetscValidHeaderSpecific(rs,RIEMANNSOLVER_CLASSID,1);
   rs->roeavg = roeavgfunct; 
   PetscFunctionReturn(0);
}