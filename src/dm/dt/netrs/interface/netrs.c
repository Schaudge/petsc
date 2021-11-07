#include <petscriemannsolver.h>       
#include <petscviewer.h>
#include <petscdraw.h>
#include <petscmat.h>
#include <petscksp.h>
#include <petsc/private/netrsimpl.h>
#include <petscnetrs.h>

#include <petsc/private/riemannsolverimpl.h> /* to be removed after adding fluxfunction class */

/*@
   NetRSSetUp - Sets up the internal data structures for the later use of a NetRS. 

   Collective on NetRS

   Input Parameter:
.  rs - the NetRS context obtained from RiemanSolverCreate()

   Notes:
   Internally called when setting the flux function as internal data structures depend on the 
   dim and numfield parameters set there. Will not normally be called by users. 

   Level: advanced

.seealso: NetRSCreate(), NetRSSetFlux()
@*/
PetscErrorCode  NetRSSetUp(NetRS rs)
{
  PetscErrorCode ierr;
  PetscInt       i; 

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rs,NETRS_CLASSID,1);
  if (rs->setupcalled) PetscFunctionReturn(0); 
  if (rs->ops->setup) {
    ierr = (*rs->ops->setup)(rs);CHKERRQ(ierr);
  }
  if (rs->numfields>-1 && rs->numedges>-1) {ierr = PetscMalloc1(rs->numedges*rs->numfields,&rs->flux_wrk);CHKERRQ(ierr);}
  if (rs->estimate) {ierr = PetscMalloc2(rs->numfields,&rs->est_wrk,rs->numfields,&rs->est_wrk2);CHKERRQ(ierr);}
  /* default value for error array is -1. This allows for knowing if the requested error array in an eval routine actually computed anything (i.e error 
  estimator is not assigned ) */ 
  ierr = PetscMalloc1(rs->numedges,&rs->error);CHKERRQ(ierr);
  for(i=0;i<rs->numedges;i++) {rs->error[i] = -1;}

  ierr = RiemannSolverSetUp(rs->rs);CHKERRQ(ierr);
  rs->setupcalled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@
   NetRSReset - Resets a NetRS context and removes any allocated internal petsc objects

   Collective on RiemanSolver

   Input Parameter:
.  rs - the RiemanmSolver context obtained from NetRSCreate()

   Level: beginner

.seealso: NetRSCreate(), NetRSSetUp(), NetRSDestroy()
@*/
PetscErrorCode  NetRSReset(NetRS rs)
{
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rs,NETRS_CLASSID,1);
  if (rs->ops->reset) {
    ierr = (*rs->ops->reset)(rs);CHKERRQ(ierr);
  }
  if (rs->flux_wrk) {ierr = PetscFree(rs->flux_wrk);CHKERRQ(ierr);} /* Not good code here */
  if (rs->est_wrk) {ierr = PetscFree2(rs->est_wrk,rs->est_wrk2);CHKERRQ(ierr);} /* also not good */
  /* Note that we should reference the RiemannSolver inside the NetRS to properly handle this reset behavior. */
  ierr = PetscFree(rs->error);CHKERRQ(ierr);
  rs->setupcalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@
   NetRSDestroy - Destroys the NetRS context that was created
   with NetRSCreate().

   Collective on NetRS

   Input Parameter:
.  rs - the NetRS context obtained from NetRSCreate()

   Level: beginner

.seealso: NetRSCreate(), NetRSSetUp()
@*/
PetscErrorCode  NetRSDestroy(NetRS *rs)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*rs) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(*rs,NETRS_CLASSID,1);
  if (--((PetscObject)(*rs))->refct > 0) {*rs = NULL; PetscFunctionReturn(0);}
  /* destory the nested netrs */ 
  ierr = NetRSDestroy(&(*rs)->fine);CHKERRQ(ierr);

  ierr = NetRSReset(*rs);CHKERRQ(ierr);
  if ((*rs)->ops->destroy) {ierr = (*(*rs)->ops->destroy)((*rs));CHKERRQ(ierr);}
  ierr = PetscHeaderDestroy(rs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  NetRSDuplicate - Create a new netrs of the same type as the original with the same settings. Still requires a call to setup after this call 
  as the intended use is to set the parameters for a "master" netrs duplicate it to other NetRS and change the types of the new netrs to the desired types. 
  This is the quick way of getting multiple netrs of different types for the same physics. 
*/

PetscErrorCode NetRSDuplicate(NetRS netrs,NetRS *newnetrs)
{
  PetscErrorCode ierr; 
  MPI_Comm       comm;
  NetRS          netrs_new; 

  PetscFunctionBegin;
  PetscValidHeaderSpecific(netrs,NETRS_CLASSID,1);
  PetscValidPointer(newnetrs,2);
  PetscValidType(netrs,1);

  netrs_new = *newnetrs;
  ierr = PetscObjectGetComm((PetscObject)netrs,&comm);CHKERRQ(ierr);
  ierr = NetRSCreate(comm,&netrs_new);CHKERRQ(ierr); 
  /* copy over the parameters and physics from netrs to newnetrs */ 

  /* physics*/
  netrs_new->user      = netrs->user; 
  netrs_new->numfields = netrs->numfields; 
  netrs_new->numedges  = netrs->numedges;
  netrs_new->rs        = netrs->rs;
  /* error estimate */
  netrs_new->estimate  = netrs->estimate;
  netrs_new->useestimator = netrs->useestimator;
  /* adaptivity*/
  netrs_new->fine = netrs->fine; 
  netrs_new->finetype = netrs->finetype;
  netrs_new->useadaptivity = netrs->useadaptivity;
  netrs_new->finetol = netrs->finetol; 
  netrs_new->coarsetol = netrs->coarsetol;
  PetscFunctionReturn(0);
}


/*@
   NetRSEvaluate - Evaluate the Riemann Solver

   Not Collective on NetRS

   Input Parameter:
.  rs  - The NetRS context obtained from NetRSCreate()
.  u - An array with rs->numfield*rs->numedges entries containing the network riemann data
.  dir - An array with the direction of the directed graph at the vertex this network riemann solver is being called 

   Output Parameter: 
.  flux     -  location to put pointer to the array of length numfields*dim containing the numerical flux. This array is owned by the 
               NetRS and should not be deallocated by the user.
.  error    -  (optional) error computed by the error esimator (if available) pass in Null if not desired. one estimate for each edge of the NetRS 
                allocated by the netRS. Values will change between calls. 

   Level: beginner

.seealso: NetRSCreate(), NetRSSetUp(), NetRSSetFlux()
@*/
PetscErrorCode  NetRSEvaluate(NetRS rs,const PetscReal *u, const EdgeDirection *dir,PetscReal **flux,PetscReal **error)
{
  PetscErrorCode ierr;
  PetscInt       e; 
  PetscReal      errest; 

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rs,NETRS_CLASSID,1);
  ierr = NetRSSetUp(rs);CHKERRQ(ierr);
  ierr = rs->ops->evaluate(rs,u,dir,rs->flux_wrk,rs->error);CHKERRQ(ierr);
  if (error) {*error = rs->error;}
  *flux = rs->flux_wrk;

    /* adaptivity */
  if(rs->useadaptivity) {
    errest = 0.0; 
    for(e=0;e<rs->numedges;e++) {
      errest = PetscMax(rs->error[e],errest); 
    }
    if(errest >= rs->finetol) { /* revaluate with fine netrs */
      if(!rs->fine) { /* fine netrs hasn't been created. Create it by duplicating rs and changing its type to finetype */
        ierr = NetRSDuplicate(rs,&rs->fine);CHKERRQ(ierr); 
        ierr = NetRSSetType(rs->fine,rs->finetype);CHKERRQ(ierr); 
        rs->fine->useadaptivity = PETSC_FALSE; /* only allow two level adaptivity for now */
        ierr = NetRSSetUp(rs->fine);CHKERRQ(ierr);
      }
      ierr = NetRSEvaluate(rs->fine,u,dir,flux,error);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/*@
   NetRSSetApplicationContext - Sets an optional user-defined context for
   the NetRS.

   Logically Collective on TS

   Input Parameters:
+  rs - the NetRS context obtained from NetRSCreate()
-  usrP - optional user context

   Level: intermediate

.seealso: NetRSGetApplicationContext()
@*/
PetscErrorCode  NetRSSetApplicationContext(NetRS rs,void *usrP)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(rs,NETRS_CLASSID,1);
  rs->user = usrP;
  PetscFunctionReturn(0);
}

/*@
    NetRSGetApplicationContext - Gets the user-defined context for the
    NetRS

    Not Collective

    Input Parameter:
.   rs - the NetRS context obtained from NetRSCreate()

    Output Parameter:
.   usrP - user context

    Level: intermediate

.seealso: NetRSSetApplicationContext()
@*/
PetscErrorCode  NetRSGetApplicationContext(NetRS rs,void *usrP)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(rs,NETRS_CLASSID,1);
  *(void**)usrP = rs->user;
  PetscFunctionReturn(0);
}

/*@
  NetRSSetFromOptions - sets parameters in a NetRS from the options database

  Collective on NetRS

  Input Parameter:
. rs - the NetRS object to set options for

  Options Database:

  Level: intermediate

.seealso 
@*/
PetscErrorCode NetRSSetFromOptions(NetRS rs)
{
  const char    *defaultType;
  char           name[256];
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rs, NETRS_CLASSID, 1);
  if (!((PetscObject) rs)->type_name) {
    defaultType = NETRSLINEAR;
  } else {
    defaultType = ((PetscObject) rs)->type_name;
  }
  if (!NetRSRegisterAllCalled) {ierr = NetRSRegisterAll();CHKERRQ(ierr);}

  ierr = PetscObjectOptionsBegin((PetscObject) rs);CHKERRQ(ierr);
  ierr = PetscOptionsFList("-netrs_type", "NetRS", "NetRSSetType", NetRSList, defaultType, name, 256, &flg);CHKERRQ(ierr);
  if (flg) {
    ierr = NetRSSetType(rs, name);CHKERRQ(ierr);
  } else if (!((PetscObject) rs)->type_name) {
    ierr = NetRSSetType(rs, defaultType);CHKERRQ(ierr);
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
    NetRSView - Prints the NetRS data structure.

    Collective on RiemannSovler

    Input Parameters:
+   rs - the NetRS context obtained from NetRSCreate()
-   viewer - visualization context

    Options Database Key:
   TODO: 
    Level: beginner

.seealso: PetscViewerASCIIOpen()
@*/
PetscErrorCode  NetRSView(NetRS rs,PetscViewer viewer)
{

  PetscFunctionBegin;
  /*
   TODO 
  */
  PetscFunctionReturn(0);
}

/*
 ----- ERROR ESTIMATOR SUPPORT  -----=
    For internal use only for now 
*/
PetscErrorCode  NetRSErrorEstimate(NetRS rs,PetscInt dir,const PetscReal *u, const PetscReal *ustar, PetscReal *errorestimate)
{
  PetscErrorCode ierr;
  void           *ctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rs,NETRS_CLASSID,1);
  ierr = NetRSSetUp(rs);CHKERRQ(ierr);
  ierr = NetRSGetApplicationContext(rs,&ctx);CHKERRQ(ierr);
  if(rs->estimate) {
    ierr = rs->estimate(ctx,rs,dir,u,ustar,errorestimate);CHKERRQ(ierr); /* meh code */
  } else {
    SETERRQ(PetscObjectComm((PetscObject)rs),PETSC_ERR_SUP,"No error estimator specified for NetRS");
  }
  PetscFunctionReturn(0);
}

PetscErrorCode NetRSSetErrorEstimate(NetRS rs, NRSErrorEstimator errorestimator)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(rs,NETRS_CLASSID,1);
  rs->estimate = errorestimator; 
  PetscFunctionReturn(0);
}

/* WIP implementation of one type of error estimator */

PetscErrorCode NetRSRoeErrorEstimate(void *ctx,NetRS rs,PetscInt dir,const PetscReal *u,const PetscReal *ustar,PetscReal *estimate)
{
  PetscErrorCode ierr;
  PetscInt       field,sgn;

  PetscFunctionBegin;
  /* compute jump */
  for (field=0; field<rs->numfields; field++) {
    rs->est_wrk[field] = ustar[field] - u[field];
  }
  ierr = RiemannSolverComputeRoeAvg(rs->rs,u,ustar,rs->est_wrk2);CHKERRQ(ierr);
  sgn = dir == EDGEIN ? -1 : 1;
  ierr = RiemannSolverCharNorm(rs->rs,rs->est_wrk2,rs->est_wrk,sgn,estimate);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Computes the L1 norm of the difference of the computed ustar value and the value on the lax curve given by laxcurve(ustar[0]) */
PetscErrorCode NetRSLaxErrorEstimate(void *ctx,NetRS rs,PetscInt dir,const PetscReal *u,const PetscReal *ustar,PetscReal *estimate)
{
  PetscErrorCode ierr;
  PetscInt       field,wavenum; 

  PetscFunctionBegin;
  /* Assumes that the lax curve is paramaterized by the first conservative variable */
  wavenum = dir == EDGEIN ? 1 : 2; /* assumes a 2 variable system */
  ierr = RiemannSolverEvalLaxCurve(rs->rs,u,ustar[0],wavenum,rs->est_wrk);CHKERRQ(ierr);
  *estimate = 0; 
  for (field = 0; field<rs->numfields; field++){
   *estimate += PetscAbsReal(rs->est_wrk[field] - ustar[field]);
  }
  PetscFunctionReturn(0);
} 
/* Simple limiter that computes the M \| (u-ustar) \|_2^2  where M is supposed to represent a bound on the 2nd derivative of the laxcurve */

/* Note: No M scaling as the it doesn't work for the current function specifications */

PetscErrorCode NetRSTaylorErrorEstimate(void *ctx,NetRS rs,PetscInt dir,const PetscReal *u,const PetscReal *ustar,PetscReal *estimate)
{
  PetscInt       field;

  PetscFunctionBegin;
  *estimate = 0; 
  for (field = 0; field<rs->numfields; field++){
    *estimate += PetscSqr(u[field] - ustar[field]);
  }
  PetscFunctionReturn(0);
}

/* ------ END OF ERROR ESTIMATOR SUPPORT ------ */

/* WIP */ 

PetscErrorCode NetRSSetRiemannSolver(NetRS nrs, RiemannSolver rs)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(nrs,NETRS_CLASSID,1);
  PetscValidHeaderSpecific(rs,RIEMANNSOLVER_CLASSID,1);
  nrs->rs = rs; /* should up the reference count to the RiemannSolver */
  nrs->numfields = rs->numfields; /* removed after flux class */
  PetscFunctionReturn(0);
}

PetscErrorCode NetRSSetNumEdges(NetRS nrs, PetscInt numedges)
{
    PetscFunctionBegin;
    PetscValidHeaderSpecific(nrs,NETRS_CLASSID,1);
    nrs->numedges = numedges;
    PetscFunctionReturn(0);
}

