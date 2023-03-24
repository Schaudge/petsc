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
  PetscInt       i; 

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rs,NETRS_CLASSID,1);
  if (rs->setupcalled) PetscFunctionReturn(0); 
  if (rs->ops->setup) {
    PetscCall((*rs->ops->setup)(rs));
  }
  if (rs->numfields>-1 && rs->numedges>-1) PetscCall(PetscMalloc1(rs->numedges*rs->numfields,&rs->flux_wrk));
  if (rs->estimate) PetscCall(PetscMalloc2(rs->numfields,&rs->est_wrk,rs->numfields,&rs->est_wrk2));
  /* default value for error array is -1. This allows for knowing if the requested error array in an eval routine actually computed anything (i.e error 
  estimator is not assigned ) */ 
  PetscCall(PetscMalloc1(rs->numedges,&rs->error));
  for(i=0;i<rs->numedges;i++) {rs->error[i] = -1;}

  PetscCall(RiemannSolverSetUp(rs->rs));
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(rs,NETRS_CLASSID,1);
  if (rs->ops->reset) {
    PetscCall((*rs->ops->reset)(rs));
  }
  if (rs->flux_wrk) PetscCall(PetscFree(rs->flux_wrk)); /* Not good code here */
  if (rs->est_wrk) PetscCall(PetscFree2(rs->est_wrk,rs->est_wrk2)); /* also not good */
  /* Note that we should reference the RiemannSolver inside the NetRS to properly handle this reset behavior. */
  PetscCall(PetscFree(rs->error));
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
  PetscFunctionBegin;
  if (!*rs) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(*rs,NETRS_CLASSID,1);
  if (--((PetscObject)(*rs))->refct > 0) {*rs = NULL; PetscFunctionReturn(0);}
  /* destory the nested netrs */ 
  PetscCall(NetRSDestroy(&(*rs)->fine));

  PetscCall(NetRSReset(*rs));
  if ((*rs)->ops->destroy) PetscCall((*(*rs)->ops->destroy)((*rs)));
  PetscCall(PetscHeaderDestroy(rs));
  PetscFunctionReturn(0);
}

/*
  NetRSDuplicate - Create a new netrs of the same type as the original with the same settings. Still requires a call to setup after this call 
  as the intended use is to set the parameters for a "master" netrs duplicate it to other NetRS and change the types of the new netrs to the desired types. 
  This is the quick way of getting multiple netrs of different types for the same physics. 
*/

PetscErrorCode NetRSDuplicate(NetRS netrs,NetRS *newnetrs)
{
  MPI_Comm       comm;
  NetRS          netrs_new; 

  PetscFunctionBegin;
  PetscValidHeaderSpecific(netrs,NETRS_CLASSID,1);
  PetscValidPointer(newnetrs,2);
  PetscValidType(netrs,1);

  PetscCall(PetscObjectGetComm((PetscObject)netrs,&comm));
  PetscCall(NetRSCreate(comm,&netrs_new)); 
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
  *newnetrs = netrs_new;  
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
.  adaption - (optional) true if adaption was used (to be removed I think)
   Level: beginner

.seealso: NetRSCreate(), NetRSSetUp(), NetRSSetFlux()
@*/
PetscErrorCode  NetRSEvaluate(NetRS rs,const PetscReal *u, const EdgeDirection *dir,PetscReal **flux,PetscReal **error,PetscBool *adaption)
{
  PetscInt       e; 
  PetscReal      errest; 

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rs,NETRS_CLASSID,1);
  PetscCall(NetRSSetUp(rs));
  PetscCall(rs->ops->evaluate(rs,u,dir,rs->flux_wrk,rs->error));
  if (error) {*error = rs->error;}
  *flux = rs->flux_wrk;
  if(adaption) {*adaption = PETSC_FALSE;}

    /* adaptivity */
  if(rs->useadaptivity) {
    errest = 0.0; 
    for(e=0;e<rs->numedges;e++) {
      errest = PetscMax(rs->error[e],errest); 
    }
    if(errest >= rs->finetol) { /* revaluate with fine netrs */
      if(!rs->fine) { /* fine netrs hasn't been created. Create it by duplicating rs and changing its type to finetype */
        PetscCall(NetRSDuplicate(rs,&rs->fine)); 
        PetscCall(NetRSSetType(rs->fine,rs->finetype)); 
        rs->fine->useadaptivity = PETSC_FALSE; /* only allow two level adaptivity for now */
        PetscCall(NetRSSetUp(rs->fine));
      }
      PetscCall(NetRSEvaluate(rs->fine,u,dir,flux,error,NULL));
      if(adaption) {*adaption = PETSC_TRUE;}
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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rs, NETRS_CLASSID, 1);
  /* Type Option */
  if (!((PetscObject) rs)->type_name) {
    defaultType = NETRSLINEAR;
  } else {
    defaultType = ((PetscObject) rs)->type_name;
  }
  if (!NetRSRegisterAllCalled) PetscCall(NetRSRegisterAll());

  PetscObjectOptionsBegin((PetscObject) rs);
  PetscCall(PetscOptionsFList("-netrs_type", "NetRS", "NetRSSetType", NetRSList, defaultType, name, 256, &flg));
  if (flg) {
    PetscCall(NetRSSetType(rs, name));
  } else if (!((PetscObject) rs)->type_name) {
    PetscCall(NetRSSetType(rs, defaultType));
  }
  /* parameter selection */
  PetscCall(PetscOptionsReal("-netrs_finetol","Tolerance to swap to fine netrs solver","",rs->finetol,&rs->finetol,NULL));
  PetscCall(PetscOptionsBool("-netrs_use_estimator","Use error estimator if available","",rs->useestimator,&rs->useestimator,NULL));
  PetscCall(PetscOptionsBool("-netrs_use_adaptivity","Use adaptivity if available","",rs->useadaptivity,&rs->useadaptivity,NULL));
  PetscCall(PetscOptionsFList("-netrs_fine", "Fine NetRS to use with adaptivity", "NetRSSetType", NetRSList, rs->finetype, name, 256, &flg));
  if (flg) {rs->finetype = name;}



  /* handle implementation specific options */
  if (rs->ops->setfromoptions) {
    PetscCall((*rs->ops->setfromoptions)(PetscOptionsObject,rs));
  }
  /* process any options handlers added with PetscObjectAddOptionsHandler() */
  PetscCall(PetscObjectProcessOptionsHandlers((PetscObject) rs,PetscOptionsObject));
  PetscOptionsEnd();
  /*
    TODO:  View from options here ? 
  */
  PetscFunctionReturn(0);
}

/*@C
    NetRSView - Prints the NetRS data structure.

    Collective on NetRS. 

    For now I use this to print error and adaptivity information to file. 

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

  PetscFunctionReturn(0);
}

/*
 ----- ERROR ESTIMATOR SUPPORT  -----=
    For internal use only for now 
*/
PetscErrorCode  NetRSErrorEstimate(NetRS rs,PetscInt dir,const PetscReal *u, const PetscReal *ustar, PetscReal *errorestimate)
{
  void           *ctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rs,NETRS_CLASSID,1);
  PetscCall(NetRSSetUp(rs));
  PetscCall(NetRSGetApplicationContext(rs,&ctx));
  if(rs->estimate) {
    PetscCall(rs->estimate(ctx,rs,dir,u,ustar,errorestimate)); /* meh code */
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
PetscErrorCode NetRSUseErrorEstimator(NetRS netrs,PetscBool useerrorestimator)
{
   PetscFunctionBegin;
  PetscValidHeaderSpecific(netrs,NETRS_CLASSID,1);
  netrs->useestimator = useerrorestimator;
  PetscFunctionReturn(0);
}
PetscErrorCode NetRSIsUsingErrorEstimator(NetRS netrs,PetscBool *useerrorestimator)
{
   PetscFunctionBegin;
  PetscValidHeaderSpecific(netrs,NETRS_CLASSID,1);
  *useerrorestimator = netrs->useestimator;
  PetscFunctionReturn(0);
}
/* WIP implementation of one type of error estimator */

PetscErrorCode NetRSRoeErrorEstimate(void *ctx,NetRS rs,PetscInt dir,const PetscReal *u,const PetscReal *ustar,PetscReal *estimate)
{
  PetscInt       field,sgn;

  PetscFunctionBegin;
  /* compute jump */
  for (field=0; field<rs->numfields; field++) {
    rs->est_wrk[field] = ustar[field] - u[field];
  }
  PetscCall(RiemannSolverComputeRoeAvg(rs->rs,u,ustar,rs->est_wrk2));
  sgn = dir == EDGEIN ? -1 : 1;
  PetscCall(RiemannSolverCharNorm(rs->rs,rs->est_wrk2,rs->est_wrk,sgn,estimate));
  PetscFunctionReturn(0);
}

/* Computes the L1 norm of the difference of the computed ustar value and the value on the lax curve given by laxcurve(ustar[0]) */
PetscErrorCode NetRSLaxErrorEstimate(void *ctx,NetRS rs,PetscInt dir,const PetscReal *u,const PetscReal *ustar,PetscReal *estimate)
{
  PetscInt       field,wavenum; 

  PetscFunctionBegin;
  /* Assumes that the lax curve is paramaterized by the first conservative variable */
  wavenum = dir == EDGEIN ? 1 : 2; /* assumes a 2 variable system */
  PetscCall(RiemannSolverEvalLaxCurve(rs->rs,u,ustar[0],wavenum,rs->est_wrk));
  *estimate = 0; 
  for (field = 1; field<rs->numfields; field++){
   *estimate += PetscAbsReal(rs->est_wrk[field] - ustar[field]);
  }
  PetscFunctionReturn(0);
} 
/* Simple limiter that computes the M \| (u[0]-ustar[0]) \|_2^2  where M is supposed to represent a bound on the 2nd derivative of the laxcurve */

/* Note: No M scaling as the it doesn't work for the current function specifications. This assumes that the lax curve is parameterized by the first 
conservation variable */

PetscErrorCode NetRSTaylorErrorEstimate(void *ctx,NetRS rs,PetscInt dir,const PetscReal *u,const PetscReal *ustar,PetscReal *estimate)
{
  PetscFunctionBegin;
  *estimate = PetscSqr(u[0] - ustar[0]);
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


/* Adaptivty Support */ 
PetscErrorCode NetRSUseAdaptivity(NetRS netrs,PetscBool useadaptivity)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(netrs,NETRS_CLASSID,1);
  netrs->useadaptivity = useadaptivity;
  PetscFunctionReturn(0);
}

PetscErrorCode NetRSIsUsingAdaptivity(NetRS netrs,PetscBool *useadaptivity)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(netrs,NETRS_CLASSID,1);
  *useadaptivity = netrs->useadaptivity;
  PetscFunctionReturn(0);
}

PetscErrorCode NetRSSetFineTol(NetRS netrs,PetscReal finetol) 
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(netrs,NETRS_CLASSID,1);
  netrs->finetol = finetol;
  PetscFunctionReturn(0);
}


