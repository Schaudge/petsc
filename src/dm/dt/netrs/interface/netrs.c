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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rs,NETRS_CLASSID,1);
  if (rs->setupcalled) PetscFunctionReturn(0); 
  if (rs->ops->setup) {
    ierr = (*rs->ops->setup)(rs);CHKERRQ(ierr);
  }
  if (rs->numfields>-1 && rs->numedges>-1) {ierr = PetscMalloc1(rs->numedges*rs->numfields,&rs->flux_wrk);CHKERRQ(ierr);}
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
  /* Note that we should reference the RiemannSolver inside the NetRS to properly handle this reset behavior. */
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

  ierr = NetRSReset(*rs);CHKERRQ(ierr);
  if ((*rs)->ops->destroy) {ierr = (*(*rs)->ops->destroy)((*rs));CHKERRQ(ierr);}
  ierr = PetscHeaderDestroy(rs);CHKERRQ(ierr);
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

   Level: beginner

.seealso: NetRSCreate(), NetRSSetUp(), NetRSSetFlux()
@*/
PetscErrorCode  NetRSEvaluate(NetRS rs,const PetscReal *u, const PetscBool *dir,PetscReal **flux)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rs,NETRS_CLASSID,1);
  ierr = NetRSSetUp(rs);CHKERRQ(ierr);
  ierr = rs->ops->evaluate(rs,u,dir,rs->flux_wrk);CHKERRQ(ierr);
  *flux = rs->flux_wrk;
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
For internal use only for now 
*/
PetscErrorCode  NetRSErrorEstimate(NetRS rs,const PetscReal *u, const PetscReal *ustar, PetscReal *errorestimate)
{
  PetscErrorCode ierr;
  void           *ctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rs,NETRS_CLASSID,1);
  ierr = NetRSSetUp(rs);CHKERRQ(ierr);
  ierr = NetRSGetApplicationContext(rs,&ctx);CHKERRQ(ierr);
  if(rs->estimate) {
    ierr = rs->estimate(ctx,u,ustar,errorestimate);CHKERRQ(ierr); /* meh code */
  } else {
    SETERRQ(PetscObjectComm((PetscObject)rs),PETSC_ERR_SUP,"No error estimator specified for NetRS");
  }
  PetscFunctionReturn(0);
}

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