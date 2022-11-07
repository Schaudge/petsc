#include <petsc/private/localnetrpimpl.h>
#include <petscnetrp.h>
#include <petsc/private/riemannsolverimpl.h> /* to be removed after adding fluxfunction class */
#include <petscsys.h>
/*@
   NetRPSetUp - Sets up the internal data structures for the later use of a NetRP. 

   Collective on NetRP

   Input Parameter:
.  rs - the NetRP context obtained from RiemanSolverCreate()

   Notes:
   Internally called when setting the flux function as internal data structures depend on the 
   dim and numfield parameters set there. Will not normally be called by users. 

   Level: advanced

.seealso: NetRPCreate(), NetRPSetFlux()
@*/
PetscErrorCode  NetRPSetUp(NetRP rp)
{
  PetscInt       i,numsubgraphs,numfield_flux, numfield_rp; 
  DM             network; 

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rp,NETRP_CLASSID,1);
  if (rp->setupcalled) PetscFunctionReturn(0);
  PetscCall(RiemannSolverSetUp(rp->flux));
  /* check riemann problem and physics consistency */
  PetscCall(RiemannSolverGetNumFields(rp->flux,&numfield_flux)); 
  switch(rp->physicsgenerality)
  {
    case Generic:
      rp->numfields = numfield_flux; 
      break;
    default:
    case Specific:
      PetscCall(NetRPGetNumFields(rp,&numfield_rp)); 
      PetscCheck(numfield_flux == numfield_rp, PetscObjectComm((PetscObject)rp),PETSC_ERR_ARG_WRONGSTATE,"The physics number of fields : %" PetscInt_FMT " is not the same as the riemann problem number of fields  %" PetscInt_FMT, numfield_flux, numfield_rp);
      break;
  }

  rp->setupcalled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

PetscErrorCode NetRPGetNumFields(NetRP rp, PetscInt *numfields) {

  PetscFunctionBegin; 
  PetscValidHeaderSpecific(rp,NETRP_CLASSID,1);
  *numfields = rp->numfields;
  PetscFunctionReturn(0); 
}

/*@
   NetRPReset - Resets a NetRP context and removes any allocated internal petsc objects

   Collective on NetRP

   Input Parameter:
.  rs - the NetRP context from NetRPCreate()

   Level: beginner

.seealso: NetRPCreate(), NetRPSetUp(), NetRPDestroy()
@*/
PetscErrorCode  NetRPReset(NetRP rp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(rp,NETRP_CLASSID,1);
  if (rp->ops->reset) {
    PetscCall((*rp->ops->reset)(rp));
  }
  PetscCall(NetRPClearCache(rp));
  rp->solvetype = Other; 
  /* Note that we should reference the RiemannSolver inside the NetRS to properly handle this reset behavior. */
  rp->setupcalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}
/*@
   NetRPReset - Clears all cached solver objects in the NetRP. 

   Not Collective on NetRP

   Input Parameter:
.  rs - the NetRP context from NetRPCreate()

   Level: advanced

.seealso: NetRPCreate(), NetRPSetUp(), NetRPDestroy()
@*/
PetscErrorCode NetRPClearCache(NetRP rp)
{
  PetscInt i, numvertdegs; 

  PetscFunctionBegin; 
  PetscValidHeaderSpecific(rp,NETRP_CLASSID,1);
  if(rp->ops->clearcache) {
    PetscCall((*rp->ops->clearcache)(rp)); 
  }
  PetscCall(NetRPGetVertexDegrees(rp,&numvertdegs,NULL)); 
  if(rp->ksp) { 
    for(i=0; i<numvertdegs; i++) {
      PetscCall(KSPDestroy(&rp->ksp[i])); 
      PetscCall(VecDestroy(&rp->vec[i]));
      PetscCall(MatDestroy(&rp->mat[i])); 
    }
    PetscFree3(rp->mat,rp->vec,rp->ksp); 
  }
  if (rp->snes) {
    for(i=0; i<numvertdegs; i++) {
      PetscCall(SNESDestroy(rp->snes[i]));
    }
    PetscFree(rp->snes); 
  }
  PetscCall(PetscHMapIClear(rp->hmap));
}
/*@
   NetRPCreateMat - Preallocate matrix structure for solving a vertdeg Riemann Problem. Does a default setup of 
   the mat, implementations may set their own details. 

   Not Collective  on NetRP

   Input Parameter:
.  rp - the NetRP context obtained from NetRPCreate()
.  vertdeg -  Vertex degree for the problem 

  Output Parameter: 

  . Mat - the created matrix for this problem. To be used in linear solvers. 

   Level: developer

.seealso: 
@*/

PetscErrorCode NetRPCreateLinear(NetRP rp, PetscInt vertdeg, Mat *mat,Vec *vec)
{
  Mat _mat;
  Vec _vec; 
  const char *prefix_netrp; 

  PetscFunctionBegin; 
  PetscValidHeaderSpecific(rp,NETRP_CLASSID,1);

  PetscCall(MatCreate(PETSC_COMM_SELF,&_mat));
  PetscCall(PetscObjectGetOptionsPrefix((PetscObject)rp,&prefix_netrp));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)_mat,prefix_netrp)); 
  PetscCall(PetscObjectAppendOptionsPrefix((PetscObject)_mat,"netrp_"));

  if(rp->ops->setupmat) PetscCall((rp->ops->setupmat)(rp,vertdeg,_mat));

  PetscCall(MatSetFromOptions(_mat));  
  PetscCall(MatSetSizes(_mat,PETSC_DECIDE,PETSC_DECIDE,vertdeg*rp->numfields,vertdeg*rp->numfields));
  PetscCall(MatSetUp(_mat));

  PetscCall(MatCreateVecs(_mat,NULL,&_vec)); 
  *mat = _mat;
  *vec = _vec; 
  PetscFunctionReturn(0); 
}

/*@
   NetRPCreateKSP - Create the KSP 

   Not Collective  on NetRP

   Input Parameter:
.  rp - the NetRP context obtained from NetRPCreate()
.  vertdeg -  Vertex degree for the problem 

  Output Parameter: 

  . ksp - the created KSP for the Riemann Problem

   Level: developer

.seealso: 
@*/

PetscErrorCode NetRPCreateKSP(NetRP rp, PetscInt vertdeg, KSP *ksp)
{
  KSP _ksp;
  PC  pc;  
  const char *prefix_netrp; 

  PetscFunctionBegin; 
  PetscValidHeaderSpecific(rp,NETRP_CLASSID,1);

  PetscCall(KSPCreate(PETSC_COMM_SELF,&_ksp));
  PetscCall(PetscObjectGetOptionsPrefix((PetscObject)rp,&prefix_netrp));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)_ksp,prefix_netrp)); 
  PetscCall(PetscObjectAppendOptionsPrefix((PetscObject)_ksp,"netrp_"));
  /* Default to direct linear solver, which makes sense for these small systems */
  PetscCall(KSPGetPC(_ksp,&pc));
  PetscCall(PCSetType(pc,PCLU)); 
  PetscCall(KSPSetType(_ksp,KSPPREONLY));

  if(rp->ops->setupksp) PetscCall((rp->ops->setupksp)(rp,vertdeg,_ksp));
  PetscCall(KSPSetFromOptions(_ksp));
  PetscCall(KSPSetUp(_ksp)); 
  *ksp = _ksp; 
  PetscFunctionReturn(0); 
}



typedef struct {
  NetRP   rp; 
  PetscInt v; 
  DM       dm; 
  Vec      U;
} NetRPSNESctx; 
/*@
   NetRPSNESWrapperFunc - Wraps the format for the nonlinear eval in NetRP class into SNES expected format. 
   Just for internal use. 

   Note that I assume that the SNes application ctx is the NetRPSNES ctx cast as void pointer. 
   This holds the DM, NetRP and vertex info needed for the function calls. Thus before calling SNESSolve, 
   correct appctx must be set, or it will not solve the correct problem. 

  Output Parameter: 

   Not Collective  on NetRP

   Input Parameter:
.  snes - the snes inside of NetRP 
.  x    - input vec 
.  ctx  - NULL

  f     - outputvec 
   Level: developer

.seealso: 
@*/

static PetscErrorCode NetRPSNESWrapperFunc(SNES snes, Vec x, Vec f, void *ctx)
{
  NetRPSNESctx *netrpsnesctx;
  NetRP  rp;
  DM     dm;
  PetscInt v;
  Vec    U; 
  void   *appctx; 

  PetscFunctionBegin; 
  PetscCall(SNESGetApplicationContext(snes,&appctx));
  netrpsnesctx = (NetRPSNESctx*) appctx; 
  rp = netrpsnesctx->rp; 
  dm = netrpsnesctx->dm; 
  v = netrpsnesctx->v; 
  U = netrpsnesctx->U; 
  PetscUseTypeMethod(rp,NonlinearEval,dm,v,U,x,f); 
  PetscFunctionReturn(0); 
}

/*@
   NetRPSNESWrapperJac - Wraps the format for the nonlinear Jac in NetRP class into SNES expected format. 
   Just for internal use. 

   Note that I assume that the SNes application ctx is the NetRPSNES ctx cast as void pointer. 
   This holds the DM, NetRP and vertex info needed for the function calls. Thus before calling SNESSolve, 
   correct appctx must be set, or it will not solve the correct problem. 

  Output Parameter: 

   Not Collective  on NetRP

   Input Parameter:
.  snes - the snes inside of NetRP 
.  x    - input vec 
.  ctx  - NULL

  f     - outputvec 
   Level: developer

.seealso: 
@*/

static PetscErrorCode NetRPSNESWrapperJac(SNES snes, Vec x, Mat Amat, Mat Pmat, void *ctx)
{
  NetRPSNESctx *netrpsnesctx;
  NetRP  rp;
  DM     dm;
  PetscInt v;
  Vec    U; 
  void   *appctx; 

  PetscFunctionBegin; 
  PetscCall(SNESGetApplicationContext(snes,&appctx));
  netrpsnesctx = (NetRPSNESctx*) appctx; 
  rp = netrpsnesctx->rp; 
  dm = netrpsnesctx->dm; 
  v = netrpsnesctx->v; 
  U = netrpsnesctx->U; 
  PetscUseTypeMethod(rp,NonlinearJac,dm,v,U,x,Amat); 
  PetscFunctionReturn(0); 
}


/*@
   NetRPCreateSNES - Create the SNES 

   Not Collective  on NetRP

   Input Parameter:
.  rp - the NetRP context obtained from NetRPCreate()
.  vertdeg -  Vertex degree for the problem 

  Output Parameter: 

  . snes - the created SNES for the Riemann Problem

   Level: developer

.seealso: 
@*/

PetscErrorCode NetRPCreateSNES(NetRP rp, PetscInt vertdeg, SNES *snes)
{
  SNES _snes;
  const char *prefix_netrp; 
  PetscInt numfield,cacheoff;
  Mat      jac; 

  PetscFunctionBegin; 
  PetscValidHeaderSpecific(rp,NETRP_CLASSID,1);

  PetscCall(SNESCreate(PETSC_COMM_SELF,&_snes));
  PetscCall(PetscObjectGetOptionsPrefix((PetscObject)rp,&prefix_netrp));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)_snes,prefix_netrp)); 
  PetscCall(PetscObjectAppendOptionsPrefix((PetscObject)_snes,"netrp_"));
  if(rp->ops->setupsnes) PetscCall((rp->ops->setupsnes)(rp,vertdeg,_snes));
  PetscCall(SNESSetFromOptions(_snes));
  PetscCall(SNESSetFunction(_snes,NULL,NetRPSNESWrapperFunc,NULL)); /* should I provide the r vec here? */

  if(rp->ops->NonlinearJac) 
  {
    /* Create Jacobian Mat */
    PetscCall(MatCreate(PETSC_COMM_SELF,&jac));
    PetscCall(PetscObjectGetOptionsPrefix((PetscObject)rp,&prefix_netrp));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject)jac,prefix_netrp)); 
    PetscCall(PetscObjectAppendOptionsPrefix((PetscObject)jac,"netrp_jac_"));
    PetscTryTypeMethod(rp,setupjac,vertdeg,jac);
    PetscCall(MatSetFromOptions(jac));  
    PetscCall(MatSetSizes(jac,PETSC_DECIDE,PETSC_DECIDE,vertdeg*numfield,vertdeg*numfield));
    PetscCall(MatSetUp(jac)); 

    PetscCall(SNESSetJacobian(_snes,jac,jac,NetRPSNESWrapperJac,NULL)); 
    PetscCall(MatDestroy(&jac)); /* dereference the jacobian mat, now ownership controlled by SNES */
  }
  *snes = _snes;
  PetscFunctionReturn(0); 
}



/*@
   NetRPAddVertexDegrees - Add vertex degrees to cache solvers for. If these degrees are already cached then
   nothing will happen. 

   Not Collective  on NetRP

   Input Parameter:
.  rp - the NetRP context obtained from NetRPCreate()
.  numdeg- the number of vertex degrees to add 
.  vertdegs -  array of vertex degrees to add. This array must be destroyed by the caller. 

   Level: intermediate

.seealso: 
@*/
PetscErrorCode NetRPAddVertexDegrees(NetRP rp, PetscInt numdegs, PetscInt *vertdegs)
{
  PetscInt i, numentries, totalnew=0,off; 
  PetscBool flg;
  Mat       *mat_new; 
  KSP       *ksp_new; 
  SNES      *snes_new;
  Vec       *vec_new;  

  PetscFunctionBegin; 
  PetscValidHeaderSpecific(rp,NETRP_CLASSID,1);
  PetscCheck(rp->setupcalled,PetscObjectComm((PetscObject)rp),PETSC_ERR_ARG_WRONGSTATE,"Call NetRPSetUp() first"); 
  PetscCall(PetscHMapIGetSize(rp->hmap,&numentries)); 
  /* count number of new entries */
  for(i=0; i<numdegs; i++) {
    PetscCheck(vertdegs[i]>0,PetscObjectComm((PetscObject)rp),PETSC_ERR_USER_INPUT,"NetRP requires vertex degrees to be greater than 0 not %" PetscInt_FMT, vertdegs[i]); 
    PetscCall(PetscHMapIHas(rp->hmap, vertdegs[i],&flg)); 
    if (flg) totalnew++; 
  }
  if (totalnew == 0) PetscFunctionReturn(0);
  /* reallocate solver cache arrays with room for new entries */
  PetscCall(PetscMalloc3(numentries+totalnew,&mat_new,numentries+totalnew,&vec_new,numentries+totalnew,&ksp_new)); 
  PetscCall(PetscMalloc(numentries+totalnew,&snes_new)); 

  PetscCall(PetscArraycpy(mat_new,rp->mat,numentries));
  PetscCall(PetscArraycpy(vec_new,rp->vec,numentries));  
  PetscCall(PetscArraycpy(ksp_new,rp->ksp,numentries));
  PetscCall(PetscArraycpy(snes_new,rp->snes,numentries)); 

  if(rp->mat){ /* if any memeory has every been allocated */
    PetscCall(PetscFree3(rp->mat,rp->vec,rp->ksp)); 
    PetscCall(PetscFree(rp->snes)); 
  }
  rp->mat = mat_new; 
  rp->ksp = ksp_new; 
  rp->snes = snes_new;
  rp->vec = vec_new; 
 
  /* add new entries */
  off = 0; 
  for (i=0; i< numdegs; i++) {
    PetscCall(PetscHMapIHas(rp->hmap, vertdegs[i],&flg)); 
    if (!flg) break;
    PetscCall(PetscHMapISet(rp->hmap,vertdegs[i],numentries+off)); 
    /* only create what is needed */
    switch(rp->solvetype)
    {
      case Nonlinear : /* assumes only usage of snes */
        PetscCall(NetRPCreateSNES(rp,vertdegs[i],&rp->snes[numentries+off])); 
        break; 
      case Linear : /* assumes only usage of Mat and KSP */
        PetscCall(NetRPCreateLinear(rp,vertdegs[i],&rp->mat[numentries+off],&rp->vec[numentries+off])); 
        PetscCall(NetRPCreateKSP(rp,vertdegs[i],&rp->ksp[numentries+off]));
        break;
      case Other: /* Create Everything */
        PetscCall(NetRPCreateLinear(rp,vertdegs[i],&rp->mat[numentries+off],&rp->vec[numentries+off])); 
        PetscCall(NetRPCreateKSP(rp,vertdegs[i],&rp->ksp[numentries+off]));
        PetscCall(NetRPCreateSNES(rp,vertdegs[i],&rp->snes[numentries+off])); 
        break;
    }
    off++; 
  }
  PetscFunctionReturn(0); 
}

/*@
   NetRPDestroy - Destroys the NetRP context that was created
   with NetRPCreate().

   Collective on NetRP

   Input Parameter:
.  rs - the NetRP context obtained from NetRPCreate()

   Level: beginner

.seealso: NetRPCreate(), NetRPSetUp()
@*/
PetscErrorCode  NetRPDestroy(NetRP *rp)
{
  PetscFunctionBegin;
  if (!*rp) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(*rp,NETRP_CLASSID,1);
  if (--((PetscObject)(*rp))->refct > 0) {*rp = NULL; PetscFunctionReturn(0);}
  PetscCall(NetRPReset(*rp));
  if ((*rp)->ops->destroy) PetscCall((*(*rp)->ops->destroy)((*rp)));
  if((*rp)->flux) RiemannSolverDestroy(&(*rp)->flux); 
  PetscCall(PetscHMapIDestroy(&(*rp)->hmap)); 
  PetscCall(PetscHeaderDestroy(rp));
  PetscFunctionReturn(0);
}

/*
  NetRPDuplicate - Create a new NetRP of the same type as the original with the same settings. Still requires a call to setup after this call 
  as the intended use is to set the parameters for a "master" NetRP duplicate it to other NetRP and change the types of the new NetRP to the desired types. 
  This is the quick way of getting multiple NetRP of different types for the same physics. 
*/

/* 
TODO: Needs a rework on what this means. 
*/
PetscErrorCode NetRPDuplicate(NetRP rp,NetRP *newrp)
{
  MPI_Comm       comm;
  NetRP          rp_new; 

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rp,NETRP_CLASSID,1);
  PetscValidPointer(newrp,2);
  PetscValidType(rp,1);

  PetscCall(PetscObjectGetComm((PetscObject)rp,&comm));
  PetscCall(NetRPCreate(comm,&rp_new)); 
  /* copy over the parameters and physics from rp to newrp */ 
  /* physics*/
  rp_new->user      = rp->user; 
  PetscCall(NetRPSetFlux(rp_new,rp->flux)); 
  *newrp = rp_new;  
  PetscFunctionReturn(0);
}

/*@
   NetRPSetApplicationContext - Sets an optional user-defined context for
   the NetRP.

   Logically Collective on NetRP

   Input Parameters:
+  rs - the NetRP context obtained from NetRPCreate()
-  usrP - optional user context

   Level: intermediate

.seealso: NetRPGetApplicationContext()
@*/
PetscErrorCode  NetRPSetApplicationContext(NetRP rp,void *usrP)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(rp,NETRP_CLASSID,1);
  rp->user = usrP;
  PetscFunctionReturn(0);
}
/*@
    NetRPGetApplicationContext - Gets the user-defined context for the
    NetRP

    Not Collective

    Input Parameter:
.   rs - the NetRP context obtained from NetRPCreate()

    Output Parameter:
.   usrP - user context

    Level: intermediate

.seealso: NetRPSetApplicationContext()
@*/
PetscErrorCode  NetRPGetApplicationContext(NetRP rp,void *usrP)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(rp,NETRP_CLASSID,1);
  *(void**)usrP = rp->user;
  PetscFunctionReturn(0);
}

/*@
  NetRPSetFromOptions - sets parameters in a NetRP from the options database

  Collective on NetRP

  Input Parameter:
. rs - the NetRP object to set options for

  Options Database:

  Level: intermediate

.seealso 
@*/
PetscErrorCode NetRPSetFromOptions(NetRP rp)
{
  const char    *defaultType;
  char           name[256];
  PetscBool      flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rp, NETRP_CLASSID, 1);
  /* Type Option */
  if (!((PetscObject) rp)->type_name) {
    defaultType = NETRPBLANK;
  } else {
    defaultType = ((PetscObject) rp)->type_name;
  }
  if (!NetRPRegisterAllCalled) PetscCall(NetRPRegisterAll());

  PetscObjectOptionsBegin((PetscObject) rp);
  PetscCall(PetscOptionsFList("-netrp_type", "NetRP", "NetRPSetType", NetRPList, defaultType, name, 256, &flg));
  if (flg) {
    PetscCall(NetRPSetType(rp, name));
  } else if (!((PetscObject) rp)->type_name) {
    PetscCall(NetRPSetType(rp, defaultType));
  }

  /* handle implementation specific options */
  if (rp->ops->setfromoptions) {
    PetscCall((*rp->ops->setfromoptions)(PetscOptionsObject,rp));
  }
  /* process any options handlerp added with PetscObjectAddOptionsHandler() */
  PetscCall(PetscObjectProcessOptionsHandlers((PetscObject) rp,PetscOptionsObject));
  PetscOptionsEnd();
  /*
    TODO:  View from options here ? 
  */
  PetscFunctionReturn(0);
}

/*@C
    NetRPView - Prints the NetRP data structure.

    Collective on NetRP. 

    For now I use this to print error and adaptivity information to file. 

    Input Parameters:
+   rs - the NetRP context obtained from NetRPCreate()
-   viewer - visualization context

    Options Database Key:
   TODO: 
    Level: beginner

.seealso: PetscViewerASCIIOpen()
@*/
PetscErrorCode  NetRPView(NetRP rp,PetscViewer viewer)
{
  PetscFunctionBegin;

  PetscFunctionReturn(0);
}

PetscErrorCode NetRPSetFlux(NetRP rp, RiemannSolver rs)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(rp,NETRP_CLASSID,1);
  PetscValidHeaderSpecific(rs,RIEMANNSOLVER_CLASSID,1); 
  if(rp->flux) PetscCall(RiemannSolverDestroy(&rp->flux)); 
  PetscCall(PetscObjectReference((PetscObject)rs)); 
  rp->flux = rs;
  PetscFunctionReturn(0);
}

PetscErrorCode NetRPGetFlux(NetRP rp, RiemannSolver *rs)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(rp,NETRP_CLASSID,1);
  PetscValidHeaderSpecific(*rs,RIEMANNSOLVER_CLASSID,1); 
  if(rp->flux) *rs = rp->flux; 
  PetscFunctionReturn(0);
}

PetscErrorCode NeetRPSetNumFields(NetRP rp, PetscInt numfields)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(rp,NETRP_CLASSID,1);
  rp->numfields = numfields; 
  PetscFunctionReturn(0); 
}

PetscErrorCode NeetRPGetNumFields(NetRP rp, PetscInt *numfields)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(rp,NETRP_CLASSID,1);
  /* NetRPSetUp contains the logic for setting computing this */
  PetscCheck(rp->setupcalled,PetscObjectComm((PetscObject)rp),PETSC_ERR_ARG_WRONGSTATE,"Call NetRPSetUp() first"); 
  *numfields = rp->numfields; 
  PetscFunctionReturn(0);
}

PetscErrorCode NetRPSetPhysicsGenerality(NetRP rp, NetRPPhysicsGenerality generality)
{
  PetscBool flg; 

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rp,NETRP_CLASSID,1);
  PetscCheck(!rp->setupcalled,PetscObjectComm((PetscObject)rp),PETSC_ERR_ARG_WRONGSTATE,"Must be called before NetRPSetUp()"); 
  PetscCall(PetscObjectTypeCompare((PetscObject)rp,NETRPBLANK,&flg));
  PetscCheck(flg,PetscObjectComm((PetscObject)rp),PETSC_ERR_ARG_WRONGSTATE,"Can only be manually set on the blank type of NetRP"); 
  rp->physicsgenerality = generality; 
  PetscFunctionReturn(0); 
}

PetscErrorCode NetRPGetPhysicsGenerality(NetRP rp, NetRPPhysicsGenerality *generality)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(rp,NETRP_CLASSID,1);
  *generality = rp->physicsgenerality;  
  PetscFunctionReturn(0); 
}

PetscErrorCode NetRPSetSolveType(NetRP rp, NetRPSolveType solvetype)
{
  PetscBool flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rp,NETRP_CLASSID,1);
  PetscCheck(!rp->setupcalled,PetscObjectComm((PetscObject)rp),PETSC_ERR_ARG_WRONGSTATE,"Solve Type must be set before calling NetRPSetUp()"); 
  /* only the blank implementation should allow for setting this, other implementations are assumed to fix the type themselves */
  PetscCall(PetscObjectTypeCompare((PetscObject)rp,NETRPBLANK,&flg));
  PetscCheck(flg,PetscObjectComm((PetscObject)rp),PETSC_ERR_ARG_WRONGSTATE,"Solve Type can only be manually set on the blank type of NetRP"); 
  rp->solvetype = solvetype; 
  PetscFunctionReturn(0);
}

PetscErrorCode NetRPGetSolveType(NetRP rp, NetRPSolveType *solvetype)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(rp,NETRP_CLASSID,1);
  *solvetype = rp->solvetype ; 
  PetscFunctionReturn(0);
}

PetscErrorCode NetRPSetSolveStar(NetRP rp, NetRPSolveStar_User solvestar)
{
  PetscBool flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rp,NETRP_CLASSID,1);
  PetscCheck(!rp->setupcalled,PetscObjectComm((PetscObject)rp),PETSC_ERR_ARG_WRONGSTATE,"Must be set before calling NetRPSetUp()"); 
  /* only the blank implementation should allow for setting this, other implementations are assumed to fix the type themselves */
  PetscCall(PetscObjectTypeCompare((PetscObject)rp,NETRPBLANK,&flg));
  PetscCheck(flg,PetscObjectComm((PetscObject)rp),PETSC_ERR_ARG_WRONGSTATE,"Can only be manually set on the blank type of NetRP"); 
  rp->ops->solveStar = solvestar;
  PetscFunctionReturn(0);
}

PetscErrorCode NetRPSetSolveFlux(NetRP rp, NetRPSolveFlux_User solveflux)
{
  PetscBool flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rp,NETRP_CLASSID,1);
  PetscCheck(!rp->setupcalled,PetscObjectComm((PetscObject)rp),PETSC_ERR_ARG_WRONGSTATE,"Must be set before calling NetRPSetUp()"); 
  /* only the blank implementation should allow for setting this, other implementations are assumed to fix the type themselves */
  PetscCall(PetscObjectTypeCompare((PetscObject)rp,NETRPBLANK,&flg));
  PetscCheck(flg,PetscObjectComm((PetscObject)rp),PETSC_ERR_ARG_WRONGSTATE,"Can only be manually set on the blank type of NetRP"); 
  rp->ops->solveFlux = solveflux;
  PetscFunctionReturn(0);
}

PetscErrorCode NetRPSetCreateMatStar(NetRP rp, NetRPCreateLinearStar linStar)
{
  PetscBool flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rp,NETRP_CLASSID,1);
  PetscCheck(!rp->setupcalled,PetscObjectComm((PetscObject)rp),PETSC_ERR_ARG_WRONGSTATE,"Must be set before calling NetRPSetUp()"); 
  /* only the blank implementation should allow for setting this, other implementations are assumed to fix the type themselves */
  PetscCall(PetscObjectTypeCompare((PetscObject)rp,NETRPBLANK,&flg));
  PetscCheck(flg,PetscObjectComm((PetscObject)rp),PETSC_ERR_ARG_WRONGSTATE,"Can only be manually set on the blank type of NetRP"); 
  rp->ops->createLinearStar = linStar;
  PetscFunctionReturn(0);
}

PetscErrorCode NetRPSetCreateMatFlux(NetRP rp, NetRPCreateLinearFlux linflux)
{
  PetscBool flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rp,NETRP_CLASSID,1);
  PetscCheck(!rp->setupcalled,PetscObjectComm((PetscObject)rp),PETSC_ERR_ARG_WRONGSTATE,"Must be set before calling NetRPSetUp()"); 
  /* only the blank implementation should allow for setting this, other implementations are assumed to fix the type themselves */
  PetscCall(PetscObjectTypeCompare((PetscObject)rp,NETRPBLANK,&flg));
  PetscCheck(flg,PetscObjectComm((PetscObject)rp),PETSC_ERR_ARG_WRONGSTATE,"Can only be manually set on the blank type of NetRP"); 
  rp->ops->createLinearFlux = linflux;
  PetscFunctionReturn(0);
}

PetscErrorCode NetRPSetNonlinearEval(NetRP rp, NetRPNonlinearEval nonlineareval)
{
  PetscBool flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rp,NETRP_CLASSID,1);
  PetscCheck(!rp->setupcalled,PetscObjectComm((PetscObject)rp),PETSC_ERR_ARG_WRONGSTATE,"Must be set before calling NetRPSetUp()"); 
  /* only the blank implementation should allow for setting this, other implementations are assumed to fix the type themselves */
  PetscCall(PetscObjectTypeCompare((PetscObject)rp,NETRPBLANK,&flg));
  PetscCheck(flg,PetscObjectComm((PetscObject)rp),PETSC_ERR_ARG_WRONGSTATE,"Can only be manually set on the blank type of NetRP"); 
  rp->ops->NonlinearEval = nonlineareval;
  PetscFunctionReturn(0);
}

PetscErrorCode NetRPSetNonlinearJac(NetRP rp, NetRPNonlinearJac nonlinearjac)
{
  PetscBool flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rp,NETRP_CLASSID,1);
  PetscCheck(!rp->setupcalled,PetscObjectComm((PetscObject)rp),PETSC_ERR_ARG_WRONGSTATE,"Must be set before calling NetRPSetUp()"); 
  /* only the blank implementation should allow for setting this, other implementations are assumed to fix the type themselves */
  PetscCall(PetscObjectTypeCompare((PetscObject)rp,NETRPBLANK,&flg));
  PetscCheck(flg,PetscObjectComm((PetscObject)rp),PETSC_ERR_ARG_WRONGSTATE,"Can only be manually set on the blank type of NetRP"); 
  rp->ops->NonlinearJac = nonlinearjac;
  PetscFunctionReturn(0);
}

/*@
   NetRPSSolveFlux - The driver function for solving for Riemann Problem fluxes. This will use the user provided functions 
   and auto cached solver objects to solve for the flux. New solver objects will be created and cached as necessary as well. 

   Not Collective on NetRP

   Input Parameter:
.  rs - the NetRP context obtained from RiemanSolverCreate()
.  network - the network that contains the vertex v with the topology of the riemann problem. 
.  v  - the vertex in network to solve the riemann problem at 
.  U  - vec containing the the deg(v)*numfield initial states of the riemman problem. Allocated by caller. 

  Output Parameter: 

. Flux - Vec Containing the deg(v)*numfield fluxes after solving the riemann problem. Allocated by caller. 

   Level: beginner

.seealso: NetRPCreate(), NetRPSetFlux()
@*/
PetscErrorCode NetRPSolveFlux(NetRP rp, DM network, PetscInt v, Vec U, Vec Flux)
{
  PetscBool flg;
  PetscInt  index; 
  NetRPSNESctx snesctx; 

  PetscFunctionBegin;
  if(!rp->setupcalled) PetscCall(NetRPSetUp(rp)); 
  /* find index of cached solvers */
  PetscCall(PetscHMapIHas(rp->hmap,v,&flg));
  if(!flg) PetscCall(NetRPAddVertexDegrees(rp,1,&v)); 
  PetscCall(PetscHMapIGet(rp->hmap,v,&index)); 

  /* switch based on type of NetRP */
  switch(rp->solvetype) 
  {
    case Linear: 
      PetscUseTypeMethod(rp,createLinearFlux,network,v,U,rp->mat[index],rp->vec[index]);
      PetscCall(KSPSetOperators(rp->ksp[index],rp->mat[index],rp->mat[index])); /* should this be moved to the creation routine? Check how PCSetUp works and if it can be reused */
      PetscCall(KSPSolve(rp->ksp[index],rp->vec[index],Flux)); 
    case Nonlinear:
      snesctx.dm = network; 
      snesctx.v  = v; 
      snesctx.rp = rp; 
      snesctx.U  = U; 
      PetscCall(SNESSetApplicationContext(rp->snes[index],(void*)&snesctx)); 
      PetscCall(SNESSolve(rp->snes[index],NULL,Flux)); 
    case Other: 
      PetscUseTypeMethod(rp,solveFlux,network,v,U,Flux); 
  }
  PetscFunctionReturn(0); 
}

/*@
   NetRPSSolveStar - The driver function for solving for Riemann Problem fluxes. This will use the user provided functions 
   and auto cached solver objects to solve for the star state. New solver objects will be created and cached as necessary as well. 

   Not Collective on NetRP

   Input Parameter:
.  rs - the NetRP context obtained from RiemanSolverCreate()
.  network - the network that contains the vertex v with the topology of the riemann problem. 
.  v  - the vertex in network to solve the riemann problem at 
.  U  - vec containing the the deg(v)*numfield initial states of the riemman problem. Allocated by caller. 

  Output Parameter: 

. Star - Vec Containing the deg(v)*numfield star states after solving the riemann problem. Allocated by caller. 

   Level: beginner

.seealso: NetRPCreate(), NetRPSetFlux()
@*/
PetscErrorCode NetRPSolveStar(NetRP rp, DM network, PetscInt v, Vec U, Vec Star)
{
  PetscBool flg;
  PetscInt  index; 
  NetRPSNESctx snesctx; 

  PetscFunctionBegin;
  if(!rp->setupcalled) PetscCall(NetRPSetUp(rp)); 
  /* find index of cached solvers */
  PetscCall(PetscHMapIHas(rp->hmap,v,&flg));
  if(!flg) PetscCall(NetRPAddVertexDegrees(rp,1,&v)); 
  PetscCall(PetscHMapIGet(rp->hmap,v,&index)); 

  /* switch based on type of NetRP */
  switch(rp->solvetype) 
  {
    case Linear: 
      PetscUseTypeMethod(rp,createLinearStar,network,v,U,rp->mat[index],rp->vec[index]);
      PetscCall(KSPSetOperators(rp->ksp[index],rp->mat[index],rp->mat[index])); /* should this be moved to the creation routine? Check how PCSetUp works and if it can be reused */
      PetscCall(KSPSolve(rp->ksp[index],rp->vec[index],Star)); 
    case Nonlinear:
      snesctx.dm = network; 
      snesctx.v  = v; 
      snesctx.rp = rp; 
      snesctx.U  = U; 
      PetscCall(SNESSetApplicationContext(rp->snes[index],(void*)&snesctx)); 
      PetscCall(SNESSolve(rp->snes[index],NULL,Star));  /* currently bugged as only one nonlinear function is allowed, need space for two. */
    case Other: 
      PetscUseTypeMethod(rp,solveStar,network,v,U,Star); 
  }
  PetscFunctionReturn(0); 
}