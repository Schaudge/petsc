#include <petsc/private/localnetrpimpl.h>
#include <petscnetrp.h> /*I "petscnetrp.h" I*/
#include <petsc/private/riemannsolverimpl.h> /* to be removed after adding fluxfunction class */

PetscLogEvent NetRP_Solve_Total;
PetscLogEvent NetRP_Solve_SetUp;
PetscLogEvent NetRP_Solve_System;

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
PetscErrorCode NetRPSetUp(NetRP rp)
{
  PetscInt numfield_flux, numfield_rp;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rp, NETRP_CLASSID, 1);
  if (rp->setupcalled) PetscFunctionReturn(0);
  rp->setupcalled = PETSC_TRUE;
  PetscCheck(rp->flux, PetscObjectComm((PetscObject)rp), PETSC_ERR_ARG_WRONGSTATE, "Requires a Flux to be set");
  PetscCall(RiemannSolverSetUp(rp->flux));
  PetscTryTypeMethod(rp, setup);
  /* check riemann problem and physics consistency */
  PetscCall(RiemannSolverGetNumFields(rp->flux, &numfield_flux));
  switch (rp->physicsgenerality) {
  case Generic: 
    rp->numfields = numfield_flux;
    break;
  default:
  case Specific:
    PetscCall(NetRPGetNumFields(rp, &numfield_rp));
    PetscCheck(numfield_flux == numfield_rp, PetscObjectComm((PetscObject)rp), PETSC_ERR_ARG_WRONGSTATE, "The physics number of fields : %" PetscInt_FMT " is not the same as the riemann problem number of fields  %" PetscInt_FMT, numfield_flux, numfield_rp);
    break;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode NetRPisSetup(NetRP rp, PetscBool *flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(rp, NETRP_CLASSID, 1);
  *flg = rp->setupcalled;
  PetscFunctionReturn(0);
}

PetscErrorCode NetRPGetNumFields(NetRP rp, PetscInt *numfields)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(rp, NETRP_CLASSID, 1);
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
PetscErrorCode NetRPReset(NetRP rp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(rp, NETRP_CLASSID, 1);
  if (rp->ops->reset) { PetscCall((*rp->ops->reset)(rp)); }
  PetscCall(NetRPClearCache(rp));
  rp->solvetype = Other;
  /* Note that we should reference the RiemannSolver inside the NetRS to properly handle this reset behavior. */
  rp->setupcalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

PetscErrorCode NetRPGetNumCached(NetRP rp, PetscInt *numcached)
{
  PetscInt off = 0;
  NetRPCacheType cachetype; 

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rp, NETRP_CLASSID, 1);
  PetscCall(NetRPGetCacheType(rp,&cachetype));
  switch(cachetype) {
    case UndirectedVDeg: 
      PetscCall(PetscHMapIGetSize(rp->hmap, numcached));
      break;
    case DirectedVDeg:
      PetscCall(PetscHMapIJGetSize(rp->dirhmap,numcached));
      break; 
  }
  PetscFunctionReturn(0);
}

/*@
   NetRPClearCache - Clears all cached solver objects in the NetRP. 

   Not Collective on NetRP

   Input Parameter:
.  rs - the NetRP context from NetRPCreate()

   Level: advanced

.seealso: NetRPCreate(), NetRPSetUp(), NetRPDestroy()
@*/
PetscErrorCode NetRPClearCache(NetRP rp)
{
  PetscInt i, numcached;
  NetRPCacheType cachetype; 

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rp, NETRP_CLASSID, 1);
  if (rp->ops->clearcache) { PetscCall((*rp->ops->clearcache)(rp)); }
  PetscCall(NetRPGetNumCached(rp, &numcached));
  if (rp->ksp) {
    switch (rp->solvetype) {
    case Linear:
      for (i = 0; i < numcached; i++) {
        PetscCall(KSPDestroy(&rp->ksp[i]));
        PetscCall(VecDestroy(&rp->vec[i]));
        PetscCall(MatDestroy(&rp->mat[i]));
      }
      break;
    case Nonlinear:
      for (i = 0; i < numcached; i++) { PetscCall(SNESDestroy(&rp->snes[i])); }
      break;
    case Optimization:
      for (i = 0; i < numcached; i++) { PetscCall(TaoDestroy(&rp->tao[i])); }
    case Other:
      break;
    }
    PetscFree4(rp->mat, rp->vec, rp->ksp, rp->snes);
  }
  PetscCall(NetRPGetCacheType(rp, &cachetype))
  switch(cachetype) {
    case UndirectedVDeg: 
      PetscCall(PetscHMapIClear(rp->hmap));
      break; 
    case DirectedVDeg: 
      PetscCall(PetscHMapIClear(rp->dirhmap));
      break; 
  }
  PetscFunctionReturn(0);
}
/*@
   NetRPCreateLinear - Preallocate matrix structure for solving a vertdeg Riemann Problem. Does a default setup of 
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

PetscErrorCode NetRPCreateLinear(NetRP rp, PetscInt vertdeg, Mat *mat, Vec *vec)
{
  Mat         _mat;
  Vec         _vec;
  const char *prefix_netrp;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rp, NETRP_CLASSID, 1);

  PetscCall(MatCreate(PETSC_COMM_SELF, &_mat));
  PetscCall(PetscObjectGetOptionsPrefix((PetscObject)rp, &prefix_netrp));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)_mat, prefix_netrp));
  PetscCall(PetscObjectAppendOptionsPrefix((PetscObject)_mat, "netrp_"));

  if (rp->ops->setupmat) PetscCall((rp->ops->setupmat)(rp, vertdeg, _mat));

  PetscCall(MatSetFromOptions(_mat));
  PetscCall(MatSetSizes(_mat, PETSC_DECIDE, PETSC_DECIDE, vertdeg * rp->numfields, vertdeg * rp->numfields));
  PetscCall(MatSetUp(_mat));

  PetscCall(MatCreateVecs(_mat, NULL, &_vec));
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
  KSP         _ksp;
  PC          pc;
  const char *prefix_netrp;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rp, NETRP_CLASSID, 1);

  PetscCall(KSPCreate(PETSC_COMM_SELF, &_ksp));
  PetscCall(PetscObjectGetOptionsPrefix((PetscObject)rp, &prefix_netrp));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)_ksp, prefix_netrp));
  PetscCall(PetscObjectAppendOptionsPrefix((PetscObject)_ksp, "netrp_"));
  /* Default to direct linear solver, which makes sense for these small systems */
  PetscCall(KSPGetPC(_ksp, &pc));
  PetscCall(PCSetType(pc, PCLU));
  PetscCall(KSPSetType(_ksp, KSPPREONLY));

  if (rp->ops->setupksp) PetscCall((rp->ops->setupksp)(rp, vertdeg, _ksp));
  PetscCall(KSPSetFromOptions(_ksp));
  // PetscCall(KSPSetUp(_ksp)); DO NOT SETUP AS KSP only "sets up" when is actually has a linear operator on it.
  *ksp = _ksp;
  PetscFunctionReturn(0);
}

typedef struct {
  NetRP      rp;
  PetscInt   vdeg;
  PetscBool *edgein;
  Vec        U;
} NetRPSNESctx;
/*C
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
C*/

static PetscErrorCode NetRPSNESWrapperFunc(SNES snes, Vec x, Vec f, void *ctx)
{
  NetRPSNESctx *netrpsnesctx;
  NetRP         rp;
  PetscInt      vdeg;
  PetscBool    *edgein;
  Vec           U;
  void         *appctx;

  PetscFunctionBegin;
  PetscCall(SNESGetApplicationContext(snes, &appctx));
  netrpsnesctx = (NetRPSNESctx *)appctx;
  rp           = netrpsnesctx->rp;
  vdeg         = netrpsnesctx->vdeg;
  edgein       = netrpsnesctx->edgein;
  U            = netrpsnesctx->U;
  PetscUseTypeMethod(rp, NonlinearEval, vdeg, edgein, U, x, f);
  PetscFunctionReturn(0);
}

/*C
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
C*/

static PetscErrorCode NetRPSNESWrapperJac(SNES snes, Vec x, Mat Amat, Mat Pmat, void *ctx)
{
  NetRPSNESctx *netrpsnesctx;
  NetRP         rp;
  PetscBool    *edgein;
  PetscInt      vdeg;
  Vec           U;
  void         *appctx;

  PetscFunctionBegin;
  PetscCall(SNESGetApplicationContext(snes, &appctx));
  netrpsnesctx = (NetRPSNESctx *)appctx;
  rp           = netrpsnesctx->rp;
  edgein       = netrpsnesctx->edgein;
  vdeg         = netrpsnesctx->vdeg;
  U            = netrpsnesctx->U;
  PetscUseTypeMethod(rp, NonlinearJac, vdeg, edgein, U, x, Amat);
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
  SNES        _snes;
  const char *prefix_netrp;
  PetscInt    numfield;
  Mat         jac;
  KSP         ksp;
  PC          pc;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rp, NETRP_CLASSID, 1);

  PetscCall(SNESCreate(PETSC_COMM_SELF, &_snes));
  PetscCall(PetscObjectGetOptionsPrefix((PetscObject)rp, &prefix_netrp));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)_snes, prefix_netrp));
  PetscCall(PetscObjectAppendOptionsPrefix((PetscObject)_snes, "netrp_"));
  if (rp->ops->setupsnes) PetscCall((rp->ops->setupsnes)(rp, vertdeg, _snes));
  PetscCall(SNESSetFromOptions(_snes));
  PetscCall(SNESGetKSP(_snes, &ksp));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)ksp, prefix_netrp));
  PetscCall(PetscObjectAppendOptionsPrefix((PetscObject)ksp, "netrp_snes_"));

  /* Default to direct linear solver, which makes sense for these small systems */
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCLU));
  PetscCall(KSPSetType(ksp, KSPPREONLY));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(SNESSetFunction(_snes, NULL, NetRPSNESWrapperFunc, NULL)); /* should I provide the r vec here? */

  if (rp->ops->NonlinearJac) {
    /* Create Jacobian Mat */
    PetscCall(MatCreate(PETSC_COMM_SELF, &jac));
    PetscCall(PetscObjectGetOptionsPrefix((PetscObject)rp, &prefix_netrp));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject)jac, prefix_netrp));
    PetscCall(PetscObjectAppendOptionsPrefix((PetscObject)jac, "netrp_jac_"));
    PetscTryTypeMethod(rp, setupjac, vertdeg, jac);
    PetscCall(MatSetFromOptions(jac));
    PetscCall(NetRPGetNumFields(rp, &numfield));
    PetscCall(MatSetSizes(jac, PETSC_DECIDE, PETSC_DECIDE, vertdeg * numfield, vertdeg * numfield));
    PetscCall(MatSetUp(jac));

    PetscCall(SNESSetJacobian(_snes, jac, jac, NetRPSNESWrapperJac, NULL));
    PetscCall(MatDestroy(&jac)); /* dereference the jacobian mat, now ownership controlled by SNES */
  }
  *snes = _snes;
  PetscFunctionReturn(0);
}
/*@
   NetRPCreateTao - Create the Tao

   Not Collective on NetRP

   Input Parameter:
.  rp - the NetRP context obtained from NetRPCreate()
.  vertdeg -  Vertex degree for the problem 

  Output Parameter: 

  . tao - the created TAO for the Riemann Problem

   Level: developer

.seealso: 
@*/

/* 
  Super needs a redo, as this is entirely designed for the traffic case 
*/
PetscErrorCode NetRPCreateTao(NetRP rp, PetscInt indeg, PetscInt outdeg, Tao *tao)
{
  Tao        _tao;
  const char *prefix_netrp;
  PetscInt    numfield; 
  Vec         LB,UB, InitialGuess; 

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rp, NETRP_CLASSID, 1);

  PetscCall(TaoCreate(PETSC_COMM_SELF, &_tao));
  PetscCall(PetscObjectGetOptionsPrefix((PetscObject)rp, &prefix_netrp));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)_tao, prefix_netrp));
  PetscCall(PetscObjectAppendOptionsPrefix((PetscObject)_tao, "netrp_"));
  if (rp->ops->setuptao) PetscCall((rp->ops->setuptao)(rp, indeg, outdeg, _tao));
  PetscCall(TaoSetFromOptions(_tao));


  /* connstruct the storage for things needed for TAO. 
     To be redone
  */
  PetscCall(NetRPGetNumFields(rp,&numfield));
  PetscCall(VecCreateSeq(PETSC_COMM_SELF,numfield*indeg,&LB)); 
  PetscCall(VecDuplicate(LB,&UB)); 

  PetscCall(TaoSetVariableBounds(_tao,LB,UB)); 
  PetscCall(VecDuplicate(LB,&InitialGuess));
  PetscCall(TaoSetSolution(_tao,InitialGuess)); 

  PetscCall(VecDestroy(&LB)); 
  PetscCall(VecDestroy(&UB)); 
  PetscCall(VecDestroy(&InitialGuess));

  *tao = _tao;
  PetscFunctionReturn(0);
}
/*@
   NetRPAddVertexDegrees_internal - Add vertex degrees to cache solvers for. If these degrees are already cached then
   nothing will happen. This is an internal function for the UndirectedVdeg Cachetype. 

   Not Collective  on NetRP

   Input Parameter:
.  rp - the NetRP context obtained from NetRPCreate()
.  numdeg- the number of vertex degrees to add 
.  vertdegs -  array of vertex degrees to add. This array must be destroyed by the caller. 

   Level: intermediate

.seealso: `NetRPAddDirVertexDegree_internal()`
@*/
PetscErrorCode NetRPAddVertexDegrees_internal(NetRP rp, PetscInt numdegs, PetscInt *vertdegs)
{
  PetscInt  i, numentries, totalnew = 0, off;
  PetscBool flg;
  Mat      *mat_new;
  KSP      *ksp_new;
  SNES     *snes_new;
  Vec      *vec_new;
  Tao      *tao_new;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rp, NETRP_CLASSID, 1);
  PetscCheck(rp->setupcalled, PetscObjectComm((PetscObject)rp), PETSC_ERR_ARG_WRONGSTATE, "Call NetRPSetUp() first");
  PetscCall(PetscHMapIGetSize(rp->hmap, &numentries));
  /* count number of new entries */
  for (i = 0; i < numdegs; i++) {
    PetscCheck(vertdegs[i] > 0, PetscObjectComm((PetscObject)rp), PETSC_ERR_USER_INPUT, "NetRP requires vertex degrees to be greater than 0 not %" PetscInt_FMT, vertdegs[i]);
    PetscCall(PetscHMapIHas(rp->hmap, vertdegs[i], &flg));
    if (!flg) totalnew++;
  }
  if (totalnew == 0) PetscFunctionReturn(0);
  /* reallocate solver cache arrays with room for new entries */
  PetscCall(PetscMalloc5(numentries + totalnew, &mat_new, numentries + totalnew, &vec_new, numentries + totalnew, &ksp_new, numentries + totalnew, &snes_new, numentries + totalnew, &tao_new));

  PetscCall(PetscArraycpy(mat_new, rp->mat, numentries));
  PetscCall(PetscArraycpy(vec_new, rp->vec, numentries));
  PetscCall(PetscArraycpy(ksp_new, rp->ksp, numentries));
  PetscCall(PetscArraycpy(snes_new, rp->snes, numentries));
  PetscCall(PetscArraycpy(tao_new, rp->tao, numentries));

  if (rp->mat) { /* if any memeory has every been allocated */
    PetscCall(PetscFree5(rp->mat, rp->vec, rp->ksp, rp->snes,rp->tao));
  }

  rp->mat  = mat_new;
  rp->ksp  = ksp_new;
  rp->snes = snes_new;
  rp->vec  = vec_new;
  rp->tao  = tao_new; 

  /* add new entries */
  off = 0;
  for (i = 0; i < numdegs; i++) {
    PetscCall(PetscHMapIHas(rp->hmap, vertdegs[i], &flg));
    if (flg) continue;
    PetscCall(PetscHMapISet(rp->hmap, vertdegs[i], numentries + off));
    /* only create what is needed */
    switch (rp->solvetype) {
    case Nonlinear: /* assumes only usage of snes */
      PetscCall(NetRPCreateSNES(rp, vertdegs[i], &rp->snes[numentries + off]));
      break;
    case Linear: /* assumes only usage of Mat and KSP */
      PetscCall(NetRPCreateLinear(rp, vertdegs[i], &rp->mat[numentries + off], &rp->vec[numentries + off]));
      PetscCall(NetRPCreateKSP(rp, vertdegs[i], &rp->ksp[numentries + off]));
      break;
    case Optimization: 
      SETERRQ(PetscObjectComm((PetscObject)rp),PETSC_ERR_SUP, "Currently Does not support UndirectVDeg cacheing for optimization solvers."); 
      break;
    case Other: /* Create Nothing */
      break;
    }
    off++;
  }
  PetscFunctionReturn(0);
}

/*@
   NetRPAddDirVertexDegrees_internal - Add directed vertex degrees to cache solvers for. If these degree pairs are already cached then
   nothing will happen. This is an internal function for the DirectedVdeg Cachetype. 

   Not Collective  on NetRP

   Input Parameter:
.  rp - the NetRP context obtained from NetRPCreate()
.  numdegs - the number of (in,out) vertex degrees to add 
.  invertdegs -  array of invertex degrees to add. This array must be destroyed by the caller. 
.  outvertdegs - array of outvertex degrees to add. This array must be destroyed by the caller.

   Level: intermediate

.seealso: `NetRPAddVertexDegree_internal()`
@*/
PetscErrorCode NetRPAddDirVertexDegrees_internal(NetRP rp, PetscInt numdegs, PetscInt *invertdegs, PetscInt *outvertdegs)
{
  PetscInt  i, numentries, totalnew = 0, off;
  PetscHashIJKey ijkey; 
  PetscBool flg;
  Mat      *mat_new;
  KSP      *ksp_new;
  SNES     *snes_new;
  Vec      *vec_new;
  Tao      *tao_new;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rp, NETRP_CLASSID, 1);
  PetscCheck(rp->setupcalled, PetscObjectComm((PetscObject)rp), PETSC_ERR_ARG_WRONGSTATE, "Call NetRPSetUp() first");
  PetscCall(PetscHMapIJGetSize(rp->dirhmap, &numentries));
  /* count number of new entries */
  for (i = 0; i < numdegs; i++) {
    PetscCheck(invertdegs[i] >= 0, PetscObjectComm((PetscObject)rp), PETSC_ERR_USER_INPUT, "NetRP requires in vertex degrees to be nonnegativ not %" PetscInt_FMT, invertdegs[i]);
    PetscCheck(outvertdegs[i] >= 0, PetscObjectComm((PetscObject)rp), PETSC_ERR_USER_INPUT, "NetRP requires out vertex degrees to be  to be nonnegative not %" PetscInt_FMT, outvertdegs[i]);
    PetscCheck(invertdegs[i]+outvertdegs[i] > 0, PetscObjectComm((PetscObject)rp), PETSC_ERR_USER_INPUT, "NetRP requires vertex degrees to be greater than 0 not %" PetscInt_FMT, invertdegs[i]+outvertdegs[i]);
    ijkey.i = invertdegs[i]; ijkey.j = outvertdegs[i];
    PetscCall(PetscHMapIJHas(rp->dirhmap, ijkey, &flg));
    if (!flg) totalnew++;
  }
  if (totalnew == 0) PetscFunctionReturn(0);
  /* reallocate solver cache arrays with room for new entries */
  PetscCall(PetscMalloc5(numentries + totalnew, &mat_new, numentries + totalnew, &vec_new, numentries + totalnew, &ksp_new, numentries + totalnew, &snes_new, numentries + totalnew, &tao_new));

  PetscCall(PetscArraycpy(mat_new, rp->mat, numentries));
  PetscCall(PetscArraycpy(vec_new, rp->vec, numentries));
  PetscCall(PetscArraycpy(ksp_new, rp->ksp, numentries));
  PetscCall(PetscArraycpy(snes_new, rp->snes, numentries));
  PetscCall(PetscArraycpy(tao_new, rp->tao, numentries));

  if (rp->mat) { /* if any memeory has every been allocated */
    PetscCall(PetscFree5(rp->mat, rp->vec, rp->ksp, rp->snes,rp->tao));
  }

  rp->mat  = mat_new;
  rp->ksp  = ksp_new;
  rp->snes = snes_new;
  rp->vec  = vec_new;
  rp->tao  = tao_new; 

  /* add new entries */
  off = 0;
  for (i = 0; i < numdegs; i++) {
    ijkey.i = invertdegs[i]; ijkey.j = outvertdegs[i];
    PetscCall(PetscHMapIJHas(rp->dirhmap, ijkey, &flg));
    if (flg) continue;
    PetscCall(PetscHMapIJSet(rp->hmap, ijkey, numentries + off));
    /* only create what is needed */
    switch (rp->solvetype) {
    case Nonlinear: /* assumes only usage of snes */
      SETERRQ(PetscObjectComm((PetscObject)rp),PETSC_ERR_SUP, "Currently Does not support UndirectVDeg cacheing for Nonlinear solvers."); 
      break;
    case Linear: /* assumes only usage of Mat and KSP */
      SETERRQ(PetscObjectComm((PetscObject)rp),PETSC_ERR_SUP, "Currently Does not support UndirectVDeg cacheing for Linear solvers."); 
      break;
    case Optimization: 
      PetscCall(NetRPCreateTao(rp,invertdegs[i],outvertdegs[j],&rp->tao[numentries + off]));
      break;
    case Other: /* Create Nothing */
      break;
    }
    off++;
  }
  PetscFunctionReturn(0);
}

/*@
   NetRPCacheSolvers - Cache space for solver objects for the given topologies. Exact solver types cached will depend
   the NetRPCacheType and NetRPSolveType.  This is the public interface for precacheing solvers that a user 
   knows will be needed in a single call. 


   Not Collective  on NetRP

   Input Parameter:
.  rp - the NetRP context obtained from NetRPCreate()
.  numdegs - the number of (in,out) vertex degrees to add 
.  invertdegs -  array of invertex degrees to add. This array must be destroyed by the caller. 
.  outvertdegs - array of outvertex degrees to add. This array must be destroyed by the caller.

   Level: intermediate

.seealso: `NetRPAddVertexDegree_internal()`
@*/
PetscErrorCode NetRPCacheSolvers(NetRP rp, PetscInt numdegs, PetscInt *invertdegs, PetscInt *outvertdegs)
{
  NetRPCacheType cachetype; 
  PetscInt       *vertdegs; 
  PetscInt       i; s

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rp, NETRP_CLASSID, 1);
  PetscCheck(rp->setupcalled, PetscObjectComm((PetscObject)rp), PETSC_ERR_ARG_WRONGSTATE, "Call NetRPSetUp() first");

  PetscCall(NetRPGetCacheType(rp,&cachetype)); 
  switch(cachetype) {
    case UndirectedVDeg: 
      PetscCall(PetscMalloc1(numdegs,&vertdegs)); 
      for(i=0; i<numdegs; i++) vertdegs[i] = invertdegs[i] + outvertdegs[i];
      PetscCall(NetRPAddVertexDegrees_internal(rp,numdegs,vertdegs)); 
      PetscCall(PetscFree1(vertdegs));
      break; 
    case DirectedVDeg: 
      PetscCall(NetRPAddDirVertexDegrees_internal(rp,numdegs,invertdegs,outvertdegs));
      break; 
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
PetscErrorCode NetRPDestroy(NetRP *rp)
{
  PetscFunctionBegin;
  if (!*rp) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(*rp, NETRP_CLASSID, 1);
  if (--((PetscObject)(*rp))->refct > 0) {
    *rp = NULL;
    PetscFunctionReturn(0);
  }
  PetscCall(NetRPReset(*rp));
  if ((*rp)->ops->destroy) PetscCall((*(*rp)->ops->destroy)((*rp)));
  if ((*rp)->flux) RiemannSolverDestroy(&(*rp)->flux);
  PetscCall(PetscHMapIDestroy(&(*rp)->hmap));
  PetscCall(PetscHMapIJDestroy(&(*rp)->dirhmap));
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
PetscErrorCode NetRPDuplicate(NetRP rp, NetRP *newrp)
{
  MPI_Comm comm;
  NetRP    rp_new;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rp, NETRP_CLASSID, 1);
  PetscValidPointer(newrp, 2);
  PetscValidType(rp, 1);

  PetscCall(PetscObjectGetComm((PetscObject)rp, &comm));
  PetscCall(NetRPCreate(comm, &rp_new));
  /* copy over the parameters and physics from rp to newrp */
  /* physics*/
  rp_new->user = rp->user;
  PetscCall(NetRPSetFlux(rp_new, rp->flux));
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
PetscErrorCode NetRPSetApplicationContext(NetRP rp, void *usrP)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(rp, NETRP_CLASSID, 1);
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
PetscErrorCode NetRPGetApplicationContext(NetRP rp, void *usrP)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(rp, NETRP_CLASSID, 1);
  *(void **)usrP = rp->user;
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
  const char *defaultType;
  char        name[256];
  PetscBool   flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rp, NETRP_CLASSID, 1);
  /* Type Option */
  if (!((PetscObject)rp)->type_name) {
    defaultType = NETRPBLANK;
  } else {
    defaultType = ((PetscObject)rp)->type_name;
  }
  if (!NetRPRegisterAllCalled) PetscCall(NetRPRegisterAll());

  PetscObjectOptionsBegin((PetscObject)rp);
  PetscCall(PetscOptionsFList("-netrp_type", "NetRP", "NetRPSetType", NetRPList, defaultType, name, 256, &flg));
  if (flg) {
    PetscCall(NetRPSetType(rp, name));
  } else if (!((PetscObject)rp)->type_name) {
    PetscCall(NetRPSetType(rp, defaultType));
  }

  /* handle implementation specific options */
  if (rp->ops->setfromoptions) { PetscCall((*rp->ops->setfromoptions)(PetscOptionsObject, rp)); }
  /* process any options handlerp added with PetscObjectAddOptionsHandler() */
  PetscCall(PetscObjectProcessOptionsHandlers((PetscObject)rp, PetscOptionsObject));
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
PetscErrorCode NetRPView(NetRP rp, PetscViewer viewer)
{
  PetscFunctionBegin;

  PetscFunctionReturn(0);
}

PetscErrorCode NetRPSetFlux(NetRP rp, RiemannSolver rs)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(rp, NETRP_CLASSID, 1);
  PetscValidHeaderSpecific(rs, RIEMANNSOLVER_CLASSID, 1);
  PetscCheck(!rp->setupcalled, PetscObjectComm((PetscObject)rs), PETSC_ERR_ARG_WRONGSTATE, "Cannot Set Flux after NetRP is setup.");
  if (rp->flux) PetscCall(RiemannSolverDestroy(&rp->flux));
  PetscCall(PetscObjectReference((PetscObject)rs));
  rp->flux = rs;
  PetscFunctionReturn(0);
}

PetscErrorCode NetRPGetFlux(NetRP rp, RiemannSolver *rs)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(rp, NETRP_CLASSID, 1);
  if (rp->flux) *rs = rp->flux;
  PetscFunctionReturn(0);
}

PetscErrorCode NeetRPSetNumFields(NetRP rp, PetscInt numfields)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(rp, NETRP_CLASSID, 1);
  rp->numfields = numfields;
  PetscFunctionReturn(0);
}

PetscErrorCode NeetRPGetNumFields(NetRP rp, PetscInt *numfields)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(rp, NETRP_CLASSID, 1);
  /* NetRPSetUp contains the logic for setting computing this */
  PetscCheck(rp->setupcalled, PetscObjectComm((PetscObject)rp), PETSC_ERR_ARG_WRONGSTATE, "Call NetRPSetUp() first");
  *numfields = rp->numfields;
  PetscFunctionReturn(0);
}

PetscErrorCode NetRPSetPhysicsGenerality(NetRP rp, NetRPPhysicsGenerality generality)
{
  PetscBool flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rp, NETRP_CLASSID, 1);
  PetscCheck(!rp->setupcalled, PetscObjectComm((PetscObject)rp), PETSC_ERR_ARG_WRONGSTATE, "Must be called before NetRPSetUp()");
  PetscCall(PetscObjectTypeCompare((PetscObject)rp, NETRPBLANK, &flg));
  PetscCheck(flg, PetscObjectComm((PetscObject)rp), PETSC_ERR_ARG_WRONGSTATE, "Can only be manually set on the blank type of NetRP");
  rp->physicsgenerality = generality;
  PetscFunctionReturn(0);
}

PetscErrorCode NetRPGetPhysicsGenerality(NetRP rp, NetRPPhysicsGenerality *generality)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(rp, NETRP_CLASSID, 1);
  *generality = rp->physicsgenerality;
  PetscFunctionReturn(0);
}

PetscErrorCode NetRPSetSolveType(NetRP rp, NetRPSolveType solvetype)
{
  PetscBool flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rp, NETRP_CLASSID, 1);
  PetscCheck(!rp->setupcalled, PetscObjectComm((PetscObject)rp), PETSC_ERR_ARG_WRONGSTATE, "Solve Type must be set before calling NetRPSetUp()");
  /* only the blank implementation should allow for setting this, other implementations are assumed to fix the type themselves */
  PetscCall(PetscObjectTypeCompare((PetscObject)rp, NETRPBLANK, &flg));
  PetscCheck(flg, PetscObjectComm((PetscObject)rp), PETSC_ERR_ARG_WRONGSTATE, "Solve Type can only be manually set on the blank type of NetRP");
  rp->solvetype = solvetype;
  PetscFunctionReturn(0);
}

PetscErrorCode NetRPGetSolveType(NetRP rp, NetRPSolveType *solvetype)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(rp, NETRP_CLASSID, 1);
  *solvetype = rp->solvetype;
  PetscFunctionReturn(0);
}

PetscErrorCode NetRPSetSolveStar(NetRP rp, NetRPSolveStar_User solvestar)
{
  PetscBool flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rp, NETRP_CLASSID, 1);
  PetscCheck(!rp->setupcalled, PetscObjectComm((PetscObject)rp), PETSC_ERR_ARG_WRONGSTATE, "Must be set before calling NetRPSetUp()");
  /* only the blank implementation should allow for setting this, other implementations are assumed to fix the type themselves */
  PetscCall(PetscObjectTypeCompare((PetscObject)rp, NETRPBLANK, &flg));
  PetscCheck(flg, PetscObjectComm((PetscObject)rp), PETSC_ERR_ARG_WRONGSTATE, "Can only be manually set on the blank type of NetRP");
  rp->ops->solveStar = solvestar;
  PetscFunctionReturn(0);
}

PetscErrorCode NetRPSetSolveFlux(NetRP rp, NetRPSolveFlux_User solveflux)
{
  PetscBool flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rp, NETRP_CLASSID, 1);
  PetscCheck(!rp->setupcalled, PetscObjectComm((PetscObject)rp), PETSC_ERR_ARG_WRONGSTATE, "Must be set before calling NetRPSetUp()");
  /* only the blank implementation should allow for setting this, other implementations are assumed to fix the type themselves */
  PetscCall(PetscObjectTypeCompare((PetscObject)rp, NETRPBLANK, &flg));
  PetscCheck(flg, PetscObjectComm((PetscObject)rp), PETSC_ERR_ARG_WRONGSTATE, "Can only be manually set on the blank type of NetRP");
  rp->ops->solveFlux = solveflux;
  PetscFunctionReturn(0);
}

PetscErrorCode NetRPSetCreateMatStar(NetRP rp, NetRPCreateLinearStar linStar)
{
  PetscBool flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rp, NETRP_CLASSID, 1);
  PetscCheck(!rp->setupcalled, PetscObjectComm((PetscObject)rp), PETSC_ERR_ARG_WRONGSTATE, "Must be set before calling NetRPSetUp()");
  /* only the blank implementation should allow for setting this, other implementations are assumed to fix the type themselves */
  PetscCall(PetscObjectTypeCompare((PetscObject)rp, NETRPBLANK, &flg));
  PetscCheck(flg, PetscObjectComm((PetscObject)rp), PETSC_ERR_ARG_WRONGSTATE, "Can only be manually set on the blank type of NetRP");
  rp->ops->createLinearStar = linStar;
  PetscFunctionReturn(0);
}

PetscErrorCode NetRPSetCreateMatFlux(NetRP rp, NetRPCreateLinearFlux linflux)
{
  PetscBool flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rp, NETRP_CLASSID, 1);
  PetscCheck(!rp->setupcalled, PetscObjectComm((PetscObject)rp), PETSC_ERR_ARG_WRONGSTATE, "Must be set before calling NetRPSetUp()");
  /* only the blank implementation should allow for setting this, other implementations are assumed to fix the type themselves */
  PetscCall(PetscObjectTypeCompare((PetscObject)rp, NETRPBLANK, &flg));
  PetscCheck(flg, PetscObjectComm((PetscObject)rp), PETSC_ERR_ARG_WRONGSTATE, "Can only be manually set on the blank type of NetRP");
  rp->ops->createLinearFlux = linflux;
  PetscFunctionReturn(0);
}

PetscErrorCode NetRPSetNonlinearEval(NetRP rp, NetRPNonlinearEval nonlineareval)
{
  PetscBool flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rp, NETRP_CLASSID, 1);
  PetscCheck(!rp->setupcalled, PetscObjectComm((PetscObject)rp), PETSC_ERR_ARG_WRONGSTATE, "Must be set before calling NetRPSetUp()");
  /* only the blank implementation should allow for setting this, other implementations are assumed to fix the type themselves */
  PetscCall(PetscObjectTypeCompare((PetscObject)rp, NETRPBLANK, &flg));
  PetscCheck(flg, PetscObjectComm((PetscObject)rp), PETSC_ERR_ARG_WRONGSTATE, "Can only be manually set on the blank type of NetRP");
  rp->ops->NonlinearEval = nonlineareval;
  PetscFunctionReturn(0);
}

PetscErrorCode NetRPSetNonlinearJac(NetRP rp, NetRPNonlinearJac nonlinearjac)
{
  PetscBool flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rp, NETRP_CLASSID, 1);
  PetscCheck(!rp->setupcalled, PetscObjectComm((PetscObject)rp), PETSC_ERR_ARG_WRONGSTATE, "Must be set before calling NetRPSetUp()");
  /* only the blank implementation should allow for setting this, other implementations are assumed to fix the type themselves */
  PetscCall(PetscObjectTypeCompare((PetscObject)rp, NETRPBLANK, &flg));
  PetscCheck(flg, PetscObjectComm((PetscObject)rp), PETSC_ERR_ARG_WRONGSTATE, "Can only be manually set on the blank type of NetRP");
  rp->ops->NonlinearJac = nonlinearjac;
  PetscFunctionReturn(0);
}

static PetscErrorCode NetRPComputeFluxInPlace_internal(NetRP rp, PetscInt vdeg, Vec Flux)
{
  PetscInt      i, numfields;
  PetscScalar  *star;
  PetscReal    *fluxval;
  RiemannSolver fluxfun;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rp, NETRP_CLASSID, 1);
  PetscCall(VecGetArray(Flux, &star));
  PetscCall(NetRPGetFlux(rp, &fluxfun));
  PetscCall(NetRPGetNumFields(rp, &numfields));
  for (i = 0; i < vdeg; i++) {
    PetscCall(RiemmanSolverEvaluateFlux(fluxfun, &star[i * numfields], &fluxval)); /* fluxval is owned by RiemannSolver */
    PetscCall(PetscArraycpy(star + i * numfields, fluxval, numfields));            /* modify in-place*/
  }
  PetscCall(VecRestoreArray(Flux, &star));
  PetscFunctionReturn(0);
}

PetscErrorCode NetRPSetCacheType(NetRP rp, NetRPCacheType cachetype)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(rp, NETRP_CLASSID, 1);
  PetscCheck(rp->setupcalled, PetscObjectComm((PetscObject)rp), PETSC_ERR_ARG_WRONGSTATE, "Must Call before NetRPSetUp()");
  rp->cachetype = cachetype; 
  PetscFunctionReturn(0);
}

/*
  Only fixed after setup has been called. Before calling this can be changed at anytime. 
  Should I only allow this to be called after setup to enforce this? 
*/
PetscErrorCode NetRPGetCacheType(NetRP rp, NetRPCacheType *cachetype)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(rp, NETRP_CLASSID, 1);
  *cachetype = rp->cachetype;  
  PetscFunctionReturn(0);
}


/* Requires that NetRP is SetUp. However this is up to the caller to check as 
  every internal usage already requires the check. */

/* Finds the index in the cached solver list for the given vertex. If it does not 
   already exist, creates the cached solvers. Always returns a valid index. 

  Exact implementation depends on the cachetype. 

*/
PetscErrorCode NetRPFindCacheIndex_internal(NetRP rp, PetscInt vdegin, PetscInt vdegout) 
{
  NetRPCacheType cachetype; 
  PetscHashIJKeyHash ijkey; 
  PetscInt       vdeg; 
  PetscBool      flg; 

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rp, NETRP_CLASSID, 1);
  PetscCall(NetRPGetCacheType(rp,&cachetype)); 
  switch(cachetype) {
    case DirectedVDeg: 
      ijkey.i = vdegin; ijkey.j = vdegout; 
      PetscCall(PetscHMapIJHas(rp->dirhmap,ijkey,&flg)); 
      if (!flg) PetscCall(NetRPAddDirVertexDegrees_internal(rp,1,&vdegin,&vdegout));
      PetscCall(PetscHMapIJGet(rp->dirhmap,ijkey,&index));
      break; 

    case UndirectedVDeg: 
      vdeg = vdegin+vdegout; 
      PetscCall(PetscHMapIHas(rp->hmap, vdeg, &flg));
      if (!flg) PetscCall(NetRPAddVertexDegrees_internal(rp, 1, &vdeg));
      PetscCall(PetscHMapIGet(rp->hmap, vdeg, &index));
      break; 
  }
  PetscFunctionReturn(0);
}

/*@
   NetRPSolveFlux - The driver function for solving for Riemann Problem fluxes. This will use the user provided functions 
   and auto cached solver objects to solve for the flux. New solver objects will be created and cached as necessary as well. 
   Always Calleable. 

   Not Collective on NetRP

   Input Parameter:
.  rp - the NetRP context obtained from RiemanSolverCreate()
.  network - the network that contains the vertex v with the topology of the riemann problem. 
.  v  - the vertex in network to solve the riemann problem at 
.  U  - vec containing the the deg(v)*numfield initial states of the riemman problem. Allocated by caller. 

  Output Parameter: 

. Flux - Vec Containing the deg(v)*numfield fluxes after solving the riemann problem. Allocated by caller. 

   Level: beginner

.seealso: NetRPCreate(), NetRPSetFlux()
@*/
PetscErrorCode NetRPSolveFlux(NetRP rp, PetscInt vdeg, PetscInt vdegin, PetscInt vdegout, PetscBool *edgein, Vec U, Vec Flux)
{
  PetscBool    flg;
  PetscInt     index; 
  NetRPSNESctx snesctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rp, NETRP_CLASSID, 1);
  PetscValidHeaderSpecific(U,VEC_CLASSID,4); 
  PetscValidHeaderSpecific(Flux,VEC_CLASSID,5); 
  if (!rp->setupcalled) PetscCall(NetRPSetUp(rp));
  PetscLogEventBegin(NetRP_Solve_Total, 0, 0, 0, 0);
  /* find index of cached solvers */
  PetscCall(NetRPFindCacheIndex_internal(rp,vdegin,vdegout));

  /* switch based on type of NetRP */
  switch (rp->solvetype) {
  case Linear: /* Assumes UndirectedVDeg Cachetype here */
    if (rp->ops->createLinearFlux) {
      PetscLogEventBegin(NetRP_Solve_System, 0, 0, 0, 0);
      PetscUseTypeMethod(rp, createLinearFlux, vdeg, edgein, U, rp->vec[index], rp->mat[index]);
      PetscCall(KSPSetOperators(rp->ksp[index], rp->mat[index], rp->mat[index])); /* should this be moved to the creation routine? Check how PCSetUp works and if it can be reused */
      PetscCall(KSPSolve(rp->ksp[index], rp->vec[index], Flux));
      PetscLogEventEnd(NetRP_Solve_System, 0, 0, 0, 0);

    } else if (rp->ops->createLinearStar) {
      PetscLogEventBegin(NetRP_Solve_System, 0, 0, 0, 0);
      PetscUseTypeMethod(rp, createLinearStar, vdeg, edgein, U, rp->vec[index], rp->mat[index]);
      PetscCall(KSPSetOperators(rp->ksp[index], rp->mat[index], rp->mat[index])); /* should this be moved to the creation routine? Check how PCSetUp works and if it can be reused */
      PetscCall(KSPSolve(rp->ksp[index], rp->vec[index], Flux));
      PetscLogEventEnd(NetRP_Solve_System, 0, 0, 0, 0); 
      /* inplace evaluate the star states in Flux by the physics flux to compute the actual flux */
      PetscCall(NetRPComputeFluxInPlace_internal(rp, vdeg, Flux));
    } else {
      SETERRQ(PetscObjectComm((PetscObject)rp), PETSC_ERR_PLIB, "No available solver for NetRPSolveFlux. This should not happen and should be caught at NetRPSetUp(). Solver Type is: LINEAR");
    }
    break;
  case Nonlinear: /* Assumes UndirectedVDeg Cachetype here */
    snesctx.edgein = edgein;
    snesctx.vdeg   = vdeg;
    snesctx.rp     = rp;
    snesctx.U      = U;
    PetscCall(SNESSetApplicationContext(rp->snes[index], (void *)&snesctx));
    PetscCall(VecCopy(U, Flux)); /* initial guess of the riemann data */
    PetscLogEventBegin(NetRP_Solve_System, 0, 0, 0, 0);
    PetscCall(SNESSolve(rp->snes[index], NULL, Flux)); /* currently assumes this solves for the star state */
    PetscLogEventEnd(NetRP_Solve_System, 0, 0, 0, 0);
    /* inplace evaluate the star states in Flux by the physics flux to compute the actual flux */
    PetscCall(NetRPComputeFluxInPlace_internal(rp, vdeg, Flux));
    break;
  case Optimization:
    /* 
      Need to set ctx here. 

      Note that Flux must be the solution vector? No... can't assume that as I do a preSolve and PostSolve 
      which can (and will) change vector sizes 
    */
    PetscCall(TaoSolve(rp->tao[index])); 
  case Other:
    PetscUseTypeMethod(rp, solveFlux, vdeg, edgein, U, Flux);
    break;
  }
  PetscLogEventEnd(NetRP_Solve_Total, 0, 0, 0, 0);

  PetscFunctionReturn(0);
}

/*@
   NetRPSolveStar - The driver function for solving for Riemann Problem fluxes. This will use the user provided functions 
   and auto cached solver objects to solve for the star state. New solver objects will be created and cached as necessary as well. 
   The type is not required to implement routines for this solver. Use `NetRPCanSolveStar()` to determine if this function can safely 
   be called. 


   Not Collective on NetRP

   Input Parameter:
.  rp - the NetRP context obtained from RiemanSolverCreate()
.  network - the network that contains the vertex v with the topology of the riemann problem. 
.  v  - the vertex in network to solve the riemann problem at 
.  U  - vec containing the the deg(v)*numfield initial states of the riemman problem. Allocated by caller. 

  Output Parameter: 

. Star - Vec Containing the deg(v)*numfield star states after solving the riemann problem. Allocated by caller. 

   Level: beginner

.seealso: NetRPCreate(), NetRPSetFlux()
@*/
PetscErrorCode NetRPSolveStar(NetRP rp, PetscInt vdeg, PetscBool *edgein, Vec U, Vec Star)
{
  PetscBool    flg;
  PetscInt     index;
  NetRPSNESctx snesctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rp, NETRP_CLASSID, 1);
  PetscValidHeaderSpecific(U,VEC_CLASSID,4); 
  PetscValidHeaderSpecific(Flux,VEC_CLASSID,5); 
  if (!rp->setupcalled) PetscCall(NetRPSetUp(rp));
  /* find index of cached solvers */
  PetscCall(PetscHMapIHas(rp->hmap, vdeg, &flg));
  if (!flg) PetscCall(NetRPAddVertexDegrees(rp, 1, &vdeg));
  PetscCall(PetscHMapIGet(rp->hmap, vdeg, &index));

  /* switch based on type of NetRP */
  switch (rp->solvetype) {
  case Linear:
    PetscUseTypeMethod(rp, createLinearStar, vdeg, edgein, U, rp->vec[index], rp->mat[index]);
    PetscCall(KSPSetOperators(rp->ksp[index], rp->mat[index], rp->mat[index])); /* should this be moved to the creation routine? Check how PCSetUp works and if it can be reused */
    PetscCall(KSPSolve(rp->ksp[index], rp->vec[index], Star));
    break;
  case Nonlinear:
    snesctx.edgein = edgein;
    snesctx.vdeg   = vdeg;
    snesctx.rp     = rp;
    snesctx.U      = U;
    PetscCall(SNESSetApplicationContext(rp->snes[index], (void *)&snesctx));
    PetscCall(VecCopy(U, Star));                       /* initial guess of the riemann data */
    PetscCall(SNESSolve(rp->snes[index], NULL, Star)); /* currently bugged as only one nonlinear function is allowed, need space for two. */
    break;
  case Optimization: 
    SETERRQ(PetscObjectComm((PetscObject)rp),PETSC_ERR_SUP,"Does not currently support NetRPSolveStar for Optimization");
  case Other:
    PetscUseTypeMethod(rp, solveStar, vdeg, edgein, U, Star);
    break;
  }
  PetscFunctionReturn(0);
}