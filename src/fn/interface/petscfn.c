
/*
   This is where the abstract PetscFn operations are defined
*/

#include <petsc/private/vecimpl.h>       /*I "petscvec.h" I*/
#include <petsc/private/matimpl.h>       /*I "petscmat.h" I*/
#include <petsc/private/fnimpl.h>        /*I "petscfn.h" I*/
#include <../src/fn/utils/fnutils.h>

/* Logging support */
PetscClassId PETSCFN_CLASSID;

/*@
   PetscFnCreate - Creates a PetscFn where the type is determined
   from either a call to PetscFnSetType() or from the options database
   with a call to PetscFnSetFromOptions(). The default PetscFn type is
   Shell. If you never call PetscFnSetType() or PetscFnSetFromOptions()
   it will generate an error when you try to use the function.

   Collective on MPI_Comm

   Input Parameter:
.  comm - MPI communicator

   Output Parameter:
.  fn - the function

   Notes:

   Level: beginner

.keywords: function, create

@*/
PetscErrorCode PetscFnCreate(MPI_Comm comm,PetscFn *fn)
{
  PetscFn        B;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(fn,2);

  *fn = NULL;
  ierr = PetscFnInitializePackage();CHKERRQ(ierr);

  ierr = PetscHeaderCreate(B,PETSCFN_CLASSID,"PetscFn","Function","PetscFn",comm,PetscFnDestroy,PetscFnView);CHKERRQ(ierr);
  ierr = PetscLayoutCreate(comm,&B->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutCreate(comm,&B->dmap);CHKERRQ(ierr);

  B->isScalar    = PETSC_FALSE;
  B->setupcalled = PETSC_FALSE;
  *fn            = B;
  PetscFunctionReturn(0);
}

/*@
   PetscFnDestroy - Frees space taken by a PetscFn.

   Collective on PetscFn

   Input Parameter:
.  fn - the function

   Level: beginner

@*/
PetscErrorCode PetscFnDestroy(PetscFn *fn)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*fn) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(*fn,PETSCFN_CLASSID,1);
  if (--((PetscObject)(*fn))->refct > 0) {*fn = NULL; PetscFunctionReturn(0);}

  if ((*fn)->ops->destroy) {
    ierr = (*(*fn)->ops->destroy)(*fn);CHKERRQ(ierr);
  }

  ierr = PetscLayoutDestroy(&(*fn)->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutDestroy(&(*fn)->dmap);CHKERRQ(ierr);
  ierr = PetscHeaderDestroy(fn);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnSetSizes(PetscFn fn, PetscInt m, PetscInt n, PetscInt M, PetscInt N)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn,PETSCFN_CLASSID,1);
  if (M > 0) PetscValidLogicalCollectiveInt(fn,M,4);
  if (N > 0) PetscValidLogicalCollectiveInt(fn,N,5);
  if (M > 0 && m > M) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Local range size %D cannot be larger than global range size %D",m,M);
  if (N > 0 && n > N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Local domain size %D cannot be larger than global domain size %D",n,N);
  if ((fn->rmap->n >= 0 && fn->rmap->N >= 0) && (fn->rmap->n != m || (M > 0 && fn->rmap->N != M))) SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot change/reset row sizes to %D local %D global after previously setting them to %D local %D global",m,M,fn->rmap->n,fn->rmap->N);
  if ((fn->dmap->n >= 0 && fn->dmap->N >= 0) && (fn->dmap->n != n || (N > 0 && fn->dmap->N != N))) SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot change/reset column sizes to %D local %D global after previously setting them to %D local %D global",n,N,fn->dmap->n,fn->dmap->N);
  fn->rmap->n = m;
  fn->dmap->n = n;
  fn->rmap->N = M > -1 ? M : fn->rmap->N;
  fn->dmap->N = N > -1 ? N : fn->dmap->N;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnGetSizes(PetscFn fn, PetscInt *m, PetscInt *n, PetscInt *M, PetscInt *N)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn,PETSCFN_CLASSID,1);
  if (m) *m = fn->rmap->n;
  if (n) *n = fn->dmap->n;
  if (M) *M = fn->rmap->N;
  if (N) *N = fn->dmap->N;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnGetLayouts(PetscFn fn,PetscLayout *rmap,PetscLayout *dmap)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn,PETSCFN_CLASSID,1);
  PetscValidType(fn,1);
  if (rmap) PetscValidPointer(rmap,2);
  if (dmap) PetscValidPointer(dmap,3);
  if (rmap) *rmap = fn->rmap;
  if (dmap) *dmap = fn->dmap;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnLayoutsSetUp(PetscFn fn)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLayoutSetUp(fn->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(fn->dmap);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnSetOptionsPrefix(PetscFn fn, const char prefix[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn,PETSCFN_CLASSID,1);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)fn,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnAppendOptionsPrefix(PetscFn fn,const char prefix[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn,PETSCFN_CLASSID,1);
  ierr = PetscObjectAppendOptionsPrefix((PetscObject)fn,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnGetOptionsPrefix(PetscFn fn,const char *prefix[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn,PETSCFN_CLASSID,1);
  ierr = PetscObjectGetOptionsPrefix((PetscObject)fn,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnSetFromOptions(PetscFn fn)
{
  PetscErrorCode ierr;
  const char     *deft = PETSCFNSHELL;
  char           type[256];
  PetscBool      test_all;
  PetscBool      flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn,PETSCFN_CLASSID,1);

  fn->setfromoptions = PETSC_TRUE;

  ierr = PetscObjectOptionsBegin((PetscObject)fn);CHKERRQ(ierr);
  ierr = PetscOptionsFList("-fn_type","Function type","PetscFnSetType",PetscFnList,deft,type,256,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscFnSetType(fn,type);CHKERRQ(ierr);
  } else if (!((PetscObject)fn)->type_name) {
    ierr = PetscFnSetType(fn,deft);CHKERRQ(ierr);
  }

  test_all = PETSC_FALSE;
  ierr = PetscOptionsBool("-fn_test_allmult","On first use, test the order of convergence of all derivative multiplications","PetscFnTestDerivativeMult",test_all,&test_all,NULL);CHKERRQ(ierr);
  if (test_all) {
    fn->test_jacmult     = PETSC_TRUE;
    fn->test_jacmultadj  = PETSC_TRUE;
    fn->test_hesmult     = PETSC_TRUE;
    fn->test_hesmultadj  = PETSC_TRUE;
    fn->test_scalgrad    = PETSC_TRUE;
    fn->test_scalhesmult = PETSC_TRUE;
  }
  ierr = PetscOptionsBool("-fn_test_jacobianmult","On first use, test the order of convergence of PetscFnJacobianMult","PetscFnTestDerivativeMult",fn->test_jacmult,&(fn->test_jacmult),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-fn_test_jacobianmultadjoint","On first use, test the order of convergence of PetscFnJacobianMultAdjoint","PetscFnTestDerivativeMult",fn->test_jacmultadj,&(fn->test_jacmultadj),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-fn_test_hessianmult","On first use, test the order of convergence of PetscFnHessianMult","PetscFnTestDerivativeMult",fn->test_hesmult,&(fn->test_hesmult),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-fn_test_hessianmultadjoint","On first use, test the order of convergence of PetscFnHessianMultAdjoint","PetscFnTestDerivativeMult",fn->test_hesmultadj,&(fn->test_hesmultadj),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-fn_test_scalargradient","On first use, test the order of convergence of PetscFnScalarGradient","PetscFnTestDerivativeMult",fn->test_scalgrad,&(fn->test_scalgrad),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-fn_test_scalarhessianmult","On first use, test the order of convergence of PetscFnScalarHessianMult","PetscFnTestDerivativeMult",fn->test_scalhesmult,&(fn->test_scalhesmult),NULL);CHKERRQ(ierr);
  test_all = PETSC_FALSE;
  ierr = PetscOptionsBool("-fn_test_allbuild","Test all built derivative matrices against matrix-free","PetscFnTestDerivativeBuild",test_all,&test_all,NULL);CHKERRQ(ierr);
  if (test_all) {
    fn->test_jacbuild     = PETSC_TRUE;
    fn->test_jacbuildadj  = PETSC_TRUE;
    fn->test_hesbuild     = PETSC_TRUE;
    fn->test_hesbuildadj  = PETSC_TRUE;
    fn->test_hesbuildswp  = PETSC_TRUE;
    fn->test_scalgrad    = PETSC_TRUE;
    fn->test_scalhesbuild = PETSC_TRUE;
  }
  ierr = PetscOptionsBool("-fn_test_jacobianbuild","On first use, test PetscFnJacobianBuild against matrix-free","PetscFnTestDerivativeBuild",fn->test_jacbuild,&(fn->test_jacbuild),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-fn_test_jacobianbuildadjoint","On first use, test PetscFnJacobianBuildAdjoint against matrix-free","PetscFnTestDerivativeBuild",fn->test_jacbuildadj,&(fn->test_jacbuildadj),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-fn_test_hessianbuild","On first use, test PetscFnHessianBuild against matrix-free","PetscFnTestDerivativeBuild",fn->test_hesbuild,&(fn->test_hesbuild),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-fn_test_hessianbuildadjoint","On first use, test PetscFnHessianBuildAdjoint against matrix-free","PetscFnTestDerivativeBuild",fn->test_hesbuildadj,&(fn->test_hesbuildadj),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-fn_test_hessianbuildswap","On first use, test PetscFnHessianBuildSwap against matrix-free","PetscFnTestDerivativeBuild",fn->test_hesbuildswp,&(fn->test_hesbuildswp),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-fn_test_scalarhessianbuild","On first use, test PetscFnScalarHessianBuild against matrix-free","PetscFnTestDerivativeBuild",fn->test_scalhesbuild,&(fn->test_scalhesbuild),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-fn_test_derfn","On first use, test the instantiated derivative PetscFns against matrix-free","PetscFnTestDerivativeFn",fn->test_derfn,&(fn->test_derfn),NULL);CHKERRQ(ierr);

  if (fn->ops->setfromoptions) {
    ierr = (*fn->ops->setfromoptions)(PetscOptionsObject,fn);CHKERRQ(ierr);
  }

  /* process any options handlers added with PetscObjectAddOptionsHandler() */
  ierr = PetscObjectProcessOptionsHandlers(PetscOptionsObject,(PetscObject)fn);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnSetUp(PetscFn fn)
{
  const char     *deft = PETSCFNSHELL;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn,PETSCFN_CLASSID,1);
  if (fn->setupcalled) PetscFunctionReturn(0);
  fn->setupcalled = PETSC_TRUE;
  if (!((PetscObject)fn)->type_name) {
    ierr = PetscFnSetType(fn, deft);CHKERRQ(ierr);
  }
  if (fn->ops->setup) {
    ierr = (*fn->ops->setup)(fn);CHKERRQ(ierr);
  }
  ierr = PetscFnLayoutsSetUp(fn);CHKERRQ(ierr);
  fn->isScalar = PETSC_FALSE;
  if (fn->rmap->N == 1) fn->isScalar = PETSC_TRUE;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnView(PetscFn fn,PetscViewer viewer)
{
  PetscErrorCode    ierr;
  PetscInt          rows,cols;
  PetscBool         iascii;
  PetscViewerFormat format;
  PetscMPIInt       size;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn,PETSCFN_CLASSID,1);
  PetscValidType(fn,1);
  if (!viewer) {
    ierr = PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)fn),&viewer);CHKERRQ(ierr);
  }
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(fn,1,viewer,2);
  ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)fn),&size);CHKERRQ(ierr);
  if (size == 1 && format == PETSC_VIEWER_LOAD_BALANCE) PetscFunctionReturn(0);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);

  if (iascii) {
    ierr = PetscObjectPrintClassNamePrefixType((PetscObject)fn,viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = PetscFnGetSizes(fn,NULL,NULL,&rows,&cols);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"range size=%D, domain size=%D\n",rows,cols);CHKERRQ(ierr);
  }
  if (fn->ops->view) {
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = (*fn->ops->view)(fn,viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  }
  if (iascii) {
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnVecCheckCompatible(Vec vec, IS is, PetscLayout layout)
{
  PetscInt       nV, nI;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  nV = vec->map->n;
  if (is) {
    ierr = ISGetLocalSize(is, &nI);CHKERRQ(ierr);
    if (nV != nI) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Vector is incompatible with index set");
  } else {
    nI = layout->n;
    if (nV != nI) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Vector is incompatible with layout");
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnMatCheckCompatible(Mat mat, IS rightIS, IS leftIS, PetscLayout rightLayout, PetscLayout leftLayout)
{
  PetscInt       nMat, mMat, nI, mI;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  nMat = mat->cmap->n;
  mMat = mat->rmap->n;
  if (leftIS) {
    ierr = ISGetLocalSize(leftIS, &nI);CHKERRQ(ierr);
    if (nI != nMat) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Matrix is incompatible with left index set");
  } else {
    nI = leftLayout->n;
    if (nI != nMat) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Matrix is incompatible with left layout");
  }
  if (rightIS) {
    ierr = ISGetLocalSize(rightIS, &mI);CHKERRQ(ierr);
    if (mI != mMat) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Matrix is incompatible with right index set");
  } else {
    mI = rightLayout->n;
    if (mI != mMat) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Matrix is incompatible with right layout");
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnCreateVec_Default(PetscFn fn, IS is, PetscLayout layout, Vec *vec)
{
  PetscInt       n, N;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCreate(PetscObjectComm((PetscObject)fn),vec);CHKERRQ(ierr);
  if (is) {
    ierr = ISGetLocalSize(is, &n);CHKERRQ(ierr);
    ierr = ISGetSize(is, &N);CHKERRQ(ierr);
    ierr = VecSetSizes(*vec, n, N);CHKERRQ(ierr);
  } else {
    ierr = VecSetLayout(*vec,layout);CHKERRQ(ierr);
  }
  ierr = VecSetType(*vec, VECSTANDARD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnCreateVecs(PetscFn fn, IS domainIS, Vec *domainVec, IS rangeIS, Vec *rangeVec)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn,PETSCFN_CLASSID,1);
  if (domainIS) PetscValidHeaderSpecific(domainIS,IS_CLASSID,2);
  if (rangeIS) PetscValidHeaderSpecific(rangeIS,IS_CLASSID,3);
  ierr = PetscFnLayoutsSetUp(fn);CHKERRQ(ierr);
  if (fn->ops->createvecs) {
    ierr = (*(fn->ops->createvecs)) (fn, domainIS, domainVec, rangeIS, rangeVec);CHKERRQ(ierr);
  } else {
    if (domainVec) {ierr = PetscFnCreateVec_Default(fn, domainIS, fn->dmap, domainVec);CHKERRQ(ierr);}
    if (rangeVec) {ierr = PetscFnCreateVec_Default(fn, rangeIS, fn->rmap, rangeVec);CHKERRQ(ierr);}
  }
#if defined(PETSC_USE_DEBUG)
  if (domainVec) {ierr = PetscFnVecCheckCompatible(*domainVec, domainIS, fn->dmap);CHKERRQ(ierr);}
  if (rangeVec) {ierr = PetscFnVecCheckCompatible(*rangeVec, rangeIS, fn->rmap);CHKERRQ(ierr);}
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnApply(PetscFn fn, Vec x, Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn,PETSCFN_CLASSID,1);
  PetscValidType(fn,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  PetscValidHeaderSpecific(y,VEC_CLASSID,3);
  ierr = PetscFnSetUp(fn);CHKERRQ(ierr);
  if (x == y) SETERRQ(PetscObjectComm((PetscObject)fn),PETSC_ERR_ARG_WRONGSTATE,"x and y must be different vectors");
  if (fn->dmap->N != x->map->N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"PetscFn fn,Vec x: global dim %D %D",fn->dmap->N,x->map->N);
  if (fn->rmap->N != y->map->N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"PetscFn fn,Vec y: global dim %D %D",fn->rmap->N,y->map->N);
  if (fn->rmap->n != y->map->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"PetscFn fn,Vec y: local dim %D %D",fn->rmap->n,y->map->n);
  VecLocked(y,3);

  ierr = VecLockPush(x);CHKERRQ(ierr);
  if (fn->ops->apply) {
    ierr = (*fn->ops->apply)(fn,x,y);CHKERRQ(ierr);
  } else if (fn->ops->derivativevec) {
    ierr = (*fn->ops->derivativevec)(fn,x,0,0,NULL,NULL,y);CHKERRQ(ierr);
  } else if (fn->isScalar && fn->ops->scalarapply) {
    PetscScalar z;

    ierr = (*fn->ops->scalarapply)(fn,x,&z);CHKERRQ(ierr);
    ierr = VecSet(y,z);CHKERRQ(ierr);
  } else if (fn->isScalar && fn->ops->scalarderivativescalar) {
    PetscScalar z;

    ierr = (*fn->ops->scalarderivativescalar)(fn,x,0,NULL,NULL,&z);CHKERRQ(ierr);
    ierr = VecSet(y,z);CHKERRQ(ierr);
  } else SETERRQ1(PetscObjectComm((PetscObject)fn), PETSC_ERR_SUP, "This PetscFn does not implement %s()", PETSC_FUNCTION_NAME);
  ierr = VecLockPop(x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnGetISVecsWithoutRange(PetscInt rangeIdx, PetscInt numISs, const IS *subsets, PetscInt numVecs, const Vec *subvecs, IS **newsubsets, Vec **newsubvecs)
{
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (rangeIdx == 0) {
    *newsubsets = (IS *) (subsets ? &subsets[1] : NULL);
    *newsubvecs = (Vec *) subvecs[1];
    PetscFunctionReturn(0);
  }
  if (rangeIdx >= numISs - 1) {
    *newsubsets = (IS *) subsets;
  } else {
    IS *sis = NULL;

    if (subsets) {
      ierr = PetscMalloc1(numISs - 1, &sis);CHKERRQ(ierr);
      for (i = 0; i < numISs - 1; i++) sis[i] = subsets[i + (i >= rangeIdx)];
    }
    *newsubsets = sis;
  }
  if (rangeIdx >= numVecs - 1) {
    *newsubvecs = (Vec *) subvecs;
  } else {
    Vec *svecs;

    ierr = PetscMalloc1(numVecs - 1, &svecs);CHKERRQ(ierr);
    for (i = 0; i < numVecs; i++) svecs[i] = subvecs[i + (i >= rangeIdx)];
    *newsubvecs = svecs;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnRestoreISVecsWithoutRange(PetscInt rangeIdx, PetscInt numISs, const IS *subsets, PetscInt numVecs, const Vec *subvecs, IS **newsubsets, Vec **newsubvecs)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (rangeIdx == 0) PetscFunctionReturn(0);
  if (rangeIdx < numISs - 1) {
    ierr = PetscFree(*newsubsets);CHKERRQ(ierr);
  }
  if (rangeIdx < numVecs - 1) {
    ierr = PetscFree(*newsubvecs);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnVecsPushVec(PetscInt numVecs, const Vec *vecs, Vec newvec, Vec **newvecs)
{
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc1(numVecs + 1, newvecs);CHKERRQ(ierr);
  for (i = 0; i < numVecs; i++) (*newvecs)[i] = vecs[i];
  (*newvecs)[numVecs] = newvec;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnISsPushIS(PetscInt numISs, const IS *subsets, IS newis, IS **newiss)
{
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (subsets == NULL && newis == NULL) {
    *newiss = NULL;
  } else {
    ierr = PetscMalloc1(numISs+1, newiss);CHKERRQ(ierr);
    for (i = 0; i < numISs; i++) (*newiss)[i] = subsets[i];
    (*newiss)[numISs] = newis;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnISVecsPushISVec(PetscInt numISs, const IS *subsets, PetscInt numVecs, const Vec *vecs, IS newis, Vec newvec, IS **newiss, Vec **newvecs)
{
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (subsets == NULL && newis == NULL) {
    *newiss = NULL;
  } else {
    ierr = PetscMalloc1(numISs+1, newiss);CHKERRQ(ierr);
    for (i = 0; i < numISs; i++) (*newiss)[i] = subsets[i];
    (*newiss)[numISs] = newis;
  }
  ierr = PetscMalloc1(numVecs+1, newvecs);CHKERRQ(ierr);
  for (i = 0; i < numVecs; i++) (*newvecs)[i] = vecs[i];
  (*newvecs)[numVecs] = newvec;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnISVecsPushFrontISVec(PetscInt numISs, const IS *subsets, PetscInt numVecs, const Vec *vecs, IS newis, Vec newvec, IS **newiss, Vec **newvecs)
{
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (subsets == NULL && newis == NULL) {
    *newiss = NULL;
  } else {
    ierr = PetscMalloc1(numISs+1, newiss);CHKERRQ(ierr);
    for (i = 0; i < numISs; i++) (*newiss)[i+1] = subsets[i];
    (*newiss)[0] = newis;
  }
  ierr = PetscMalloc1(numVecs+1, newvecs);CHKERRQ(ierr);
  for (i = 0; i < numVecs; i++) (*newvecs)[i+1] = vecs[i];
  (*newvecs)[0] = newvec;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnISsGetConcat(PetscInt numISs1, const IS *subsets1, PetscInt numISs2, const IS *subsets2, IS **newISs)
{
  PetscInt       i;
  IS             *ISs;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!subsets1 && !subsets2) {
    *newISs = NULL;
  } else if (!numISs1) {
    *newISs = (IS *) subsets2;
  } else if (!numISs2) {
    *newISs = (IS *) subsets1;
  } else {
    ierr = PetscMalloc1(numISs1 + numISs2, &ISs);CHKERRQ(ierr);
    for (i = 0; i < numISs1; i++) ISs[i] = subsets1[i];
    for (i = 0; i < numISs2; i++) ISs[i+numISs1] = subsets2[i];
    *newISs = ISs;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnISsRestoreConcat(PetscInt numISs1, const IS *subsets1, PetscInt numISs2, const IS *subsets2, IS **newISs)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (numISs1 && numISs2 && (subsets1 || subsets2)) {
    ierr = PetscFree(*newISs);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnVecsGetConcat(PetscInt numVecs1, const Vec *subsets1, PetscInt numVecs2, const Vec *subsets2, Vec **newVecs)
{
  PetscInt       i;
  Vec            *Vecs;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!subsets1 && !subsets2) {
    *newVecs = NULL;
  } else if (!numVecs1) {
    *newVecs = (Vec *) subsets2;
  } else if (!numVecs2) {
    *newVecs = (Vec *) subsets1;
  } else {
    ierr = PetscMalloc1(numVecs1 + numVecs2, &Vecs);CHKERRQ(ierr);
    for (i = 0; i < numVecs1; i++) Vecs[i] = subsets1[i];
    for (i = 0; i < numVecs2; i++) Vecs[i+numVecs1] = subsets2[i];
    *newVecs = Vecs;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnVecsRestoreConcat(PetscInt numVecs1, const Vec *subsets1, PetscInt numVecs2, const Vec *subsets2, Vec **newVecs)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (numVecs1 && numVecs2 && (subsets1 || subsets2)) {
    ierr = PetscFree(*newVecs);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnDerivativeScalar(PetscFn fn, Vec x, PetscInt der, PetscInt rangeIdx, const IS subsets[], const Vec subvecs[], PetscScalar *z)
{
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn,PETSCFN_CLASSID,1);
  PetscValidType(fn,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  PetscValidLogicalCollectiveInt(fn,der,3);
  PetscValidLogicalCollectiveInt(fn,rangeIdx,4);
  ierr = PetscFnSetUp(fn);CHKERRQ(ierr);
  if (der < 0) SETERRQ(PetscObjectComm((PetscObject)fn),PETSC_ERR_ARG_OUTOFRANGE,"derivative must be non-negative");
  if (rangeIdx < 0 || rangeIdx > der) SETERRQ2(PetscObjectComm((PetscObject)fn),PETSC_ERR_ARG_OUTOFRANGE,"range vector index %D not in range [0,%D]",rangeIdx,der);
  /* there are der+1 subsets and der+1 subvecs */
#if defined(PETSC_USE_DEBUG)
  for (i = 0; i < der + 1; i++) {ierr = PetscFnVecCheckCompatible(subvecs[i], subsets ? subsets[i] : NULL, (i == rangeIdx) ? fn->rmap : fn->dmap);CHKERRQ(ierr);}
#endif
  ierr = VecLockPush(x);CHKERRQ(ierr);
  for (i = 0; i < der+1; i++) {
    ierr = VecLockPush(subvecs[i]);CHKERRQ(ierr);
  }
  if (fn->ops->derivativescalar) {
    ierr = (*fn->ops->derivativescalar)(fn,x,der,rangeIdx,subsets,subvecs,z);CHKERRQ(ierr);
  } else if (fn->isScalar && fn->ops->scalarderivativescalar) {
    /* for a functional derivative to produce a scalar, it must contract against der vectors */
    PetscScalar  s;
    IS          *scalarsubsets = NULL;
    Vec         *scalarvecs;

    ierr = PetscFnGetISVecsWithoutRange(rangeIdx, der+1, subsets, der+1, subvecs, &scalarsubsets, &scalarvecs);CHKERRQ(ierr);
    ierr = (*fn->ops->scalarderivativescalar) (fn,x,der,scalarsubsets,scalarvecs,z);CHKERRQ(ierr);
    ierr = VecScalarBcast(subvecs[rangeIdx], &s);CHKERRQ(ierr);
    *z *= s;
    ierr = PetscFnRestoreISVecsWithoutRange(rangeIdx, der+1, subsets, der+1, subvecs, &scalarsubsets, &scalarvecs);CHKERRQ(ierr);
  } else if (fn->ops->derivativevec) {
    /* compute by getting a vector output and dotting with one of the input vectors.
     * always choose a domainvec for the output vector if possible */
    Vec          outVec;

    ierr = VecDuplicate(subvecs[der], &outVec);CHKERRQ(ierr);
    ierr = (*fn->ops->derivativevec)(fn, x, der, rangeIdx, subsets, subvecs, outVec);CHKERRQ(ierr);
    ierr = VecDot(subvecs[der], outVec, z);CHKERRQ(ierr);
    ierr = VecDestroy(&outVec);CHKERRQ(ierr);
  } else if (fn->isScalar && der > 0 && fn->ops->scalarderivativevec) {
    Vec *vecsubvecs;
    IS  *vecsubsets = NULL;
    PetscScalar s;
    Vec  outVec;

    ierr = PetscFnGetISVecsWithoutRange(rangeIdx, der+1, subsets, der+1, subvecs, &vecsubsets, &vecsubvecs);CHKERRQ(ierr);
    ierr = VecDuplicate(vecsubvecs[der-1],&outVec);CHKERRQ(ierr);
    ierr = (*fn->ops->scalarderivativevec)(fn,x,der,vecsubsets,vecsubvecs,outVec);CHKERRQ(ierr);
    ierr = VecDot(vecsubvecs[der-1],outVec,z);CHKERRQ(ierr);
    ierr = VecScalarBcast(subvecs[rangeIdx], &s);CHKERRQ(ierr);
    *z *= s;
    ierr = PetscFnRestoreISVecsWithoutRange(rangeIdx, der+1, subsets, der+1, subvecs, &vecsubsets, &vecsubvecs);CHKERRQ(ierr);
  } else SETERRQ1(PetscObjectComm((PetscObject)fn), PETSC_ERR_SUP, "This PetscFn does not implement %s()", PETSC_FUNCTION_NAME);
  for (i = 0; i < der+1; i++) {
    ierr = VecLockPop(subvecs[i]);CHKERRQ(ierr);
  }
  ierr = VecLockPop(x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnDerivativeVec(PetscFn fn, Vec x, PetscInt der, PetscInt rangeIdx, const IS subsets[], const Vec subvecs[], Vec y)
{
  PetscInt       i;
  IS             yIS;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn,PETSCFN_CLASSID,1);
  PetscValidType(fn,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  PetscValidLogicalCollectiveInt(fn,der,3);
  PetscValidLogicalCollectiveInt(fn,rangeIdx,4);
  ierr = PetscFnSetUp(fn);CHKERRQ(ierr);
  if (der < 0) SETERRQ(PetscObjectComm((PetscObject)fn),PETSC_ERR_ARG_OUTOFRANGE,"derivative must be non-negative");
  if (rangeIdx < 0 || rangeIdx > der) SETERRQ2(PetscObjectComm((PetscObject)fn),PETSC_ERR_ARG_OUTOFRANGE,"range vector index %D not in range [0,%D]",rangeIdx,der);
  yIS = subsets ? subsets[der] : NULL;
  /* there are der+1 subsets, der input vecs and 1 output vec */
#if defined(PETSC_USE_DEBUG)
  for (i = 0; i < der; i++) {ierr = PetscFnVecCheckCompatible(subvecs[i], subsets ? subsets[i] : NULL, (i == rangeIdx) ? fn->rmap : fn->dmap);CHKERRQ(ierr);}
  ierr = PetscFnVecCheckCompatible(y, yIS, (der == rangeIdx) ? fn->rmap : fn->dmap);CHKERRQ(ierr);
#endif
  ierr = VecLockPush(x);CHKERRQ(ierr);
  for (i = 0; i < der; i++) {
    ierr = VecLockPush(subvecs[i]);CHKERRQ(ierr);
  }
  if (fn->ops->derivativevec) {
    ierr = (*fn->ops->derivativevec)(fn,x,der,rangeIdx,subsets,subvecs,y);CHKERRQ(ierr);
  } else if (fn->isScalar && rangeIdx < der && fn->ops->scalarderivativevec) {
    /* for a scalar functional derivative to produce a vector, it must contract against der-1 vectors */
    IS          *scalarsubsets = NULL;
    Vec         *scalarvecs = NULL;
    PetscScalar z;

    ierr = PetscFnGetISVecsWithoutRange(rangeIdx, der+1, subsets, der+1, subvecs, &scalarsubsets, &scalarvecs);CHKERRQ(ierr);
    ierr = (*fn->ops->scalarderivativevec)(fn,x,der,scalarsubsets,scalarvecs,y);CHKERRQ(ierr);
    ierr = VecScalarBcast(subvecs[rangeIdx], &z);CHKERRQ(ierr);
    ierr = VecScale(y, z);CHKERRQ(ierr);
    ierr = PetscFnRestoreISVecsWithoutRange(rangeIdx, der+1, subsets, der+1, subvecs, &scalarsubsets, &scalarvecs);CHKERRQ(ierr);
  } else if (fn->isScalar && rangeIdx == der && fn->ops->derivativescalar) {
    Vec         onevec;
    Vec         *newvecs;
    PetscScalar z;

    ierr = VecDuplicate(y, &onevec);CHKERRQ(ierr);
    ierr = VecSet(onevec, 1.);CHKERRQ(ierr);
    ierr = PetscFnVecsPushVec(der, subvecs, onevec, &newvecs);CHKERRQ(ierr);
    ierr = (*fn->ops->derivativescalar)(fn, x, der, rangeIdx, subsets, newvecs, &z);CHKERRQ(ierr);
    ierr = PetscFree(newvecs);CHKERRQ(ierr);
    ierr = VecDestroy(&onevec);CHKERRQ(ierr);
    ierr = VecSet(y,z);CHKERRQ(ierr);
  } else if (fn->isScalar && rangeIdx == der && fn->ops->scalarderivativescalar) {
    PetscScalar z;

    ierr = (*fn->ops->scalarderivativescalar)(fn, x, der, subsets, subvecs, &z);CHKERRQ(ierr);
    ierr = VecSet(y,z);CHKERRQ(ierr);
  } else SETERRQ1(PetscObjectComm((PetscObject)fn), PETSC_ERR_SUP, "This PetscFn does not implement %s()", PETSC_FUNCTION_NAME);
  for (i = 0; i < der; i++) {
    ierr = VecLockPop(subvecs[i]);CHKERRQ(ierr);
  }
  ierr = VecLockPop(x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnDerivativeMat(PetscFn fn, Vec x, PetscInt der, PetscInt rangeIdx, const IS subsets[], const Vec subvecs[], MatReuse reuse, Mat *M, Mat *Mpre)
{
  PetscInt       i;
  IS             rightIS, leftIS, rangeIS;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn,PETSCFN_CLASSID,1);
  PetscValidType(fn,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  PetscValidLogicalCollectiveInt(fn,der,3);
  PetscValidLogicalCollectiveInt(fn,rangeIdx,4);
  ierr = PetscFnSetUp(fn);CHKERRQ(ierr);
  if (der < 1) SETERRQ(PetscObjectComm((PetscObject)fn),PETSC_ERR_ARG_OUTOFRANGE,"derivative must be positive");
  if (rangeIdx < 0 || rangeIdx > der) SETERRQ2(PetscObjectComm((PetscObject)fn),PETSC_ERR_ARG_OUTOFRANGE,"range vector index %D not in range [0,%D]",rangeIdx,der);
  rightIS = subsets ? subsets[der-1]    : NULL;
  leftIS  = subsets ? subsets[der]      : NULL;
  rangeIS = subsets ? subsets[rangeIdx] : NULL;
  /* there are der+1 subsets, der-1 input vecs and 1 output mat */
#if defined(PETSC_USE_DEBUG)
  for (i = 0; i < der; i++) {ierr = PetscFnVecCheckCompatible(subvecs[i], subsets ? subsets[i] : NULL, (i == rangeIdx) ? fn->rmap : fn->dmap);CHKERRQ(ierr);}
  if (reuse == MAT_REUSE_MATRIX) {
    PetscLayout    rightLayout = (der-1 == rangeIdx) ? fn->rmap : fn->dmap;
    PetscLayout    leftLayout = (der == rangeIdx) ? fn->rmap : fn->dmap;

    if (M) {ierr = PetscFnMatCheckCompatible(*M, rightIS, leftIS, rightLayout, leftLayout);CHKERRQ(ierr);}
    if (Mpre) {ierr = PetscFnMatCheckCompatible(*Mpre, rightIS, leftIS, rightLayout, leftLayout);CHKERRQ(ierr);}
  }
#endif
  if (!M && !Mpre) PetscFunctionReturn(0);
  ierr = VecLockPush(x);CHKERRQ(ierr);
  for (i = 0; i < der - 1; i++) {
    ierr = VecLockPush(subvecs[i]);CHKERRQ(ierr);
  }
  if (fn->ops->derivativemat) {
    ierr = (*fn->ops->derivativemat)(fn,x,der,rangeIdx,subsets,subvecs,reuse,M,Mpre);CHKERRQ(ierr);
  } else if (fn->isScalar && rangeIdx >= der - 1 && fn->ops->derivativevec) {
    IS          *subsets = NULL;
    Vec         *newvecs = NULL;
    Vec         onevec;
    Vec         grad;
    Mat         *mat = M ? M : Mpre;
    PetscBool   colVec = (der == rangeIdx) ? PETSC_FALSE: PETSC_TRUE;

    ierr = PetscFnCreateVecs(fn, (der == rangeIdx) ? rightIS : leftIS, &grad, rangeIS, &onevec);CHKERRQ(ierr);
    ierr = VecSet(onevec, 1.);CHKERRQ(ierr);
    ierr = PetscFnVecsPushVec(der-1, subvecs, onevec, &newvecs);CHKERRQ(ierr);
    ierr = (*fn->ops->derivativevec)(fn,x,der,rangeIdx,subsets,newvecs,grad);CHKERRQ(ierr);
    if (reuse == MAT_INITIAL_MATRIX) {
      ierr = MatCreateDenseVecs(PetscObjectComm((PetscObject)grad),1,&grad,colVec,mat);CHKERRQ(ierr);
    } else {
      ierr = MatSetValuesVec(*mat, grad, 0, colVec, INSERT_VALUES);CHKERRQ(ierr);
    }
    if (Mpre && Mpre != mat) {
      ierr = MatDuplicateOrCopy(*mat, reuse, Mpre);CHKERRQ(ierr);
    }
    ierr = PetscFree(newvecs);CHKERRQ(ierr);
    ierr = VecDestroy(&onevec);CHKERRQ(ierr);
    ierr = VecDestroy(&grad);CHKERRQ(ierr);
  } else if (fn->isScalar && rangeIdx >= der - 1 && fn->ops->scalarderivativevec) {
    /* for a scalar functional derivative to produce a vector, it must contract against der-1 vectors */
    IS          *scalarsubsets = NULL;
    Vec         *scalarvecs = NULL;
    Vec         grad;
    Mat         *mat = M ? M : Mpre;
    PetscBool   colVec = (der == rangeIdx) ? PETSC_FALSE: PETSC_TRUE;

    ierr = PetscFnGetISVecsWithoutRange(rangeIdx, der+1, subsets, der-1, subvecs, &scalarsubsets, &scalarvecs);CHKERRQ(ierr);
    ierr = PetscFnCreateVecs(fn, (der == rangeIdx) ? rightIS : leftIS, &grad, NULL, NULL);CHKERRQ(ierr);
    ierr = (*fn->ops->scalarderivativevec)(fn,x,der,scalarsubsets,scalarvecs,grad);CHKERRQ(ierr);
    if (reuse == MAT_INITIAL_MATRIX) {
      ierr = MatCreateDenseVecs(PetscObjectComm((PetscObject)grad),1,&grad,colVec,mat);CHKERRQ(ierr);
    } else {
      ierr = MatSetValuesVec(*mat, grad, 0, colVec, INSERT_VALUES);CHKERRQ(ierr);
    }
    if (Mpre && Mpre != mat) {
      ierr = MatDuplicateOrCopy(*mat, reuse, Mpre);CHKERRQ(ierr);
    }
    ierr = VecDestroy(&grad);CHKERRQ(ierr);
    ierr = PetscFnRestoreISVecsWithoutRange(rangeIdx, der+1, subsets, der-1, subvecs, &scalarsubsets, &scalarvecs);CHKERRQ(ierr);
  } else if (fn->isScalar && rangeIdx < der - 1 && fn->ops->scalarderivativemat) {
    /* for a scalar functional derivative to produce a matrix, it must contract against der-2 vectors */
    IS          *scalarsubsets = NULL;
    Vec         *scalarvecs = NULL;
    PetscScalar z;

    ierr = PetscFnGetISVecsWithoutRange(rangeIdx, der+1, subsets, der-1, subvecs, &scalarsubsets, &scalarvecs);CHKERRQ(ierr);
    ierr = (*fn->ops->scalarderivativemat)(fn,x,der,scalarsubsets,scalarvecs,reuse,M,Mpre);CHKERRQ(ierr);
    ierr = VecScalarBcast(subvecs[rangeIdx], &z);CHKERRQ(ierr);
    if (M) {
      ierr = MatScale(*M, z);CHKERRQ(ierr);
    }
    if (Mpre && Mpre != M) {
      ierr = MatScale(*Mpre, z);CHKERRQ(ierr);
    }
    ierr = PetscFnRestoreISVecsWithoutRange(rangeIdx, der+1, subsets, der-1, subvecs, &scalarsubsets, &scalarvecs);CHKERRQ(ierr);
  } else SETERRQ1(PetscObjectComm((PetscObject)fn), PETSC_ERR_SUP, "This PetscFn does not implement %s()", PETSC_FUNCTION_NAME);
  for (i = 0; i < der-1; i++) {
    ierr = VecLockPop(subvecs[i]);CHKERRQ(ierr);
  }
  ierr = VecLockPop(x);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
  if (reuse == MAT_INITIAL_MATRIX) {
    PetscLayout    rightLayout = (der-1 == rangeIdx) ? fn->rmap : fn->dmap;
    PetscLayout    leftLayout = (der == rangeIdx) ? fn->rmap : fn->dmap;

    if (M) {ierr = PetscFnMatCheckCompatible(*M, rightIS, leftIS, rightLayout, leftLayout);CHKERRQ(ierr);}
    if (Mpre) {ierr = PetscFnMatCheckCompatible(*Mpre, rightIS, leftIS, rightLayout, leftLayout);CHKERRQ(ierr);}
  }
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnScalarDerivativeScalar(PetscFn fn, Vec x, PetscInt der, const IS subsets[], const Vec subvecs[], PetscScalar *z)
{
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn,PETSCFN_CLASSID,1);
  PetscValidType(fn,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  PetscValidLogicalCollectiveInt(fn,der,3);
  ierr = PetscFnSetUp(fn);CHKERRQ(ierr);
  if (!(fn->isScalar)) SETERRQ(PetscObjectComm((PetscObject)fn),PETSC_ERR_ARG_SIZ, "PetscFn is not a scalar function");
  if (der < 0) SETERRQ(PetscObjectComm((PetscObject)fn),PETSC_ERR_ARG_OUTOFRANGE,"derivative must be non-negative");
  /* there are der subsets and der subvecs */
#if defined(PETSC_USE_DEBUG)
  for (i = 0; i < der; i++) {ierr = PetscFnVecCheckCompatible(subvecs[i], subsets ? subsets[i] : NULL, fn->dmap);CHKERRQ(ierr);}
#endif
  ierr = VecLockPush(x);CHKERRQ(ierr);
  for (i = 0; i < der; i++) {
    ierr = VecLockPush(subvecs[i]);CHKERRQ(ierr);
  }
  if (fn->ops->scalarderivativescalar) {
    ierr = (*fn->ops->scalarderivativescalar)(fn,x,der,subsets,subvecs,z);CHKERRQ(ierr);
  } else if (fn->ops->derivativescalar) {
    IS          *vecsubsets = NULL;
    Vec         *vecsubvecs = NULL;
    Vec         onevec;

    ierr = PetscFnCreateVecs(fn, NULL, NULL, NULL, &onevec);CHKERRQ(ierr);
    ierr = VecSet(onevec, 1.);CHKERRQ(ierr);
    ierr = PetscFnISVecsPushISVec(der, subsets, der, subvecs, NULL, onevec, &vecsubsets, &vecsubvecs);CHKERRQ(ierr);
    ierr = (*fn->ops->derivativescalar)(fn,x,der,der,vecsubsets,vecsubvecs,z);CHKERRQ(ierr);
    ierr = PetscFree(vecsubvecs);CHKERRQ(ierr);
    ierr = PetscFree(vecsubsets);CHKERRQ(ierr);
    ierr = VecDestroy(&onevec);CHKERRQ(ierr);
  } else if (der > 0 && fn->ops->scalarderivativevec) {
    Vec         grad;

    ierr = VecDuplicate(subvecs[der], &grad);CHKERRQ(ierr);
    ierr = (*fn->ops->scalarderivativevec)(fn,x,der,subsets,subvecs,grad);CHKERRQ(ierr);
    ierr = VecDot(subvecs[der], grad, z);CHKERRQ(ierr);
    ierr = VecDestroy(&grad);CHKERRQ(ierr);
  } else if (fn->ops->derivativevec) {
    IS          *vecsubsets = NULL;
    Vec         grad;

    ierr = PetscFnCreateVecs(fn, NULL, &grad, NULL, NULL);CHKERRQ(ierr);
    ierr = PetscFnISsPushIS(der, subsets, NULL, &vecsubsets);CHKERRQ(ierr);
    ierr = (*fn->ops->derivativevec)(fn,x,der,der,vecsubsets,subvecs,grad);CHKERRQ(ierr);
    ierr = VecScalarBcast(grad, z);CHKERRQ(ierr);
    ierr = PetscFree(vecsubsets);CHKERRQ(ierr);
    ierr = VecDestroy(&grad);CHKERRQ(ierr);
  } else SETERRQ1(PetscObjectComm((PetscObject)fn), PETSC_ERR_SUP, "This PetscFn does not implement %s()", PETSC_FUNCTION_NAME);
  for (i = 0; i < der; i++) {
    ierr = VecLockPop(subvecs[i]);CHKERRQ(ierr);
  }
  ierr = VecLockPop(x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnScalarDerivativeVec(PetscFn fn, Vec x, PetscInt der, const IS subsets[], const Vec subvecs[], Vec y)
{
  PetscInt       i;
  IS             yIS;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn,PETSCFN_CLASSID,1);
  PetscValidType(fn,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  PetscValidLogicalCollectiveInt(fn,der,3);
  ierr = PetscFnSetUp(fn);CHKERRQ(ierr);
  if (!(fn->isScalar)) SETERRQ(PetscObjectComm((PetscObject)fn),PETSC_ERR_ARG_SIZ, "PetscFn is not a scalar function");
  if (der < 1) SETERRQ(PetscObjectComm((PetscObject)fn),PETSC_ERR_ARG_OUTOFRANGE,"derivative must be non-negative");
  yIS = subsets ? subsets[der-1] : NULL;
  /* there are der subsets, der-1 input vecs and 1 output vec */
#if defined(PETSC_USE_DEBUG)
  for (i = 0; i < der-1; i++) {ierr = PetscFnVecCheckCompatible(subvecs[i], subsets ? subsets[i] : NULL, fn->dmap);CHKERRQ(ierr);}
  ierr = PetscFnVecCheckCompatible(y, yIS, fn->dmap);CHKERRQ(ierr);
#endif
  ierr = VecLockPush(x);CHKERRQ(ierr);
  for (i = 0; i < der-1; i++) {
    ierr = VecLockPush(subvecs[i]);CHKERRQ(ierr);
  }
  if (fn->ops->scalarderivativevec) {
    ierr = (*fn->ops->scalarderivativevec)(fn,x,der,subsets,subvecs,y);CHKERRQ(ierr);
  } else if (fn->ops->derivativevec) {
    IS          *vecsubsets = NULL;
    Vec         *vecsubvecs = NULL;
    Vec         onevec;

    ierr = PetscFnCreateVecs(fn, NULL, NULL, NULL, &onevec);CHKERRQ(ierr);
    ierr = VecSet(onevec, 1.);CHKERRQ(ierr);
    ierr = PetscFnISVecsPushFrontISVec(der, subsets, der-1, subvecs, NULL, onevec, &vecsubsets, &vecsubvecs);CHKERRQ(ierr);
    ierr = (*fn->ops->derivativevec)(fn,x,der,0,vecsubsets,vecsubvecs,y);CHKERRQ(ierr);
    ierr = PetscFree(vecsubvecs);CHKERRQ(ierr);
    ierr = PetscFree(vecsubsets);CHKERRQ(ierr);
    ierr = VecDestroy(&onevec);CHKERRQ(ierr);
  } else SETERRQ1(PetscObjectComm((PetscObject)fn), PETSC_ERR_SUP, "This PetscFn does not implement %s()", PETSC_FUNCTION_NAME);
  for (i = 0; i < der; i++) {
    ierr = VecLockPop(subvecs[i]);CHKERRQ(ierr);
  }
  ierr = VecLockPop(x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnScalarDerivativeMat(PetscFn fn, Vec x, PetscInt der, const IS subsets[], const Vec subvecs[], MatReuse reuse, Mat *M, Mat *Mpre)
{
  PetscInt       i;
  IS             rightIS, leftIS;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn,PETSCFN_CLASSID,1);
  PetscValidType(fn,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  PetscValidLogicalCollectiveInt(fn,der,3);
  ierr = PetscFnSetUp(fn);CHKERRQ(ierr);
  if (!(fn->isScalar)) SETERRQ(PetscObjectComm((PetscObject)fn),PETSC_ERR_ARG_SIZ, "PetscFn is not a scalar function");
  rightIS = subsets ? subsets[der-2] : NULL;
  leftIS  = subsets ? subsets[der-1] : NULL;
  /* there are der subsets, der-2 input vecs and 1 output mat */
#if defined(PETSC_USE_DEBUG)
  for (i = 0; i < der-2; i++) {ierr = PetscFnVecCheckCompatible(subvecs[i], subsets ? subsets[i] : NULL, fn->dmap);CHKERRQ(ierr);}
  if (reuse == MAT_REUSE_MATRIX) {
    PetscLayout    rightLayout = fn->dmap;
    PetscLayout    leftLayout = fn->dmap;

    if (M) {ierr = PetscFnMatCheckCompatible(*M, rightIS, leftIS, rightLayout, leftLayout);CHKERRQ(ierr);}
    if (Mpre) {ierr = PetscFnMatCheckCompatible(*Mpre, rightIS, leftIS, rightLayout, leftLayout);CHKERRQ(ierr);}
  }
#endif
  if (!M && !Mpre) PetscFunctionReturn(0);
  ierr = VecLockPush(x);CHKERRQ(ierr);
  for (i = 0; i < der - 2; i++) {
    ierr = VecLockPush(subvecs[i]);CHKERRQ(ierr);
  }
  if (fn->ops->scalarderivativemat) {
    ierr = (*fn->ops->scalarderivativemat)(fn,x,der,subsets,subvecs,reuse,M,Mpre);CHKERRQ(ierr);
  } else if (fn->ops->derivativemat) {
    IS          *vecsubsets = NULL;
    Vec         *vecsubvecs = NULL;
    Vec         onevec;

    ierr = PetscFnCreateVecs(fn, NULL, NULL, NULL, &onevec);CHKERRQ(ierr);
    ierr = VecSet(onevec, 1.);CHKERRQ(ierr);
    ierr = PetscFnISVecsPushFrontISVec(der, subsets, der-2, subvecs, NULL, onevec, &vecsubsets, &vecsubvecs);CHKERRQ(ierr);
    ierr = (*fn->ops->derivativemat)(fn,x,der,0,vecsubsets,vecsubvecs,reuse,M,Mpre);CHKERRQ(ierr);
  } else SETERRQ1(PetscObjectComm((PetscObject)fn), PETSC_ERR_SUP, "This PetscFn does not implement %s()", PETSC_FUNCTION_NAME);
  for (i = 0; i < der-1; i++) {
    ierr = VecLockPop(subvecs[i]);CHKERRQ(ierr);
  }
  ierr = VecLockPop(x);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
  if (reuse == MAT_INITIAL_MATRIX) {
    PetscLayout    rightLayout = fn->dmap;
    PetscLayout    leftLayout = fn->dmap;

    if (M) {ierr = PetscFnMatCheckCompatible(*M, rightIS, leftIS, rightLayout, leftLayout);CHKERRQ(ierr);}
    if (Mpre) {ierr = PetscFnMatCheckCompatible(*Mpre, rightIS, leftIS, rightLayout, leftLayout);CHKERRQ(ierr);}
  }
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnJacobianMult(PetscFn fn, Vec x, Vec xhat, Vec Jxhat)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnDerivativeVec(fn, x, 1, 1, NULL, &xhat, Jxhat);CHKERRQ(ierr);
  if (fn->test_jacmult) {
    PetscReal rate;

    fn->test_jacmult = PETSC_FALSE;
    ierr = PetscFnTestDerivativeMult(fn,PETSCFNOP_JACOBIANMULT,x,xhat,NULL,NULL,PETSC_DEFAULT,PETSC_DEFAULT,&rate);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnJacobianMultAdjoint(PetscFn fn, Vec x, Vec v, Vec Jadjv)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn,PETSCFN_CLASSID,1);
  PetscValidType(fn,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  PetscValidHeaderSpecific(v,VEC_CLASSID,3);
  PetscValidHeaderSpecific(Jadjv,VEC_CLASSID,4);
  ierr = PetscFnSetUp(fn);CHKERRQ(ierr);
  if (x == v) SETERRQ(PetscObjectComm((PetscObject)fn),PETSC_ERR_ARG_WRONGSTATE,"x and v must be different vectors");
  if (x == Jadjv) SETERRQ(PetscObjectComm((PetscObject)fn),PETSC_ERR_ARG_WRONGSTATE,"x and Jadjv must be different vectors");
  if (v == Jadjv) SETERRQ(PetscObjectComm((PetscObject)fn),PETSC_ERR_ARG_WRONGSTATE,"v and Jadjv must be different vectors");
  if (fn->dmap->N != x->map->N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"PetscFn fn,Vec x: global dim %D %D",fn->dmap->N,x->map->N);
  if (fn->rmap->N != v->map->N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"PetscFn fn,Vec v: global dim %D %D",fn->rmap->N,v->map->N);
  if (fn->dmap->N != Jadjv->map->N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"PetscFn fn,Vec Jadjv: global dim %D %D",fn->dmap->N,Jadjv->map->N);
  if (fn->dmap->n != Jadjv->map->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"PetscFn fn,Vec Jadjv: local dim %D %D",fn->dmap->n,Jadjv->map->n);
  VecLocked(Jadjv,4);

  ierr = VecLockPush(x);CHKERRQ(ierr);
  ierr = VecLockPush(v);CHKERRQ(ierr);
  if (fn->ops->jacobianmultadjoint) {
    ierr = (*fn->ops->jacobianmultadjoint)(fn,x,v,Jadjv);CHKERRQ(ierr);
  } else if (fn->isScalar && fn->ops->scalargradient) {
    PetscScalar z;

    ierr = (*fn->ops->scalargradient) (fn, x, Jadjv); CHKERRQ(ierr);
    ierr = VecScalarBcast(v, &z);CHKERRQ(ierr);
    ierr = VecScale(Jadjv, z);CHKERRQ(ierr);
  } else SETERRQ1(PetscObjectComm((PetscObject)fn), PETSC_ERR_SUP, "This PetscFn does not implement %s()", PETSC_FUNCTION_NAME);
  ierr = VecLockPop(v);CHKERRQ(ierr);
  ierr = VecLockPop(x);CHKERRQ(ierr);
  if (fn->test_jacmultadj) {
    PetscReal rate;

    fn->test_jacmultadj = PETSC_FALSE;
    ierr = PetscFnTestDerivativeMult(fn,PETSCFNOP_JACOBIANMULTADJOINT,x,NULL,v,NULL,PETSC_DEFAULT,PETSC_DEFAULT,&rate);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnVecToMat(Vec g, PetscBool colVec, MatReuse reuse, Mat *A, Mat *Apre)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (reuse == MAT_REUSE_MATRIX) {
    if (A) {
      ierr = MatSetValuesVec(*A, g, 0, colVec, INSERT_VALUES);CHKERRQ(ierr);
      ierr = MatAssemblyBegin(*A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyEnd(*A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    }
    if (Apre && Apre != A) {
      ierr = MatSetValuesVec(*A, g, 0, colVec, INSERT_VALUES);CHKERRQ(ierr);
      ierr = MatAssemblyBegin(*Apre,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyEnd(*Apre,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    }
  } else {
    if (A) {
      ierr = MatCreateDenseVecs(PetscObjectComm((PetscObject)g), 1, &g, colVec, A);CHKERRQ(ierr);
    }
    if (Apre) {
      ierr = MatCreateDenseVecs(PetscObjectComm((PetscObject)g), 1, &g, colVec, Apre);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnJacobianBuild(PetscFn fn, Vec x, MatReuse reuse, Mat *J, Mat *Jpre)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn,PETSCFN_CLASSID,1);
  PetscValidType(fn,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  ierr = PetscFnSetUp(fn);CHKERRQ(ierr);
  if (!J && !Jpre) PetscFunctionReturn(0);
  ierr = VecLockPush(x);CHKERRQ(ierr);
  if (fn->ops->jacobianbuild) {
    ierr = (*fn->ops->jacobianbuild)(fn,x,reuse,J,Jpre);CHKERRQ(ierr);
  } else if (fn->isScalar && fn->ops->scalargradient) {
    Vec                g;

    ierr = VecDuplicate(x, &g);CHKERRQ(ierr);
    ierr = (*fn->ops->scalargradient) (fn, x, g); CHKERRQ(ierr);
    ierr = PetscFnVecToMat(g, PETSC_FALSE, reuse, J, Jpre);CHKERRQ(ierr);
    ierr = VecDestroy(&g);CHKERRQ(ierr);
  } else SETERRQ1(PetscObjectComm((PetscObject)fn), PETSC_ERR_SUP, "This PetscFn does not implement %s()", PETSC_FUNCTION_NAME);
  ierr = VecLockPop(x);CHKERRQ(ierr);
  if (fn->test_jacbuild) {
    PetscReal norm, err;

    fn->test_jacbuild = PETSC_FALSE;
    if (J) {ierr = PetscFnTestDerivativeBuild(fn,PETSCFNOP_JACOBIANBUILD,*J,x,NULL,NULL,NULL,&norm,&err);CHKERRQ(ierr);}
    if (Jpre) {ierr = PetscFnTestDerivativeBuild(fn,PETSCFNOP_JACOBIANBUILD,*Jpre,x,NULL,NULL,NULL,&norm,&err);CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnJacobianBuildAdjoint(PetscFn fn, Vec x, MatReuse reuse, Mat *Jadj, Mat *Jadjpre)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn,PETSCFN_CLASSID,1);
  PetscValidType(fn,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  ierr = PetscFnSetUp(fn);CHKERRQ(ierr);
  if (!Jadj && !Jadjpre) PetscFunctionReturn(0);
  ierr = VecLockPush(x);CHKERRQ(ierr);
  if (fn->ops->jacobianbuildadjoint) {
    ierr = (*fn->ops->jacobianbuildadjoint)(fn,x,reuse,Jadj,Jadjpre);CHKERRQ(ierr);
  } else if (fn->isScalar && fn->ops->scalargradient) {
    Vec                g;

    ierr = VecDuplicate(x, &g);CHKERRQ(ierr);
    ierr = (*fn->ops->scalargradient) (fn, x, g); CHKERRQ(ierr);
    ierr = PetscFnVecToMat(g, PETSC_TRUE, reuse, Jadj, Jadjpre);CHKERRQ(ierr);
    ierr = VecDestroy(&g);CHKERRQ(ierr);
  } else SETERRQ1(PetscObjectComm((PetscObject)fn), PETSC_ERR_SUP, "This PetscFn does not implement %s()", PETSC_FUNCTION_NAME);
  ierr = VecLockPop(x);CHKERRQ(ierr);
  if (fn->test_jacbuildadj) {
    PetscReal norm, err;

    fn->test_jacbuildadj = PETSC_FALSE;
    if (Jadj) {ierr = PetscFnTestDerivativeBuild(fn,PETSCFNOP_JACOBIANBUILDADJOINT,*Jadj,x,NULL,NULL,NULL,&norm,&err);CHKERRQ(ierr);}
    if (Jadjpre) {ierr = PetscFnTestDerivativeBuild(fn,PETSCFNOP_JACOBIANBUILDADJOINT,*Jadjpre,x,NULL,NULL,NULL,&norm,&err);CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnHessianMult(PetscFn fn, Vec x, Vec xhat, Vec xdot, Vec Hxhatxdot)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn,PETSCFN_CLASSID,1);
  PetscValidType(fn,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  PetscValidHeaderSpecific(xhat,VEC_CLASSID,3);
  PetscValidHeaderSpecific(xdot,VEC_CLASSID,4);
  PetscValidHeaderSpecific(Hxhatxdot,VEC_CLASSID,5);
  ierr = PetscFnSetUp(fn);CHKERRQ(ierr);
  if (x == Hxhatxdot) SETERRQ(PetscObjectComm((PetscObject)fn),PETSC_ERR_ARG_WRONGSTATE,"x and Hxhatxdot must be different vectors");
  if (x == Hxhatxdot) SETERRQ(PetscObjectComm((PetscObject)fn),PETSC_ERR_ARG_WRONGSTATE,"v and Hxhatxdot must be different vectors");
  if (xhat == Hxhatxdot) SETERRQ(PetscObjectComm((PetscObject)fn),PETSC_ERR_ARG_WRONGSTATE,"xhat and Hxhatxdot must be different vectors");
  if (fn->dmap->N != x->map->N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"PetscFn fn,Vec x: global dim %D %D",fn->dmap->N,x->map->N);
  if (fn->dmap->N != xhat->map->N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"PetscFn fn,Vec xhat: global dim %D %D",fn->dmap->N,xhat->map->N);
  if (fn->dmap->N != xdot->map->N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"PetscFn fn,Vec xdot: global dim %D %D",fn->dmap->N,xdot->map->N);
  if (fn->rmap->N != Hxhatxdot->map->N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"PetscFn fn,Vec Hxhatxdot: global dim %D %D",fn->rmap->N,Hxhatxdot->map->N);
  if (fn->rmap->n != Hxhatxdot->map->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"PetscFn fn,Vec Hxhatxdot: local dim %D %D",fn->rmap->n,Hxhatxdot->map->n);
  VecLocked(Hxhatxdot,5);

  ierr = VecLockPush(x);CHKERRQ(ierr);
  ierr = VecLockPush(xhat);CHKERRQ(ierr);
  ierr = VecLockPush(xdot);CHKERRQ(ierr);
  if (fn->ops->hessianmult) {
    ierr = (*fn->ops->hessianmult)(fn,x,xhat,xdot,Hxhatxdot);CHKERRQ(ierr);
  } else if (fn->isScalar && fn->ops->scalarhessianmult) {
    PetscScalar z;
    Vec         Hxhat;

    ierr = VecDuplicate(xhat, &Hxhat);CHKERRQ(ierr);
    ierr = (*fn->ops->scalarhessianmult) (fn, x, xhat, Hxhat); CHKERRQ(ierr);
    ierr = VecDot(Hxhat, xdot, &z);CHKERRQ(ierr);
    ierr = VecSet(Hxhatxdot, z);CHKERRQ(ierr);
    ierr = VecDestroy(&Hxhat);CHKERRQ(ierr);
  } else SETERRQ1(PetscObjectComm((PetscObject)fn), PETSC_ERR_SUP, "This PetscFn does not implement %s()", PETSC_FUNCTION_NAME);
  ierr = VecLockPop(xdot);CHKERRQ(ierr);
  ierr = VecLockPop(xhat);CHKERRQ(ierr);
  ierr = VecLockPop(x);CHKERRQ(ierr);
  if (fn->test_hesmult) {
    PetscReal rate;

    fn->test_hesmult = PETSC_FALSE;
    ierr = PetscFnTestDerivativeMult(fn,PETSCFNOP_HESSIANMULT,x,xhat,xdot,NULL,PETSC_DEFAULT,PETSC_DEFAULT,&rate);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnHessianMultAdjoint(PetscFn fn, Vec x, Vec v, Vec xhat, Vec Hadjvxhat)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn,PETSCFN_CLASSID,1);
  PetscValidType(fn,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  PetscValidHeaderSpecific(v,VEC_CLASSID,3);
  PetscValidHeaderSpecific(xhat,VEC_CLASSID,4);
  PetscValidHeaderSpecific(Hadjvxhat,VEC_CLASSID,4);
  ierr = PetscFnSetUp(fn);CHKERRQ(ierr);
  if (x == Hadjvxhat) SETERRQ(PetscObjectComm((PetscObject)fn),PETSC_ERR_ARG_WRONGSTATE,"x and Hadjvxhat must be different vectors");
  if (v == Hadjvxhat) SETERRQ(PetscObjectComm((PetscObject)fn),PETSC_ERR_ARG_WRONGSTATE,"v and Hadjvxhat must be different vectors");
  if (xhat == Hadjvxhat) SETERRQ(PetscObjectComm((PetscObject)fn),PETSC_ERR_ARG_WRONGSTATE,"xhat and Hadjvxhat must be different vectors");
  if (fn->dmap->N != x->map->N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"PetscFn fn,Vec x: global dim %D %D",fn->dmap->N,x->map->N);
  if (fn->rmap->N != v->map->N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"PetscFn fn,Vec v: global dim %D %D",fn->rmap->N,v->map->N);
  if (fn->dmap->N != xhat->map->N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"PetscFn fn,Vec xhat: global dim %D %D",fn->dmap->N,xhat->map->N);
  if (fn->dmap->N != Hadjvxhat->map->N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"PetscFn fn,Vec Hadjvxhat: global dim %D %D",fn->dmap->N,Hadjvxhat->map->N);
  if (fn->dmap->n != Hadjvxhat->map->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"PetscFn fn,Vec Hadjvxhat: local dim %D %D",fn->dmap->n,Hadjvxhat->map->n);
  VecLocked(Hadjvxhat,5);

  ierr = VecLockPush(x);CHKERRQ(ierr);
  ierr = VecLockPush(v);CHKERRQ(ierr);
  ierr = VecLockPush(xhat);CHKERRQ(ierr);
  if (fn->ops->hessianmultadjoint) {
    ierr = (*fn->ops->hessianmultadjoint)(fn,x,v,xhat,Hadjvxhat);CHKERRQ(ierr);
  } else if (fn->isScalar && fn->ops->scalarhessianmult) {
    PetscScalar z;

    ierr = (*fn->ops->scalarhessianmult) (fn, x, xhat, Hadjvxhat); CHKERRQ(ierr);
    ierr = VecScalarBcast(v, &z);CHKERRQ(ierr);
    ierr = VecScale(Hadjvxhat, z);CHKERRQ(ierr);
  } else SETERRQ1(PetscObjectComm((PetscObject)fn), PETSC_ERR_SUP, "This PetscFn does not implement %s()", PETSC_FUNCTION_NAME);
  ierr = VecLockPop(xhat);CHKERRQ(ierr);
  ierr = VecLockPop(v);CHKERRQ(ierr);
  ierr = VecLockPop(x);CHKERRQ(ierr);
  if (fn->test_hesmultadj) {
    PetscReal rate;

    fn->test_hesmultadj = PETSC_FALSE;
    ierr = PetscFnTestDerivativeMult(fn,PETSCFNOP_HESSIANMULTADJOINT,x,xhat,v,NULL,PETSC_DEFAULT,PETSC_DEFAULT,&rate);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnHessianBuild(PetscFn fn, Vec x, Vec xhat, MatReuse reuse, Mat *H, Mat *Hpre)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn,PETSCFN_CLASSID,1);
  PetscValidType(fn,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  PetscValidHeaderSpecific(xhat,VEC_CLASSID,3);
  ierr = PetscFnSetUp(fn);CHKERRQ(ierr);
  if (!H && !Hpre) PetscFunctionReturn(0);
  ierr = VecLockPush(x);CHKERRQ(ierr);
  ierr = VecLockPush(xhat);CHKERRQ(ierr);
  if (fn->ops->hessianbuild) {
    ierr = (*fn->ops->hessianbuild)(fn,x,xhat,reuse,H,Hpre);CHKERRQ(ierr);
  } else if (fn->isScalar && fn->ops->scalarhessianmult) {
    Vec                Hxhat;

    ierr = VecDuplicate(xhat, &Hxhat);CHKERRQ(ierr);
    ierr = (*fn->ops->scalarhessianmult) (fn, x, xhat, Hxhat);CHKERRQ(ierr);
    ierr = PetscFnVecToMat(Hxhat, PETSC_FALSE, reuse, H, Hpre);CHKERRQ(ierr);
    ierr = VecDestroy(&Hxhat);CHKERRQ(ierr);
  } else SETERRQ1(PetscObjectComm((PetscObject)fn), PETSC_ERR_SUP, "This PetscFn does not implement %s()", PETSC_FUNCTION_NAME);
  ierr = VecLockPop(xhat);CHKERRQ(ierr);
  ierr = VecLockPop(x);CHKERRQ(ierr);
  if (fn->test_hesbuild) {
    PetscReal norm, err;

    fn->test_hesbuild = PETSC_FALSE;
    if (H) {ierr = PetscFnTestDerivativeBuild(fn,PETSCFNOP_HESSIANBUILD,*H,x,xhat,NULL,NULL,&norm,&err);CHKERRQ(ierr);}
    if (Hpre) {ierr = PetscFnTestDerivativeBuild(fn,PETSCFNOP_HESSIANBUILD,*Hpre,x,xhat,NULL,NULL,&norm,&err);CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnHessianBuildSwap(PetscFn fn, Vec x, Vec xhat, MatReuse reuse, Mat *Hswp, Mat *Hswppre)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn,PETSCFN_CLASSID,1);
  PetscValidType(fn,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  PetscValidHeaderSpecific(xhat,VEC_CLASSID,3);
  ierr = PetscFnSetUp(fn);CHKERRQ(ierr);
  if (!Hswp && !Hswppre) PetscFunctionReturn(0);
  ierr = VecLockPush(x);CHKERRQ(ierr);
  ierr = VecLockPush(xhat);CHKERRQ(ierr);
  if (fn->ops->hessianbuildswap) {
    ierr = (*fn->ops->hessianbuildswap)(fn,x,xhat,reuse,Hswp,Hswppre);CHKERRQ(ierr);
  } else if (fn->isScalar && fn->ops->scalarhessianmult) {
    Vec                Hxhat;

    ierr = VecDuplicate(xhat, &Hxhat);CHKERRQ(ierr);
    ierr = (*fn->ops->scalarhessianmult) (fn, x, xhat, Hxhat);CHKERRQ(ierr);
    ierr = PetscFnVecToMat(Hxhat, PETSC_TRUE, reuse, Hswp, Hswppre);CHKERRQ(ierr);
    ierr = VecDestroy(&Hxhat);CHKERRQ(ierr);
  } else SETERRQ1(PetscObjectComm((PetscObject)fn), PETSC_ERR_SUP, "This PetscFn does not implement %s()", PETSC_FUNCTION_NAME);
  ierr = VecLockPop(xhat);CHKERRQ(ierr);
  ierr = VecLockPop(x);CHKERRQ(ierr);
  if (fn->test_hesbuildswp) {
    PetscReal norm, err;

    fn->test_hesbuildswp = PETSC_FALSE;
    if (Hswp) {ierr = PetscFnTestDerivativeBuild(fn,PETSCFNOP_HESSIANBUILDSWAP,*Hswp,x,xhat,NULL,NULL,&norm,&err);CHKERRQ(ierr);}
    if (Hswppre) {ierr = PetscFnTestDerivativeBuild(fn,PETSCFNOP_HESSIANBUILDSWAP,*Hswppre,x,xhat,NULL,NULL,&norm,&err);CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnHessianBuildAdjoint(PetscFn fn, Vec x, Vec v, MatReuse reuse, Mat *Hadj, Mat *Hadjpre)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn,PETSCFN_CLASSID,1);
  PetscValidType(fn,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  PetscValidHeaderSpecific(v,VEC_CLASSID,3);
  ierr = PetscFnSetUp(fn);CHKERRQ(ierr);
  if (!Hadj && !Hadjpre) PetscFunctionReturn(0);
  ierr = VecLockPush(x);CHKERRQ(ierr);
  ierr = VecLockPush(v);CHKERRQ(ierr);
  if (fn->ops->hessianbuildadjoint) {
    ierr = (*fn->ops->hessianbuildadjoint)(fn,x,v,reuse,Hadj,Hadjpre);CHKERRQ(ierr);
  } else if (fn->isScalar && fn->ops->scalarhessianbuild) {
    PetscScalar z;

    ierr = (*fn->ops->scalarhessianbuild) (fn, x, reuse, Hadj, Hadjpre);CHKERRQ(ierr);
    ierr = VecScalarBcast(v, &z);CHKERRQ(ierr);
    if (Hadj) {ierr = MatScale(*Hadj,z);CHKERRQ(ierr);}
    if (Hadjpre && Hadjpre != Hadj) {ierr = MatScale(*Hadjpre,z);CHKERRQ(ierr);}
  } else SETERRQ1(PetscObjectComm((PetscObject)fn), PETSC_ERR_SUP, "This PetscFn does not implement %s()", PETSC_FUNCTION_NAME);
  ierr = VecLockPop(v);CHKERRQ(ierr);
  ierr = VecLockPop(x);CHKERRQ(ierr);
  if (fn->test_hesbuildadj) {
    PetscReal norm, err;

    fn->test_hesbuildadj = PETSC_FALSE;
    if (Hadj) {ierr = PetscFnTestDerivativeBuild(fn,PETSCFNOP_HESSIANBUILDADJOINT,*Hadj,x,v,NULL,NULL,&norm,&err);CHKERRQ(ierr);}
    if (Hadjpre) {ierr = PetscFnTestDerivativeBuild(fn,PETSCFNOP_HESSIANBUILDADJOINT,*Hadjpre,x,v,NULL,NULL,&norm,&err);CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnScalarApply(PetscFn fn, Vec x, PetscScalar *z)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn,PETSCFN_CLASSID,1);
  PetscValidType(fn,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  ierr = PetscFnSetUp(fn);CHKERRQ(ierr);
  if (!(fn->isScalar)) SETERRQ(PetscObjectComm((PetscObject)fn),PETSC_ERR_ARG_SIZ, "PetscFn is not a scalar function");
  if (fn->dmap->N != x->map->N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"PetscFn fn,Vec x: global dim %D %D",fn->dmap->N,x->map->N);

  ierr = VecLockPush(x);CHKERRQ(ierr);
  if (fn->ops->scalarapply) {
    ierr = (*fn->ops->scalarapply)(fn,x,z);CHKERRQ(ierr);
  } else if (fn->ops->scalarderivativescalar) {
    ierr = (*fn->ops->scalarderivativescalar)(fn,x,0,NULL,NULL,z);CHKERRQ(ierr);
  } else if (fn->ops->apply) {
    Vec y;

    ierr = PetscFnCreateVecs(fn, NULL, NULL, NULL, &y);CHKERRQ(ierr);
    ierr = (*fn->ops->apply)(fn,x,y);CHKERRQ(ierr);
    ierr = VecScalarBcast(y,z);CHKERRQ(ierr);
    ierr = VecDestroy(&y);CHKERRQ(ierr);
  } else if (fn->ops->derivativescalar) {
    Vec y;

    ierr = PetscFnCreateVecs(fn, NULL, NULL, NULL, &y);CHKERRQ(ierr);
    ierr = VecSet(y, 1.);CHKERRQ(ierr);
    ierr = (*fn->ops->derivativescalar)(fn,x,0,0,NULL,&y,z);CHKERRQ(ierr);
    ierr = VecDestroy(&y);CHKERRQ(ierr);
  } else if (fn->ops->derivativevec) {
    Vec y;

    ierr = PetscFnCreateVecs(fn, NULL, NULL, NULL, &y);CHKERRQ(ierr);
    ierr = (*fn->ops->derivativevec)(fn,x,0,0,NULL,NULL,y);CHKERRQ(ierr);
    ierr = VecScalarBcast(y,z);CHKERRQ(ierr);
    ierr = VecDestroy(&y);CHKERRQ(ierr);
  } else SETERRQ1(PetscObjectComm((PetscObject)fn), PETSC_ERR_SUP, "This PetscFn does not implement %s()", PETSC_FUNCTION_NAME);
  ierr = VecLockPop(x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnScalarGradient(PetscFn fn, Vec x, Vec g)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn,PETSCFN_CLASSID,1);
  PetscValidType(fn,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  ierr = PetscFnSetUp(fn);CHKERRQ(ierr);
  if (!(fn->isScalar)) SETERRQ(PetscObjectComm((PetscObject)fn),PETSC_ERR_ARG_SIZ, "PetscFn is not a scalar function");
  if (fn->dmap->N != x->map->N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"PetscFn fn,Vec x: global dim %D %D",fn->dmap->N,x->map->N);
  if (fn->dmap->N != g->map->N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"PetscFn fn,Vec g: global dim %D %D",fn->dmap->N,g->map->N);
  if (fn->dmap->n != g->map->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"PetscFn fn,Vec g: local dim %D %D",fn->dmap->n,g->map->n);

  ierr = VecLockPush(x);CHKERRQ(ierr);
  if (fn->ops->scalargradient) {
    ierr = (*fn->ops->scalargradient)(fn,x,g);CHKERRQ(ierr);
  } else if (fn->ops->jacobianbuildadjoint) {
    Mat Jadj;
    PetscInt i, iStart, iEnd, *ia;
    PetscInt zero = 0;
    PetscScalar *ga;

    ierr = (*fn->ops->jacobianbuildadjoint)(fn,x,MAT_INITIAL_MATRIX,&Jadj,NULL);CHKERRQ(ierr);
    ierr = VecGetOwnershipRange(x,&iStart,&iEnd);CHKERRQ(ierr);
    ierr = PetscMalloc1(iEnd - iStart,&ia);CHKERRQ(ierr);
    for (i = 0; i < iEnd - iStart; i++) ia[i] = i + iStart;
    ierr = VecGetArray(g, &ga);CHKERRQ(ierr);
    ierr = MatGetValues(Jadj,iEnd - iStart, ia, 1, &zero, ga);CHKERRQ(ierr);
    ierr = VecRestoreArray(g, &ga);CHKERRQ(ierr);
    ierr = PetscFree(ia);CHKERRQ(ierr);
    ierr = MatDestroy(&Jadj);CHKERRQ(ierr);
  } else SETERRQ1(PetscObjectComm((PetscObject)fn), PETSC_ERR_SUP, "This PetscFn does not implement %s()", PETSC_FUNCTION_NAME);
  ierr = VecLockPop(x);CHKERRQ(ierr);
  if (fn->test_scalgrad) {
    PetscReal rate;

    fn->test_scalgrad = PETSC_FALSE;
    ierr = PetscFnTestDerivativeMult(fn,PETSCFNOP_SCALARGRADIENT,x,NULL,NULL,NULL,PETSC_DEFAULT,PETSC_DEFAULT,&rate);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnScalarHessianMult(PetscFn fn, Vec x, Vec xhat, Vec Hxhat)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn,PETSCFN_CLASSID,1);
  PetscValidType(fn,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  ierr = PetscFnSetUp(fn);CHKERRQ(ierr);
  if (!(fn->isScalar)) SETERRQ(PetscObjectComm((PetscObject)fn),PETSC_ERR_ARG_SIZ, "PetscFn is not a scalar function");
  if (fn->dmap->N != x->map->N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"PetscFn fn,Vec x: global dim %D %D",fn->dmap->N,x->map->N);
  if (fn->dmap->N != xhat->map->N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"PetscFn fn,Vec xhat: global dim %D %D",fn->dmap->N,xhat->map->N);
  if (fn->dmap->N != Hxhat->map->N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"PetscFn fn,Vec Hxhat: global dim %D %D",fn->dmap->N,Hxhat->map->N);
  if (fn->dmap->n != Hxhat->map->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"PetscFn fn,Vec Hxhat: local dim %D %D",fn->dmap->n,Hxhat->map->n);

  ierr = VecLockPush(x);CHKERRQ(ierr);
  ierr = VecLockPush(xhat);CHKERRQ(ierr);
  if (fn->ops->scalarhessianmult) {
    ierr = (*fn->ops->scalarhessianmult)(fn,x,xhat,Hxhat);CHKERRQ(ierr);
  } else if (fn->ops->hessianmultadjoint) {
    Vec v;

    ierr = PetscFnCreateVecs(fn, NULL, NULL, NULL, &v);CHKERRQ(ierr);
    ierr = VecSet(v, 1.);CHKERRQ(ierr);
    ierr = (*fn->ops->hessianmultadjoint)(fn,x,v,xhat,Hxhat);CHKERRQ(ierr);
    ierr = VecDestroy(&v);CHKERRQ(ierr);
  } else SETERRQ1(PetscObjectComm((PetscObject)fn), PETSC_ERR_SUP, "This PetscFn does not implement %s()", PETSC_FUNCTION_NAME);
  ierr = VecLockPop(xhat);CHKERRQ(ierr);
  ierr = VecLockPop(x);CHKERRQ(ierr);
  if (fn->test_scalhesmult) {
    PetscReal rate;

    fn->test_scalhesmult = PETSC_FALSE;
    ierr = PetscFnTestDerivativeMult(fn,PETSCFNOP_SCALARHESSIANMULT,x,xhat,NULL,NULL,PETSC_DEFAULT,PETSC_DEFAULT,&rate);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnScalarHessianBuild(PetscFn fn, Vec x, MatReuse reuse, Mat *H, Mat *Hpre)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn,PETSCFN_CLASSID,1);
  PetscValidType(fn,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  ierr = PetscFnSetUp(fn);CHKERRQ(ierr);
  if (!(fn->isScalar)) SETERRQ(PetscObjectComm((PetscObject)fn),PETSC_ERR_ARG_SIZ, "PetscFn is not a scalar function");
  if (fn->dmap->N != x->map->N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"PetscFn fn,Vec x: global dim %D %D",fn->dmap->N,x->map->N);
  if (!H && !Hpre) PetscFunctionReturn(0);

  ierr = VecLockPush(x);CHKERRQ(ierr);
  if (fn->ops->scalarhessianbuild) {
    ierr = (*fn->ops->scalarhessianbuild)(fn,x,reuse,H,Hpre);CHKERRQ(ierr);
  } else if (fn->ops->hessianbuildadjoint) {
    Vec v;

    ierr = PetscFnCreateVecs(fn, NULL, NULL, NULL, &v);CHKERRQ(ierr);
    ierr = VecSet(v, 1.);CHKERRQ(ierr);
    ierr = (*fn->ops->hessianbuildadjoint)(fn,x,v,reuse,H,Hpre);CHKERRQ(ierr);
    ierr = VecDestroy(&v);CHKERRQ(ierr);
  } else SETERRQ1(PetscObjectComm((PetscObject)fn), PETSC_ERR_SUP, "This PetscFn does not implement %s()", PETSC_FUNCTION_NAME);
  ierr = VecLockPop(x);CHKERRQ(ierr);
  if (fn->test_scalhesbuild) {
    PetscReal norm, err;

    fn->test_scalhesbuild = PETSC_FALSE;
    if (H) {ierr = PetscFnTestDerivativeBuild(fn,PETSCFNOP_SCALARHESSIANBUILD,*H,x,NULL,NULL,NULL,&norm,&err);CHKERRQ(ierr);}
    if (Hpre) {ierr = PetscFnTestDerivativeBuild(fn,PETSCFNOP_SCALARHESSIANBUILD,*Hpre,x,NULL,NULL,NULL,&norm,&err);CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnIsScalar(PetscFn fn, PetscBool *isScalar)
{
  PetscFunctionBegin;
  if (fn->setupcalled) {
    *isScalar = fn->isScalar;
    PetscFunctionReturn(0);
  } else {
    *isScalar = (fn->rmap->N == 1) ? PETSC_TRUE : PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}

const char *PetscFnOperations[] = {
                                  "createvecs",
                                  "apply",
                                  "jacobianmult",
                                  "jacobianmultadjoint",
                                  "jacobianbuild",
                                  "jacobianbuildadjoint",
                                  "hessianmult",
                                  "hessianmultadjoint",
                                  "hessianbuild",
                                  "hessianbuildadjoint",
                                  "hessianbuildswap",
                                  "scalarapply",
                                  "scalargradient",
                                  "scalarhessianmult",
                                  "scalarhessianbuild",
                                  "createsubfns",
                                  "destroysubfns",
                                  "createsubfn",
                                  "createderivativefn",
                                  "destroy",
                                  "view",
                                  };

PetscErrorCode PetscFnTestDerivativeMult(PetscFn fn, PetscFnOperation op, Vec x, Vec xhat, Vec dot, PetscRandom rand, PetscReal e1, PetscReal e2, PetscReal * rate)
{
  PetscRandom    rorig = rand;
  Vec            xorig = x;
  Vec            xhatorig = xhat;
  Vec            dotorig = dot;
  PetscInt       i;
  Vec            xtilde[2];
  Vec            f0, fmeas, fpred, der;
  PetscScalar    r0, rmeas, rpred, rder;
  PetscReal      diff[2];
  PetscReal      e[2];
  PetscBool      anyRandom;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  e[0] = e1;
  e[1] = e2;
  if (fn->setfromoptions && e1 < 0. && e2 < 0.) {
    PetscInt two = 2;

    ierr = PetscOptionsGetRealArray(((PetscObject)fn)->options,((PetscObject)fn)->prefix,"-fn_test_derivative_offsets",e,&two,NULL);CHKERRQ(ierr);
  }
  if (e[0] < 0.) {e[0] = 2. * PetscSqrtReal(PETSC_SMALL);}
  if (e[1] < 0.) {e[1] = PetscSqrtReal(PETSC_SMALL);}
  anyRandom = PETSC_FALSE;
  if (!x) anyRandom = PETSC_TRUE;
  if (!xhat) anyRandom = PETSC_TRUE;
  if (!dot) anyRandom = PETSC_TRUE;
  if (anyRandom && !rand) {
    ierr = PetscRandomCreate(PetscObjectComm((PetscObject)fn),&rand);CHKERRQ(ierr);
    if (fn->setfromoptions) {
      ierr = PetscObjectSetOptionsPrefix((PetscObject)rand,((PetscObject)fn)->prefix);CHKERRQ(ierr);
      ierr = PetscObjectAppendOptionsPrefix((PetscObject)rand,"fn_test_derivative_");CHKERRQ(ierr);
      ierr = PetscRandomSetFromOptions(rand);CHKERRQ(ierr);
    }
  }
  if (!x) {
    ierr = PetscFnCreateVecs(fn, NULL, &x, NULL, NULL);CHKERRQ(ierr);
    ierr = VecSetRandom(x, rand);CHKERRQ(ierr);
  }
  if (!xhat) {
    ierr = PetscFnCreateVecs(fn, NULL, &xhat, NULL, NULL);CHKERRQ(ierr);
    ierr = VecSetRandom(xhat, rand);CHKERRQ(ierr);
  }
  if (!dot) {
    if (op == PETSCFNOP_JACOBIANMULTADJOINT || op == PETSCFNOP_HESSIANMULTADJOINT) {
      ierr = PetscFnCreateVecs(fn, NULL, NULL, NULL, &dot);CHKERRQ(ierr);
    } else {
      ierr = PetscFnCreateVecs(fn, NULL, &dot, NULL, NULL);CHKERRQ(ierr);
    }
    ierr = VecSetRandom(dot, rand);CHKERRQ(ierr);
  }
  for (i = 0; i < 2; i++) {
    ierr = VecDuplicate(x, &xtilde[i]);CHKERRQ(ierr);
    ierr = VecWAXPY(xtilde[i],e[i],xhat,x);CHKERRQ(ierr);
  }
  switch (op) {
  case PETSCFNOP_JACOBIANMULT:
    ierr = PetscFnCreateVecs(fn, NULL, NULL, NULL, &f0);CHKERRQ(ierr);
    ierr = PetscFnApply(fn, x, f0);CHKERRQ(ierr);
    ierr = VecDuplicate(f0, &der);CHKERRQ(ierr);
    ierr = PetscFnJacobianMult(fn, x, xhat, der);CHKERRQ(ierr);
    ierr = VecDuplicate(f0, &fmeas);CHKERRQ(ierr);
    ierr = VecDuplicate(f0, &fpred);CHKERRQ(ierr);
    for (i = 0; i < 2; i++) {
      ierr = PetscFnApply(fn, xtilde[i], fmeas);CHKERRQ(ierr);
      ierr = VecWAXPY(fpred,e[i],der,f0);CHKERRQ(ierr);
      ierr = VecAXPY(fpred,-1.,fmeas);CHKERRQ(ierr);
      ierr = VecNorm(fpred,NORM_2,&diff[i]);CHKERRQ(ierr);
    }
    ierr = VecDestroy(&fpred);CHKERRQ(ierr);
    ierr = VecDestroy(&fmeas);CHKERRQ(ierr);
    ierr = VecDestroy(&der);CHKERRQ(ierr);
    ierr = VecDestroy(&f0);CHKERRQ(ierr);
    break;
  case PETSCFNOP_JACOBIANMULTADJOINT:
    ierr = PetscFnCreateVecs(fn, NULL, NULL, NULL, &f0);CHKERRQ(ierr);
    ierr = PetscFnApply(fn, x, f0);CHKERRQ(ierr);
    ierr = VecDuplicate(x, &der);CHKERRQ(ierr);
    ierr = VecDot(dot, f0, &r0);CHKERRQ(ierr);
    ierr = PetscFnJacobianMultAdjoint(fn, x, dot, der);CHKERRQ(ierr);
    ierr = VecDot(der, xhat, &rder);CHKERRQ(ierr);
    ierr = VecDuplicate(f0, &fmeas);CHKERRQ(ierr);
    for (i = 0; i < 2; i++) {
      ierr = PetscFnApply(fn, xtilde[i], fmeas);CHKERRQ(ierr);
      ierr = VecDot(dot, fmeas, &rmeas);CHKERRQ(ierr);
      rpred = r0 + e[i] * rder;
      diff[i] = PetscAbsScalar(rpred - rmeas);
    }
    ierr = VecDestroy(&fmeas);CHKERRQ(ierr);
    ierr = VecDestroy(&der);CHKERRQ(ierr);
    ierr = VecDestroy(&f0);CHKERRQ(ierr);
    break;
  case PETSCFNOP_HESSIANMULT:
    ierr = PetscFnCreateVecs(fn, NULL, NULL, NULL, &f0);CHKERRQ(ierr);
    ierr = PetscFnJacobianMult(fn, x, dot, f0);CHKERRQ(ierr);
    ierr = VecDuplicate(f0, &der);CHKERRQ(ierr);
    ierr = PetscFnHessianMult(fn, x, dot, xhat, der);CHKERRQ(ierr);
    ierr = VecDuplicate(f0, &fmeas);CHKERRQ(ierr);
    ierr = VecDuplicate(f0, &fpred);CHKERRQ(ierr);
    for (i = 0; i < 2; i++) {
      ierr = PetscFnJacobianMult(fn, xtilde[i], dot, fmeas);CHKERRQ(ierr);
      ierr = VecWAXPY(fpred,e[i],der,f0);CHKERRQ(ierr);
      ierr = VecAXPY(fpred,-1.,fmeas);CHKERRQ(ierr);
      ierr = VecNorm(fpred,NORM_2,&diff[i]);CHKERRQ(ierr);
    }
    ierr = VecDestroy(&fpred);CHKERRQ(ierr);
    ierr = VecDestroy(&fmeas);CHKERRQ(ierr);
    ierr = VecDestroy(&der);CHKERRQ(ierr);
    ierr = VecDestroy(&f0);CHKERRQ(ierr);
    break;
  case PETSCFNOP_HESSIANMULTADJOINT:
    ierr = PetscFnCreateVecs(fn, NULL, &f0, NULL, NULL);CHKERRQ(ierr);
    ierr = PetscFnJacobianMultAdjoint(fn, x, dot, f0);CHKERRQ(ierr);
    ierr = VecDuplicate(f0, &der);CHKERRQ(ierr);
    ierr = PetscFnHessianMultAdjoint(fn, x, dot, xhat, der);CHKERRQ(ierr);
    ierr = VecDuplicate(f0, &fmeas);CHKERRQ(ierr);
    ierr = VecDuplicate(f0, &fpred);CHKERRQ(ierr);
    for (i = 0; i < 2; i++) {
      ierr = PetscFnJacobianMultAdjoint(fn, xtilde[i], dot, fmeas);CHKERRQ(ierr);
      ierr = VecWAXPY(fpred,e[i],der,f0);CHKERRQ(ierr);
      ierr = VecAXPY(fpred,-1.,fmeas);CHKERRQ(ierr);
      ierr = VecNorm(fpred,NORM_2,&diff[i]);CHKERRQ(ierr);
    }
    ierr = VecDestroy(&fpred);CHKERRQ(ierr);
    ierr = VecDestroy(&fmeas);CHKERRQ(ierr);
    ierr = VecDestroy(&der);CHKERRQ(ierr);
    ierr = VecDestroy(&f0);CHKERRQ(ierr);
    break;
  case PETSCFNOP_SCALARGRADIENT:
    ierr = PetscFnCreateVecs(fn, NULL, &der, NULL, NULL);CHKERRQ(ierr);
    ierr = PetscFnScalarGradient(fn,x,der);CHKERRQ(ierr);
    ierr = PetscFnScalarApply(fn,x,&r0);CHKERRQ(ierr);
    ierr = VecDot(der,xhat,&rder);CHKERRQ(ierr);
    for (i = 0; i < 2; i++) {
      ierr = PetscFnScalarApply(fn,xtilde[i],&rmeas);CHKERRQ(ierr);
      rpred = r0 + e[i] * rder;
      diff[i] = PetscAbsScalar(rpred - rmeas);
    }
    ierr = VecDestroy(&der);CHKERRQ(ierr);
    break;
  case PETSCFNOP_SCALARHESSIANMULT:
    ierr = PetscFnCreateVecs(fn, NULL, &f0, NULL, NULL);CHKERRQ(ierr);
    ierr = PetscFnScalarGradient(fn,x,f0);CHKERRQ(ierr);
    ierr = VecDuplicate(f0, &der);CHKERRQ(ierr);
    ierr = PetscFnScalarHessianMult(fn,x,xhat,der);CHKERRQ(ierr);
    ierr = VecDuplicate(f0, &fmeas);CHKERRQ(ierr);
    ierr = VecDuplicate(f0, &fpred);CHKERRQ(ierr);
    for (i = 0; i < 2; i++) {
      ierr = PetscFnScalarGradient(fn,xtilde[i],fmeas);CHKERRQ(ierr);
      ierr = VecWAXPY(fpred,e[i],der,f0);CHKERRQ(ierr);
      ierr = VecAXPY(fpred,-1.,fmeas);CHKERRQ(ierr);
      ierr = VecNorm(fpred,NORM_2,&diff[i]);CHKERRQ(ierr);
    }
    ierr = VecDestroy(&fpred);CHKERRQ(ierr);
    ierr = VecDestroy(&fmeas);CHKERRQ(ierr);
    ierr = VecDestroy(&der);CHKERRQ(ierr);
    ierr = VecDestroy(&f0);CHKERRQ(ierr);
    break;
  default:
    SETERRQ1(PetscObjectComm((PetscObject)fn), PETSC_ERR_ARG_OUTOFRANGE, "%s cannot be called on this PetscFnOperation", PETSC_FUNCTION_NAME);
  }
  *rate = PetscLog2Real(diff[1] / diff[0]) / PetscLog2Real(e[1] / e[0]);
  for (i = 0; i < 2; i++) {ierr = VecDestroy(&xtilde[i]);CHKERRQ(ierr);}
  if (dotorig != dot) {ierr = VecDestroy(&dot);CHKERRQ(ierr);}
  if (xhatorig != xhat) {ierr = VecDestroy(&xhat);CHKERRQ(ierr);}
  if (xorig != x) {ierr = VecDestroy(&x);CHKERRQ(ierr);}
  if (rorig != rand) {ierr = PetscRandomDestroy(&rand);CHKERRQ(ierr);}
  if (fn->setfromoptions) {
    PetscBool view = PETSC_FALSE;

    ierr = PetscOptionsGetBool(((PetscObject)fn)->options,((PetscObject)fn)->prefix,"-fn_test_derivative_view",&view,NULL);CHKERRQ(ierr);
    if (view) {
      MPI_Comm comm = PetscObjectComm((PetscObject)fn);
      ierr = PetscPrintf(comm, "%s: Tested convergence of %s at offsets (%g, %g); tangents differ by (%g, %g): measured rate %g\n", PETSC_FUNCTION_NAME, PetscFnOperations[op], e[0], e[1], diff[0], diff[1], *rate);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnTestDerivativeBuild(PetscFn fn, PetscFnOperation op, Mat M, Vec x, Vec dot, Vec var, PetscRandom rand, PetscReal *norm, PetscReal *err)
{
  PetscRandom    rorig = rand;
  Vec            dotorig = dot;
  Vec            varorig = var;
  Vec            b, c;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn, PETSCFN_CLASSID, 1);
  PetscValidHeaderSpecific(M, MAT_CLASSID, 3);
  PetscValidHeaderSpecific(x, VEC_CLASSID, 4);
  if (!x) SETERRQ(PetscObjectComm((PetscObject)fn),PETSC_ERR_ARG_NULL, "Need x at value where matrix was constructed");
  if (op == PETSCFNOP_HESSIANBUILD || op == PETSCFNOP_HESSIANBUILDADJOINT) {
    if (!dot) {
      if (op == PETSCFNOP_HESSIANBUILD) SETERRQ(PetscObjectComm((PetscObject)fn),PETSC_ERR_ARG_NULL, "Need dot to be primal direction where matrix was constructed");
      else SETERRQ(PetscObjectComm((PetscObject)fn),PETSC_ERR_ARG_NULL, "Need dot to be adjoint where matrix was constructed");
    }
  }
  if (!var && !rand) {
    ierr = PetscRandomCreate(PetscObjectComm((PetscObject)fn),&rand);CHKERRQ(ierr);
    if (fn->setfromoptions) {
      ierr = PetscObjectSetOptionsPrefix((PetscObject)rand,((PetscObject)fn)->prefix);CHKERRQ(ierr);
      ierr = PetscObjectAppendOptionsPrefix((PetscObject)rand,"fn_test_derivativemat_");CHKERRQ(ierr);
      ierr = PetscRandomSetFromOptions(rand);CHKERRQ(ierr);
    }
  }
  if (!var) {
    if (op == PETSCFNOP_JACOBIANBUILDADJOINT || op == PETSCFNOP_HESSIANBUILDSWAP) {
      ierr = PetscFnCreateVecs(fn, NULL, NULL, NULL, &var);CHKERRQ(ierr);
    }
    else {
      ierr = PetscFnCreateVecs(fn, NULL, &var, NULL, NULL);CHKERRQ(ierr);
    }
    ierr = VecSetRandom(var, rand);CHKERRQ(ierr);
  }
  if (op == PETSCFNOP_JACOBIANBUILD || op == PETSCFNOP_HESSIANBUILD) {
    ierr = PetscFnCreateVecs(fn, NULL, NULL, NULL, &b);CHKERRQ(ierr);
    ierr = PetscFnCreateVecs(fn, NULL, NULL, NULL, &c);CHKERRQ(ierr);
  } else {
    ierr = PetscFnCreateVecs(fn, NULL, &b, NULL, NULL);CHKERRQ(ierr);
    ierr = PetscFnCreateVecs(fn, NULL, &c, NULL, NULL);CHKERRQ(ierr);
  }
  ierr = MatMult(M, var, b);CHKERRQ(ierr);
  switch (op) {
  case PETSCFNOP_JACOBIANBUILD:
    ierr = PetscFnJacobianMult(fn,x,var,c);CHKERRQ(ierr);
    break;
  case PETSCFNOP_JACOBIANBUILDADJOINT:
    ierr = PetscFnJacobianMultAdjoint(fn,x,var,c);CHKERRQ(ierr);
    break;
  case PETSCFNOP_HESSIANBUILD:
    ierr = PetscFnHessianMult(fn,x,dot,var,c);CHKERRQ(ierr);
    break;
  case PETSCFNOP_HESSIANBUILDADJOINT:
    ierr = PetscFnHessianMultAdjoint(fn,x,dot,var,c);CHKERRQ(ierr);
    break;
  case PETSCFNOP_HESSIANBUILDSWAP:
    ierr = PetscFnHessianMultAdjoint(fn,x,var,dot,c);CHKERRQ(ierr);
    break;
  case PETSCFNOP_SCALARHESSIANBUILD:
    ierr = PetscFnScalarHessianMult(fn,x,var,c);CHKERRQ(ierr);
    break;
  default:
    SETERRQ1(PetscObjectComm((PetscObject)fn), PETSC_ERR_ARG_OUTOFRANGE, "%s cannot be called on this PetscFnOperation", PETSC_FUNCTION_NAME);
  }
  ierr = VecAXPY(b, -1., c);CHKERRQ(ierr);
  ierr = VecNorm(c, NORM_2, norm);CHKERRQ(ierr);
  ierr = VecNorm(b, NORM_2, err);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = VecDestroy(&c);CHKERRQ(ierr);
  if (var != varorig) {ierr = VecDestroy(&var);CHKERRQ(ierr);}
  if (dot != dotorig) {ierr = VecDestroy(&dot);CHKERRQ(ierr);}
  if (rorig != rand) {ierr = PetscRandomDestroy(&rand);CHKERRQ(ierr);}
  if (fn->setfromoptions) {
    PetscBool view = PETSC_FALSE;

    ierr = PetscOptionsGetBool(((PetscObject)fn)->options,((PetscObject)fn)->prefix,"-fn_test_derivativemat_view",&view,NULL);CHKERRQ(ierr);
    if (view) {
      MPI_Comm comm = PetscObjectComm((PetscObject)fn);
      ierr = PetscPrintf(comm, "%s: Tested %s matrix against matrix-free action: norm %g, error %g\n", PETSC_FUNCTION_NAME, PetscFnOperations[op], (double) *norm, (double) *err);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnTestDerivativeFn(PetscFn fn, PetscFn df, PetscFnOperation op, PetscInt numDots, const Vec dotVecs[], Vec x, PetscReal *norm, PetscReal *err)
{
  Vec            dfOut, fnOut, fnOutRange, fnOutDomain;
  PetscReal      dfz, fnz;
  PetscInt       maxDots;
  MPI_Comm       comm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn, PETSCFN_CLASSID, 1);
  PetscValidHeaderSpecific(x, VEC_CLASSID, 6);
  comm = PetscObjectComm((PetscObject)fn);
  switch (op) {
  case PETSCFNOP_SCALARAPPLY:
    maxDots = 0;
    break;
  case PETSCFNOP_APPLY:
  case PETSCFNOP_SCALARGRADIENT:
    maxDots = 1;
    break;
  case PETSCFNOP_JACOBIANMULT:
  case PETSCFNOP_JACOBIANMULTADJOINT:
  case PETSCFNOP_SCALARHESSIANMULT:
    maxDots = 2;
    break;
  case PETSCFNOP_HESSIANMULT:
  case PETSCFNOP_HESSIANMULTADJOINT:
    maxDots = 3;
    break;
  default:
    SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "%s cannot be called on this PetscFnOperation", PETSC_FUNCTION_NAME);
  }
  if (numDots < maxDots - 1 || numDots > maxDots) SETERRQ2(comm, PETSC_ERR_ARG_OUTOFRANGE, "This operation contracts with at most %D vectors: %D given", maxDots, numDots);
  ierr = PetscFnCreateVecs(fn, NULL, &fnOutDomain, NULL, &fnOutRange);CHKERRQ(ierr);
  ierr = PetscFnCreateVecs(df, NULL, NULL, NULL, &dfOut);CHKERRQ(ierr);
  if (numDots == maxDots) {
    ierr = PetscFnScalarApply(df, x, &dfz);CHKERRQ(ierr);
  } else {
    ierr = PetscFnApply(df, x, dfOut);CHKERRQ(ierr);
  }
  switch (op) {
  case PETSCFNOP_APPLY:
    ierr = PetscFnApply(fn,x, fnOutRange);CHKERRQ(ierr);
    fnOut = fnOutRange;
    break;
  case PETSCFNOP_JACOBIANMULT:
    ierr = PetscFnJacobianMult(fn, x, dotVecs[0], fnOutRange);CHKERRQ(ierr);
    fnOut = fnOutRange;
    break;
  case PETSCFNOP_JACOBIANMULTADJOINT:
    ierr = PetscFnJacobianMultAdjoint(fn, x, dotVecs[0], fnOutDomain);CHKERRQ(ierr);
    fnOut = fnOutDomain;
    break;
  case PETSCFNOP_HESSIANMULT:
    ierr = PetscFnHessianMult(fn, x, dotVecs[0], dotVecs[1], fnOutRange);CHKERRQ(ierr);
    fnOut = fnOutRange;
    break;
  case PETSCFNOP_HESSIANMULTADJOINT:
    ierr = PetscFnHessianMultAdjoint(fn, x, dotVecs[0], dotVecs[1], fnOutDomain);CHKERRQ(ierr);
    fnOut = fnOutDomain;
    break;
  case PETSCFNOP_SCALARAPPLY:
    ierr = PetscFnScalarApply(fn, x, &fnz);CHKERRQ(ierr);
    break;
  case PETSCFNOP_SCALARGRADIENT:
    ierr = PetscFnScalarGradient(fn, x, fnOutDomain);CHKERRQ(ierr);
    fnOut = fnOutDomain;
    break;
  case PETSCFNOP_SCALARHESSIANMULT:
    ierr = PetscFnScalarHessianMult(fn, x, dotVecs[0], fnOutDomain);CHKERRQ(ierr);
    fnOut = fnOutDomain;
    break;
  default:
    SETERRQ1(PetscObjectComm((PetscObject)fn), PETSC_ERR_ARG_OUTOFRANGE, "%s cannot be called on this PetscFnOperation", PETSC_FUNCTION_NAME);
  }
  if (numDots == maxDots) {
    if (op != PETSCFNOP_SCALARAPPLY) {
      ierr = VecDot(dotVecs[maxDots - 1], fnOut, &fnz);CHKERRQ(ierr);
    }
    *norm = PetscAbsScalar(fnz);
    *err = PetscAbsScalar(fnz - dfz);
  } else {
    ierr = VecAXPY(dfOut, -1., fnOut);CHKERRQ(ierr);
    ierr = VecNorm(fnOut, NORM_2, norm);CHKERRQ(ierr);
    ierr = VecNorm(dfOut, NORM_2, err);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&dfOut);CHKERRQ(ierr);
  ierr = VecDestroy(&fnOutRange);CHKERRQ(ierr);
  ierr = VecDestroy(&fnOutDomain);CHKERRQ(ierr);
  if (fn->setfromoptions) {
    PetscBool view = PETSC_FALSE;

    ierr = PetscOptionsGetBool(((PetscObject)fn)->options,((PetscObject)fn)->prefix,"-fn_test_derivativefn_view",&view,NULL);CHKERRQ(ierr);
    if (view) {
      MPI_Comm comm = PetscObjectComm((PetscObject)fn);
      ierr = PetscPrintf(comm, "%s: Tested %s instantiated function against matrix-free action: norm %g, error %g\n", PETSC_FUNCTION_NAME, PetscFnOperations[op], (double) *norm, (double) *err);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

typedef struct
{
  PetscFn origFn;
  PetscFnOperation op;
  PetscInt numDots;
  PetscInt maxDots;
  PetscInt der;
  PetscInt rangeIdx;
  Vec dotVecs[3];
  Vec workVecs[1];
  IS  *dotISs;
} PetscFnDerShell;

static PetscErrorCode PetscFnShellDestroy_DerShell(PetscFn fn)
{
  PetscFnDerShell *derShell;
  PetscInt         i;
  PetscFn          origFn;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscFnShellGetContext(fn, (void **) &derShell);CHKERRQ(ierr);
  origFn = derShell->origFn;
  for (i = 0; i < derShell->numDots; i++) {
    ierr = VecDestroy(&(derShell->dotVecs[i]));CHKERRQ(ierr);
  }
  ierr = VecDestroy(&(derShell->workVecs[0]));CHKERRQ(ierr);
  ierr = PetscFree(derShell);CHKERRQ(ierr);
  ierr = PetscFnDestroy(&origFn);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnShellCreateVecs_DerShell(PetscFn fn, IS domainIS, Vec *domainVec, IS rangeIS, Vec *rangeVec)
{
  PetscFnDerShell *derShell;
  PetscFn          origFn;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscFnShellGetContext(fn, (void **) &derShell);CHKERRQ(ierr);
  origFn = derShell->origFn;
  if (domainVec) {
    ierr = PetscFnCreateVecs(origFn, domainIS, domainVec, NULL, NULL);CHKERRQ(ierr);
  }
  if (!rangeVec) PetscFunctionReturn(0);
  if (fn->rmap == origFn->rmap) {
    ierr = PetscFnCreateVecs(origFn, NULL, NULL, rangeIS, rangeVec);CHKERRQ(ierr);
  } else if (fn->rmap == origFn->dmap) {
    ierr = PetscFnCreateVecs(origFn, rangeIS, rangeVec, NULL, NULL);CHKERRQ(ierr);
  } else {
    ierr = PetscFnCreateVec_Default(fn, rangeIS, fn->rmap, rangeVec);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnShellScalarApply_DerShell(PetscFn fn, Vec x, PetscScalar *z)
{
  PetscFnDerShell *derShell;
  PetscFn          origFn;
  PetscInt         numDots;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscFnShellGetContext(fn, (void **) &derShell);CHKERRQ(ierr);
  origFn = derShell->origFn;
  numDots = derShell->numDots;
  switch (derShell->op) {
  case PETSCFNOP_APPLY:
    ierr = PetscFnApply(origFn, x, derShell->workVecs[0]);CHKERRQ(ierr);
    break;
  case PETSCFNOP_JACOBIANMULT:
    ierr = PetscFnJacobianMult(origFn, x, derShell->dotVecs[0], derShell->workVecs[0]);CHKERRQ(ierr);
    break;
  case PETSCFNOP_JACOBIANMULTADJOINT:
    ierr = PetscFnJacobianMultAdjoint(origFn, x, derShell->dotVecs[0], derShell->workVecs[0]);CHKERRQ(ierr);
    break;
  case PETSCFNOP_HESSIANMULT:
    ierr = PetscFnHessianMult(origFn, x, derShell->dotVecs[0], derShell->dotVecs[1], derShell->workVecs[0]);CHKERRQ(ierr);
    break;
  case PETSCFNOP_HESSIANMULTADJOINT:
    ierr = PetscFnHessianMultAdjoint(origFn, x, derShell->dotVecs[0], derShell->dotVecs[1], derShell->workVecs[0]);CHKERRQ(ierr);
    break;
  case PETSCFNOP_SCALARGRADIENT:
    ierr = PetscFnScalarGradient(origFn, x, derShell->workVecs[0]);CHKERRQ(ierr);
    break;
  case PETSCFNOP_SCALARHESSIANMULT:
    ierr = PetscFnScalarHessianMult(origFn, x, derShell->dotVecs[0], derShell->workVecs[0]);CHKERRQ(ierr);
    break;
  default:
    SETERRQ1(PetscObjectComm((PetscObject)fn), PETSC_ERR_ARG_OUTOFRANGE, "%s cannot be called on this PetscFnOperation", PETSC_FUNCTION_NAME);
  }
  ierr = VecDot(derShell->dotVecs[numDots - 1], derShell->workVecs[0], z);CHKERRQ(ierr);
  if (fn->test_self_as_derfn) {
    PetscReal norm, err;

    fn->test_self_as_derfn = PETSC_FALSE;
    ierr = PetscFnTestDerivativeFn(origFn, fn, derShell->op, numDots, derShell->dotVecs, x, &norm, &err);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnShellApply_DerShell(PetscFn fn, Vec x, Vec y)
{
  PetscFnDerShell *derShell;
  PetscFn          origFn;
  PetscScalar      z;
  PetscInt         numDots, maxDots;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscFnShellGetContext(fn, (void **) &derShell);CHKERRQ(ierr);
  origFn = derShell->origFn;
  numDots = derShell->numDots;
  maxDots = derShell->maxDots;
  if (numDots == maxDots) {
    ierr = PetscFnShellScalarApply_DerShell(fn, x, &z);CHKERRQ(ierr);
    ierr = VecSet(y,z);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  switch (derShell->op) {
  case PETSCFNOP_JACOBIANMULT:
    ierr = PetscFnJacobianMult(origFn, x, derShell->dotVecs[0], y);CHKERRQ(ierr);
    break;
  case PETSCFNOP_JACOBIANMULTADJOINT:
    ierr = PetscFnJacobianMultAdjoint(origFn, x, derShell->dotVecs[0], y);CHKERRQ(ierr);
    break;
  case PETSCFNOP_HESSIANMULT:
    ierr = PetscFnHessianMult(origFn, x, derShell->dotVecs[0], derShell->dotVecs[1], y);CHKERRQ(ierr);
    break;
  case PETSCFNOP_HESSIANMULTADJOINT:
    ierr = PetscFnHessianMultAdjoint(origFn, x, derShell->dotVecs[0], derShell->dotVecs[1], y);CHKERRQ(ierr);
    break;
  case PETSCFNOP_SCALARHESSIANMULT:
    ierr = PetscFnScalarHessianMult(origFn, x, derShell->dotVecs[0], y);CHKERRQ(ierr);
    break;
  default:
    SETERRQ1(PetscObjectComm((PetscObject)fn), PETSC_ERR_ARG_OUTOFRANGE, "%s cannot be called on this PetscFnOperation", PETSC_FUNCTION_NAME);
  }
  if (fn->test_self_as_derfn) {
    PetscReal norm, err;

    fn->test_self_as_derfn = PETSC_FALSE;
    ierr = PetscFnTestDerivativeFn(origFn, fn, derShell->op, numDots, derShell->dotVecs, x, &norm, &err);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnShellJacobianMult_DerShell(PetscFn fn, Vec x, Vec xhat, Vec y)
{
  PetscFnDerShell *derShell;
  PetscFn          origFn;
  PetscScalar      z;
  PetscInt         numDots, maxDots;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscFnShellGetContext(fn, (void **) &derShell);CHKERRQ(ierr);
  origFn = derShell->origFn;
  numDots = derShell->numDots;
  maxDots = derShell->maxDots;
  switch (derShell->op) {
  case PETSCFNOP_APPLY:
    ierr = PetscFnJacobianMult(origFn, x, xhat, derShell->workVecs[0]);CHKERRQ(ierr);
    break;
  case PETSCFNOP_JACOBIANMULT:
    if (numDots == 1) {
      ierr = PetscFnHessianMult(origFn, x, derShell->dotVecs[0], xhat, y);CHKERRQ(ierr);
    } else {
      ierr = PetscFnHessianMult(origFn, x, derShell->dotVecs[0], xhat, derShell->workVecs[0]);CHKERRQ(ierr);
    }
    break;
  case PETSCFNOP_JACOBIANMULTADJOINT:
    if (numDots == 1) {
      ierr = PetscFnHessianMultAdjoint(origFn, x, derShell->dotVecs[0], xhat, y);CHKERRQ(ierr);
    } else {
      ierr = PetscFnHessianMultAdjoint(origFn, x, derShell->dotVecs[0], xhat, derShell->workVecs[0]);CHKERRQ(ierr);
    }
    break;
  case PETSCFNOP_SCALARGRADIENT:
    ierr = PetscFnScalarHessianMult(origFn, x, xhat, derShell->workVecs[0]);CHKERRQ(ierr);
    break;
  default:
    SETERRQ1(PetscObjectComm((PetscObject)fn), PETSC_ERR_ARG_OUTOFRANGE, "%s cannot be called on this PetscFnOperation", PETSC_FUNCTION_NAME);
  }
  if (numDots == maxDots) {
    ierr = VecDot(derShell->dotVecs[numDots - 1], derShell->workVecs[0], &z);CHKERRQ(ierr);
    ierr = VecSet(y,z);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnShellScalarGradient_DerShell(PetscFn fn, Vec x, Vec y)
{
  PetscFnDerShell *derShell;
  PetscFn          origFn;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscFnShellGetContext(fn, (void **) &derShell);CHKERRQ(ierr);
  origFn = derShell->origFn;
  switch (derShell->op) {
  case PETSCFNOP_APPLY:
    ierr = PetscFnJacobianMultAdjoint(origFn, x, derShell->dotVecs[0], y);CHKERRQ(ierr);
    break;
  case PETSCFNOP_JACOBIANMULT:
    ierr = PetscFnHessianMultAdjoint(origFn, x, derShell->dotVecs[1], derShell->dotVecs[0], y);CHKERRQ(ierr);
    break;
  case PETSCFNOP_JACOBIANMULTADJOINT:
    ierr = PetscFnHessianMultAdjoint(origFn, x, derShell->dotVecs[0], derShell->dotVecs[1], y);CHKERRQ(ierr);
    break;
  case PETSCFNOP_SCALARGRADIENT:
    ierr = PetscFnScalarGradient(origFn, x, y);CHKERRQ(ierr);
    break;
  default:
    SETERRQ1(PetscObjectComm((PetscObject)fn), PETSC_ERR_ARG_OUTOFRANGE, "%s cannot be called on this PetscFnOperation", PETSC_FUNCTION_NAME);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnShellJacobianMultAdjoint_DerShell(PetscFn fn, Vec x, Vec v, Vec y)
{
  PetscFnDerShell *derShell;
  PetscFn          origFn;
  PetscScalar      z;
  PetscInt         numDots, maxDots;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscFnShellGetContext(fn, (void **) &derShell);CHKERRQ(ierr);
  origFn = derShell->origFn;
  numDots = derShell->numDots;
  maxDots = derShell->maxDots;
  if (numDots == maxDots) {
    ierr = PetscFnShellScalarGradient_DerShell(fn, x, y);CHKERRQ(ierr);
    ierr = VecScalarBcast(v, &z);CHKERRQ(ierr);
    ierr = VecScale(y,z);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  switch (derShell->op) {
  case PETSCFNOP_JACOBIANMULT:
    /* dot0 is domain vector */
    ierr = PetscFnHessianMultAdjoint(origFn, x, v, derShell->dotVecs[0], y);CHKERRQ(ierr);
    break;
  case PETSCFNOP_JACOBIANMULTADJOINT:
    /* dot0 is range vector, v is a domain vector */
    ierr = PetscFnHessianMultAdjoint(origFn, x, derShell->dotVecs[0], v, y);CHKERRQ(ierr);
    break;
  default:
    SETERRQ1(PetscObjectComm((PetscObject)fn), PETSC_ERR_ARG_OUTOFRANGE, "%s cannot be called on this PetscFnOperation", PETSC_FUNCTION_NAME);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnShellHessianMult_DerShell(PetscFn fn, Vec x, Vec xhat, Vec xdot, Vec y)
{
  PetscFnDerShell *derShell;
  PetscFn          origFn;
  PetscScalar      z;
  PetscInt         numDots, maxDots;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscFnShellGetContext(fn, (void **) &derShell);CHKERRQ(ierr);
  origFn = derShell->origFn;
  numDots = derShell->numDots;
  maxDots = derShell->maxDots;
  switch (derShell->op) {
  case PETSCFNOP_APPLY:
    /* dot0 is range vector */
    ierr = PetscFnHessianMult(origFn, x, xhat, xdot, derShell->workVecs[0]);CHKERRQ(ierr);
    break;
  default:
    SETERRQ1(PetscObjectComm((PetscObject)fn), PETSC_ERR_ARG_OUTOFRANGE, "%s cannot be called on this PetscFnOperation", PETSC_FUNCTION_NAME);
  }
  if (numDots == maxDots) {
    ierr = VecDot(derShell->dotVecs[numDots - 1], derShell->workVecs[0], &z);CHKERRQ(ierr);
    ierr = VecSet(y,z);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnShellHessianMultAdjoint_DerShell(PetscFn fn, Vec x, Vec v, Vec xhat, Vec y)
{
  PetscFnDerShell *derShell;
  PetscFn          origFn;
  PetscScalar      z;
  PetscInt         numDots, maxDots;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscFnShellGetContext(fn, (void **) &derShell);CHKERRQ(ierr);
  origFn = derShell->origFn;
  numDots = derShell->numDots;
  maxDots = derShell->maxDots;
  switch (derShell->op) {
  case PETSCFNOP_APPLY:
    /* dot0 is range vector */
    ierr = PetscFnHessianMultAdjoint(origFn, x, derShell->dotVecs[0], xhat, y);CHKERRQ(ierr);
    break;
  default:
    SETERRQ1(PetscObjectComm((PetscObject)fn), PETSC_ERR_ARG_OUTOFRANGE, "%s cannot be called on this PetscFnOperation", PETSC_FUNCTION_NAME);
  }
  if (numDots == maxDots) {
    ierr = VecScalarBcast(v, &z);CHKERRQ(ierr);
    ierr = VecScale(y,z);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnShellScalarHessianMult_DerShell(PetscFn fn, Vec x, Vec xhat, Vec y)
{
  PetscFnDerShell *derShell;
  PetscFn          origFn;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscFnShellGetContext(fn, (void **) &derShell);CHKERRQ(ierr);
  origFn = derShell->origFn;
  switch (derShell->op) {
  case PETSCFNOP_APPLY:
    /* dot0 is range vector */
    ierr = PetscFnHessianMultAdjoint(origFn, x, derShell->dotVecs[0], xhat, y);CHKERRQ(ierr);
    break;
  default:
    SETERRQ1(PetscObjectComm((PetscObject)fn), PETSC_ERR_ARG_OUTOFRANGE, "%s cannot be called on this PetscFnOperation", PETSC_FUNCTION_NAME);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnShellJacobianBuild_DerShell(PetscFn fn, Vec x, MatReuse reuse, Mat *J, Mat *Jpre)
{
  PetscFnDerShell *derShell;
  PetscFn          origFn;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscFnShellGetContext(fn, (void **) &derShell);CHKERRQ(ierr);
  origFn = derShell->origFn;
  /* we assume that if numDots == maxDots, then a PETSCFNOP_JACOBIANBUILD method is not passed */
  switch (derShell->op) {
  case PETSCFNOP_JACOBIANMULT:
    /* The shape of the matrix will be the same, so it must be Hessian */
    ierr = PetscFnHessianBuild(origFn, x, derShell->dotVecs[0], reuse, J, Jpre);CHKERRQ(ierr);
    break;
  case PETSCFNOP_JACOBIANMULTADJOINT:
    /* The shape of the matrix will be the square, so it must be HessianAdjoint */
    ierr = PetscFnHessianBuildAdjoint(origFn, x, derShell->dotVecs[0], reuse, J, Jpre);CHKERRQ(ierr);
    break;
  case PETSCFNOP_SCALARGRADIENT:
    ierr = PetscFnScalarHessianBuild(origFn, x, reuse, J, Jpre);CHKERRQ(ierr);
    break;
  default:
    SETERRQ1(PetscObjectComm((PetscObject)fn), PETSC_ERR_ARG_OUTOFRANGE, "%s cannot be called on this PetscFnOperation", PETSC_FUNCTION_NAME);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnShellJacobianBuildAdjoint_DerShell(PetscFn fn, Vec x, MatReuse reuse, Mat *J, Mat *Jpre)
{
  PetscFnDerShell *derShell;
  PetscFn          origFn;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscFnShellGetContext(fn, (void **) &derShell);CHKERRQ(ierr);
  origFn = derShell->origFn;
  /* we assume that if numDots == maxDots, then a PETSCFNOP_JACOBIANBUILDADJOINT method is not passed */
  switch (derShell->op) {
  case PETSCFNOP_JACOBIANMULT:
    ierr = PetscFnHessianBuildSwap(origFn, x, derShell->dotVecs[0], reuse, J, Jpre);CHKERRQ(ierr);
    break;
  case PETSCFNOP_JACOBIANMULTADJOINT:
    ierr = PetscFnHessianBuildAdjoint(origFn, x, derShell->dotVecs[0], reuse, J, Jpre);CHKERRQ(ierr);
    break;
  case PETSCFNOP_SCALARGRADIENT:
    ierr = PetscFnScalarHessianBuild(origFn, x, reuse, J, Jpre);CHKERRQ(ierr);
    break;
  default:
    SETERRQ1(PetscObjectComm((PetscObject)fn), PETSC_ERR_ARG_OUTOFRANGE, "%s cannot be called on this PetscFnOperation", PETSC_FUNCTION_NAME);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnShellScalarHessianBuild_DerShell(PetscFn fn, Vec x, MatReuse reuse, Mat *H, Mat *Hpre)
{
  PetscFnDerShell *derShell;
  PetscFn          origFn;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscFnShellGetContext(fn, (void **) &derShell);CHKERRQ(ierr);
  origFn = derShell->origFn;
  if (derShell->op != PETSCFNOP_APPLY) SETERRQ1(PetscObjectComm((PetscObject)fn), PETSC_ERR_ARG_OUTOFRANGE, "%s cannot be called on this PetscFnOperation", PETSC_FUNCTION_NAME);
  ierr = PetscFnHessianBuildAdjoint(origFn, x, derShell->dotVecs[0], reuse, H, Hpre);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnDerivativeScalar_DerShell(PetscFn fn, Vec x, PetscInt der, PetscInt rangeIdx, const IS subsets[], const Vec subvecs[], PetscScalar *z)
{
  PetscFnDerShell *derShell;
  PetscFn          origFn;
  PetscInt         totalDer;
  IS        *totalSubsets;
  Vec       *totalSubvecs;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscFnShellGetContext(fn, (void **) &derShell);CHKERRQ(ierr);
  totalDer = derShell->der + der;
  origFn = derShell->origFn;
  ierr = PetscFnISsGetConcat(derShell->numDots, derShell->dotISs, der + 1, subsets, &totalSubsets);CHKERRQ(ierr);
  ierr = PetscFnVecsGetConcat(derShell->numDots, derShell->dotVecs, der + 1, subvecs, &totalSubvecs);CHKERRQ(ierr);
  if (derShell->rangeIdx < 0) {
    ierr = PetscFnScalarDerivativeScalar(origFn, x, totalDer, totalSubsets, totalSubvecs, z);CHKERRQ(ierr);
  } else if (derShell->rangeIdx < derShell->numDots) {
    ierr = PetscFnDerivativeScalar(origFn, x, totalDer, derShell->rangeIdx, totalSubsets, totalSubvecs, z);CHKERRQ(ierr);
  } else {
    ierr = PetscFnDerivativeScalar(origFn, x, totalDer, rangeIdx + derShell->numDots, totalSubsets, totalSubvecs, z);CHKERRQ(ierr);
  }
  ierr = PetscFnVecsRestoreConcat(derShell->numDots, derShell->dotVecs, der + 1, subvecs, &totalSubvecs);CHKERRQ(ierr);
  ierr = PetscFnISsRestoreConcat(derShell->numDots, derShell->dotISs, der + 1, subsets, &totalSubsets);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnDerivativeVec_DerShell(PetscFn fn, Vec x, PetscInt der, PetscInt rangeIdx, const IS subsets[], const Vec subvecs[], Vec y)
{
  PetscFnDerShell *derShell;
  PetscFn          origFn;
  PetscInt         totalDer;
  IS        *totalSubsets;
  Vec       *totalSubvecs;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscFnShellGetContext(fn, (void **) &derShell);CHKERRQ(ierr);
  totalDer = derShell->der + der;
  origFn = derShell->origFn;
  ierr = PetscFnISsGetConcat(derShell->numDots, derShell->dotISs, der + 1, subsets, &totalSubsets);CHKERRQ(ierr);
  ierr = PetscFnVecsGetConcat(derShell->numDots, derShell->dotVecs, der, subvecs, &totalSubvecs);CHKERRQ(ierr);
  if (derShell->rangeIdx < 0) {
    ierr = PetscFnScalarDerivativeVec(origFn, x, totalDer, totalSubsets, totalSubvecs, y);CHKERRQ(ierr);
  } else if (derShell->rangeIdx < derShell->numDots) {
    ierr = PetscFnDerivativeVec(origFn, x, totalDer, derShell->rangeIdx, totalSubsets, totalSubvecs, y);CHKERRQ(ierr);
  } else {
    ierr = PetscFnDerivativeVec(origFn, x, totalDer, rangeIdx + derShell->numDots, totalSubsets, totalSubvecs, y);CHKERRQ(ierr);
  }
  ierr = PetscFnVecsRestoreConcat(derShell->numDots, derShell->dotVecs, der, subvecs, &totalSubvecs);CHKERRQ(ierr);
  ierr = PetscFnISsRestoreConcat(derShell->numDots, derShell->dotISs, der + 1, subsets, &totalSubsets);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnDerivativeMat_DerShell(PetscFn fn, Vec x, PetscInt der, PetscInt rangeIdx, const IS subsets[], const Vec subvecs[], MatReuse reuse, Mat *M, Mat *Mpre)
{
  PetscFnDerShell *derShell;
  PetscFn          origFn;
  PetscInt         totalDer;
  IS        *totalSubsets;
  Vec       *totalSubvecs;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscFnShellGetContext(fn, (void **) &derShell);CHKERRQ(ierr);
  totalDer = derShell->der + der;
  origFn = derShell->origFn;
  ierr = PetscFnISsGetConcat(derShell->numDots, derShell->dotISs, der + 1, subsets, &totalSubsets);CHKERRQ(ierr);
  ierr = PetscFnVecsGetConcat(derShell->numDots, derShell->dotVecs, der - 1, subvecs, &totalSubvecs);CHKERRQ(ierr);
  if (derShell->rangeIdx < 0) {
    ierr = PetscFnScalarDerivativeMat(origFn, x, totalDer, totalSubsets, totalSubvecs, reuse, M, Mpre);CHKERRQ(ierr);
  } else if (derShell->rangeIdx < derShell->numDots) {
    ierr = PetscFnDerivativeMat(origFn, x, totalDer, derShell->rangeIdx, totalSubsets, totalSubvecs, reuse, M, Mpre);CHKERRQ(ierr);
  } else {
    ierr = PetscFnDerivativeMat(origFn, x, totalDer, rangeIdx + derShell->numDots, totalSubsets, totalSubvecs, reuse, M, Mpre);CHKERRQ(ierr);
  }
  ierr = PetscFnVecsRestoreConcat(derShell->numDots, derShell->dotVecs, der - 1, subvecs, &totalSubvecs);CHKERRQ(ierr);
  ierr = PetscFnISsRestoreConcat(derShell->numDots, derShell->dotISs, der + 1, subsets, &totalSubsets);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnScalarDerivativeScalar_DerShell(PetscFn fn, Vec x, PetscInt der, const IS subsets[], const Vec subvecs[], PetscScalar *z)
{
  PetscFnDerShell *derShell;
  PetscFn          origFn;
  PetscInt         totalDer;
  IS        *totalSubsets;
  Vec       *totalSubvecs;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscFnShellGetContext(fn, (void **) &derShell);CHKERRQ(ierr);
  totalDer = derShell->der + der;
  origFn = derShell->origFn;
  ierr = PetscFnISsGetConcat(derShell->numDots, derShell->dotISs, der, subsets, &totalSubsets);CHKERRQ(ierr);
  ierr = PetscFnVecsGetConcat(derShell->numDots, derShell->dotVecs, der, subvecs, &totalSubvecs);CHKERRQ(ierr);
  if (derShell->rangeIdx < 0) {
    ierr = PetscFnScalarDerivativeScalar(origFn, x, totalDer, totalSubsets, totalSubvecs, z);CHKERRQ(ierr);
  } else {
    ierr = PetscFnDerivativeScalar(origFn, x, totalDer, derShell->rangeIdx, totalSubsets, totalSubvecs, z);CHKERRQ(ierr);
  }
  ierr = PetscFnVecsRestoreConcat(derShell->numDots, derShell->dotVecs, der, subvecs, &totalSubvecs);CHKERRQ(ierr);
  ierr = PetscFnISsRestoreConcat(derShell->numDots, derShell->dotISs, der, subsets, &totalSubsets);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnScalarDerivativeVec_DerShell(PetscFn fn, Vec x, PetscInt der, const IS subsets[], const Vec subvecs[], Vec y)
{
  PetscFnDerShell *derShell;
  PetscFn          origFn;
  PetscInt         totalDer;
  IS        *totalSubsets;
  Vec       *totalSubvecs;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscFnShellGetContext(fn, (void **) &derShell);CHKERRQ(ierr);
  totalDer = derShell->der + der;
  origFn = derShell->origFn;
  ierr = PetscFnISsGetConcat(derShell->numDots, derShell->dotISs, der, subsets, &totalSubsets);CHKERRQ(ierr);
  ierr = PetscFnVecsGetConcat(derShell->numDots, derShell->dotVecs, der - 1, subvecs, &totalSubvecs);CHKERRQ(ierr);
  if (derShell->rangeIdx < 0) {
    ierr = PetscFnScalarDerivativeVec(origFn, x, totalDer, totalSubsets, totalSubvecs, y);CHKERRQ(ierr);
  } else {
    ierr = PetscFnDerivativeVec(origFn, x, totalDer, derShell->rangeIdx, totalSubsets, totalSubvecs, y);CHKERRQ(ierr);
  }
  ierr = PetscFnVecsRestoreConcat(derShell->numDots, derShell->dotVecs, der - 1, subvecs, &totalSubvecs);CHKERRQ(ierr);
  ierr = PetscFnISsRestoreConcat(derShell->numDots, derShell->dotISs, der, subsets, &totalSubsets);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnScalarDerivativeMat_DerShell(PetscFn fn, Vec x, PetscInt der, const IS subsets[], const Vec subvecs[], MatReuse reuse, Mat *M, Mat *Mpre)
{
  PetscFnDerShell *derShell;
  PetscFn          origFn;
  PetscInt         totalDer;
  IS        *totalSubsets;
  Vec       *totalSubvecs;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscFnShellGetContext(fn, (void **) &derShell);CHKERRQ(ierr);
  totalDer = derShell->der + der;
  origFn = derShell->origFn;
  ierr = PetscFnISsGetConcat(derShell->numDots, derShell->dotISs, der, subsets, &totalSubsets);CHKERRQ(ierr);
  ierr = PetscFnVecsGetConcat(derShell->numDots, derShell->dotVecs, der - 2, subvecs, &totalSubvecs);CHKERRQ(ierr);
  if (derShell->rangeIdx < 0) {
    ierr = PetscFnScalarDerivativeMat(origFn, x, totalDer, totalSubsets, totalSubvecs, reuse, M, Mpre);CHKERRQ(ierr);
  } else {
    ierr = PetscFnDerivativeMat(origFn, x, totalDer, derShell->rangeIdx, totalSubsets, totalSubvecs, reuse, M, Mpre);CHKERRQ(ierr);
  }
  ierr = PetscFnVecsRestoreConcat(derShell->numDots, derShell->dotVecs, der - 2, subvecs, &totalSubvecs);CHKERRQ(ierr);
  ierr = PetscFnISsRestoreConcat(derShell->numDots, derShell->dotISs, der, subsets, &totalSubsets);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnCreateDerivativeFn_DerShell(PetscFn fn, PetscFnOperation op, PetscInt numDots, const Vec dotVecs[], PetscFn *derFn)
{
  PetscInt         maxDots;
  PetscInt         derIsScalar;
  PetscInt         i, m, M, der;
  PetscFn          df;
  PetscFnDerShell *derShell;
  PetscLayout      rangeLayout = NULL;
  PetscInt         rangeIdx;
  MPI_Comm         comm;

  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn, PETSCFN_CLASSID, 1);
  comm = PetscObjectComm((PetscObject)fn);
  switch (op) {
  case PETSCFNOP_APPLY:
    rangeLayout = fn->rmap;
    maxDots     = 1;
    rangeIdx    = 0;
    der         = 0;
    break;
  case PETSCFNOP_JACOBIANMULT:
    rangeLayout = fn->rmap;
    maxDots     = 2;
    rangeIdx    = 1;
    der         = 1;
    break;
  case PETSCFNOP_JACOBIANMULTADJOINT:
    rangeLayout = fn->dmap;
    maxDots     = 2;
    rangeIdx    = 0;
    der         = 1;
    break;
  case PETSCFNOP_HESSIANMULT:
    rangeLayout = fn->rmap;
    maxDots     = 3;
    rangeIdx    = 2;
    der         = 2;
    break;
  case PETSCFNOP_HESSIANMULTADJOINT:
    rangeLayout = fn->dmap;
    maxDots     = 3;
    rangeIdx    = 0;
    der         = 2;
    break;
  case PETSCFNOP_SCALARAPPLY:
    rangeLayout = NULL;
    maxDots     = 0;
    rangeIdx    = -1;
    der         = 0;
    break;
  case PETSCFNOP_SCALARGRADIENT:
    rangeLayout = fn->dmap;
    maxDots     = 1;
    rangeIdx    = -1;
    der         = 1;
    break;
  case PETSCFNOP_SCALARHESSIANMULT:
    rangeLayout = fn->dmap;
    maxDots     = 2;
    rangeIdx    = -1;
    der         = 2;
    break;
  default:
    SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "%s cannot be called on this PetscFnOperation", PETSC_FUNCTION_NAME);
  }
  if (numDots < maxDots - 1 || numDots > maxDots) SETERRQ2(comm, PETSC_ERR_ARG_OUTOFRANGE, "This operation contracts with at most %D vectors: %D given", maxDots, numDots);
  if (op == PETSCFNOP_SCALARGRADIENT || (op == PETSCFNOP_APPLY && numDots == 0)) {
    ierr = PetscObjectReference((PetscObject)fn);CHKERRQ(ierr);
    *derFn = fn;
    PetscFunctionReturn(0);
  }
  derIsScalar = (numDots == maxDots) ? PETSC_TRUE : PETSC_FALSE;
  if (derIsScalar) {
    m = PETSC_DECIDE;
    M = 1;
  } else {
    m = rangeLayout->n;
    M = rangeLayout->N;
  }
  ierr = PetscFnCreate(comm, &df);CHKERRQ(ierr);
  ierr = PetscFnSetSizes(df, m, fn->dmap->n, M, fn->dmap->N);CHKERRQ(ierr);
  ierr = PetscLayoutReference(fn->dmap,&(df->dmap));CHKERRQ(ierr);
  if (!derIsScalar) {
    ierr = PetscLayoutReference(rangeLayout,&(df->rmap));CHKERRQ(ierr);
  }
  ierr = PetscFnSetType(df, PETSCFNSHELL);CHKERRQ(ierr);
  ierr = PetscNewLog(df, &derShell);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)fn);CHKERRQ(ierr);
  derShell->origFn   = fn;
  derShell->op       = op;
  derShell->numDots  = numDots;
  derShell->maxDots  = maxDots;
  derShell->dotISs   = NULL;
  derShell->rangeIdx = rangeIdx;
  derShell->der      = der;
  for (i = 0; i < numDots; i++) {
    ierr = PetscObjectReference((PetscObject)dotVecs[i]);CHKERRQ(ierr);
    derShell->dotVecs[i] = dotVecs[i];
  }
  derShell->workVecs[0] = NULL;
  if (numDots) {ierr = VecDuplicate(dotVecs[numDots - 1], &(derShell->workVecs[0]));CHKERRQ(ierr);}
  ierr = PetscFnShellSetContext(df, (void *) derShell);CHKERRQ(ierr);
  ierr = PetscFnShellSetOperation(df, PETSCFNOP_DESTROY,                (void (*)(void)) PetscFnShellDestroy_DerShell);CHKERRQ(ierr);
  ierr = PetscFnShellSetOperation(df, PETSCFNOP_CREATEVECS,             (void (*)(void)) PetscFnShellCreateVecs_DerShell);CHKERRQ(ierr);
  ierr = PetscFnShellSetOperation(df, PETSCFNOP_APPLY,                  (void (*)(void)) PetscFnShellApply_DerShell);CHKERRQ(ierr);
  ierr = PetscFnShellSetOperation(df, PETSCFNOP_JACOBIANMULT,           (void (*)(void)) PetscFnShellJacobianMult_DerShell);CHKERRQ(ierr);
  ierr = PetscFnShellSetOperation(df, PETSCFNOP_JACOBIANMULTADJOINT,    (void (*)(void)) PetscFnShellJacobianMultAdjoint_DerShell);CHKERRQ(ierr);
  ierr = PetscFnShellSetOperation(df, PETSCFNOP_HESSIANMULT,            (void (*)(void)) PetscFnShellHessianMult_DerShell);CHKERRQ(ierr);
  ierr = PetscFnShellSetOperation(df, PETSCFNOP_HESSIANMULTADJOINT,     (void (*)(void)) PetscFnShellHessianMultAdjoint_DerShell);CHKERRQ(ierr);
  if (derIsScalar) {
    ierr = PetscFnShellSetOperation(df, PETSCFNOP_SCALARAPPLY,            (void (*)(void)) PetscFnShellScalarApply_DerShell);CHKERRQ(ierr);
    ierr = PetscFnShellSetOperation(df, PETSCFNOP_SCALARGRADIENT,         (void (*)(void)) PetscFnShellScalarGradient_DerShell);CHKERRQ(ierr);
    ierr = PetscFnShellSetOperation(df, PETSCFNOP_SCALARHESSIANMULT,      (void (*)(void)) PetscFnShellScalarHessianMult_DerShell);CHKERRQ(ierr);
    ierr = PetscFnShellSetOperation(df, PETSCFNOP_SCALARHESSIANBUILD,     (void (*)(void)) PetscFnShellScalarHessianBuild_DerShell);CHKERRQ(ierr);
    ierr = PetscFnShellSetOperation(df, PETSCFNOP_SCALARDERIVATIVESCALAR, (void (*)(void)) PetscFnScalarDerivativeScalar_DerShell);CHKERRQ(ierr);
    ierr = PetscFnShellSetOperation(df, PETSCFNOP_SCALARDERIVATIVEVEC,    (void (*)(void)) PetscFnScalarDerivativeVec_DerShell);CHKERRQ(ierr);
    ierr = PetscFnShellSetOperation(df, PETSCFNOP_SCALARDERIVATIVEMAT,    (void (*)(void)) PetscFnScalarDerivativeMat_DerShell);CHKERRQ(ierr);
  } else {
    ierr = PetscFnShellSetOperation(df, PETSCFNOP_JACOBIANBUILD,        (void (*)(void)) PetscFnShellJacobianBuild_DerShell);CHKERRQ(ierr);
    ierr = PetscFnShellSetOperation(df, PETSCFNOP_JACOBIANBUILDADJOINT, (void (*)(void)) PetscFnShellJacobianBuildAdjoint_DerShell);CHKERRQ(ierr);
    ierr = PetscFnShellSetOperation(df, PETSCFNOP_DERIVATIVESCALAR, (void (*)(void)) PetscFnDerivativeScalar_DerShell);CHKERRQ(ierr);
    ierr = PetscFnShellSetOperation(df, PETSCFNOP_DERIVATIVEVEC,    (void (*)(void)) PetscFnDerivativeVec_DerShell);CHKERRQ(ierr);
    ierr = PetscFnShellSetOperation(df, PETSCFNOP_DERIVATIVEMAT,    (void (*)(void)) PetscFnDerivativeMat_DerShell);CHKERRQ(ierr);
  }
  *derFn = df;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnCreateDerivativeFn(PetscFn fn, PetscFnOperation op, PetscInt numDots, const Vec dotVecs[], PetscFn *derFn)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn, PETSCFN_CLASSID, 1);
  ierr = PetscFnSetUp(fn);CHKERRQ(ierr);
  if (fn->ops->createderivativefn) {
    ierr = (*fn->ops->createderivativefn) (fn, op, numDots, dotVecs, derFn);CHKERRQ(ierr);
  } else {
    ierr = PetscFnCreateDerivativeFn_DerShell(fn, op, numDots, dotVecs, derFn);CHKERRQ(ierr);
  }
  if (fn->test_derfn) {
    (*derFn)->test_self_as_derfn = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}
