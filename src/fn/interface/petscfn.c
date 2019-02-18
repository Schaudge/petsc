
/*
   This is where the abstract PetscFn operations are defined
*/

#include <petsc/private/vecimpl.h>       /*I "petscvec.h" I*/
#include <petsc/private/matimpl.h>       /*I "petscmat.h" I*/
#include <petsc/private/fnimpl.h>        /*I "petscfn.h" I*/

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
  ierr = PetscStrallocpy(VECSTANDARD,(char**)&B->rangeType);CHKERRQ(ierr);
  ierr = PetscStrallocpy(VECSTANDARD,(char**)&B->domainType);CHKERRQ(ierr);
  ierr = PetscStrallocpy(MATAIJ,(char**)&B->jacType);CHKERRQ(ierr);
  ierr = PetscStrallocpy(MATAIJ,(char**)&B->jacPreType);CHKERRQ(ierr);
  ierr = PetscStrallocpy(MATAIJ,(char**)&B->adjType);CHKERRQ(ierr);
  ierr = PetscStrallocpy(MATAIJ,(char**)&B->adjPreType);CHKERRQ(ierr);
  ierr = PetscStrallocpy(MATAIJ,(char**)&B->hesType);CHKERRQ(ierr);
  ierr = PetscStrallocpy(MATAIJ,(char**)&B->hesPreType);CHKERRQ(ierr);

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

  ierr = PetscFree((*fn)->rangeType);CHKERRQ(ierr);
  ierr = PetscFree((*fn)->domainType);CHKERRQ(ierr);
  ierr = PetscFree((*fn)->jacType);CHKERRQ(ierr);
  ierr = PetscFree((*fn)->jacPreType);CHKERRQ(ierr);
  ierr = PetscFree((*fn)->adjType);CHKERRQ(ierr);
  ierr = PetscFree((*fn)->adjPreType);CHKERRQ(ierr);
  ierr = PetscFree((*fn)->hesType);CHKERRQ(ierr);
  ierr = PetscFree((*fn)->hesPreType);CHKERRQ(ierr);

  ierr = PetscLayoutDestroy(&(*fn)->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutDestroy(&(*fn)->dmap);CHKERRQ(ierr);
  ierr = PetscHeaderDestroy(fn);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnSetSizes(PetscFn fn, PetscInt m, PetscInt M, PetscInt n, PetscInt N)
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

PetscErrorCode PetscFnGetSize(PetscFn fn, PetscInt *m, PetscInt *n)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn,PETSCFN_CLASSID,1);
  if (m) *m = fn->rmap->N;
  if (n) *n = fn->dmap->N;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnGetLocalSize(PetscFn fn, PetscInt *m, PetscInt *n)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn,PETSCFN_CLASSID,1);
  if (m) PetscValidIntPointer(m,2);
  if (n) PetscValidIntPointer(n,3);
  if (m) *m = fn->rmap->n;
  if (n) *n = fn->dmap->n;
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
  PetscBool      flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn,PETSCFN_CLASSID,1);

  ierr = PetscObjectOptionsBegin((PetscObject)fn);CHKERRQ(ierr);

  ierr = PetscOptionsFList("-fn_type","Function type","PetscFnSetType",PetscFnList,deft,type,256,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscFnSetType(fn,type);CHKERRQ(ierr);
  } else if (!((PetscObject)fn)->type_name) {
    ierr = PetscFnSetType(fn,deft);CHKERRQ(ierr);
  }

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
  ierr = PetscLayoutSetUp(fn->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(fn->dmap);CHKERRQ(ierr);
  fn->isScalar = PETSC_FALSE;
  if (fn->dmap->N == 1) fn->isScalar = PETSC_TRUE;
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
    if (format == PETSC_VIEWER_ASCII_INFO || format == PETSC_VIEWER_ASCII_INFO_DETAIL) {

      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = PetscFnGetSize(fn,&rows,&cols);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"range size=%D, domain size=%D\n",rows,cols);CHKERRQ(ierr);
    }
  }
  if (fn->ops->view) {
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = (*fn->ops->view)(fn,viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  }
  if (iascii) {
    ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
    if (format == PETSC_VIEWER_ASCII_INFO || format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnSetVecTypes(PetscFn fn, VecType rangeType, VecType domainType)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn,PETSCFN_CLASSID,1);
  if (rangeType) {
    ierr = PetscFree(fn->rangeType);CHKERRQ(ierr);
    ierr = PetscStrallocpy(rangeType,(char**)&fn->rangeType);CHKERRQ(ierr);
  }
  if (domainType) {
    ierr = PetscFree(fn->domainType);CHKERRQ(ierr);
    ierr = PetscStrallocpy(domainType,(char**)&fn->domainType);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnGetVecTypes(PetscFn fn, VecType *rangeType, VecType *domainType)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn,PETSCFN_CLASSID,1);
  if (rangeType) {
    *rangeType = fn->rangeType;
  }
  if (domainType) {
    *domainType = fn->domainType;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnSetJacobianMatTypes(PetscFn fn, MatType jacType, MatType jacPreType, MatType adjType, MatType adjPreType)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn,PETSCFN_CLASSID,1);
  if (jacType) {
    ierr = PetscFree(fn->jacType);CHKERRQ(ierr);
    ierr = PetscStrallocpy(jacType,(char**)&fn->jacType);CHKERRQ(ierr);
  }
  if (jacPreType) {
    ierr = PetscFree(fn->jacPreType);CHKERRQ(ierr);
    ierr = PetscStrallocpy(jacPreType,(char**)&fn->jacPreType);CHKERRQ(ierr);
  }
  if (adjType) {
    ierr = PetscFree(fn->adjType);CHKERRQ(ierr);
    ierr = PetscStrallocpy(adjType,(char**)&fn->adjType);CHKERRQ(ierr);
  }
  if (adjPreType) {
    ierr = PetscFree(fn->adjPreType);CHKERRQ(ierr);
    ierr = PetscStrallocpy(adjPreType,(char**)&fn->adjPreType);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnGetJacobianMatTypes(PetscFn fn, MatType *jacType, MatType *jacPreType, MatType *adjType, MatType *adjPreType)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn,PETSCFN_CLASSID,1);
  if (jacType) {
    *jacType = fn->jacType;
  }
  if (jacPreType) {
    *jacPreType = fn->jacPreType;
  }
  if (adjType) {
    *adjType = fn->adjType;
  }
  if (adjPreType) {
    *adjPreType = fn->adjPreType;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnSetHessianMatTypes(PetscFn fn, MatType hesType, MatType hesPreType)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn,PETSCFN_CLASSID,1);
  if (hesType) {
    ierr = PetscFree(fn->hesType);CHKERRQ(ierr);
    ierr = PetscStrallocpy(hesType,(char**)&fn->hesType);CHKERRQ(ierr);
  }
  if (hesPreType) {
    ierr = PetscFree(fn->hesPreType);CHKERRQ(ierr);
    ierr = PetscStrallocpy(hesPreType,(char**)&fn->hesPreType);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnGetHessianMatTypes(PetscFn fn, MatType *hesType, MatType *hesPreType)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn,PETSCFN_CLASSID,1);
  if (hesType) {
    *hesType = fn->hesType;
  }
  if (hesPreType) {
    *hesPreType = fn->hesPreType;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnCreateVecs(PetscFn fn, Vec *rangeVec, Vec *domainVec)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn,PETSCFN_CLASSID,1);
  if (fn->ops->createvecs) {
    ierr = (*(fn->ops->createvecs)) (fn, rangeVec, domainVec);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
    if (rangeVec) {
      VecType   type = fn->rangeType;
      VecType   rettype;
      PetscBool same,standard;
      PetscLayout retLayout;

      ierr = VecGetType(*rangeVec, &rettype);CHKERRQ(ierr);
      ierr = PetscStrcmp(type,VECSTANDARD,&standard);CHKERRQ(ierr);
      if (standard) {
        PetscInt size;

        ierr = MPI_Comm_size(PetscObjectComm((PetscObject)fn), &size);CHKERRQ(ierr);
        type = size > 1 ? VECMPI : VECSEQ;
      }
      ierr = PetscStrcmp(rettype,type,&same);CHKERRQ(ierr);
      if (!same) SETERRQ2(PetscObjectComm((PetscObject)fn),PETSC_ERR_USER,"User supplied PETSCFNOP_CREATEVECS returned type %s, not %s set with PetscFnSetVecTypes()", rettype, type);
      ierr = VecGetLayout(*rangeVec, &retLayout);CHKERRQ(ierr);
      ierr = PetscLayoutCompare(retLayout, fn->rmap, &same);CHKERRQ(ierr);
      if (!same) SETERRQ(PetscObjectComm((PetscObject)fn),PETSC_ERR_USER,"User supplied PETSCFNOP_CREATEVECS returned vec of wrong shape");
    }
    if (domainVec) {
      VecType   type = fn->domainType;
      VecType   rettype;
      PetscBool same,standard;
      PetscLayout retLayout;

      ierr = VecGetType(*domainVec, &rettype);CHKERRQ(ierr);
      ierr = PetscStrcmp(type,VECSTANDARD,&standard);CHKERRQ(ierr);
      if (standard) {
        PetscInt size;

        ierr = MPI_Comm_size(PetscObjectComm((PetscObject)fn), &size);CHKERRQ(ierr);
        type = size > 1 ? VECMPI : VECSEQ;
      }
      ierr = PetscStrcmp(rettype,type,&same);CHKERRQ(ierr);
      if (!same) SETERRQ2(PetscObjectComm((PetscObject)fn),PETSC_ERR_USER,"User supplied PETSCFNOP_CREATEVECS returned type %s, not %s set with PetscFnSetVecTypes()", rettype, type);
      ierr = VecGetLayout(*domainVec, &retLayout);CHKERRQ(ierr);
      ierr = PetscLayoutCompare(retLayout, fn->dmap, &same);CHKERRQ(ierr);
      if (!same) SETERRQ(PetscObjectComm((PetscObject)fn),PETSC_ERR_USER,"User supplied PETSCFNOP_CREATEVECS returned vec of wrong shape");
    }
#endif
  } else {
    if (rangeVec) {
      ierr = VecCreate(PetscObjectComm((PetscObject)fn),rangeVec);CHKERRQ(ierr);
      ierr = VecSetLayout(*rangeVec,fn->rmap);CHKERRQ(ierr);
    }
    if (domainVec) {
      ierr = VecCreate(PetscObjectComm((PetscObject)fn),domainVec);CHKERRQ(ierr);
      ierr = VecSetLayout(*domainVec,fn->dmap);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnCreateMats(PetscFn fn, Mat *jac, Mat *jacPre, Mat *adj, Mat *adjPre, Mat *hes, Mat *hesPre)
{
  PetscInt       i;
  MatType        types[6];
  PetscLayout    layouts[3][2];
  Mat*           mats[6];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn,PETSCFN_CLASSID,1);
  types[0] = fn->jacType;
  types[1] = fn->jacPreType;
  types[2] = fn->adjType;
  types[3] = fn->adjPreType;
  types[4] = fn->hesType;
  types[5] = fn->hesPreType;
  layouts[0][0] = fn->rmap;
  layouts[0][1] = fn->dmap;
  layouts[1][0] = fn->dmap;
  layouts[1][1] = fn->rmap;
  layouts[2][0] = fn->dmap;
  layouts[2][1] = fn->dmap;
  if (fn->ops->createmats) {
    ierr = (*(fn->ops->createmats)) (fn, jac, jacPre, adj, adjPre, hes, hesPre);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
    mats[0] = jac;
    mats[1] = jacPre;
    mats[2] = adj;
    mats[3] = adjPre;
    mats[4] = hes;
    mats[5] = hesPre;
    for (i = 0; i < 6; i++) {
      PetscBool same;
      MatType   rettype;
      Mat       dummy;
      PetscLayout rowmap, colmap;

      if (!mats[i]) continue;
      ierr = MatCreate(PetscObjectComm((PetscObject)fn), &dummy);CHKERRQ(ierr);
      ierr = MatSetType(dummy, types[i]);CHKERRQ(ierr);
      ierr = MatGetType(dummy, &types[i]);CHKERRQ(ierr);
      ierr = MatGetType(*(mats[i]), &rettype);CHKERRQ(ierr);
      ierr = PetscObjectTypeCompare((PetscObject) *(mats[i]), types[i], &same);CHKERRQ(ierr);
      if (!same) SETERRQ2(PetscObjectComm((PetscObject)fn),PETSC_ERR_USER,"User supplied PETSCFNOP_CREATEMATS returned type %s, not %s", rettype, types[i]);
      ierr = MatGetLayouts(*(mats[i]), &rowmap, &colmap);CHKERRQ(ierr);
      same = (layouts[i/2][0]->n == rowmap->n && layouts[i/2][0]->N == rowmap->N) ? PETSC_TRUE : PETSC_FALSE;
      if (!same) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"User supplied PETSCFNOP_CREATEMATS returned mat of wrong row shape");
      same = (layouts[i/2][1]->n == colmap->n && layouts[i/2][1]->N == colmap->N) ? PETSC_TRUE : PETSC_FALSE;
      if (!same) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"User supplied PETSCFNOP_CREATEMATS returned mat of wrong column shape");
      ierr = MatDestroy(&dummy);CHKERRQ(ierr);
    }
#endif
  }
  else {
    mats[0] = jac;
    mats[1] = jacPre;
    mats[2] = adj;
    mats[3] = adjPre;
    mats[4] = hes;
    mats[5] = hesPre;
    for (i = 0; i < 6; i++) {
      PetscInt m, M, n, N;
      if (!mats[i]) continue;
      ierr = MatCreate(PetscObjectComm((PetscObject)fn),mats[i]);CHKERRQ(ierr);
      ierr = MatSetType(*(mats[i]),types[i]);CHKERRQ(ierr);
      ierr = PetscLayoutGetSize(layouts[i/2][0],&N);CHKERRQ(ierr);
      ierr = PetscLayoutGetLocalSize(layouts[i/2][0],&n);CHKERRQ(ierr);
      ierr = PetscLayoutGetSize(layouts[i/2][1],&M);CHKERRQ(ierr);
      ierr = PetscLayoutGetLocalSize(layouts[i/2][1],&m);CHKERRQ(ierr);
      ierr = MatSetSizes(*(mats[i]),n,m,N,M);CHKERRQ(ierr);
    }
  }
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
  if (!fn->ops->apply) {
    ierr = (*fn->ops->apply)(fn,x,y);CHKERRQ(ierr);
  } else if (fn->isScalar && fn->ops->scalarapply) {
    PetscScalar z;

    ierr = (*fn->ops->scalarapply)(fn,x,&z);CHKERRQ(ierr);
    ierr = VecSet(y,z);CHKERRQ(ierr);
  } else SETERRQ1(PetscObjectComm((PetscObject)fn), PETSC_ERR_SUP, "This PetscFn does not implement %s()", PETSC_FUNCTION_NAME);
  ierr = VecLockPop(x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnJacobianMult(PetscFn fn, Vec x, Vec xhat, Vec Jxhat)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn,PETSCFN_CLASSID,1);
  PetscValidType(fn,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  PetscValidHeaderSpecific(xhat,VEC_CLASSID,3);
  PetscValidHeaderSpecific(Jxhat,VEC_CLASSID,4);
  ierr = PetscFnSetUp(fn);CHKERRQ(ierr);
  if (x == xhat) SETERRQ(PetscObjectComm((PetscObject)fn),PETSC_ERR_ARG_WRONGSTATE,"x and xhat must be different vectors");
  if (x == Jxhat) SETERRQ(PetscObjectComm((PetscObject)fn),PETSC_ERR_ARG_WRONGSTATE,"x and Jxhat must be different vectors");
  if (xhat == Jxhat) SETERRQ(PetscObjectComm((PetscObject)fn),PETSC_ERR_ARG_WRONGSTATE,"xhat and Jxhat must be different vectors");
  if (fn->dmap->N != x->map->N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"PetscFn fn,Vec x: global dim %D %D",fn->dmap->N,x->map->N);
  if (fn->dmap->N != xhat->map->N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"PetscFn fn,Vec xhat: global dim %D %D",fn->dmap->N,xhat->map->N);
  if (fn->rmap->N != Jxhat->map->N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"PetscFn fn,Vec Jxhat: global dim %D %D",fn->rmap->N,Jxhat->map->N);
  if (fn->rmap->n != Jxhat->map->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"PetscFn fn,Vec Jxhat: local dim %D %D",fn->rmap->n,Jxhat->map->n);
  VecLocked(Jxhat,4);

  ierr = VecLockPush(x);CHKERRQ(ierr);
  ierr = VecLockPush(xhat);CHKERRQ(ierr);
  if (fn->ops->jacobianmult) {
    ierr = (*fn->ops->jacobianmult)(fn,x,xhat,Jxhat);CHKERRQ(ierr);
  } else if (fn->isScalar && fn->ops->scalargradient) {
    Vec         g;
    PetscScalar z;

    ierr = VecDuplicate(x, &g);CHKERRQ(ierr);
    ierr = (*fn->ops->scalargradient) (fn, x, g);CHKERRQ(ierr);
    ierr = VecDot(g, xhat, &z);CHKERRQ(ierr);
    ierr = VecSet(Jxhat, z);CHKERRQ(ierr);
    ierr = VecDestroy(&g);CHKERRQ(ierr);
  } else SETERRQ1(PetscObjectComm((PetscObject)fn), PETSC_ERR_SUP, "This PetscFn does not implement %s()", PETSC_FUNCTION_NAME);
  ierr = VecLockPop(xhat);CHKERRQ(ierr);
  ierr = VecLockPop(x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnVecScalarBcast(Vec v, PetscScalar *zp)
{
  MPI_Comm       comm;
  PetscLayout    map;
  PetscMPIInt    rank;
  PetscInt       broot;
  PetscScalar    z;
  const PetscScalar *zv;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  comm = PetscObjectComm((PetscObject)v);
  ierr = VecGetLayout(v, &map);CHKERRQ(ierr);
  ierr = PetscLayoutFindOwner(map, 0, &broot);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = VecGetArrayRead(v, &zv);CHKERRQ(ierr);
  z    = ((PetscInt) broot == rank) ? zv[0] : 0.;
  ierr = VecRestoreArrayRead(v, &zv);CHKERRQ(ierr);
  ierr = MPI_Bcast(&z, 1, MPIU_REAL, broot, comm);CHKERRQ(ierr);
  *zp  = z;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnJacobianMultAdjoint(PetscFn fn, Vec x, Vec v, Vec JTv)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn,PETSCFN_CLASSID,1);
  PetscValidType(fn,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  PetscValidHeaderSpecific(v,VEC_CLASSID,3);
  PetscValidHeaderSpecific(JTv,VEC_CLASSID,4);
  ierr = PetscFnSetUp(fn);CHKERRQ(ierr);
  if (x == v) SETERRQ(PetscObjectComm((PetscObject)fn),PETSC_ERR_ARG_WRONGSTATE,"x and v must be different vectors");
  if (x == JTv) SETERRQ(PetscObjectComm((PetscObject)fn),PETSC_ERR_ARG_WRONGSTATE,"x and JTv must be different vectors");
  if (v == JTv) SETERRQ(PetscObjectComm((PetscObject)fn),PETSC_ERR_ARG_WRONGSTATE,"v and JTv must be different vectors");
  if (fn->dmap->N != x->map->N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"PetscFn fn,Vec x: global dim %D %D",fn->dmap->N,x->map->N);
  if (fn->rmap->N != v->map->N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"PetscFn fn,Vec v: global dim %D %D",fn->rmap->N,v->map->N);
  if (fn->dmap->N != JTv->map->N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"PetscFn fn,Vec JTv: global dim %D %D",fn->dmap->N,JTv->map->N);
  if (fn->dmap->n != JTv->map->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"PetscFn fn,Vec JTv: local dim %D %D",fn->dmap->n,JTv->map->n);
  VecLocked(JTv,4);

  ierr = VecLockPush(x);CHKERRQ(ierr);
  ierr = VecLockPush(v);CHKERRQ(ierr);
  if (fn->ops->jacobianmultadjoint) {
    ierr = (*fn->ops->jacobianmultadjoint)(fn,x,v,JTv);CHKERRQ(ierr);
  } else if (fn->isScalar && fn->ops->scalargradient) {
    PetscScalar z;

    ierr = (*fn->ops->scalargradient) (fn, x, JTv); CHKERRQ(ierr);
    ierr = PetscFnVecScalarBcast(v, &z);CHKERRQ(ierr);
    ierr = VecScale(JTv, z);CHKERRQ(ierr);
  } else SETERRQ1(PetscObjectComm((PetscObject)fn), PETSC_ERR_SUP, "This PetscFn does not implement %s()", PETSC_FUNCTION_NAME);
  ierr = VecLockPop(v);CHKERRQ(ierr);
  ierr = VecLockPop(x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnJacobianCreate(PetscFn fn, Vec x, Mat J, Mat Jpre)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn,PETSCFN_CLASSID,1);
  PetscValidType(fn,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  ierr = PetscFnSetUp(fn);CHKERRQ(ierr);
  if (J) {
    PetscValidHeaderSpecific(J,MAT_CLASSID,3);
    if (fn->dmap->N != J->cmap->N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"PetscFn fn,Mat J: global domain/column dim %D %D",fn->dmap->N,J->cmap->N);
    if (fn->dmap->n != J->cmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"PetscFn fn,Mat J: local domain/column dim %D %D",fn->dmap->n,J->cmap->n);
    if (fn->rmap->N != J->rmap->N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"PetscFn fn,Mat J: global range/row dim %D %D",fn->rmap->N,J->rmap->N);
    if (fn->rmap->n != J->rmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"PetscFn fn,Mat J: local range/row dim %D %D",fn->rmap->n,J->rmap->n);
  }
  if (Jpre) {
    PetscValidHeaderSpecific(Jpre,VEC_CLASSID,4);
    if (fn->dmap->N != Jpre->cmap->N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"PetscFn fn,Mat Jpre: global domain/column dim %D %D",fn->dmap->N,Jpre->cmap->N);
    if (fn->dmap->n != Jpre->cmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"PetscFn fn,Mat Jpre: local domain/column dim %D %D",fn->dmap->n,Jpre->cmap->n);
    if (fn->rmap->N != Jpre->rmap->N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"PetscFn fn,Mat Jpre: global range/row dim %D %D",fn->rmap->N,Jpre->rmap->N);
    if (fn->rmap->n != Jpre->rmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"PetscFn fn,Mat Jpre: local range/row dim %D %D",fn->rmap->n,Jpre->rmap->n);
  }
  if (!J && !Jpre) PetscFunctionReturn(0);
  ierr = VecLockPush(x);CHKERRQ(ierr);
  if (fn->ops->jacobiancreate) {
    ierr = (*fn->ops->jacobiancreate)(fn,x,J,Jpre);CHKERRQ(ierr);
  } else if (fn->isScalar && fn->ops->scalargradient) {
    Mat                jac = J ? J : Jpre;
    const PetscScalar *ga;
    Vec                g;
    PetscInt           zero = 0;
    PetscInt           i, iStart, iEnd, *ia;

    ierr = VecDuplicate(x, &g);CHKERRQ(ierr);
    ierr = (*fn->ops->scalargradient) (fn, x, g); CHKERRQ(ierr);
    ierr = VecGetOwnershipRange(g, &iStart, &iEnd);CHKERRQ(ierr);
    ierr = PetscMalloc1(iEnd - iStart, &ia);CHKERRQ(ierr);
    for (i = 0; i < iEnd - iStart; i++) ia[i] = i + iStart;
    ierr = VecGetArrayRead(g, &ga);CHKERRQ(ierr);
    ierr = MatSetValues(jac, 1, &zero, iEnd - iStart, ia, ga, INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(g, &ga);CHKERRQ(ierr);
    ierr = PetscFree(ia);CHKERRQ(ierr);
    ierr = VecDestroy(&g);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    if (J && J != jac) {ierr = MatCopy(jac, J, DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);}
    if (Jpre && Jpre != jac) {ierr = MatCopy(jac, Jpre, DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);}
  } else SETERRQ1(PetscObjectComm((PetscObject)fn), PETSC_ERR_SUP, "This PetscFn does not implement %s()", PETSC_FUNCTION_NAME);
  ierr = VecLockPop(x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnJacobianCreateAdjoint(PetscFn fn, Vec x, Mat A, Mat Apre)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn,PETSCFN_CLASSID,1);
  PetscValidType(fn,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  ierr = PetscFnSetUp(fn);CHKERRQ(ierr);
  if (A) {
    PetscValidHeaderSpecific(A,MAT_CLASSID,3);
    if (fn->dmap->N != A->rmap->N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"PetscFn fn,Mat A: global domain/row dim %D %D",fn->dmap->N,A->rmap->N);
    if (fn->dmap->n != A->rmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"PetscFn fn,Mat A: local domain/row dim %D %D",fn->dmap->n,A->rmap->n);
    if (fn->rmap->N != A->cmap->N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"PetscFn fn,Mat A: global range/column dim %D %D",fn->rmap->N,A->cmap->N);
    if (fn->rmap->n != A->cmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"PetscFn fn,Mat A: local range/column dim %D %D",fn->rmap->n,A->cmap->n);
  }
  if (Apre) {
    PetscValidHeaderSpecific(Apre,VEC_CLASSID,4);
    if (fn->dmap->N != Apre->rmap->N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"PetscFn fn,Mat Apre: global domain/row dim %D %D",fn->dmap->N,Apre->rmap->N);
    if (fn->dmap->n != Apre->rmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"PetscFn fn,Mat Apre: local domain/row dim %D %D",fn->dmap->n,Apre->rmap->n);
    if (fn->rmap->N != Apre->cmap->N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"PetscFn fn,Mat Apre: global range/column dim %D %D",fn->rmap->N,Apre->cmap->N);
    if (fn->rmap->n != Apre->cmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"PetscFn fn,Mat Apre: local range/column dim %D %D",fn->rmap->n,Apre->cmap->n);
  }
  if (!A && !Apre) PetscFunctionReturn(0);
  ierr = VecLockPush(x);CHKERRQ(ierr);
  if (fn->ops->jacobiancreateadjoint) {
    ierr = (*fn->ops->jacobiancreateadjoint)(fn,x,A,Apre);CHKERRQ(ierr);
  } else if (fn->isScalar && fn->ops->scalargradient) {
    Mat                adj = A ? A : Apre;
    const PetscScalar *ga;
    Vec                g;
    PetscInt           zero = 0;
    PetscInt           i, iStart, iEnd, *ia;

    ierr = VecDuplicate(x, &g);CHKERRQ(ierr);
    ierr = (*fn->ops->scalargradient) (fn, x, g); CHKERRQ(ierr);
    ierr = VecGetOwnershipRange(g, &iStart, &iEnd);CHKERRQ(ierr);
    ierr = PetscMalloc1(iEnd - iStart, &ia);CHKERRQ(ierr);
    for (i = 0; i < iEnd - iStart; i++) ia[i] = i + iStart;
    ierr = VecGetArrayRead(g, &ga);CHKERRQ(ierr);
    ierr = MatSetValues(adj, iEnd - iStart, ia, 1, &zero, ga, INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(g, &ga);CHKERRQ(ierr);
    ierr = PetscFree(ia);CHKERRQ(ierr);
    ierr = VecDestroy(&g);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(adj,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(adj,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    if (A && A != adj) {ierr = MatCopy(adj, A, DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);}
    if (Apre && Apre != adj) {ierr = MatCopy(adj, Apre, DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);}
  } else SETERRQ1(PetscObjectComm((PetscObject)fn), PETSC_ERR_SUP, "This PetscFn does not implement %s()", PETSC_FUNCTION_NAME);
  ierr = VecLockPop(x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnHessianMult(PetscFn fn, Vec x, Vec v, Vec xhat, Vec vHxhat)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn,PETSCFN_CLASSID,1);
  PetscValidType(fn,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  PetscValidHeaderSpecific(v,VEC_CLASSID,3);
  PetscValidHeaderSpecific(xhat,VEC_CLASSID,4);
  PetscValidHeaderSpecific(vHxhat,VEC_CLASSID,4);
  ierr = PetscFnSetUp(fn);CHKERRQ(ierr);
  if (x == vHxhat) SETERRQ(PetscObjectComm((PetscObject)fn),PETSC_ERR_ARG_WRONGSTATE,"x and vHxhat must be different vectors");
  if (v == vHxhat) SETERRQ(PetscObjectComm((PetscObject)fn),PETSC_ERR_ARG_WRONGSTATE,"v and vHxhat must be different vectors");
  if (xhat == vHxhat) SETERRQ(PetscObjectComm((PetscObject)fn),PETSC_ERR_ARG_WRONGSTATE,"xhat and vHxhat must be different vectors");
  if (fn->dmap->N != x->map->N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"PetscFn fn,Vec x: global dim %D %D",fn->dmap->N,x->map->N);
  if (fn->rmap->N != v->map->N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"PetscFn fn,Vec v: global dim %D %D",fn->rmap->N,v->map->N);
  if (fn->dmap->N != xhat->map->N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"PetscFn fn,Vec xhat: global dim %D %D",fn->dmap->N,xhat->map->N);
  if (fn->dmap->N != vHxhat->map->N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"PetscFn fn,Vec vHxhat: global dim %D %D",fn->dmap->N,vHxhat->map->N);
  if (fn->dmap->n != vHxhat->map->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"PetscFn fn,Vec vHxhat: local dim %D %D",fn->dmap->n,vHxhat->map->n);
  VecLocked(vHxhat,5);

  ierr = VecLockPush(x);CHKERRQ(ierr);
  ierr = VecLockPush(v);CHKERRQ(ierr);
  ierr = VecLockPush(xhat);CHKERRQ(ierr);
  if (fn->ops->hessianmult) {
    ierr = (*fn->ops->hessianmult)(fn,x,v,xhat,vHxhat);CHKERRQ(ierr);
  } else if (fn->isScalar && fn->ops->scalarhessianmult) {
    PetscScalar z;

    ierr = (*fn->ops->scalarhessianmult) (fn, x, xhat, vHxhat); CHKERRQ(ierr);
    ierr = PetscFnVecScalarBcast(v, &z);CHKERRQ(ierr);
    ierr = VecScale(vHxhat, z);CHKERRQ(ierr);
  } else SETERRQ1(PetscObjectComm((PetscObject)fn), PETSC_ERR_SUP, "This PetscFn does not implement %s()", PETSC_FUNCTION_NAME);
  ierr = VecLockPop(xhat);CHKERRQ(ierr);
  ierr = VecLockPop(v);CHKERRQ(ierr);
  ierr = VecLockPop(x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnHessianCreate(PetscFn fn, Vec x, Vec v, Mat H, Mat Hpre)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn,PETSCFN_CLASSID,1);
  PetscValidType(fn,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  PetscValidHeaderSpecific(v,VEC_CLASSID,3);
  ierr = PetscFnSetUp(fn);CHKERRQ(ierr);
  if (H) {
    PetscValidHeaderSpecific(H,MAT_CLASSID,4);
    if (fn->dmap->N != H->cmap->N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"PetscFn fn,Mat H: global domain/column dim %D %D",fn->dmap->N,H->cmap->N);
    if (fn->dmap->n != H->cmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"PetscFn fn,Mat H: local domain/column dim %D %D",fn->dmap->n,H->cmap->n);
    if (fn->dmap->N != H->rmap->N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"PetscFn fn,Mat H: global domain/row dim %D %D",fn->dmap->N,H->rmap->N);
    if (fn->dmap->n != H->rmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"PetscFn fn,Mat H: local domain/row dim %D %D",fn->dmap->n,H->rmap->n);
  }
  if (Hpre) {
    PetscValidHeaderSpecific(Hpre,VEC_CLASSID,5);
    if (fn->dmap->N != Hpre->cmap->N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"PetscFn fn,Mat Hpre: global domain/column dim %D %D",fn->dmap->N,Hpre->cmap->N);
    if (fn->dmap->n != Hpre->cmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"PetscFn fn,Mat Hpre: local domain/column dim %D %D",fn->dmap->n,Hpre->cmap->n);
    if (fn->dmap->N != Hpre->rmap->N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"PetscFn fn,Mat Hpre: global domain/row dim %D %D",fn->dmap->N,Hpre->rmap->N);
    if (fn->dmap->n != Hpre->rmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"PetscFn fn,Mat Hpre: local domain/row dim %D %D",fn->dmap->n,Hpre->rmap->n);
  }
  if (!H && !Hpre) PetscFunctionReturn(0);
  ierr = VecLockPush(x);CHKERRQ(ierr);
  ierr = VecLockPush(v);CHKERRQ(ierr);
  if (fn->ops->hessiancreate) {
    ierr = (*fn->ops->hessiancreate)(fn,x,v,H,Hpre);CHKERRQ(ierr);
  } else if (fn->isScalar && fn->ops->scalarhessiancreate) {
    PetscScalar z;

    ierr = (*fn->ops->scalarhessiancreate) (fn, x, H, Hpre);CHKERRQ(ierr);
    ierr = PetscFnVecScalarBcast(v, &z);CHKERRQ(ierr);
    if (H) {ierr = MatScale(H,z);CHKERRQ(ierr);}
    if (Hpre && Hpre != H) {ierr = MatScale(Hpre,z);CHKERRQ(ierr);}
  } else SETERRQ1(PetscObjectComm((PetscObject)fn), PETSC_ERR_SUP, "This PetscFn does not implement %s()", PETSC_FUNCTION_NAME);
  ierr = VecLockPop(v);CHKERRQ(ierr);
  ierr = VecLockPop(x);CHKERRQ(ierr);
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
  if (!fn->ops->scalarapply) {
    ierr = (*fn->ops->scalarapply)(fn,x,z);CHKERRQ(ierr);
  } else if (fn->ops->apply) {
    Vec y;

    ierr = PetscFnCreateVecs(fn, &y, NULL);CHKERRQ(ierr);
    ierr = (*fn->ops->apply)(fn,x,y);CHKERRQ(ierr);
    ierr = PetscFnVecScalarBcast(y,z);CHKERRQ(ierr);
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
  if (!fn->ops->scalargradient) {
    ierr = (*fn->ops->scalargradient)(fn,x,g);CHKERRQ(ierr);
  } else if (fn->ops->jacobiancreateadjoint) {
    Mat JT;
    PetscInt i, iStart, iEnd, *ia;
    PetscInt zero = 0;
    PetscScalar *ga;

    ierr = PetscFnCreateMats(fn, NULL, NULL, &JT, NULL, NULL, NULL);CHKERRQ(ierr);
    ierr = (*fn->ops->jacobiancreateadjoint)(fn,x,JT,NULL);CHKERRQ(ierr);
    ierr = VecGetOwnershipRange(x,&iStart,&iEnd);CHKERRQ(ierr);
    ierr = PetscMalloc1(iEnd - iStart,&ia);CHKERRQ(ierr);
    for (i = 0; i < iEnd - iStart; i++) ia[i] = i + iStart;
    ierr = VecGetArray(g, &ga);CHKERRQ(ierr);
    ierr = MatGetValues(JT,iEnd - iStart, ia, 1, &zero, ga);CHKERRQ(ierr);
    ierr = VecRestoreArray(g, &ga);CHKERRQ(ierr);
    ierr = PetscFree(ia);CHKERRQ(ierr);
    ierr = MatDestroy(&JT);CHKERRQ(ierr);
  } else SETERRQ1(PetscObjectComm((PetscObject)fn), PETSC_ERR_SUP, "This PetscFn does not implement %s()", PETSC_FUNCTION_NAME);
  ierr = VecLockPop(x);CHKERRQ(ierr);
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
  if (!fn->ops->scalarhessianmult) {
    ierr = (*fn->ops->scalarhessianmult)(fn,x,xhat,Hxhat);CHKERRQ(ierr);
  } else if (fn->ops->hessianmult) {
    Vec v;

    ierr = PetscFnCreateVecs(fn, &v, NULL);CHKERRQ(ierr);
    ierr = VecSet(v, 1.);CHKERRQ(ierr);
    ierr = (*fn->ops->hessianmult)(fn,x,v,xhat,Hxhat);CHKERRQ(ierr);
    ierr = VecDestroy(&v);CHKERRQ(ierr);
  } else SETERRQ1(PetscObjectComm((PetscObject)fn), PETSC_ERR_SUP, "This PetscFn does not implement %s()", PETSC_FUNCTION_NAME);
  ierr = VecLockPop(xhat);CHKERRQ(ierr);
  ierr = VecLockPop(x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnScalarHessianCreate(PetscFn fn, Vec x, Mat H, Mat Hpre)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn,PETSCFN_CLASSID,1);
  PetscValidType(fn,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  ierr = PetscFnSetUp(fn);CHKERRQ(ierr);
  if (!(fn->isScalar)) SETERRQ(PetscObjectComm((PetscObject)fn),PETSC_ERR_ARG_SIZ, "PetscFn is not a scalar function");
  if (fn->dmap->N != x->map->N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"PetscFn fn,Vec x: global dim %D %D",fn->dmap->N,x->map->N);
  if (H) {
    PetscValidHeaderSpecific(H,MAT_CLASSID,4);
    if (fn->dmap->N != H->cmap->N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"PetscFn fn,Mat H: global domain/column dim %D %D",fn->dmap->N,H->cmap->N);
    if (fn->dmap->n != H->cmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"PetscFn fn,Mat H: local domain/column dim %D %D",fn->dmap->n,H->cmap->n);
    if (fn->dmap->N != H->rmap->N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"PetscFn fn,Mat H: global domain/row dim %D %D",fn->dmap->N,H->rmap->N);
    if (fn->dmap->n != H->rmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"PetscFn fn,Mat H: local domain/row dim %D %D",fn->dmap->n,H->rmap->n);
  }
  if (Hpre) {
    PetscValidHeaderSpecific(Hpre,VEC_CLASSID,5);
    if (fn->dmap->N != Hpre->cmap->N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"PetscFn fn,Mat Hpre: global domain/column dim %D %D",fn->dmap->N,Hpre->cmap->N);
    if (fn->dmap->n != Hpre->cmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"PetscFn fn,Mat Hpre: local domain/column dim %D %D",fn->dmap->n,Hpre->cmap->n);
    if (fn->dmap->N != Hpre->rmap->N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"PetscFn fn,Mat Hpre: global domain/row dim %D %D",fn->dmap->N,Hpre->rmap->N);
    if (fn->dmap->n != Hpre->rmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"PetscFn fn,Mat Hpre: local domain/row dim %D %D",fn->dmap->n,Hpre->rmap->n);
  }
  if (!H && !Hpre) PetscFunctionReturn(0);

  ierr = VecLockPush(x);CHKERRQ(ierr);
  if (!fn->ops->scalarhessiancreate) {
    ierr = (*fn->ops->scalarhessiancreate)(fn,x,H,Hpre);CHKERRQ(ierr);
  } else if (fn->ops->hessiancreate) {
    Vec v;

    ierr = PetscFnCreateVecs(fn, &v, NULL);CHKERRQ(ierr);
    ierr = VecSet(v, 1.);CHKERRQ(ierr);
    ierr = (*fn->ops->hessiancreate)(fn,x,v,H,Hpre);CHKERRQ(ierr);
    ierr = VecDestroy(&v);CHKERRQ(ierr);
  } else SETERRQ1(PetscObjectComm((PetscObject)fn), PETSC_ERR_SUP, "This PetscFn does not implement %s()", PETSC_FUNCTION_NAME);
  ierr = VecLockPop(x);CHKERRQ(ierr);
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
