
/*
   This is where the abstract PetscFn operations are defined
*/

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
