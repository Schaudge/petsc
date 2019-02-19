
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
  ierr = PetscFree((*fn)->jacadjType);CHKERRQ(ierr);
  ierr = PetscFree((*fn)->jacadjPreType);CHKERRQ(ierr);
  ierr = PetscFree((*fn)->hesType);CHKERRQ(ierr);
  ierr = PetscFree((*fn)->hesPreType);CHKERRQ(ierr);
  ierr = PetscFree((*fn)->hesadjType);CHKERRQ(ierr);
  ierr = PetscFree((*fn)->hesadjPreType);CHKERRQ(ierr);

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

static PetscErrorCode PetscFnSetUp_MatType(MPI_Comm comm, MatType typein, char *typeout[])
{
  Mat            A;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  typein = typein ? typein : MATAIJ;
  ierr = MatCreate(comm, &A);CHKERRQ(ierr);
  ierr = MatSetSizes(A, 0, 0, 0, 0);CHKERRQ(ierr);
  ierr = MatSetType(A, typein);CHKERRQ(ierr);
  ierr = MatGetType(A, &typein);CHKERRQ(ierr);
  ierr = PetscFree(*typeout);CHKERRQ(ierr);
  ierr = PetscStrallocpy(typein, typeout);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnSetUp_VecType(MPI_Comm comm, VecType typein, char *typeout[])
{
  Vec            A;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  typein = typein ? typein : VECSTANDARD;
  ierr = VecCreate(comm, &A);CHKERRQ(ierr);
  ierr = VecSetSizes(A, 0, 0);CHKERRQ(ierr);
  ierr = VecSetType(A, typein);CHKERRQ(ierr);
  ierr = VecGetType(A, &typein);CHKERRQ(ierr);
  ierr = PetscFree(*typeout);CHKERRQ(ierr);
  ierr = PetscStrallocpy(typein, typeout);CHKERRQ(ierr);
  ierr = VecDestroy(&A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnSetUp(PetscFn fn)
{
  MPI_Comm       comm;
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
  if (fn->rmap->N == 1) fn->isScalar = PETSC_TRUE;
  comm = PetscObjectComm((PetscObject)fn);
  ierr = PetscFnSetUp_VecType(comm, fn->rangeType,  (char **) &(fn->rangeType));CHKERRQ(ierr);
  ierr = PetscFnSetUp_VecType(comm, fn->domainType, (char **) &(fn->domainType));CHKERRQ(ierr);
  ierr = PetscFnSetUp_MatType(comm, fn->jacType,    (char **) &(fn->jacType));CHKERRQ(ierr);
  ierr = PetscFnSetUp_MatType(comm, fn->jacadjType, (char **) &(fn->jacadjType));CHKERRQ(ierr);
  ierr = PetscFnSetUp_MatType(comm, fn->hesType,    (char **) &(fn->hesType));CHKERRQ(ierr);
  ierr = PetscFnSetUp_MatType(comm, fn->hesadjType, (char **) &(fn->hesadjType));CHKERRQ(ierr);
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

PetscErrorCode PetscFnSetJacobianMatTypes(PetscFn fn, MatType jacType, MatType jacPreType, MatType jacadjType, MatType jacadjPreType)
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
  if (jacadjType) {
    ierr = PetscFree(fn->jacadjType);CHKERRQ(ierr);
    ierr = PetscStrallocpy(jacadjType,(char**)&fn->jacadjType);CHKERRQ(ierr);
  }
  if (jacadjPreType) {
    ierr = PetscFree(fn->jacadjPreType);CHKERRQ(ierr);
    ierr = PetscStrallocpy(jacadjPreType,(char**)&fn->jacadjPreType);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnGetJacobianMatTypes(PetscFn fn, MatType *jacType, MatType *jacPreType, MatType *jacadjType, MatType *jacadjPreType)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn,PETSCFN_CLASSID,1);
  if (jacType) {
    *jacType = fn->jacType;
  }
  if (jacPreType) {
    *jacPreType = fn->jacPreType;
  }
  if (jacadjType) {
    *jacadjType = fn->jacadjType;
  }
  if (jacadjPreType) {
    *jacadjPreType = fn->jacadjPreType;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnSetHessianMatTypes(PetscFn fn, MatType hesType, MatType hesPreType, MatType hesadjType, MatType hesadjPreType)
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
  if (hesadjType) {
    ierr = PetscFree(fn->hesadjType);CHKERRQ(ierr);
    ierr = PetscStrallocpy(hesadjType,(char**)&fn->hesadjType);CHKERRQ(ierr);
  }
  if (hesadjPreType) {
    ierr = PetscFree(fn->hesadjPreType);CHKERRQ(ierr);
    ierr = PetscStrallocpy(hesadjPreType,(char**)&fn->hesadjPreType);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


PetscErrorCode PetscFnGetHessianMatTypes(PetscFn fn, MatType *hesType, MatType *hesPreType, MatType *hesadjType, MatType *hesadjPreType)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn,PETSCFN_CLASSID,1);
  if (hesType) {
    *hesType = fn->hesType;
  }
  if (hesPreType) {
    *hesPreType = fn->hesPreType;
  }
  if (hesadjType) {
    *hesadjType = fn->hesadjType;
  }
  if (hesadjPreType) {
    *hesadjPreType = fn->hesadjPreType;
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

static PetscErrorCode PetscFnCreateMats_Internal(PetscFn fn, Mat *mats[4], PetscLayout layouts[3][2], MatType types[4], PetscErrorCode (*op) (PetscFn,Mat*,Mat*,Mat*,Mat*))
{
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (op) {
    ierr = (*op)(fn, mats[0], mats[1], mats[2], mats[3]);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
    for (i = 0; i < 4; i++) {
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
  } else {
    for (i = 0; i < 4; i++) {
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

PetscErrorCode PetscFnCreateJacobianMats(PetscFn fn, Mat *jac, Mat *jacPre, Mat *jacadj, Mat *jacadjPre)
{
  MatType        types[4];
  PetscLayout    layouts[3][2];
  Mat*           mats[4];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn,PETSCFN_CLASSID,1);
  types[0] = fn->jacType;
  types[1] = fn->jacPreType;
  types[2] = fn->jacadjType;
  types[3] = fn->jacadjPreType;
  layouts[0][0] = fn->rmap;
  layouts[0][1] = fn->dmap;
  layouts[1][0] = fn->dmap;
  layouts[1][1] = fn->rmap;
  mats[0] = jac;
  mats[1] = jacPre;
  mats[2] = jacadj;
  mats[3] = jacadjPre;
  ierr = PetscFnCreateMats_Internal(fn, mats, layouts, types, fn->ops->createjacobianmats);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnCreateHessianMats(PetscFn fn, Mat *hes, Mat *hesPre, Mat *hesadj, Mat *hesadjPre)
{
  MatType        types[4];
  PetscLayout    layouts[3][2];
  Mat*           mats[4];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn,PETSCFN_CLASSID,1);
  types[0] = fn->hesType;
  types[1] = fn->hesPreType;
  types[2] = fn->hesadjType;
  types[3] = fn->hesadjPreType;
  layouts[0][0] = fn->rmap;
  layouts[0][1] = fn->dmap;
  layouts[1][0] = fn->dmap;
  layouts[1][1] = fn->dmap;
  mats[0] = hes;
  mats[1] = hesPre;
  mats[2] = hesadj;
  mats[3] = hesadjPre;
  ierr = PetscFnCreateMats_Internal(fn, mats, layouts, types, fn->ops->createhessianmats);CHKERRQ(ierr);
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
    ierr = PetscFnVecScalarBcast(v, &z);CHKERRQ(ierr);
    ierr = VecScale(Jadjv, z);CHKERRQ(ierr);
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

PetscErrorCode PetscFnJacobianCreateAdjoint(PetscFn fn, Vec x, Mat Jadj, Mat Jadjpre)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn,PETSCFN_CLASSID,1);
  PetscValidType(fn,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  ierr = PetscFnSetUp(fn);CHKERRQ(ierr);
  if (Jadj) {
    PetscValidHeaderSpecific(Jadj,MAT_CLASSID,3);
    if (fn->dmap->N != Jadj->rmap->N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"PetscFn fn,Mat Jadj: global domain/row dim %D %D",fn->dmap->N,Jadj->rmap->N);
    if (fn->dmap->n != Jadj->rmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"PetscFn fn,Mat Jadj: local domain/row dim %D %D",fn->dmap->n,Jadj->rmap->n);
    if (fn->rmap->N != Jadj->cmap->N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"PetscFn fn,Mat Jadj: global range/column dim %D %D",fn->rmap->N,Jadj->cmap->N);
    if (fn->rmap->n != Jadj->cmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"PetscFn fn,Mat Jadj: local range/column dim %D %D",fn->rmap->n,Jadj->cmap->n);
  }
  if (Jadjpre) {
    PetscValidHeaderSpecific(Jadjpre,VEC_CLASSID,4);
    if (fn->dmap->N != Jadjpre->rmap->N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"PetscFn fn,Mat Jadjpre: global domain/row dim %D %D",fn->dmap->N,Jadjpre->rmap->N);
    if (fn->dmap->n != Jadjpre->rmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"PetscFn fn,Mat Jadjpre: local domain/row dim %D %D",fn->dmap->n,Jadjpre->rmap->n);
    if (fn->rmap->N != Jadjpre->cmap->N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"PetscFn fn,Mat Jadjpre: global range/column dim %D %D",fn->rmap->N,Jadjpre->cmap->N);
    if (fn->rmap->n != Jadjpre->cmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"PetscFn fn,Mat Jadjpre: local range/column dim %D %D",fn->rmap->n,Jadjpre->cmap->n);
  }
  if (!Jadj && !Jadjpre) PetscFunctionReturn(0);
  ierr = VecLockPush(x);CHKERRQ(ierr);
  if (fn->ops->jacobiancreateadjoint) {
    ierr = (*fn->ops->jacobiancreateadjoint)(fn,x,Jadj,Jadjpre);CHKERRQ(ierr);
  } else if (fn->isScalar && fn->ops->scalargradient) {
    Mat                jacadj = Jadj ? Jadj : Jadjpre;
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
    ierr = MatSetValues(jacadj, iEnd - iStart, ia, 1, &zero, ga, INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(g, &ga);CHKERRQ(ierr);
    ierr = PetscFree(ia);CHKERRQ(ierr);
    ierr = VecDestroy(&g);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(jacadj,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(jacadj,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    if (Jadj && Jadj != jacadj) {ierr = MatCopy(jacadj, Jadj, DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);}
    if (Jadjpre && Jadjpre != jacadj) {ierr = MatCopy(jacadj, Jadjpre, DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);}
  } else SETERRQ1(PetscObjectComm((PetscObject)fn), PETSC_ERR_SUP, "This PetscFn does not implement %s()", PETSC_FUNCTION_NAME);
  ierr = VecLockPop(x);CHKERRQ(ierr);
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
    ierr = PetscFnVecScalarBcast(v, &z);CHKERRQ(ierr);
    ierr = VecScale(Hadjvxhat, z);CHKERRQ(ierr);
  } else SETERRQ1(PetscObjectComm((PetscObject)fn), PETSC_ERR_SUP, "This PetscFn does not implement %s()", PETSC_FUNCTION_NAME);
  ierr = VecLockPop(xhat);CHKERRQ(ierr);
  ierr = VecLockPop(v);CHKERRQ(ierr);
  ierr = VecLockPop(x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnHessianCreate(PetscFn fn, Vec x, Vec xhat, Mat H, Mat Hpre)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn,PETSCFN_CLASSID,1);
  PetscValidType(fn,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  PetscValidHeaderSpecific(xhat,VEC_CLASSID,3);
  ierr = PetscFnSetUp(fn);CHKERRQ(ierr);
  if (H) {
    PetscValidHeaderSpecific(H,MAT_CLASSID,4);
    if (fn->dmap->N != H->cmap->N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"PetscFn fn,Mat H: global domain/column dim %D %D",fn->dmap->N,H->cmap->N);
    if (fn->dmap->n != H->cmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"PetscFn fn,Mat H: local domain/column dim %D %D",fn->dmap->n,H->cmap->n);
    if (fn->rmap->N != H->rmap->N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"PetscFn fn,Mat H: global range/row dim %D %D",fn->rmap->N,H->rmap->N);
    if (fn->rmap->n != H->rmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"PetscFn fn,Mat H: local range/row dim %D %D",fn->rmap->n,H->rmap->n);
  }
  if (Hpre) {
    PetscValidHeaderSpecific(Hpre,VEC_CLASSID,5);
    if (fn->dmap->N != Hpre->cmap->N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"PetscFn fn,Mat Hpre: global domain/column dim %D %D",fn->dmap->N,Hpre->cmap->N);
    if (fn->dmap->n != Hpre->cmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"PetscFn fn,Mat Hpre: local domain/column dim %D %D",fn->dmap->n,Hpre->cmap->n);
    if (fn->rmap->N != Hpre->rmap->N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"PetscFn fn,Mat Hpre: global range/row dim %D %D",fn->rmap->N,Hpre->rmap->N);
    if (fn->rmap->n != Hpre->rmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"PetscFn fn,Mat Hpre: local range/row dim %D %D",fn->rmap->n,Hpre->rmap->n);
  }
  if (!H && !Hpre) PetscFunctionReturn(0);
  ierr = VecLockPush(x);CHKERRQ(ierr);
  ierr = VecLockPush(xhat);CHKERRQ(ierr);
  if (fn->ops->hessiancreate) {
    ierr = (*fn->ops->hessiancreate)(fn,x,xhat,H,Hpre);CHKERRQ(ierr);
  } else if (fn->isScalar && fn->ops->scalarhessianmult) {
    Mat                hes = H ? H : Hpre;
    Vec                Hxhat;
    PetscInt           i, iStart, iEnd, *ia, zero = 0;
    const PetscScalar *ga;

    ierr = VecDuplicate(xhat, &Hxhat);CHKERRQ(ierr);
    ierr = (*fn->ops->scalarhessianmult) (fn, x, xhat, Hxhat);CHKERRQ(ierr);
    ierr = VecGetOwnershipRange(Hxhat, &iStart, &iEnd);CHKERRQ(ierr);
    ierr = PetscMalloc1(iEnd - iStart, &ia);CHKERRQ(ierr);
    for (i = 0; i < iEnd - iStart; i++) ia[i] = i + iStart;
    ierr = VecGetArrayRead(Hxhat, &ga);CHKERRQ(ierr);
    ierr = MatSetValues(hes, iEnd - iStart, ia, 1, &zero, ga, INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(Hxhat, &ga);CHKERRQ(ierr);
    ierr = PetscFree(ia);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(hes,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(hes,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    if (H && H != hes) {ierr = MatCopy(hes, H, DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);}
    if (Hpre && Hpre != hes) {ierr = MatCopy(hes, Hpre, DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);}
    ierr = VecDestroy(&Hxhat);CHKERRQ(ierr);
  } else SETERRQ1(PetscObjectComm((PetscObject)fn), PETSC_ERR_SUP, "This PetscFn does not implement %s()", PETSC_FUNCTION_NAME);
  ierr = VecLockPop(xhat);CHKERRQ(ierr);
  ierr = VecLockPop(x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnHessianCreateAdjoint(PetscFn fn, Vec x, Vec v, Mat Hadj, Mat Hadjpre)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn,PETSCFN_CLASSID,1);
  PetscValidType(fn,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  PetscValidHeaderSpecific(v,VEC_CLASSID,3);
  ierr = PetscFnSetUp(fn);CHKERRQ(ierr);
  if (Hadj) {
    PetscValidHeaderSpecific(Hadj,MAT_CLASSID,4);
    if (fn->dmap->N != Hadj->cmap->N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"PetscFn fn,Mat Hadj: global domain/column dim %D %D",fn->dmap->N,Hadj->cmap->N);
    if (fn->dmap->n != Hadj->cmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"PetscFn fn,Mat Hadj: local domain/column dim %D %D",fn->dmap->n,Hadj->cmap->n);
    if (fn->dmap->N != Hadj->rmap->N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"PetscFn fn,Mat Hadj: global domain/row dim %D %D",fn->dmap->N,Hadj->rmap->N);
    if (fn->dmap->n != Hadj->rmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"PetscFn fn,Mat Hadj: local domain/row dim %D %D",fn->dmap->n,Hadj->rmap->n);
  }
  if (Hadjpre) {
    PetscValidHeaderSpecific(Hadjpre,VEC_CLASSID,5);
    if (fn->dmap->N != Hadjpre->cmap->N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"PetscFn fn,Mat Hadjpre: global domain/column dim %D %D",fn->dmap->N,Hadjpre->cmap->N);
    if (fn->dmap->n != Hadjpre->cmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"PetscFn fn,Mat Hadjpre: local domain/column dim %D %D",fn->dmap->n,Hadjpre->cmap->n);
    if (fn->dmap->N != Hadjpre->rmap->N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"PetscFn fn,Mat Hadjpre: global domain/row dim %D %D",fn->dmap->N,Hadjpre->rmap->N);
    if (fn->dmap->n != Hadjpre->rmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"PetscFn fn,Mat Hadjpre: local domain/row dim %D %D",fn->dmap->n,Hadjpre->rmap->n);
  }
  if (!Hadj && !Hadjpre) PetscFunctionReturn(0);
  ierr = VecLockPush(x);CHKERRQ(ierr);
  ierr = VecLockPush(v);CHKERRQ(ierr);
  if (fn->ops->hessiancreateadjoint) {
    ierr = (*fn->ops->hessiancreateadjoint)(fn,x,v,Hadj,Hadjpre);CHKERRQ(ierr);
  } else if (fn->isScalar && fn->ops->scalarhessiancreate) {
    PetscScalar z;

    ierr = (*fn->ops->scalarhessiancreate) (fn, x, Hadj, Hadjpre);CHKERRQ(ierr);
    ierr = PetscFnVecScalarBcast(v, &z);CHKERRQ(ierr);
    if (Hadj) {ierr = MatScale(Hadj,z);CHKERRQ(ierr);}
    if (Hadjpre && Hadjpre != Hadj) {ierr = MatScale(Hadjpre,z);CHKERRQ(ierr);}
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
  if (fn->ops->scalarapply) {
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
  if (fn->ops->scalargradient) {
    ierr = (*fn->ops->scalargradient)(fn,x,g);CHKERRQ(ierr);
  } else if (fn->ops->jacobiancreateadjoint) {
    Mat Jadj;
    PetscInt i, iStart, iEnd, *ia;
    PetscInt zero = 0;
    PetscScalar *ga;

    ierr = PetscFnCreateJacobianMats(fn, NULL, NULL, &Jadj, NULL);CHKERRQ(ierr);
    ierr = (*fn->ops->jacobiancreateadjoint)(fn,x,Jadj,NULL);CHKERRQ(ierr);
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

    ierr = PetscFnCreateVecs(fn, &v, NULL);CHKERRQ(ierr);
    ierr = VecSet(v, 1.);CHKERRQ(ierr);
    ierr = (*fn->ops->hessianmultadjoint)(fn,x,v,xhat,Hxhat);CHKERRQ(ierr);
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
  if (fn->ops->scalarhessiancreate) {
    ierr = (*fn->ops->scalarhessiancreate)(fn,x,H,Hpre);CHKERRQ(ierr);
  } else if (fn->ops->hessiancreateadjoint) {
    Vec v;

    ierr = PetscFnCreateVecs(fn, &v, NULL);CHKERRQ(ierr);
    ierr = VecSet(v, 1.);CHKERRQ(ierr);
    ierr = (*fn->ops->hessiancreateadjoint)(fn,x,v,H,Hpre);CHKERRQ(ierr);
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
