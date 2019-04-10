
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
  ierr = PetscHMapIJCreate(&B->testedscalar);CHKERRQ(ierr);
  ierr = PetscHMapIJCreate(&B->testedvec);CHKERRQ(ierr);
  ierr = PetscHMapIJCreate(&B->testedmat);CHKERRQ(ierr);
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
  ierr = PetscHMapIJDestroy(&(*fn)->testedscalar);CHKERRQ(ierr);
  ierr = PetscHMapIJDestroy(&(*fn)->testedvec);CHKERRQ(ierr);
  ierr = PetscHMapIJDestroy(&(*fn)->testedmat);CHKERRQ(ierr);
  ierr = PetscHeaderDestroy(fn);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscFnSetSizes - Sets the local and global sizes, and checks to determine compatibility

  Collective on PetscFn

  Input Parameters:
+  A - the PetscFn function
.  m - number of local range degrees of freedom (or PETSC_DECIDE)
.  n - number of local domain degrees of freedom (or PETSC_DECIDE)
.  M - number of global range degrees of freedom (or PETSC_DETERMINE)
-  N - number of global domain degrees of freedom (or PETSC_DETERMINE)

   Notes:
   m (n) and M (N) cannot be both PETSC_DECIDE
   If one processor calls this with M (N) of PETSC_DECIDE then all processors must, otherwise the program will hang.

   If PETSC_DECIDE is not used for the arguments 'm' and 'n', then the
   user must ensure that they are chosen to be compatible with the
   vectors. To do this, one first considers the function application
   'y = A(x)'. The 'm' that is used in the above routine must match the
   local size used in the vector creation routine VecCreateMPI() for 'y'.
   Likewise, the 'n' used must match that used as the local size in
   VecCreateMPI() for 'x'.

   You cannot change the sizes once they have been set.

   The sizes must be set before PetscFnSetUp().

   If the number of global range degrees of freedom is 1, then the function
   is scalar valued: PetscFnIsScalar() will yield PETSC_TRUE, and
   scalar methods (PetscFnScalarApply(), PetscFnScalarDerivativeVec(), etc.)
   can be used with this function.  Vector valued methods
   (PetscFnApply(), PetscFnDerivativeVec(), etc.), can be used on
   all PetscFn functions of any number of global range degrees of freedom.

  Level: beginner

.seealso: PetscFnGetSizes(), PetscFnIsScalar()
@*/
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

/*@
  PetscFnGetSizes - Gets the local and global sizes

  Not Collective

  Input Parameters:
.  A - the PetscFn function

  Output Parameters:
+  m - number of local range degrees of freedom (optional)
.  n - number of local domain degrees of freedom (optional)
.  M - number of global range degrees of freedom (optional)
-  N - number of global range degrees of freedom (optional)

   Notes:
   See PetscFnSetSizes() for more description of these values

  Level: beginner

.seealso: PetscFnSetSizes(), PetscFnIsScalar()
@*/
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

/*@
   PetscFnGetLayouts - Gets the PetscLayout objects for range and domain vectors

   Not Collective

   Input Parameters:
.  fn - the PetscFn function

   Output Parameters:
+ rmap - range layout (optional)
- dmap - domain layout (optional)

   Level: advanced

   Notes: Does not increment the reference counts of the returned PetscLayouts, they should not be destroyed

.seealso:  PetscFnCreateVecs()
@*/
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

/*@
   PetscFnLayoutsSetUp - Set up the PetscLayouts internal to a PetscFn

   Logically Collective on PetscFn

   Input Parameters:
.  fn - the PetscFn function

   Level: advanced

   Notes:
   This is provided so that implementations of PetscFn
   may use the layouts prior to PetscFnSetUp() being called.

.seealso:  PetscFnGetLayouts()
@*/
PetscErrorCode PetscFnLayoutsSetUp(PetscFn fn)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLayoutSetUp(fn->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(fn->dmap);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PetscFnSetOptionsPrefix - Sets the prefix used for searching for all
   PetscFn options in the database.

   Logically Collective on PetscFn

   Input Parameter:
+  fn - the PetscFn function
-  prefix - the prefix to prepend to all option names

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the hyphen.

   Level: advanced

.keywords: PetscFn, set, options, prefix, database

.seealso: PetscFnSetFromOptions()
@*/
PetscErrorCode PetscFnSetOptionsPrefix(PetscFn fn, const char prefix[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn,PETSCFN_CLASSID,1);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)fn,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PetscFnAppendOptionsPrefix - Appends to the prefix used for searching for all
   PetscFn options in the database.

   Logically Collective on PetscFn

   Input Parameters:
+  fn - the PetscFn function
-  prefix - the prefix to prepend to all option names

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the hyphen.

   Level: advanced

.keywords: PetscFn, append, options, prefix, database

.seealso: PetscFnGetOptionsPrefix()
@*/
PetscErrorCode PetscFnAppendOptionsPrefix(PetscFn fn,const char prefix[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn,PETSCFN_CLASSID,1);
  ierr = PetscObjectAppendOptionsPrefix((PetscObject)fn,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PetscFnGetOptionsPrefix - Sets the prefix used for searching for all
   PetscFn options in the database.

   Not Collective

   Input Parameter:
.  fn - the PetscFn function

   Output Parameter:
.  prefix - pointer to the prefix string used

   Notes:
    On the fortran side, the user should pass in a string 'prefix' of
   sufficient length to hold the prefix.

   Level: advanced

.keywords: PetscFn, get, options, prefix, database

.seealso: PetscFnAppendOptionsPrefix()
@*/
PetscErrorCode PetscFnGetOptionsPrefix(PetscFn fn,const char *prefix[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn,PETSCFN_CLASSID,1);
  ierr = PetscObjectGetOptionsPrefix((PetscObject)fn,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   PetscFnSetFromOptions - Sets various PetscFn parameters from user options.

   Collective on PetscFn

   Input Parameter:
.  fn - the PetscFn function

   Options Database Keys:
+    -fn_type <type> - (PetscFnType for a complete list)
.    -fn_test_scalar <bool> - On first use of PetscFnDerivativeScalar() or PetscFnScalarDerivativeScalar() for a given derivative order, perform a simple
finite difference test of the computed derivative using PetscFnTestsDerivativeScalar() or PetscFnTestScalarDerivativeScalar()
.    -fn_test_vec <bool> - On first use of PetscFnDerivativeVec() or PetscFnScalarDerivativeVec() for a given derivative order, perform a simple
finite difference test of the computed derivative using PetscFnTestsDerivativeVec() or PetscFnTestScalarDerivativeVec()
.    -fn_test_mat <bool> - On first use of PetscFnDerivativeMat() or PetscFnScalarDerivativeMat() for a given derivative order, compare
the matrix-vector product of the computed matrix against the same product computed using PetscFnDerivativeVec() or PetscFnScalarDerivativeVec()
+    -fn_test_fn <bool> - On first use of a PetscFn created by PetscFnCreateDerivativeFn() or PetscFnCreateScalarDerivativeFn(), compare
the action of that derive Fn against the equivalent action of the original PetscFn

   Even More Options Database Keys:
   See the manpages for particular formats
   for additional format-specific options.

   Level: beginner

.keywords: PetscFn,

.seealso: PetscFnTestDerivativeScalar(), PetscFnTestScalarDerivativeScalar(),
          PetscFnTestDerivativeVec(), PetscFnTestScalarDerivativeVec(),
          PetscFnTestDerivativeMat(), PetscFnTestScalarDerivativeMat(),
          PetscFnTestDerivativeFn(), PetscFnTestScalarDerivativeFn()
@*/
PetscErrorCode PetscFnSetFromOptions(PetscFn fn)
{
  PetscErrorCode ierr;
  const char     *deft = PETSCFNSHELL;
  char           type[256];
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

  ierr = PetscOptionsBool("-fn_test_scalar","On first use, test the order of convergence of derivative scalars","PetscFnTestDerivativeScalar",fn->test_scalar,&fn->test_scalar,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-fn_test_vec","On first use, test the order of convergence of derivative vectors","PetscFnTestDerivativeVec",fn->test_vec,&fn->test_vec,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-fn_test_mat","On first use, test derivative matrices against derivative vectors","PetscFnTestDerivativeMat",fn->test_mat,&fn->test_mat,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-fn_test_fn","On first use, test the instantiated derivative PetscFns against matrix-free","PetscFnTestDerivativeFn",fn->test_derfn,&(fn->test_derfn),NULL);CHKERRQ(ierr);


  if (fn->ops->setfromoptions) {
    ierr = (*fn->ops->setfromoptions)(PetscOptionsObject,fn);CHKERRQ(ierr);
  }

  /* process any options handlers added with PetscObjectAddOptionsHandler() */
  ierr = PetscObjectProcessOptionsHandlers(PetscOptionsObject,(PetscObject)fn);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   PetscFnSetUp - Sets up the internal PetscFn data structures for the later use.

   Collective on PetscFn

   Input Parameters:
.  fn - the PetscFn function

   Notes:

   Level: beginner

.keywords: PetscFn, setup

.seealso: PetscFnCreate(), PetscFnDestroy()
@*/
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

/*@C
   PetscFnView - Visualizes a PetscFn object.

   Collective on PetscFn

   Input Parameters:
+  fn - the PetscFn function
-  viewer - visualization context

   Level: beginner

   Notes:
   Because the one thing that can be described about all PetscFn functions is the
   size of their domain and range spaces, most of the detail in PetscFnView() must
   come from individual implementations of PetscFn

.seealso: PetscViewerPushFormat(), PetscViewerASCIIOpen(), PetscViewerDrawOpen(),
          PetscViewerSocketOpen(), PetscViewerBinaryOpen(),
@*/
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
  nMat = mat->rmap->n;
  mMat = mat->cmap->n;
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

/*@C
   PetscFnCreateVecs - Create vector(s) compatible with the function (with the same
     parallel layout), or with subsets of the range and domain space

   Collective on PetscFn

   Input Parameter:
+  fn - the PetscFn function
.  domainIS - indicate the subset of the domain degrees of freedom that the domainVec should match (optional, if NULL the domain Vector will be full-sized)
-  rangeIS - indicate the subset of the range degrees of freedom that the rangeVec should match (optional, if NULL the range vector will be full-sized)

   Output Parameter:
+   domainVec - (optional) vector that can be domain value where a PetscFn and its derivatives can be evaluated, or a domain variation that contracts with a derivative (see e.g. PetscFnDerivativeVec() or PetscFnScalarDerivativeVec() for more information on contracting variations)
-   left - (optional) vector that the can be the range value that a PetscFn evaluates to, or a range variation that contracts with a derivative (see e.g. PetscFnDerivativeVec() for more information on contracting variations)

   Notes:
   A range vector can be created for a scalar function: it will be a vector containing a single value.

   Developer Note:
   A PetscFn can be "rectangular" (have different range and domain sizes), and the intention is to allow easy handling of "domains" that are unions of simpler domains, for example
   a parameterized function $y = f(x;m)$ can have a domain that is all $(x,m)$ pairs.  To make it easier to handle partial derivatives like $\frac{\partial f}{\partial m}[\hat{m}]$,
   the derivative methods of PetscFn take subsets that indicate which portion of the full derivative is desired.  The subset arguments in PetscFnCreateVecs() are present to
   allow PetscFn implementations to create just $\hat{m}$ when needed.

  Level: advanced

.seealso: PetscFnCreate(), VecDestroy()
@*/
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

/*@
   PetscFnApply - Computes the PetscFn action, y = F(x).

   Collective on PetscFn and Vec

   Input Parameters:
+  fn - the PetscFn function
-  x  - the domain vector

   Output Parameters:
.  y - the result

   Notes:
   The vectors x and y cannot be the same.  I.e., one cannot
   call PetscFnApply(fn,y,y).

   This can be called even for a scalar-valued function:
   in all cases, the appropriate result vector can be
   created with PetscFnCreateVecs().

   Level: beginner
@*/
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

  ierr = VecLockReadPush(x);CHKERRQ(ierr);
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
  ierr = VecLockReadPop(x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnGetISVecsWithoutRange(PetscInt rangeIdx, PetscInt numISs, const IS *subsets, PetscInt numVecs, const Vec *subvecs, IS **newsubsets, Vec **newsubvecs)
{
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (rangeIdx == 0) {
    *newsubsets = (IS *) (subsets ? &subsets[1] : NULL);
    *newsubvecs = (Vec *) (subvecs ? &subvecs[1] : NULL);
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
    for (i = 0; i < numVecs - 1; i++) svecs[i] = subvecs[i + (i >= rangeIdx)];
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
      ierr = MatSetValuesVec(*Apre, g, 0, colVec, INSERT_VALUES);CHKERRQ(ierr);
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

/*@
   PetscFnDerivativeScalar - Computes a partial derivative of a PetscFn action, fully contracted with variations.

   If $X$ is the domain space of a function $f$, and $W$ is its range, $D^{(k)}f(x)$ can be viewed as a $k+1$-order
   tensor, $D^{(k)}f(x):X\times\dots\times X\times W' \to \mathbb{Z}$ ($\mathbb{Z}$ is the field of PetscScalar). 

   If the tensor is contracted against the $k+1$ variations that are provided, one if which is a variation from $W'$,
   the result is a scalar. 

   Partial derivatives of $f$ can be computed by using variations over subsets of $X$ or $W'$.

   Collective on PetscFn and Vec

   Input Parameters:
+  fn - the PetscFn function
.  x  - the domain vector where $D^{(k)} f$ is evaluated
.  der - the order $k >= 0$ of the derivative.
.  rangeIdx - indicates which of the variations in the subvecs is the range-space variation
.  subsets - (optional) if a partial derivative is desired, a list of (der + 1) subsets can be given that indicates
which partial derivative is desired.  If subsets[i] is NULL, it indicates that the full derivative is desired for that
variation.  If subsets is NULL, it indicates that the full derivative is desired for all variations.
-  subvecs - (der + 1) variations that contract against the derivative tensor.


   Output Parameters:
.  z - the result

   Level: advanced
@*/
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
  ierr = VecLockReadPush(x);CHKERRQ(ierr);
  for (i = 0; i < der+1; i++) {
    ierr = VecLockReadPush(subvecs[i]);CHKERRQ(ierr);
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
    ierr = VecLockReadPop(subvecs[i]);CHKERRQ(ierr);
  }
  ierr = VecLockReadPop(x);CHKERRQ(ierr);
  if (der > 0 && fn->test_scalar) {
    PetscHashIJKey key;
    PetscBool      missing;

    key.i = der;
    key.j = rangeIdx;
    ierr = PetscHMapIJQuerySet(fn->testedscalar, key, 1, &missing);CHKERRQ(ierr);
    if (missing) {
      PetscReal rate;

      ierr = PetscFnTestDerivativeScalar(fn, x, der, rangeIdx, subsets, subvecs, *z, PETSC_DEFAULT, PETSC_DEFAULT, &rate);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/*@
   PetscFnDerivativeVec - Computes a partial derivative of a PetscFn action, partially contracted with variations,
   resulting in a vector y.

   If $X$ is the domain space of a function $f$, and $W$ is its range, $D^{(k)}f(x)$ can be viewed as a $k+1$-order
   tensor, $D^{(k)}f(x):X\times\dots\times X\times W' \to \mathbb{Z}$ ($\mathbb{Z}$ is the field of PetscScalar). 

   If the tensor is contracted against the $k$ variations that are provided, the result is a vector $y$.
   The dot product of $y$ with another vector $z$ will give the same scalar value as if $z$ had been
   included in the $k+1$ variations given to PetscFnDerivativeScalar().

   If the $k$ variations do not contain a range-space vector from $W'$, $y$ will be a vector in the range space $W$.

   If the $k$ variations do contain a range-space vector from $W'$, $y$ will be a functional on the domain space $X$.

   Partial derivatives of $f$ can be computed by using variations over subsets of $X$ or $W'$.

   Collective on PetscFn and Vec

   Input Parameters:
+  fn - the PetscFn function
.  x  - the domain vector where $D^{(k)} f$ is evaluated
.  der - the order $k >= 0$ of the derivative.
.  rangeIdx - indicates which of the variations in the subvecs is the range-space variation. If rangeIdx is der,
   it indicates that the output vector y is in the range space.
.  subsets - (optional) if a partial derivative is desired, a list of (der + 1) subsets can be given that indicates
which partial derivative is desired.  If subsets[i] is NULL, it indicates that the full derivative is desired for that
variation.  If subsets is NULL, it indicates that the full derivative is desired for all variations.
-  subvecs - der variations that contract against the derivative tensor.


   Output Parameters:
.  y - the result vector.  If rangeIdx == der, it is in the range space, otherwise it is a functional on the domain space.
   If subsets[der] != NULL, it is a partial vector.

   Notes:

   If der == 0, this should be equivalent to PetscFnApply(), except that a subset of the result can be returned.
   If der == 1 && rangeIdx == 1, this is the action of the Jacobian of f on a domain-space variation
   If der == 1 && rangeIdx == 0, this is the action of the adjoint of the Jacobian of f on a range-space variation
   If der == 2 && rangeIdx == 2, this is the bilinear action of the Hessian of f on two domain-space variations
   If der == 2 && rangeIdx == 0, this is the bilinear action of the adjoint Hessian of f on one range-space variation and one domain-space variation

   Level: advanced
@*/
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
  ierr = VecLockReadPush(x);CHKERRQ(ierr);
  for (i = 0; i < der; i++) {
    ierr = VecLockReadPush(subvecs[i]);CHKERRQ(ierr);
  }
  if (fn->ops->derivativevec) {
    ierr = (*fn->ops->derivativevec)(fn,x,der,rangeIdx,subsets,subvecs,y);CHKERRQ(ierr);
  } else if (fn->isScalar && rangeIdx < der && fn->ops->scalarderivativevec) {
    /* for a scalar functional derivative to produce a vector, it must contract against der-1 vectors */
    IS          *scalarsubsets = NULL;
    Vec         *scalarvecs = NULL;
    PetscScalar z;

    ierr = PetscFnGetISVecsWithoutRange(rangeIdx, der+1, subsets, der, subvecs, &scalarsubsets, &scalarvecs);CHKERRQ(ierr);
    ierr = (*fn->ops->scalarderivativevec)(fn,x,der,scalarsubsets,scalarvecs,y);CHKERRQ(ierr);
    ierr = VecScalarBcast(subvecs[rangeIdx], &z);CHKERRQ(ierr);
    ierr = VecScale(y, z);CHKERRQ(ierr);
    ierr = PetscFnRestoreISVecsWithoutRange(rangeIdx, der+1, subsets, der, subvecs, &scalarsubsets, &scalarvecs);CHKERRQ(ierr);
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
    ierr = VecLockReadPop(subvecs[i]);CHKERRQ(ierr);
  }
  ierr = VecLockReadPop(x);CHKERRQ(ierr);
  if (der > 0 && fn->test_vec) {
    PetscHashIJKey key;
    PetscBool      missing;

    key.i = der;
    key.j = rangeIdx;
    ierr = PetscHMapIJQuerySet(fn->testedvec, key, 1, &missing);CHKERRQ(ierr);
    if (missing) {
      PetscReal rate;

      ierr = PetscFnTestDerivativeVec(fn, x, der, rangeIdx, subsets, subvecs, y, PETSC_DEFAULT, PETSC_DEFAULT, &rate);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/*@
   PetscFnDerivativeMat - Computes a partial derivative of a PetscFn action, partially contracted with variations,
   resulting in a matrix and an approximation of that matrix for use in preconditioning.

   If $X$ is the domain space of a function $f$, and $W$ is its range, $D^{(k)}f(x)$ can be viewed as a $k+1$-order
   tensor, $D^{(k)}f(x):X\times\dots\times X\times W' \to \mathbb{Z}$ ($\mathbb{Z}$ is the field of PetscScalar). 

   If the tensor is contracted against the $k-1$ variations that are provided, the result is a matrix $M$.
   The matrix vector product of $M$ with another vector $z$ will give the same vector value as if $z$
   had been included in the $k$ variations given to PetscFnDerivativeVec().

   If the $k-1$ variations do not contain a range-space vector from $W'$, $M$ will be a rectangular matrix.

   If the $k-1$ variations do contain a range-space vector from $W'$, $M$ will be a symmetric matrix acting
   on domain-space vectors.

   A matrix of partial derivatives of $f$ can be computed by using variations over subsets of $X$ or $W'$.

   Collective on PetscFn and Vec

   Input Parameters:
+  fn - the PetscFn function
.  x  - the domain vector where $D^{(k)} f$ is evaluated
.  der - the order $k >= 1$ of the derivative.
.  rangeIdx - indicates which of the variations in the subvecs is the range-space variation. If rangeIdx is der-1 or der,
   it indicates that the output matrix M will have a column or row space equivalent to the range space, respectively.
.  subsets - (optional) if a partial derivative is desired, a list of (der + 1) subsets can be given that indicates
which partial derivative is desired.  If subsets[i] is NULL, it indicates that the full derivative is desired for that
variation.  If subsets is NULL, it indicates that the full derivative is desired for all variations.
.  subvecs - (der-1) variations that contract against the derivative tensor.
-  reuse - indicate whether the matrices should be created (MAT_INITIAL_MATRIX) or reused (MAT_REUSE_MATRIX).


   Output Parameters:
+  M - the result matrix.  If rangeIdx == der-1, its column space is the range space and its row space is the domain space;
   if rangeidx == der, its column space is the domain space and its row space is the range space; otherwise both
   its column and row spaces are the domain space and it is symmetric.
-  Mpre - an optional preconditioner for M.

   Notes:

   If der == 1 && rangeIdx == 1, this is the Jacobian of f
   If der == 1 && rangeIdx == 0, this is the adjoint of the Jacobian of f (the [Hermitian, if complex valued] transpose)
   If der == 2 && rangeIdx == 2, this is the action Hessian of f on one domain-space variation
   If der == 2 && rangeIdx == 1, is the [Hermitian] transpose of the above
   If der == 2 && rangeIdx == 0, this is the action of the adjoint Hessian of f on one range-space variation

   Level: advanced
@*/
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
  for (i = 0; i < der-1; i++) {ierr = PetscFnVecCheckCompatible(subvecs[i], subsets ? subsets[i] : NULL, (i == rangeIdx) ? fn->rmap : fn->dmap);CHKERRQ(ierr);}
  if (reuse == MAT_REUSE_MATRIX) {
    PetscLayout    rightLayout = (der-1 == rangeIdx) ? fn->rmap : fn->dmap;
    PetscLayout    leftLayout = (der == rangeIdx) ? fn->rmap : fn->dmap;

    if (M) {ierr = PetscFnMatCheckCompatible(*M, rightIS, leftIS, rightLayout, leftLayout);CHKERRQ(ierr);}
    if (Mpre) {ierr = PetscFnMatCheckCompatible(*Mpre, rightIS, leftIS, rightLayout, leftLayout);CHKERRQ(ierr);}
  }
#endif
  if (!M && !Mpre) PetscFunctionReturn(0);
  ierr = VecLockReadPush(x);CHKERRQ(ierr);
  for (i = 0; i < der - 1; i++) {
    ierr = VecLockReadPush(subvecs[i]);CHKERRQ(ierr);
  }
  if (fn->ops->derivativemat) {
    ierr = (*fn->ops->derivativemat)(fn,x,der,rangeIdx,subsets,subvecs,reuse,M,Mpre);CHKERRQ(ierr);
  } else if (fn->isScalar && rangeIdx >= der - 1 && fn->ops->derivativevec) {
    IS          *subsets = NULL;
    Vec         *newvecs = NULL;
    Vec         onevec;
    Vec         grad;
    PetscBool   colVec = (der == rangeIdx) ? PETSC_FALSE: PETSC_TRUE;

    ierr = PetscFnCreateVecs(fn, (der == rangeIdx) ? rightIS : leftIS, &grad, rangeIS, &onevec);CHKERRQ(ierr);
    ierr = VecSet(onevec, 1.);CHKERRQ(ierr);
    ierr = PetscFnVecsPushVec(der-1, subvecs, onevec, &newvecs);CHKERRQ(ierr);
    ierr = (*fn->ops->derivativevec)(fn,x,der,rangeIdx,subsets,newvecs,grad);CHKERRQ(ierr);
    ierr = PetscFnVecToMat(grad, colVec, reuse, M, Mpre);CHKERRQ(ierr);
    ierr = PetscFree(newvecs);CHKERRQ(ierr);
    ierr = VecDestroy(&onevec);CHKERRQ(ierr);
    ierr = VecDestroy(&grad);CHKERRQ(ierr);
  } else if (fn->isScalar && rangeIdx >= der - 1 && fn->ops->scalarderivativevec) {
    /* for a scalar functional derivative to produce a vector, it must contract against der-1 vectors */
    IS          *scalarsubsets = NULL;
    Vec         *scalarvecs = NULL;
    Vec         grad;
    PetscBool   colVec = (der == rangeIdx) ? PETSC_FALSE: PETSC_TRUE;

    ierr = PetscFnGetISVecsWithoutRange(rangeIdx, der+1, subsets, der-1, subvecs, &scalarsubsets, &scalarvecs);CHKERRQ(ierr);
    ierr = PetscFnCreateVecs(fn, (der == rangeIdx) ? rightIS : leftIS, &grad, NULL, NULL);CHKERRQ(ierr);
    ierr = (*fn->ops->scalarderivativevec)(fn,x,der,scalarsubsets,scalarvecs,grad);CHKERRQ(ierr);
    ierr = PetscFnVecToMat(grad, colVec, reuse, M, Mpre);CHKERRQ(ierr);
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
    if (Mpre && (!M || *Mpre != *M)) {
      ierr = MatScale(*Mpre, z);CHKERRQ(ierr);
    }
    ierr = PetscFnRestoreISVecsWithoutRange(rangeIdx, der+1, subsets, der-1, subvecs, &scalarsubsets, &scalarvecs);CHKERRQ(ierr);
  } else SETERRQ1(PetscObjectComm((PetscObject)fn), PETSC_ERR_SUP, "This PetscFn does not implement %s()", PETSC_FUNCTION_NAME);
  for (i = 0; i < der-1; i++) {
    ierr = VecLockReadPop(subvecs[i]);CHKERRQ(ierr);
  }
  ierr = VecLockReadPop(x);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
  if (reuse == MAT_INITIAL_MATRIX) {
    PetscLayout    rightLayout = (der-1 == rangeIdx) ? fn->rmap : fn->dmap;
    PetscLayout    leftLayout = (der == rangeIdx) ? fn->rmap : fn->dmap;

    if (M) {ierr = PetscFnMatCheckCompatible(*M, rightIS, leftIS, rightLayout, leftLayout);CHKERRQ(ierr);}
    if (Mpre) {ierr = PetscFnMatCheckCompatible(*Mpre, rightIS, leftIS, rightLayout, leftLayout);CHKERRQ(ierr);}
  }
#endif
  if (der > 0 && fn->test_mat) {
    PetscHashIJKey key;
    PetscBool      missing;

    key.i = der;
    key.j = rangeIdx;
    ierr = PetscHMapIJQuerySet(fn->testedmat, key, 1, &missing);CHKERRQ(ierr);
    if (missing) {
      PetscReal norm, err;

      if (M && *M) {ierr = PetscFnTestDerivativeMat(fn, x, der, rangeIdx, subsets, subvecs, *M, NULL, &norm, &err);CHKERRQ(ierr);}
      if (Mpre && *Mpre) {ierr = PetscFnTestDerivativeMat(fn, x, der, rangeIdx, subsets, subvecs, *Mpre, NULL, &norm, &err);CHKERRQ(ierr);}
    }
  }
  PetscFunctionReturn(0);
}

/*@
   PetscFnScalarDerivativeScalar - Computes a partial derivative of a PetscFn scalar function, fully contracted with variations.

   If $X$ is the domain space of a scalar function $f$, $D^{(k)}f(x)$ can be viewed as a $k$-order
   tensor, $D^{(k)}f(x):X\times\dots\times X \to \mathbb{Z}$ ($\mathbb{Z}$ is the field of PetscScalar). 

   If the tensor is contracted against the $k$ variations that are provided, the result is a scalar. 

   Partial derivatives of $f$ can be computed by using variations over subsets of $X$.

   Collective on PetscFn and Vec

   Input Parameters:
+  fn - the PetscFn function
.  x  - the domain vector where $D^{(k)} f$ is evaluated
.  der - the order $k >= 0$ of the derivative.
.  subsets - (optional) if a partial derivative is desired, a list of (der) subsets can be given that indicates
which partial derivative is desired.  If subsets[i] is NULL, it indicates that the full derivative is desired for that
variation.  If subsets is NULL, it indicates that the full derivative is desired for all variations.
-  subvecs - (der) variations that contract against the derivative tensor.


   Output Parameters:
.  z - the result

   Notes:
   If der == 0, this is equivalent to PetscFnScalarApply()
   If der == 1, this is the inner product of the gradient of $f$ on a variation
   If der == 2, this is the bilinear action of the Hessian of $f$ on two variations

   Level: advanced
@*/
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
  ierr = VecLockReadPush(x);CHKERRQ(ierr);
  for (i = 0; i < der; i++) {
    ierr = VecLockReadPush(subvecs[i]);CHKERRQ(ierr);
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
    ierr = VecLockReadPop(subvecs[i]);CHKERRQ(ierr);
  }
  ierr = VecLockReadPop(x);CHKERRQ(ierr);
  if (der > 0 && fn->test_scalar) {
    PetscHashIJKey key;
    PetscBool      missing;

    key.i = der;
    key.j = -1;
    ierr = PetscHMapIJQuerySet(fn->testedscalar, key, 1, &missing);CHKERRQ(ierr);
    if (missing) {
      PetscReal rate;

      ierr = PetscFnTestScalarDerivativeScalar(fn, x, der, subsets, subvecs, *z, PETSC_DEFAULT, PETSC_DEFAULT, &rate);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/*@
   PetscFnScalarDerivativeVec - Computes a partial derivative of a PetscFn scalar function, partially contracted with variations
   resulting in a vector y.

   If $X$ is the domain space of a scalar function $f$, $D^{(k)}f(x)$ can be viewed as a $k$-order
   tensor, $D^{(k)}f(x):X\times\dots\times X \to \mathbb{Z}$ ($\mathbb{Z}$ is the field of PetscScalar). 

   If the tensor is contracted against the $k-1$ variations that are provided, the result is a vector $y$. 
   The dot product of $y$ with another vector $z$ will give the same scalar value as if $z$ had been
   included in the $k$ variations given to PetscFnScalarDerivativeScalar().

   Partial derivatives of $f$ can be computed by using variations over subsets of $X$.

   Collective on PetscFn and Vec

   Input Parameters:
+  fn - the PetscFn function
.  x  - the domain vector where $D^{(k)} f$ is evaluated
.  der - the order $k >= 1$ of the derivative.
.  subsets - (optional) if a partial derivative is desired, a list of (der) subsets can be given that indicates
which partial derivative is desired.  If subsets[i] is NULL, it indicates that the full derivative is desired for that
variation.  If subsets is NULL, it indicates that the full derivative is desired for all variations.
-  subvecs - (der-1) variations that contract against the derivative tensor.

   Output Parameters:
.  y - the result

   Notes:
   If der == 1, this is gradient of $f$
   If der == 2, this is action of the Hessian of $f$ on a varation

   Level: advanced
@*/
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
  ierr = VecLockReadPush(x);CHKERRQ(ierr);
  for (i = 0; i < der-1; i++) {
    ierr = VecLockReadPush(subvecs[i]);CHKERRQ(ierr);
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
  for (i = 0; i < der-1; i++) {
    ierr = VecLockReadPop(subvecs[i]);CHKERRQ(ierr);
  }
  ierr = VecLockReadPop(x);CHKERRQ(ierr);
  if (der > 0 && fn->test_vec) {
    PetscHashIJKey key;
    PetscBool      missing;

    key.i = der;
    key.j = -1;
    ierr = PetscHMapIJQuerySet(fn->testedvec, key, 1, &missing);CHKERRQ(ierr);
    if (missing) {
      PetscReal rate;

      ierr = PetscFnTestScalarDerivativeVec(fn, x, der, subsets, subvecs, y, PETSC_DEFAULT, PETSC_DEFAULT, &rate);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/*@
   PetscFnScalarDerivativeMat - Computes a partial derivative of a PetscFn scalar function, partially contracted with variations
   resulting in a matrix and an approximation of that matrix for use in preconditioning.

   If $X$ is the domain space of a scalar function $f$, $D^{(k)}f(x)$ can be viewed as a $k$-order
   tensor, $D^{(k)}f(x):X\times\dots\times X \to \mathbb{Z}$ ($\mathbb{Z}$ is the field of PetscScalar). 

   If the tensor is contracted against the $k-2$ variations that are provided, the result is a matrix $M$.
   The matrix vector product of $M$ with another vector $z$ will give the same vector value as if $z$
   had been included in the $k-1$ variations given to PetscFnScalarDerivativeVec().

   Partial derivatives of $f$ can be computed by using variations over subsets of $X$.

   Collective on PetscFn and Vec

   Input Parameters:
+  fn - the PetscFn function
.  x  - the domain vector where $D^{(k)} f$ is evaluated
.  der - the order $k >= 2$ of the derivative.
.  subsets - (optional) if a partial derivative is desired, a list of (der) subsets can be given that indicates
which partial derivative is desired.  If subsets[i] is NULL, it indicates that the full derivative is desired for that
variation.  If subsets is NULL, it indicates that the full derivative is desired for all variations.
-  subvecs - (der-2) variations that contract against the derivative tensor.

   Output Parameters:
+  M - the result matrix.
-  Mpre - an optional preconditioner for M.

   Notes:
   If der == 2, this is the Hessian of $f$

   Level: advanced
@*/
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
  ierr = VecLockReadPush(x);CHKERRQ(ierr);
  for (i = 0; i < der - 2; i++) {
    ierr = VecLockReadPush(subvecs[i]);CHKERRQ(ierr);
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
    ierr = PetscFree(vecsubvecs);CHKERRQ(ierr);
    ierr = PetscFree(vecsubsets);CHKERRQ(ierr);
    ierr = VecDestroy(&onevec);CHKERRQ(ierr);
  } else SETERRQ1(PetscObjectComm((PetscObject)fn), PETSC_ERR_SUP, "This PetscFn does not implement %s()", PETSC_FUNCTION_NAME);
  for (i = 0; i < der-2; i++) {
    ierr = VecLockReadPop(subvecs[i]);CHKERRQ(ierr);
  }
  ierr = VecLockReadPop(x);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
  if (reuse == MAT_INITIAL_MATRIX) {
    PetscLayout    rightLayout = fn->dmap;
    PetscLayout    leftLayout = fn->dmap;

    if (M) {ierr = PetscFnMatCheckCompatible(*M, rightIS, leftIS, rightLayout, leftLayout);CHKERRQ(ierr);}
    if (Mpre) {ierr = PetscFnMatCheckCompatible(*Mpre, rightIS, leftIS, rightLayout, leftLayout);CHKERRQ(ierr);}
  }
#endif
  if (der > 0 && fn->test_mat) {
    PetscHashIJKey key;
    PetscBool      missing;

    key.i = der;
    key.j = -1;
    ierr = PetscHMapIJQuerySet(fn->testedmat, key, 1, &missing);CHKERRQ(ierr);
    if (missing) {
      PetscReal norm, err;

      if (M && *M) {ierr = PetscFnTestScalarDerivativeMat(fn, x, der, subsets, subvecs, *M, NULL, &norm, &err);CHKERRQ(ierr);}
      if (Mpre && *Mpre) {ierr = PetscFnTestScalarDerivativeMat(fn, x, der, subsets, subvecs, *Mpre, NULL, &norm, &err);CHKERRQ(ierr);}
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnJacobianMult(PetscFn fn, Vec x, Vec xhat, Vec Jxhat)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnDerivativeVec(fn, x, 1, 1, NULL, &xhat, Jxhat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnJacobianMultAdjoint(PetscFn fn, Vec x, Vec v, Vec Jadjv)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnDerivativeVec(fn, x, 1, 0, NULL, &v, Jadjv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnJacobianBuild(PetscFn fn, Vec x, MatReuse reuse, Mat *J, Mat *Jpre)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnDerivativeMat(fn, x, 1, 1, NULL, NULL, reuse, J, Jpre);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnJacobianBuildAdjoint(PetscFn fn, Vec x, MatReuse reuse, Mat *Jadj, Mat *Jadjpre)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnDerivativeMat(fn, x, 1, 0, NULL, NULL, reuse, Jadj, Jadjpre);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnHessianMult(PetscFn fn, Vec x, Vec xhat, Vec xdot, Vec Hxhatxdot)
{
  Vec            subvecs[2];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  subvecs[0] = xhat;
  subvecs[1] = xdot;
  ierr = PetscFnDerivativeVec(fn, x, 2, 2, NULL, subvecs, Hxhatxdot);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnHessianMultAdjoint(PetscFn fn, Vec x, Vec v, Vec xhat, Vec Hadjvxhat)
{
  Vec            subvecs[2];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  subvecs[0] = v;
  subvecs[1] = xhat;
  ierr = PetscFnDerivativeVec(fn, x, 2, 0, NULL, subvecs, Hadjvxhat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnHessianBuild(PetscFn fn, Vec x, Vec xhat, MatReuse reuse, Mat *H, Mat *Hpre)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnDerivativeMat(fn, x, 2, 2, NULL, &xhat, reuse, H, Hpre);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnHessianBuildSwap(PetscFn fn, Vec x, Vec xhat, MatReuse reuse, Mat *Hswp, Mat *Hswppre)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnDerivativeMat(fn, x, 2, 1, NULL, &xhat, reuse, Hswp, Hswppre);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnHessianBuildAdjoint(PetscFn fn, Vec x, Vec v, MatReuse reuse, Mat *Hadj, Mat *Hadjpre)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnDerivativeMat(fn, x, 2, 0, NULL, &v, reuse, Hadj, Hadjpre);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   PetscFnApply - Computes the PetscFn scalar-valued action, z = F(x).

   Collective on PetscFn and Vec

   Input Parameters:
+  fn - the PetscFn function
-  x  - the domain vector

   Output Parameters:
.  z - the result

   Level: beginner
@*/
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

  ierr = VecLockReadPush(x);CHKERRQ(ierr);
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
  ierr = VecLockReadPop(x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnScalarGradient(PetscFn fn, Vec x, Vec g)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnScalarDerivativeVec(fn, x, 1, NULL, NULL, g);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnScalarHessianMult(PetscFn fn, Vec x, Vec xhat, Vec Hxhat)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnScalarDerivativeVec(fn, x, 2, NULL, &xhat, Hxhat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnScalarHessianBuild(PetscFn fn, Vec x, MatReuse reuse, Mat *H, Mat *Hpre)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnScalarDerivativeMat(fn, x, 2, NULL, NULL, reuse, H, Hpre);CHKERRQ(ierr);
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
                                  "scalarapply",
                                  "createsubfns",
                                  "destroysubfns",
                                  "createsubfn",
                                  "createderivativefn",
                                  "destroy",
                                  "view",
                                  };

PetscErrorCode PetscFnTestDerivativeVec(PetscFn fn, Vec x, PetscInt der, PetscInt rangeIdx, const IS subsets[], const Vec subvecs[], Vec y, PetscReal e1, PetscReal e2, PetscReal * rate)
{
  PetscInt       i;
  PetscReal      diff[2];
  PetscReal      e[2];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn, PETSCFN_CLASSID, 1);
  PetscValidHeaderSpecific(x, VEC_CLASSID, 2);
  e[0] = e1;
  e[1] = e2;
  if (fn->setfromoptions && e1 < 0. && e2 < 0.) {
    PetscInt two = 2;

    ierr = PetscOptionsGetRealArray(((PetscObject)fn)->options,((PetscObject)fn)->prefix,"-fn_test_derivative_offsets",e,&two,NULL);CHKERRQ(ierr);
  }
  if (e[0] < 0.) {e[0] = 2. * PetscSqrtReal(PETSC_SMALL);}
  if (e[1] < 0.) {e[1] = PetscSqrtReal(PETSC_SMALL);}
  if (rangeIdx == der) {
    Vec xhat;
    const IS *newsubsets;
    const Vec *newsubvecs;
    Vec y0, ye, ypred;
    Vec xe;
    PetscScalar xhatdot;

    if (!subsets || subsets[0] == NULL) {
      ierr = VecDuplicate(subvecs[0], &xhat);CHKERRQ(ierr);
      ierr = VecCopy(subvecs[0], xhat);CHKERRQ(ierr);
    } else {
      const PetscScalar *va;
      const PetscInt *idx;
      PetscInt n;

      ierr = VecDuplicate(x, &xhat);CHKERRQ(ierr);
      ierr = VecGetArrayRead(subvecs[0], &va);CHKERRQ(ierr);
      ierr = ISGetLocalSize(subsets[0], &n);CHKERRQ(ierr);
      ierr = ISGetIndices(subsets[0], &idx);CHKERRQ(ierr);
      ierr = VecSetValues(xhat, n, idx, va, INSERT_VALUES);CHKERRQ(ierr);
      ierr = ISRestoreIndices(subsets[0], &idx);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(subvecs[0], &va);CHKERRQ(ierr);
      ierr = VecAssemblyBegin(xhat);CHKERRQ(ierr);
      ierr = VecAssemblyEnd(xhat);CHKERRQ(ierr);
    }
    ierr = VecNormalize(xhat, &xhatdot);CHKERRQ(ierr);

    newsubsets = (const IS *) (subsets ? &(subsets[1]) : NULL);
    newsubvecs = (const Vec *) (der > 1 ? &(subvecs[1]) : NULL);
    ierr = VecDuplicate(y, &y0);CHKERRQ(ierr);
    ierr = PetscFnDerivativeVec(fn, x, der - 1, rangeIdx - 1, newsubsets, newsubvecs, y0);CHKERRQ(ierr);
    ierr = VecDuplicate(y, &ye);CHKERRQ(ierr);
    ierr = VecDuplicate(y, &ypred);CHKERRQ(ierr);
    ierr = VecDuplicate(x, &xe);CHKERRQ(ierr);
    for (i = 0; i < 2; i++) {
      ierr = VecCopy(y0, ypred);CHKERRQ(ierr);
      ierr = VecAXPY(ypred,e[i]/xhatdot,y);CHKERRQ(ierr);
      ierr = VecCopy(x, xe);CHKERRQ(ierr);
      ierr = VecAXPY(xe,e[i],xhat);CHKERRQ(ierr);
      ierr = PetscFnDerivativeVec(fn, xe, der-1, rangeIdx - 1, newsubsets, newsubvecs, ye);CHKERRQ(ierr);
      ierr = VecAXPY(ye,-1.,ypred);CHKERRQ(ierr);
      ierr = VecNorm(ye,NORM_2,&diff[i]);CHKERRQ(ierr);
    }
    ierr = VecDestroy(&xe);CHKERRQ(ierr);
    ierr = VecDestroy(&ypred);CHKERRQ(ierr);
    ierr = VecDestroy(&ye);CHKERRQ(ierr);
    ierr = VecDestroy(&y0);CHKERRQ(ierr);
    ierr = VecDestroy(&xhat);CHKERRQ(ierr);
  } else {
    Vec xhat;
    PetscScalar z0, ze, zpred;
    PetscScalar xhatdot;
    Vec xe;

    if (!subsets || subsets[der] == NULL) {
      ierr = VecDuplicate(y, &xhat);CHKERRQ(ierr);
      ierr = VecCopy(y, xhat);CHKERRQ(ierr);
    } else {
      const PetscScalar *va;
      const PetscInt *idx;
      PetscInt n;

      ierr = VecDuplicate(x, &xhat);CHKERRQ(ierr);
      ierr = VecGetArrayRead(y, &va);CHKERRQ(ierr);
      ierr = ISGetLocalSize(subsets[der], &n);CHKERRQ(ierr);
      ierr = ISGetIndices(subsets[der], &idx);CHKERRQ(ierr);
      ierr = VecSetValues(xhat, n, idx, va, INSERT_VALUES);CHKERRQ(ierr);
      ierr = ISRestoreIndices(subsets[der], &idx);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(y, &va);CHKERRQ(ierr);
      ierr = VecAssemblyBegin(xhat);CHKERRQ(ierr);
      ierr = VecAssemblyEnd(xhat);CHKERRQ(ierr);
    }
    ierr = VecNormalize(xhat, &xhatdot);CHKERRQ(ierr);
    if (rangeIdx < 0) {
      ierr = PetscFnScalarDerivativeScalar(fn, x, der - 1, subsets, subvecs, &z0);CHKERRQ(ierr);
    } else {
      ierr = PetscFnDerivativeScalar(fn, x, der - 1, rangeIdx, subsets, subvecs, &z0);CHKERRQ(ierr);
    }
    ierr = VecDuplicate(x, &xe);CHKERRQ(ierr);
    for (i = 0; i < 2; i++) {
      zpred = z0 + e[i] * xhatdot;
      ierr = VecCopy(x, xe);CHKERRQ(ierr);
      ierr = VecAXPY(xe,e[i],xhat);CHKERRQ(ierr);
      if (rangeIdx < 0) {
        ierr = PetscFnScalarDerivativeScalar(fn, xe, der-1, subsets, subvecs, &ze);CHKERRQ(ierr);
      } else {
        ierr = PetscFnDerivativeScalar(fn, xe, der-1, rangeIdx, subsets, subvecs, &ze);CHKERRQ(ierr);
      }
      diff[i] = PetscAbsScalar(zpred - ze);
    }
    ierr = VecDestroy(&xe);CHKERRQ(ierr);
    ierr = VecDestroy(&xhat);CHKERRQ(ierr);
  }
  *rate = PetscLog2Real(diff[1] / diff[0]) / PetscLog2Real(e[1] / e[0]);
  if (fn->setfromoptions) {
    PetscBool view = PETSC_FALSE;

    ierr = PetscOptionsGetBool(((PetscObject)fn)->options,((PetscObject)fn)->prefix,"-fn_test_derivative_view",&view,NULL);CHKERRQ(ierr);
    if (view) {
      MPI_Comm comm = PetscObjectComm((PetscObject)fn);
      if (rangeIdx >= 0) {
        ierr = PetscPrintf(comm, "%s: Derivative order %D, range index %D, tested at offsets (%g, %g); tangents differ by (%g, %g): measured rate %g\n", PETSC_FUNCTION_NAME, der, rangeIdx, e[0], e[1], diff[0], diff[1], *rate);CHKERRQ(ierr);
      } else {
        ierr = PetscPrintf(comm, "%s: Derivative order %D, scalar function, tested at offsets (%g, %g); tangents differ by (%g, %g): measured rate %g\n", PETSC_FUNCTION_NAME, der, e[0], e[1], diff[0], diff[1], *rate);CHKERRQ(ierr);
      }
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnTestScalarDerivativeScalar(PetscFn fn, Vec x, PetscInt der, const IS subsets[], const Vec subvecs[], PetscScalar z, PetscReal e1, PetscReal e2, PetscReal *rate)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnTestDerivativeScalar(fn, x, der, -1, subsets, subvecs, z, e1, e2, rate);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnTestDerivativeScalar(PetscFn fn, Vec x, PetscInt der, PetscInt rangeIdx, const IS subsets[], const Vec subvecs[], PetscScalar z, PetscReal e1, PetscReal e2, PetscReal * rate)
{
  PetscInt       i;
  PetscReal      diff[2];
  PetscReal      e[2];
  PetscErrorCode ierr;
  Vec xhat, xe;
  const IS *newsubsets;
  const Vec *newsubvecs;
  PetscScalar z0, ze, zpred;
  PetscScalar xhatdot;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn, PETSCFN_CLASSID, 1);
  PetscValidHeaderSpecific(x, VEC_CLASSID, 2);
  e[0] = e1;
  e[1] = e2;
  if (fn->setfromoptions && e1 < 0. && e2 < 0.) {
    PetscInt two = 2;

    ierr = PetscOptionsGetRealArray(((PetscObject)fn)->options,((PetscObject)fn)->prefix,"-fn_test_derivative_offsets",e,&two,NULL);CHKERRQ(ierr);
  }
  if (e[0] < 0.) {e[0] = 2. * PetscSqrtReal(PETSC_SMALL);}
  if (e[1] < 0.) {e[1] = PetscSqrtReal(PETSC_SMALL);}
  if (rangeIdx == 0) {
    newsubsets = subsets;
    newsubvecs = subvecs;
    if (!subsets || subsets[der] == NULL) {
      ierr = VecDuplicate(subvecs[der], &xhat);CHKERRQ(ierr);
      ierr = VecCopy(subvecs[der], xhat);CHKERRQ(ierr);
    } else {
      const PetscScalar *va;
      const PetscInt *idx;
      PetscInt n;

      ierr = VecDuplicate(x, &xhat);CHKERRQ(ierr);
      ierr = VecGetArrayRead(subvecs[der], &va);CHKERRQ(ierr);
      ierr = ISGetLocalSize(subsets[der], &n);CHKERRQ(ierr);
      ierr = ISGetIndices(subsets[der], &idx);CHKERRQ(ierr);
      ierr = VecSetValues(xhat, n, idx, va, INSERT_VALUES);CHKERRQ(ierr);
      ierr = ISRestoreIndices(subsets[der], &idx);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(subvecs[der], &va);CHKERRQ(ierr);
      ierr = VecAssemblyBegin(xhat);CHKERRQ(ierr);
      ierr = VecAssemblyEnd(xhat);CHKERRQ(ierr);
    }
  } else {
    newsubsets = (const IS *) (subsets ? &subsets[1] : NULL);
    newsubvecs = (const Vec *) (subvecs ? &subvecs[1] : NULL);
    if (!subsets || subsets[0] == NULL) {
      ierr = VecDuplicate(subvecs[0], &xhat);CHKERRQ(ierr);
      ierr = VecCopy(subvecs[0], xhat);CHKERRQ(ierr);
    } else {
      const PetscScalar *va;
      const PetscInt *idx;
      PetscInt n;

      ierr = VecDuplicate(x, &xhat);CHKERRQ(ierr);
      ierr = VecGetArrayRead(subvecs[0], &va);CHKERRQ(ierr);
      ierr = ISGetLocalSize(subsets[0], &n);CHKERRQ(ierr);
      ierr = ISGetIndices(subsets[0], &idx);CHKERRQ(ierr);
      ierr = VecSetValues(xhat, n, idx, va, INSERT_VALUES);CHKERRQ(ierr);
      ierr = ISRestoreIndices(subsets[0], &idx);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(subvecs[0], &va);CHKERRQ(ierr);
      ierr = VecAssemblyBegin(xhat);CHKERRQ(ierr);
      ierr = VecAssemblyEnd(xhat);CHKERRQ(ierr);
    }
  }
  ierr = VecNormalize(xhat, &xhatdot);CHKERRQ(ierr);
  ierr = VecDuplicate(x, &xe);CHKERRQ(ierr);

  if (rangeIdx < 0) {
    ierr = PetscFnScalarDerivativeScalar(fn, x, der - 1, newsubsets, newsubvecs, &z0);CHKERRQ(ierr);
  } else {
    ierr = PetscFnDerivativeScalar(fn, x, der - 1, rangeIdx == 0 ? 0 : rangeIdx - 1, newsubsets, newsubvecs, &z0);CHKERRQ(ierr);
  }
  for (i = 0; i < 2; i++) {
    zpred = z0 + e[i] * z / xhatdot;
    ierr = VecCopy(x, xe);CHKERRQ(ierr);
    ierr = VecAXPY(xe,e[i],xhat);CHKERRQ(ierr);
    if (rangeIdx < 0) {
      ierr = PetscFnScalarDerivativeScalar(fn, xe, der-1, newsubsets, newsubvecs, &ze);CHKERRQ(ierr);
    } else {
      ierr = PetscFnDerivativeScalar(fn, xe, der-1, rangeIdx == 0 ? 0 : rangeIdx - 1, newsubsets, newsubvecs, &ze);CHKERRQ(ierr);
    }
    diff[i] = PetscAbsScalar(zpred - ze);
  }
  ierr = VecDestroy(&xe);CHKERRQ(ierr);
  ierr = VecDestroy(&xhat);CHKERRQ(ierr);
  *rate = PetscLog2Real(diff[1] / diff[0]) / PetscLog2Real(e[1] / e[0]);
  if (fn->setfromoptions) {
    PetscBool view = PETSC_FALSE;

    ierr = PetscOptionsGetBool(((PetscObject)fn)->options,((PetscObject)fn)->prefix,"-fn_test_derivative_view",&view,NULL);CHKERRQ(ierr);
    if (view) {
      MPI_Comm comm = PetscObjectComm((PetscObject)fn);
      if (rangeIdx >= 0) {
        ierr = PetscPrintf(comm, "%s: Derivative order %D, range index %D, tested at offsets (%g, %g); tangents differ by (%g, %g): measured rate %g\n", PETSC_FUNCTION_NAME, der, rangeIdx, e[0], e[1], diff[0], diff[1], *rate);CHKERRQ(ierr);
      } else {
        ierr = PetscPrintf(comm, "%s: Derivative order %D, scalar function, tested at offsets (%g, %g); tangents differ by (%g, %g): measured rate %g\n", PETSC_FUNCTION_NAME, der, e[0], e[1], diff[0], diff[1], *rate);CHKERRQ(ierr);
      }
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnTestScalarDerivativeVec(PetscFn fn, Vec x, PetscInt der, const IS subsets[], const Vec subvecs[], Vec y, PetscReal e1, PetscReal e2, PetscReal *rate)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnTestDerivativeVec(fn, x, der, -1, subsets, subvecs, y, e1, e2, rate);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnTestDerivativeMat(PetscFn fn, Vec x, PetscInt der, PetscInt rangeIdx, const IS subsets[], const Vec subvecs[], Mat M, PetscRandom rand, PetscReal *norm, PetscReal *err)
{
  PetscRandom    rorig = rand;
  Vec            b, c, cvec;
  Vec            *newsubvecs;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn, PETSCFN_CLASSID, 1);
  PetscValidHeaderSpecific(x, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(M, MAT_CLASSID, 7);
  if (!rand) {
    ierr = PetscRandomCreate(PetscObjectComm((PetscObject)fn),&rand);CHKERRQ(ierr);
    if (fn->setfromoptions) {
      ierr = PetscObjectSetOptionsPrefix((PetscObject)rand,((PetscObject)fn)->prefix);CHKERRQ(ierr);
      ierr = PetscObjectAppendOptionsPrefix((PetscObject)rand,"fn_test_derivativemat_");CHKERRQ(ierr);
      ierr = PetscRandomSetFromOptions(rand);CHKERRQ(ierr);
    }
  }
  ierr = MatCreateVecs(M, &b, &c);CHKERRQ(ierr);
  ierr = VecSetRandom(b, rand);CHKERRQ(ierr);
  ierr = PetscFnVecsPushVec(rangeIdx < 0 ? der - 2 : der - 1, subvecs, b, &newsubvecs);CHKERRQ(ierr);
  ierr = MatMult(M, b, c);CHKERRQ(ierr);
  ierr = VecDuplicate(c, &cvec);CHKERRQ(ierr);
  if (rangeIdx < 0) {
    ierr = PetscFnScalarDerivativeVec(fn, x, der, subsets, newsubvecs, cvec);CHKERRQ(ierr);
  } else {
    ierr = PetscFnDerivativeVec(fn, x, der, rangeIdx, subsets, newsubvecs, cvec);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = VecAXPY(cvec, -1., c);CHKERRQ(ierr);
  ierr = VecNorm(c, NORM_2, norm);CHKERRQ(ierr);
  ierr = VecNorm(cvec, NORM_2, err);CHKERRQ(ierr);
  ierr = VecDestroy(&cvec);CHKERRQ(ierr);
  ierr = VecDestroy(&c);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = PetscFree(newsubvecs);CHKERRQ(ierr);
  if (rorig != rand) {ierr = PetscRandomDestroy(&rand);CHKERRQ(ierr);}
  if (fn->setfromoptions) {
    PetscBool view = PETSC_FALSE;

    ierr = PetscOptionsGetBool(((PetscObject)fn)->options,((PetscObject)fn)->prefix,"-fn_test_derivativemat_view",&view,NULL);CHKERRQ(ierr);
    if (view) {
      MPI_Comm comm = PetscObjectComm((PetscObject)fn);
      if (rangeIdx >= 0) {
        ierr = PetscPrintf(comm, "%s: Derivative order %D, range index %D, tested matrix against matrix-free action: norm %g, error %g\n", PETSC_FUNCTION_NAME, der, rangeIdx, (double) *norm, (double) *err);CHKERRQ(ierr);
      } else {
        ierr = PetscPrintf(comm, "%s: Derivative order %D, scalar function, tested matrix against matrix-free action: norm %g, error %g\n", PETSC_FUNCTION_NAME, der, (double) *norm, (double) *err);CHKERRQ(ierr);
      }
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnTestScalarDerivativeMat(PetscFn fn, Vec x, PetscInt der, const IS subsets[], const Vec subvecs[], Mat M, PetscRandom rand, PetscReal *norm, PetscReal *err)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnTestDerivativeMat(fn, x, der, -1, subsets, subvecs, M, rand, norm, err);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnTestDerivativeFn(PetscFn fn, PetscFn df, PetscInt der, PetscInt rangeIdx, PetscInt numDots, const IS dotISs[], const Vec dotVecs[], Vec x, PetscReal *norm, PetscReal *err)
{
  PetscInt       maxDots;
  MPI_Comm       comm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn, PETSCFN_CLASSID, 1);
  PetscValidHeaderSpecific(x, VEC_CLASSID, 6);
  comm = PetscObjectComm((PetscObject)fn);
  if (rangeIdx < 0) maxDots = der;
  else              maxDots = der + 1;
  if (numDots < maxDots - 1 || numDots > maxDots) SETERRQ2(comm, PETSC_ERR_ARG_OUTOFRANGE, "This operation contracts with at most %D vectors: %D given", maxDots, numDots);
  if (numDots == maxDots) {
    PetscReal      dfz, fnz;

    ierr = PetscFnScalarApply(df, x, &dfz);CHKERRQ(ierr);
    if (rangeIdx < 0) {
      ierr = PetscFnScalarDerivativeScalar(fn, x, der, dotISs, dotVecs, &fnz);CHKERRQ(ierr);
    } else {
      ierr = PetscFnDerivativeScalar(fn, x, der, rangeIdx, dotISs, dotVecs, &fnz);CHKERRQ(ierr);
    }
    *norm = PetscAbsScalar(fnz);
    *err = PetscAbsScalar(fnz - dfz);
  } else {
    IS *newISs;
    Vec            dfOut, fnOut;

    if (rangeIdx < maxDots - 1) {
      ierr = PetscFnCreateVecs(fn, NULL, &fnOut, NULL, NULL);CHKERRQ(ierr);
    } else {
      ierr = PetscFnCreateVecs(fn, NULL, NULL, NULL, &fnOut);CHKERRQ(ierr);
    }
    ierr = PetscFnCreateVecs(df, NULL, NULL, NULL, &dfOut);CHKERRQ(ierr);
    ierr = PetscFnApply(df, x, dfOut);CHKERRQ(ierr);
    ierr = PetscFnISsPushIS(numDots, dotISs, NULL, &newISs);CHKERRQ(ierr);
    if (rangeIdx < 0) {
      ierr = PetscFnScalarDerivativeVec(fn, x, der, newISs, dotVecs, fnOut);CHKERRQ(ierr);
    } else {
      ierr = PetscFnDerivativeVec(fn, x, der, rangeIdx, newISs, dotVecs, fnOut);CHKERRQ(ierr);
    }
    ierr = PetscFree(newISs);CHKERRQ(ierr);
    ierr = VecAXPY(dfOut, -1., fnOut);CHKERRQ(ierr);
    ierr = VecNorm(fnOut, NORM_2, norm);CHKERRQ(ierr);
    ierr = VecNorm(dfOut, NORM_2, err);CHKERRQ(ierr);
    ierr = VecDestroy(&dfOut);CHKERRQ(ierr);
    ierr = VecDestroy(&fnOut);CHKERRQ(ierr);
  }
  if (fn->setfromoptions) {
    PetscBool view = PETSC_FALSE;

    ierr = PetscOptionsGetBool(((PetscObject)fn)->options,((PetscObject)fn)->prefix,"-fn_test_derivativefn_view",&view,NULL);CHKERRQ(ierr);
    if (view) {
      MPI_Comm comm = PetscObjectComm((PetscObject)fn);
      if (rangeIdx >= 0) {
        ierr = PetscPrintf(comm, "%s: Derivative order %D, range index %D, tested instantiated function against matrix-free action: norm %g, error %g\n", PETSC_FUNCTION_NAME, der, rangeIdx, (double) *norm, (double) *err);CHKERRQ(ierr);
      } else {
        ierr = PetscPrintf(comm, "%s: Derivative order %D, scalar function, tested instantiated function against matrix-free action: norm %g, error %g\n", PETSC_FUNCTION_NAME, der, (double) *norm, (double) *err);CHKERRQ(ierr);
      }
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnTestScalarDerivativeFn(PetscFn fn, PetscFn df, PetscInt der, PetscInt numDots, const IS dotISs[], const Vec dotVecs[], Vec x, PetscReal *norm, PetscReal *err)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnTestDerivativeFn(fn, df, der, -1, numDots, dotISs, dotVecs, x, norm, err);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

typedef struct
{
  PetscFn origFn;
  PetscInt numDots;
  PetscInt maxDots;
  PetscInt der;
  PetscInt rangeIdx;
  Vec *dotVecs;
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
  ierr = PetscFree(derShell->dotVecs);CHKERRQ(ierr);
  if (derShell->dotISs) {
    for (i = 0; i < derShell->numDots; i++) {
      ierr = ISDestroy(&(derShell->dotISs[i]));CHKERRQ(ierr);
    }
  }
  ierr = PetscFree(derShell->dotISs);CHKERRQ(ierr);
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
  if (der == 0 && fn->test_self_as_derfn) {
    PetscReal norm, err;

    fn->test_self_as_derfn = PETSC_FALSE;
    ierr = PetscFnTestDerivativeFn(origFn, fn, derShell->der, derShell->rangeIdx, derShell->numDots, derShell->dotISs, derShell->dotVecs, x, &norm, &err);CHKERRQ(ierr);
  }
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
  if (der == 0 && fn->test_self_as_derfn) {
    PetscReal norm, err;

    fn->test_self_as_derfn = PETSC_FALSE;
    ierr = PetscFnTestDerivativeFn(origFn, fn, derShell->der, derShell->rangeIdx, derShell->numDots, derShell->dotISs, derShell->dotVecs, x, &norm, &err);CHKERRQ(ierr);
  }
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
  if (der == 0 && fn->test_self_as_derfn) {
    PetscReal norm, err;

    fn->test_self_as_derfn = PETSC_FALSE;
    ierr = PetscFnTestDerivativeFn(origFn, fn, derShell->der, derShell->rangeIdx, derShell->numDots, derShell->dotISs, derShell->dotVecs, x, &norm, &err);CHKERRQ(ierr);
  }
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
  if (der == 0 && fn->test_self_as_derfn) {
    PetscReal norm, err;

    fn->test_self_as_derfn = PETSC_FALSE;
    ierr = PetscFnTestDerivativeFn(origFn, fn, derShell->der, derShell->rangeIdx, derShell->numDots, derShell->dotISs, derShell->dotVecs, x, &norm, &err);CHKERRQ(ierr);
  }
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

static PetscErrorCode PetscFnCreateDerivativeFn_DerShell(PetscFn fn, PetscInt der, PetscInt rangeIdx, PetscInt numVecs, const IS subsets[], const Vec subvecs[], PetscFn *derFn)
{
  PetscInt         maxVecs;
  PetscInt         derIsScalar;
  PetscInt         i, m, M;
  PetscFn          df;
  PetscFnDerShell *derShell;
  PetscLayout      rangeLayout = NULL;
  MPI_Comm         comm;

  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn, PETSCFN_CLASSID, 1);
  comm = PetscObjectComm((PetscObject)fn);
  if (rangeIdx < 0) maxVecs = der;
  else              maxVecs = der + 1;
  if (numVecs < maxVecs - 1 || numVecs > maxVecs) SETERRQ3(comm, PETSC_ERR_ARG_OUTOFRANGE, "This operation contracts with at %D or %Dvectors: %D given", maxVecs - 1, maxVecs, numVecs);
  derIsScalar = (numVecs == maxVecs) ? PETSC_TRUE : PETSC_FALSE;
  if (derIsScalar) {
    rangeLayout = NULL;
  } else if (rangeIdx == maxVecs - 1) {
    rangeLayout = fn->rmap;
  } else {
    rangeLayout = fn->dmap;
  }
  if (der == 0 && numVecs == 0) {
    ierr = PetscObjectReference((PetscObject)fn);CHKERRQ(ierr);
    *derFn = fn;
    PetscFunctionReturn(0);
  }
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
  ierr = PetscMalloc1(numVecs, &derShell->dotVecs);CHKERRQ(ierr);
  if (subsets) {
    ierr = PetscMalloc1(numVecs, &derShell->dotISs);CHKERRQ(ierr);
  }
  derShell->origFn   = fn;
  derShell->numDots  = numVecs;
  derShell->maxDots  = maxVecs;
  derShell->dotISs   = NULL;
  derShell->rangeIdx = rangeIdx;
  derShell->der      = der;
  for (i = 0; i < numVecs; i++) {
    ierr = PetscObjectReference((PetscObject)subvecs[i]);CHKERRQ(ierr);
    derShell->dotVecs[i] = subvecs[i];
    if (subsets && subsets[i]) {
      ierr = PetscObjectReference((PetscObject)subsets[i]);CHKERRQ(ierr);
      derShell->dotISs[i] = subsets[i];
    }
  }
  ierr = PetscFnShellSetContext(df, (void *) derShell);CHKERRQ(ierr);
  ierr = PetscFnShellSetOperation(df, PETSCFNOP_DESTROY,                (void (*)(void)) PetscFnShellDestroy_DerShell);CHKERRQ(ierr);
  ierr = PetscFnShellSetOperation(df, PETSCFNOP_CREATEVECS,             (void (*)(void)) PetscFnShellCreateVecs_DerShell);CHKERRQ(ierr);
  if (derIsScalar) {
    ierr = PetscFnShellSetOperation(df, PETSCFNOP_SCALARDERIVATIVESCALAR, (void (*)(void)) PetscFnScalarDerivativeScalar_DerShell);CHKERRQ(ierr);
    ierr = PetscFnShellSetOperation(df, PETSCFNOP_SCALARDERIVATIVEVEC,    (void (*)(void)) PetscFnScalarDerivativeVec_DerShell);CHKERRQ(ierr);
    ierr = PetscFnShellSetOperation(df, PETSCFNOP_SCALARDERIVATIVEMAT,    (void (*)(void)) PetscFnScalarDerivativeMat_DerShell);CHKERRQ(ierr);
  } else {
    ierr = PetscFnShellSetOperation(df, PETSCFNOP_DERIVATIVESCALAR, (void (*)(void)) PetscFnDerivativeScalar_DerShell);CHKERRQ(ierr);
    ierr = PetscFnShellSetOperation(df, PETSCFNOP_DERIVATIVEVEC,    (void (*)(void)) PetscFnDerivativeVec_DerShell);CHKERRQ(ierr);
    ierr = PetscFnShellSetOperation(df, PETSCFNOP_DERIVATIVEMAT,    (void (*)(void)) PetscFnDerivativeMat_DerShell);CHKERRQ(ierr);
  }
  *derFn = df;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnCreateDerivativeFn(PetscFn fn, PetscInt der, PetscInt rangeIdx, PetscInt numVecs, const IS subsets[], const Vec subvecs[], PetscFn *derFn)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn, PETSCFN_CLASSID, 1);
  ierr = PetscFnSetUp(fn);CHKERRQ(ierr);
  if (fn->ops->derivativefn) {
    ierr = (*fn->ops->derivativefn) (fn, der, rangeIdx, numVecs, subsets, subvecs, derFn);CHKERRQ(ierr);
  } else {
    ierr = PetscFnCreateDerivativeFn_DerShell(fn, der, rangeIdx, numVecs, subsets, subvecs, derFn);CHKERRQ(ierr);
  }
  if (fn->test_derfn) {
    (*derFn)->test_self_as_derfn = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnCreateScalarDerivativeFn(PetscFn fn, PetscInt der, PetscInt numVecs, const IS subsets[], const Vec subvecs[], PetscFn *derFn)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn, PETSCFN_CLASSID, 1);
  ierr = PetscFnSetUp(fn);CHKERRQ(ierr);
  if (fn->ops->scalarderivativefn) {
    ierr = (*fn->ops->scalarderivativefn) (fn, der, numVecs, subsets, subvecs, derFn);CHKERRQ(ierr);
  } else {
    ierr = PetscFnCreateDerivativeFn_DerShell(fn, der, -1, numVecs, subsets, subvecs, derFn);CHKERRQ(ierr);
  }
  if (fn->test_derfn) {
    (*derFn)->test_self_as_derfn = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}
