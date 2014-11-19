#include <../src/mat/impls/elemental/matelemsparseimpl.h> /*I "petscmat.h" I*/

#undef __FUNCT__
#define __FUNCT__ "MatGetInfo_ElemSparse"
static PetscErrorCode MatGetInfo_ElemSparse(Mat A,MatInfoType flag,MatInfo *info)
{
  Mat_ElemSparse *a = (Mat_ElemSparse*)A->data;

  PetscFunctionBegin;
  info->block_size     = 1.0;

  if (flag == MAT_LOCAL) {
  } else if (flag == MAT_GLOBAL_MAX) {
  } else if (flag == MAT_GLOBAL_SUM) {
  }

  info->nz_unneeded       = 0.0;
  info->assemblies        = (double)A->num_ass;
  info->mallocs           = 0;
  info->memory            = ((PetscObject)A)->mem;
  info->fill_ratio_given  = 0; /* determined by Elemental */
  info->fill_ratio_needed = 0;
  info->factor_mallocs    = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSetUp_ElemSparse"
static PetscErrorCode MatSetUp_ElemSparse(Mat A)
{
  Mat_ElemSparse *elem = (Mat_ElemSparse*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLayoutSetUp(A->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(A->cmap);CHKERRQ(ierr);
  /* Set up the elemental matrix */
  elem->cmat  = new El::DistSparseMatrix<PetscElemScalar>(A->rmap->N,A->cmap->N,PetscObjectComm((PetscObject)A));
  elem->cvecr = new El::DistMultiVec<PetscElemScalar>(A->cmap->N,1,PetscObjectComm((PetscObject)A));
  elem->cvecl = new El::DistMultiVec<PetscElemScalar>(A->rmap->N,1,PetscObjectComm((PetscObject)A));
  if (A->rmap->rstart != elem->cmat->FirstLocalRow() || A->rmap->rend != elem->cmat->FirstLocalRow()+elem->cmat->LocalHeight()) {
    SETERRQ4(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONG,"matrix rowblock distribution does not match! [%D,%D] != [%D,%D]",
             A->rmap->rstart,A->rmap->rend,elem->cmat->FirstLocalRow(),elem->cmat->FirstLocalRow()+elem->cmat->LocalHeight());
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatGetRow_ElemSparse"
static PetscErrorCode MatGetRow_ElemSparse(Mat A,PetscInt row,PetscInt *nz,PetscInt **idx,PetscScalar **v)
{
  Mat_ElemSparse *a = (Mat_ElemSparse*)A->data;
  PetscInt       lrow,nzp;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  lrow=row-A->rmap->rstart;
  if (lrow<0) PetscFunctionReturn(0);
  nzp=a->cmat->NumConnections(lrow);
  if (nz) *nz = nzp;
  if (v) {
    *v = 0;
    if (nzp) {
      PetscInt i,off;
      off=(PetscInt)a->cmat->EntryOffset(lrow);
      ierr = PetscMalloc1(nzp,v);CHKERRQ(ierr);
      for (i=0;i<nzp;i++) *(*v + i) = a->cmat->Value(i+off);
    }
  }
  if (idx) {
    *idx = 0;
    if (nzp) {
      PetscInt i,off;
      off=(PetscInt)a->cmat->EntryOffset(lrow);
      ierr = PetscMalloc1(nzp,idx);CHKERRQ(ierr);
      for (i=0;i<nzp;i++) *(*idx + i) = a->cmat->Col(i+off);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatRestoreRow_ElemSparse"
static PetscErrorCode MatRestoreRow_ElemSparse(Mat A,PetscInt row,PetscInt *nz,PetscInt **idx,PetscScalar **v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (idx) {ierr = PetscFree(*idx);CHKERRQ(ierr);}
  if (v)   {ierr = PetscFree(*v);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatAssemblyBegin_ElemSparse"
static PetscErrorCode MatAssemblyBegin_ElemSparse(Mat A,MatAssemblyType type)
{
  Mat_ElemSparse *a = (Mat_ElemSparse*)A->data;

  PetscFunctionBegin;
  a->cmat->MakeConsistent();
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatAssemblyEnd_ElemSparse"
static PetscErrorCode MatAssemblyEnd_ElemSparse(Mat A,MatAssemblyType type)
{
  PetscFunctionBegin;
  /* Nothing to be done */
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSetValues_ElemSparse"
static PetscErrorCode MatSetValues_ElemSparse(Mat A,PetscInt nr,const PetscInt *rows,PetscInt nc,const PetscInt *cols,const PetscScalar *vals,InsertMode imode)
{
  Mat_ElemSparse *a = (Mat_ElemSparse*)A->data;
  PetscInt       i,j;

  PetscFunctionBegin;
  if (imode == INSERT_VALUES) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONG,"INSERT_VALUES is not currently supported with MATELEMSPARSE! use ADD_VALUES instead");
  for (i=0;i<nr;i++) {
    if (rows[i] < A->rmap->rstart || rows[i] >= A->rmap->rend) {
      SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONG,"Insert unowned rows is not currently supported with MATELEMSPARSE");
    }
    for (j=0;j<nc;j++) {
      a->cmat->QueueLocalUpdate(rows[i]-A->rmap->rstart,cols[j],vals[i*nc+j]);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatConvert_ElemSparse_AIJ"
PETSC_EXTERN PetscErrorCode MatConvert_ElemSparse_AIJ(Mat A, MatType newtype,MatReuse reuse,Mat *newmat)
{
  Mat_ElemSparse *elem=(Mat_ElemSparse*)A->data;
  Mat            mat_aij;
  PetscInt       row,rstart=A->rmap->rstart,rend=A->rmap->rend,lr=A->rmap->n,lc=A->cmap->n;
  PetscInt       *dnnz,*onnz;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreate(PetscObjectComm((PetscObject)A),&mat_aij);CHKERRQ(ierr);
  ierr = MatSetSizes(mat_aij,lr,PETSC_DECIDE,A->rmap->N,A->cmap->N);CHKERRQ(ierr);
  ierr = MatSetType(mat_aij,MATAIJ);CHKERRQ(ierr);
  /* preallocation (overestimated for MPIAIJ) */
  ierr = PetscMalloc2(lr,&dnnz,lr,&onnz);CHKERRQ(ierr);
  for (row=0;row<lr;row++) {
    dnnz[row]=PetscMin(elem->cmat->NumConnections(row),lc);
    onnz[row]=PetscMin(elem->cmat->NumConnections(row),A->cmap->N-lc);
  }
  ierr = MatSeqAIJSetPreallocation(mat_aij,0,dnnz);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(mat_aij,0,dnnz,0,onnz);CHKERRQ(ierr);
  ierr = MatSetUp(mat_aij);CHKERRQ(ierr);
  ierr = PetscFree2(dnnz,onnz);CHKERRQ(ierr);
  /* fill matrix values */
  for (row=rstart; row<rend; row++) {
    PetscInt          ncols;
    const PetscInt    *cols;
    const PetscScalar *vals;
    ierr = MatGetRow(A,row,&ncols,&cols,&vals);CHKERRQ(ierr);
    ierr = MatSetValues(mat_aij,1,&row,ncols,cols,vals,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatRestoreRow(A,row,&ncols,&cols,&vals);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(mat_aij,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat_aij,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  if (reuse == MAT_REUSE_MATRIX) {
    ierr = MatHeaderReplace(A,mat_aij);CHKERRQ(ierr);
  } else {
    *newmat = mat_aij;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatConvert_AIJ_ElemSparse"
PETSC_EXTERN PetscErrorCode MatConvert_AIJ_ElemSparse(Mat A, MatType newtype,MatReuse reuse,Mat *newmat)
{
  Mat            mat_elemental;
  Mat_ElemSparse *elem;
  PetscErrorCode ierr;
  PetscInt       row,rstart=A->rmap->rstart,rend=A->rmap->rend,rstart_el,rend_el;
  MatInfo        info;

  PetscFunctionBegin;
  ierr = MatCreate(PetscObjectComm((PetscObject)A),&mat_elemental);CHKERRQ(ierr);
  ierr = MatSetSizes(mat_elemental,PETSC_DECIDE,PETSC_DECIDE,A->rmap->N,A->cmap->N);CHKERRQ(ierr);
  ierr = MatSetType(mat_elemental,MATELEMSPARSE);CHKERRQ(ierr);
  ierr = MatSetUp(mat_elemental);CHKERRQ(ierr);
  /* check */
  ierr = MatGetOwnershipRange(A,&rstart,&rend);CHKERRQ(ierr);
  elem = (Mat_ElemSparse*)mat_elemental->data;
  rstart_el = elem->cmat->FirstLocalRow();
  rend_el = elem->cmat->FirstLocalRow()+elem->cmat->LocalHeight();
  if (rstart != rstart_el || rend != rend_el) {
    SETERRQ4(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONG,"matrix rowblock distribution does not match! [%D,%D] != [%D,%D]",rstart,rend,rstart_el,rend_el);
  }
  /* elemental preallocation */
  ierr = MatGetInfo(A,MAT_LOCAL,&info);CHKERRQ(ierr);
  elem->cmat->Reserve((PetscInt)info.nz_used);CHKERRQ(ierr);
  /* fill matrix values */
  for (row=rstart; row<rend; row++) {
    PetscInt          ncols;
    const PetscInt    *cols;
    const PetscScalar *vals;
    ierr = MatGetRow(A,row,&ncols,&cols,&vals);CHKERRQ(ierr);
    ierr = MatSetValues(mat_elemental,1,&row,ncols,cols,vals,ADD_VALUES);CHKERRQ(ierr);
    ierr = MatRestoreRow(A,row,&ncols,&cols,&vals);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(mat_elemental,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat_elemental,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  if (reuse == MAT_REUSE_MATRIX) {
    ierr = MatHeaderReplace(A,mat_elemental);CHKERRQ(ierr);
  } else {
    *newmat = mat_elemental;
  }
  PetscFunctionReturn(0);
}

/*
 Provides an interface to Elemental sparse solver
 This code has been copied and modified from the previous interface to Clique
*/

/*
  MatConvertToElemSparse: Convert Petsc aij matrix to ElemSparse matrix

  input:
+   A     - matrix in seqaij or mpiaij format
-   reuse - denotes if the destination matrix is to be created or reused. Currently
            MAT_REUSE_MATRIX is only supported for inplace conversion, otherwise use MAT_INITIAL_MATRIX.

  output:
.   matelem - ElemSparse context
*/
#undef __FUNCT__
#define __FUNCT__ "MatConvertToElemSparse"
PetscErrorCode MatConvertToElemSparse(Mat A,MatReuse reuse,Mat_ElemSparse *matelem)
{
  PetscErrorCode                        ierr;
  PetscInt                              i,rstart,rend,ncols;
  MatInfo                               info;
  El::DistSparseMatrix<PetscElemScalar> *cmat;

  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX){
    /* create ElemSparse matrix */
    cmat = new El::DistSparseMatrix<PetscElemScalar>(A->rmap->N,PetscObjectComm((PetscObject)A));
    matelem->cmat = cmat;
  } else {
    cmat = matelem->cmat;
  }

  /* fill matrix values */
  ierr = MatGetOwnershipRange(A,&rstart,&rend);CHKERRQ(ierr);
  const int firstLocalRow = cmat->FirstLocalRow();
  const int localHeight = cmat->LocalHeight();
  if (rstart != firstLocalRow || rend-rstart != localHeight) {
    SETERRQ4(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONG,"matrix rowblock distribution does not match! [%D,%D] != [%D,%D]",rstart,rend,firstLocalRow,firstLocalRow+localHeight);
  }
  /* elemental preallocation */
  if (reuse == MAT_INITIAL_MATRIX) {
    ierr = MatGetInfo(A,MAT_LOCAL,&info);CHKERRQ(ierr);
    cmat->Reserve((PetscInt)info.nz_used);CHKERRQ(ierr);
  }

  /* insert values */
  for (i=0; i<rend-rstart; i++){
    PetscInt              j;
    const PetscInt        *cols;
    const PetscElemScalar *vals;
    ierr = MatGetRow(A,i+rstart,&ncols,&cols,&vals);CHKERRQ(ierr);
    for (j=0; j<ncols; j++){
      cmat->QueueLocalUpdate(i,cols[j],vals[j]);
    }
    ierr = MatRestoreRow(A,i+rstart,&ncols,&cols,&vals);CHKERRQ(ierr);
  }
  /* elemental assembly */
  cmat->MakeConsistent();
  //Display(*cmat,"MATRIX");
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMult_ElemSparse"
static PetscErrorCode MatMult_ElemSparse(Mat A,Vec X,Vec Y)
{
  Mat_ElemSparse        *elem=(Mat_ElemSparse*)A->data;
  PetscErrorCode        ierr;
  PetscInt              i;
  const PetscElemScalar *array;

  PetscFunctionBegin;
  if (!elem->cmat) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_NULL,"Elemental sparse matrix has not been created yet. Maybe you forgot to call MatSetUp");

  ierr = VecGetArrayRead(X,&array);CHKERRQ(ierr);
  for (i=0; i<A->cmap->n; i++) {
    elem->cvecr->SetLocal(i,0,array[i]);
  }
  El::Multiply(El::NORMAL,1.0,*(elem->cmat),*(elem->cvecr),0.0,*(elem->cvecl));
  ierr = VecRestoreArrayRead(X,&array);CHKERRQ(ierr);
  for (i=0; i<A->rmap->n; i++) {
    ierr = VecSetValue(Y,i+A->rmap->rstart,elem->cvecl->GetLocal(i,0),INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(Y);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(Y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatView_ElemSparse"
PetscErrorCode MatView_ElemSparse(Mat A,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool      iascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    PetscViewerFormat format;
    ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
    if (format == PETSC_VIEWER_ASCII_INFO) {
      ierr = PetscViewerASCIIPrintf(viewer,"ElemSparse run parameters:\n");CHKERRQ(ierr);
    } else if (format == PETSC_VIEWER_DEFAULT) { /* matrix A is factored matrix, remove this block */
      Mat Aaij;
      ierr = PetscViewerASCIIUseTabs(viewer,PETSC_FALSE);CHKERRQ(ierr);
      ierr = PetscViewerASCIIUseTabs(viewer,PETSC_TRUE);CHKERRQ(ierr);
      ierr = PetscPrintf(PetscObjectComm((PetscObject)viewer),"ElemSparse matrix\n");CHKERRQ(ierr);
      ierr = MatComputeExplicitOperator(A,&Aaij);CHKERRQ(ierr);
      ierr = MatView(Aaij,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
      ierr = MatDestroy(&Aaij);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatDestroy_ElemSparse_private"
static PetscErrorCode MatDestroy_ElemSparse_private(Mat_ElemSparse* elem)
{
  PetscFunctionBegin;
  if (elem->cmat) delete elem->cmat;
  if (elem->cvecr) delete elem->cvecr;
  if (elem->cvecl) delete elem->cvecl;
  if (elem->sepTree) delete elem->sepTree;
  if (elem->map) delete elem->map;
  if (elem->frontTree) delete elem->frontTree;
  if (elem->rhs) delete elem->rhs;
  if (elem->xNodal) delete elem->xNodal;
  if (elem->info) delete elem->info;
  if (elem->inverseMap) delete elem->inverseMap;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatDestroy_ElemSparse"
static PetscErrorCode MatDestroy_ElemSparse(Mat A)
{
  PetscErrorCode ierr;
  Mat_ElemSparse *elem;
  PetscBool      iselemsparse;

  PetscFunctionBegin;
  /* clear composed functions */
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatFactorGetSolverPackage_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatConvert_elemsparse_mpiaij_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatConvert_elemsparse_seqaij_C",NULL);CHKERRQ(ierr);

  ierr = PetscObjectTypeCompare((PetscObject)A,MATELEMSPARSE,&iselemsparse);CHKERRQ(ierr);
  if (iselemsparse) {
    elem = (Mat_ElemSparse*)A->data;
    ierr = MatDestroy_ElemSparse_private(elem);
    ierr = PetscFree(elem);CHKERRQ(ierr);
  }
  if (A->spptr) {
    elem = (Mat_ElemSparse*)A->spptr;
    ierr = MatDestroy_ElemSparse_private(elem);
    if (elem->Destroy) {
      ierr = (elem->Destroy)(A);CHKERRQ(ierr);
    }
    ierr = PetscFree(elem);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSolve_ElemSparse"
PetscErrorCode MatSolve_ElemSparse(Mat A,Vec B,Vec X)
{
  PetscErrorCode        ierr;
  PetscInt              i;
  const PetscScalar     *array;
  Mat_ElemSparse        *elem=(Mat_ElemSparse*)A->spptr;
  El::DistMultiVec<PetscElemScalar> *bc=elem->rhs;
  El::DistNodalMultiVec<PetscElemScalar> *xNodal=elem->xNodal;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(B,&array);CHKERRQ(ierr);
  for (i=0; i<A->rmap->n; i++) {
    bc->SetLocal(i,0,array[i]);
  }
  ierr = VecRestoreArrayRead(B,&array);CHKERRQ(ierr);

  xNodal->Pull( *elem->inverseMap, *elem->info, *bc );
  El::Solve( *elem->info, *elem->frontTree, *xNodal);
  xNodal->Push( *elem->inverseMap, *elem->info, *bc );

  for (i=0; i<A->cmap->n; i++) {
    ierr = VecSetValue(X,i+A->cmap->rstart,bc->GetLocal(i,0),INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(X);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(X);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCholeskyFactorNumeric_ElemSparse"
PetscErrorCode MatCholeskyFactorNumeric_ElemSparse(Mat F,Mat A,const MatFactorInfo *info)
{
  PetscErrorCode    ierr;
  Mat_ElemSparse        *cliq=(Mat_ElemSparse*)F->spptr;
  PETSC_UNUSED
  El::DistSparseMatrix<PetscElemScalar> *cmat;

  PetscFunctionBegin;
  cmat = cliq->cmat;
  if (cliq->matstruc == SAME_NONZERO_PATTERN){ /* successing numerical factorization */
    /* Update cmat */
    ierr = MatConvertToElemSparse(A,MAT_REUSE_MATRIX,cliq);CHKERRQ(ierr);
  }

  /* Numeric factorization */
  if (cliq->frontTree) delete cliq->frontTree;
  cliq->frontTree = new El::DistSymmFrontTree<PetscElemScalar>( *cmat, *cliq->map, *cliq->sepTree, *cliq->info );
  El::LDL( *cliq->info, *cliq->frontTree, El::LDL_1D);
  //L.frontType = cliq::SYMM_2D;

  // refactor
  //cliq::ChangeFrontType( *cliq->frontTree, cliq::LDL_2D );
  //*(cliq->frontTree.frontType) = cliq::LDL_2D;
  //cliq::LDL( *cliq->info, *cliq->frontTree, cliq::LDL_2D );

  cliq->matstruc = SAME_NONZERO_PATTERN;
  F->assembled   = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCholeskyFactorSymbolic_ElemSparse"
PetscErrorCode MatCholeskyFactorSymbolic_ElemSparse(Mat F,Mat A,IS r,const MatFactorInfo *info)
{
  PetscErrorCode                        ierr;
  Mat_ElemSparse                        *Acliq=(Mat_ElemSparse*)F->spptr;
  El::DistSparseMatrix<PetscElemScalar> *cmat;
  El::BisectCtrl                        ctrl;

  PetscFunctionBegin;
  /* Convert A to Aclique */
  ierr = MatConvertToElemSparse(A,MAT_INITIAL_MATRIX,Acliq);CHKERRQ(ierr);
  cmat = Acliq->cmat;

  ctrl.sequential = PETSC_TRUE;
  ctrl.numSeqSeps = Acliq->numSeqSeps;
  ctrl.numDistSeps = Acliq->numDistSeps;
  ctrl.cutoff = Acliq->cutoff;
  El::NestedDissection( cmat->DistGraph(), *Acliq->map, *Acliq->sepTree, *Acliq->info, ctrl);
  Acliq->map->FormInverse( *Acliq->inverseMap );

  Acliq->matstruc      = DIFFERENT_NONZERO_PATTERN;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatGetFactor_aij_elemental"
PETSC_EXTERN PetscErrorCode MatGetFactor_aij_elemental(Mat A,MatFactorType ftype,Mat *F)
{
  Mat            B;
  Mat_ElemSparse *elem;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreate(PetscObjectComm((PetscObject)A),&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,A->rmap->n,A->cmap->n,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetType(B,((PetscObject)A)->type_name);CHKERRQ(ierr);
  ierr = MatSetUp(B);CHKERRQ(ierr);
  ierr = PetscElementalInitializePackage();
  ierr = PetscNewLog(B,&elem);CHKERRQ(ierr);
  B->spptr            = (void*)elem;
  elem->rhs           = new El::DistMultiVec<PetscElemScalar>(A->rmap->N,1,PetscObjectComm((PetscObject)A));
  elem->xNodal        = new El::DistNodalMultiVec<PetscElemScalar>();
  elem->info          = new El::DistSymmInfo;
  elem->inverseMap    = new El::DistMap;
  elem->map           = new El::DistMap;
  elem->sepTree       = new El::DistSeparatorTree;
  elem->Destroy       = B->ops->destroy;

  B->ops->view    = MatView_ElemSparse;
  B->ops->solve   = MatSolve_ElemSparse;

  B->ops->destroy = MatDestroy_ElemSparse;
  B->factortype   = ftype;
  B->assembled    = PETSC_FALSE;
  B->ops->choleskyfactorsymbolic = MatCholeskyFactorSymbolic_ElemSparse;
  B->ops->choleskyfactornumeric  = MatCholeskyFactorNumeric_ElemSparse;
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatFactorGetSolverPackage_C",MatFactorGetSolverPackage_elemental);CHKERRQ(ierr);

  /* Set Clique options */
  ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)A),((PetscObject)A)->prefix,"ElemSparse Options","Mat");CHKERRQ(ierr);
  elem->cutoff      = 128;  /* maximum size of leaf node */
  elem->numDistSeps = 1;    /* number of distributed separators to try */
  elem->numSeqSeps  = 1;    /* number of sequential separators to try */
  PetscOptionsEnd();

  *F = B;
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------*/
static struct _MatOps MatOps_Values = {
       MatSetValues_ElemSparse,
       MatGetRow_ElemSparse,
       MatRestoreRow_ElemSparse,
       MatMult_ElemSparse,
/* 4*/ 0, //MatMultAdd_ElemDense,
       0, //MatMultTranspose_ElemDense,
       0, //MatMultTransposeAdd_ElemDense,
       MatSolve_ElemSparse,
       0, //MatSolveAdd_ElemDense,
       0, //MatSolveTranspose_ElemDense,
/*10*/ 0, //MatSolveTransposeAdd_ElemDense,
       0, //MatLUFactor_ElemDense,
       0, //MatCholeskyFactor_ElemDense,
       0,
       0, //MatTranspose_ElemDense,
/*15*/ MatGetInfo_ElemSparse,
       0,
       0, //MatGetDiagonal_ElemDense,
       0, //MatDiagonalScale_ElemDense,
       0, //MatNorm_ElemDense,
/*20*/ MatAssemblyBegin_ElemSparse,
       MatAssemblyEnd_ElemSparse,
       0, //MatSetOption_ElemDense,
       0, //MatZeroEntries_ElemDense,
/*24*/ 0,
       0, //MatLUFactorSymbolic_ElemDense,
       0, //MatLUFactorNumeric_ElemDense,
       MatCholeskyFactorSymbolic_ElemSparse,
       MatCholeskyFactorNumeric_ElemSparse,
/*29*/ MatSetUp_ElemSparse,
       0,
       0,
       0,
       0,
/*34*/ 0, //MatDuplicate_ElemDense,
       0,
       0,
       0,
       0,
/*39*/ 0, //MatAXPY_ElemDense,
       0,
       0,
       0,
       0, //MatCopy_ElemDense,
/*44*/ 0,
       0, //MatScale_ElemDense,
       0,
       0,
       0,
/*49*/ 0,
       0,
       0,
       0,
       0,
/*54*/ 0,
       0,
       0,
       0,
       0,
/*59*/ 0,
       MatDestroy_ElemSparse,
       MatView_ElemSparse,
       0,
       0,
/*64*/ 0,
       0,
       0,
       0, //MatSetValuesLocal_ElemSparse,
       0,
/*69*/ 0,
       0,
       0, //MatConvert_ElemDense_Dense,
       0,
       0,
/*74*/ 0,
       0,
       0,
       0,
       0,
/*79*/ 0,
       0,
       0,
       0,
       0,
/*84*/ 0,
       0,
       0,
       0,
       0,
/*89*/ 0, //MatMatMult_ElemDense,
       0, //MatMatMultSymbolic_ElemDense,
       0, //MatMatMultNumeric_ElemDense,
       0,
       0,
/*94*/ 0,
       0, //MatMatTransposeMult_ElemDense,
       0, //MatMatTransposeMultSymbolic_ElemDense,
       0, //MatMatTransposeMultNumeric_ElemDense,
       0,
/*99*/ 0,
       0,
       0,
       0, //MatConjugate_ElemDense,
       0,
/*104*/0,
       0,
       0,
       0,
       0,
/*109*/0, //MatMatSolve_ElemDense,
       0,
       0,
       0,
       0,
/*114*/0,
       0,
       0,
       0,
       0,
/*119*/0,
       0, //MatHermitianTranspose_ElemDense,
       0,
       0,
       0,
/*124*/0,
       0,
       0,
       0,
       0,
/*129*/0,
       0,
       0,
       0,
       0,
/*134*/0,
       0,
       0,
       0,
       0
};

/*MC
   MATELEMSPARSE = "elemsparse" - A matrix type for sparse matrices using the Elemental package

   Options Database Keys:
+ -mat_type elemsparse - sets the matrix type to "elemsparse" during a call to MatSetFromOptions()

  Level: beginner

.seealso:
M*/

#undef __FUNCT__
#define __FUNCT__ "MatCreate_ElemSparse"
PETSC_EXTERN PetscErrorCode MatCreate_ElemSparse(Mat A)
{
  Mat_ElemSparse      *a;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscElementalInitializePackage();CHKERRQ(ierr);
  ierr = PetscMemcpy(A->ops,&MatOps_Values,sizeof(struct _MatOps));CHKERRQ(ierr);
  ierr = PetscNewLog(A,&a);CHKERRQ(ierr);
  A->data = (void*)a;
  ierr = PetscObjectChangeTypeName((PetscObject)A,MATELEMSPARSE);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatConvert_elemsparse_mpiaij_C",MatConvert_ElemSparse_AIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatConvert_elemsparse_seqaij_C",MatConvert_ElemSparse_AIJ);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
