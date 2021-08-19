#include <petsc/private/petscginkgo.h>

const char GinkgoCitation[] = "@Article{Ginkgo2020,\n"
"  author  = {Anzt, Hartwig and Cojean, Terry and Chen, Yen-Chen and Flegar, Goran and G{\"o}bel, Fritz and Gr{\"u}tzmacher, Thomas and Nayak, Pratik and Ribizel, Tobias and Tsai, Yu-Hsiang},\n"
"  title   = {Ginkgo: A high performance numerical linear algebra library},\n"
"  journal = {Journal of Open Source Software},\n"
"  volume  = {5},\n"
"  number  = {52},\n"
"  pages   = {2260},\n"
"  year    = {2020}\n"
"}\n";
static PetscBool GinkgoCite = PETSC_FALSE;


/* TODO: SEK */
static PetscErrorCode MatView_GinkgoCSR(Mat A,PetscViewer viewer)
{
  PetscErrorCode ierr;
  Mat_GinkgoCSR     *a = (Mat_GinkgoCSR*)A->data;
  PetscBool      iascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    PetscViewerFormat format;
    ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
    if (format == PETSC_VIEWER_ASCII_INFO) {
      /* call Ginkgo viewing function */
      ierr = PetscViewerASCIIPrintf(viewer,"Ginkgo run parameters:\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  allocated entries=%d\n",(*a->emat).AllocatedMemory());CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  grid height=%d, grid width=%d\n",(*a->emat).Grid().Height(),(*a->emat).Grid().Width());CHKERRQ(ierr);
      if (format == PETSC_VIEWER_ASCII_FACTOR_INFO) {
        /* call Ginkgo viewing function */
        ierr = PetscPrintf(PetscObjectComm((PetscObject)viewer),"test matview_ginkgo 2\n");CHKERRQ(ierr);
      }

    } else if (format == PETSC_VIEWER_DEFAULT) {
      ierr = PetscViewerASCIIUseTabs(viewer,PETSC_FALSE);CHKERRQ(ierr);
      /* TODO: SEK
      gko::Print( *a->emat, "Ginkgo matrix (cyclic ordering)");
      ierr = PetscViewerASCIIUseTabs(viewer,PETSC_TRUE);CHKERRQ(ierr);
      if (A->factortype == MAT_FACTOR_NONE) {
        Mat Adense;
        ierr = MatConvert(A,MATDENSE,MAT_INITIAL_MATRIX,&Adense);CHKERRQ(ierr);
        ierr = MatView(Adense,viewer);CHKERRQ(ierr);
        ierr = MatDestroy(&Adense);CHKERRQ(ierr);
      }
    } else SETERRQ(PetscObjectComm((PetscObject)viewer),PETSC_ERR_SUP,"Format"); */
  } else {
    /* convert to dense format and call MatView() */
    Mat Adense;
    ierr = MatConvert(A,MATDENSE,MAT_INITIAL_MATRIX,&Adense);CHKERRQ(ierr);
    ierr = MatView(Adense,viewer);CHKERRQ(ierr);
    ierr = MatDestroy(&Adense);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatGetInfo_GinkgoCSR(Mat A,MatInfoType flag,MatInfo *info)
{
  Mat_GinkgoCSR  *a = (Mat_GinkgoCSR*)A->data;

  PetscFunctionBegin;
  info->block_size = 1.0;

  if (flag == MAT_LOCAL) {
    info->nz_allocated   = (*a->emat).AllocatedMemory(); /* locally allocated */
    info->nz_used        = info->nz_allocated;
  } else if (flag == MAT_GLOBAL_MAX) {
    //ierr = MPIU_Allreduce(isend,irecv,5,MPIU_REAL,MPIU_MAX,PetscObjectComm((PetscObject)matin));CHKERRMPI(ierr);
    /* see MatGetInfo_MPIAIJ() for getting global info->nz_allocated! */
    //SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP," MAT_GLOBAL_MAX not written yet");
  } else if (flag == MAT_GLOBAL_SUM) {
    //SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP," MAT_GLOBAL_SUM not written yet");
    info->nz_allocated   = (*a->emat).AllocatedMemory(); /* locally allocated */
    info->nz_used        = info->nz_allocated; /* assume Ginkgo does accurate allocation */
    //ierr = MPIU_Allreduce(isend,irecv,1,MPIU_REAL,MPIU_SUM,PetscObjectComm((PetscObject)A));CHKERRMPI(ierr);
    //PetscPrintf(PETSC_COMM_SELF,"    ... [%d] locally allocated %g\n",rank,info->nz_allocated);
  }

  info->nz_unneeded       = 0.0;
  info->assemblies        = A->num_ass;
  info->mallocs           = 0;
  info->memory            = ((PetscObject)A)->mem;
  info->fill_ratio_given  = 0; /* determined by Ginkgo */
  info->fill_ratio_needed = 0;
  info->factor_mallocs    = 0;
  PetscFunctionReturn(0);
}

PetscErrorCode MatSetOption_Ginkgo(Mat A,MatOption op,PetscBool flg)
{
  PetscGinkgoScalar *a = A->get_values();

  PetscFunctionBegin;
  switch (op) {
  case MAT_NEW_NONZERO_LOCATIONS:
  case MAT_NEW_NONZERO_LOCATION_ERR:
  case MAT_NEW_NONZERO_ALLOCATION_ERR:
  case MAT_SYMMETRIC:
  case MAT_SORTED_FULL:
  case MAT_HERMITIAN:
  case MAT_ROW_ORIENTED:
  default:
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"unknown option %s",MatOptions[op]);
  }
  PetscFunctionReturn(0);
}


/*MC
   MATGINKGOCSR = "ginkgo" - A matrix type for sparse matrices using the Ginkgo package

  Use ./configure --download-ginkgo to install PETSc to use Ginkgo

  Use -pc_type ilu -pc_factor_mat_solver_type ginkgo to use this solver

   Options Database Keys:
+ -mat_type ginkgo - sets the matrix type to "ginkgo" during a call to MatSetFromOptions()

  Level: beginner

.seealso: MATSEQAIJ
M*/

PETSC_EXTERN PetscErrorCode MatCreate_Ginkgo(Mat A)
{
  Mat_GinkgoCSR      *a;
  PetscErrorCode     ierr;
  PetscBool          flg,flg1;
  Mat_GinkgoCSR_Grid *commgrid;
  MPI_Comm           icomm;
  PetscInt           optv1;

  PetscFunctionBegin;
  ierr = PetscMemcpy(A->ops,&MatOps_Values,sizeof(struct _MatOps));CHKERRQ(ierr);
  A->insertmode = NOT_SET_VALUES;

  ierr = PetscNewLog(A,&a);CHKERRQ(ierr);
  A->data = (void*)a;

  /* Set up the ginkgo matrix */
  El::mpi::Comm cxxcomm(PetscObjectComm((PetscObject)A));

  /* Grid needs to be shared between multiple Mats on the same communicator, implement by attribute caching on the MPI_Comm */
  if (Petsc_Ginkgo_keyval == MPI_KEYVAL_INVALID) {
    ierr = MPI_Comm_create_keyval(MPI_COMM_NULL_COPY_FN,MPI_COMM_NULL_DELETE_FN,&Petsc_Ginkgo_keyval,(void*)0);CHKERRMPI(ierr);
    ierr = PetscCitationsRegister(GinkgoCitation,&GinkgoCite);CHKERRQ(ierr);
  }
  ierr = PetscCommDuplicate(cxxcomm.comm,&icomm,NULL);CHKERRQ(ierr);
  ierr = MPI_Comm_get_attr(icomm,Petsc_Ginkgo_keyval,(void**)&commgrid,(int*)&flg);CHKERRMPI(ierr);
  if (!flg) {
    ierr = PetscNewLog(A,&commgrid);CHKERRQ(ierr);

    ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)A),((PetscObject)A)->prefix,"Ginkgo Options","Mat");CHKERRQ(ierr);
    /* displayed default grid sizes (CommSize,1) are set by us arbitrarily until El::Grid() is called */
    ierr = PetscOptionsInt("-mat_ginkgo_grid_height","Grid Height","None",El::mpi::Size(cxxcomm),&optv1,&flg1);CHKERRQ(ierr);
    if (flg1) {
      if (El::mpi::Size(cxxcomm) % optv1) SETERRQ2(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_INCOMP,"Grid Height %D must evenly divide CommSize %D",optv1,(PetscInt)El::mpi::Size(cxxcomm));
      commgrid->grid = new El::Grid(cxxcomm,optv1); /* use user-provided grid height */
    } else {
      commgrid->grid = new El::Grid(cxxcomm); /* use Ginkgo default grid sizes */
      /* printf("new commgrid->grid = %p\n",commgrid->grid);  -- memory leak revealed by valgrind? */
    }
    commgrid->grid_refct = 1;
    ierr = MPI_Comm_set_attr(icomm,Petsc_Ginkgo_keyval,(void*)commgrid);CHKERRMPI(ierr);

    a->pivoting    = 1;
    ierr = PetscOptionsInt("-mat_ginkgo_pivoting","Pivoting","None",a->pivoting,&a->pivoting,NULL);CHKERRQ(ierr);

    ierr = PetscOptionsEnd();CHKERRQ(ierr);
  } else {
    commgrid->grid_refct++;
  }
  ierr = PetscCommDestroy(&icomm);CHKERRQ(ierr);
  a->grid        = commgrid->grid;
  a->emat        = new El::DistMatrix<PetscGinkgoScalar>(*a->grid);
  a->roworiented = PETSC_TRUE;

  ierr = PetscObjectComposeFunction((PetscObject)A,"MatGetOwnershipIS_C",MatGetOwnershipIS_Ginkgo);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatProductSetFromOptions_ginkgo_mpidense_C",MatProductSetFromOptions_Ginkgo_MPIDense);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)A,MATGINKGO);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}static PetscErrorCode MatSetValues_Ginkgo(Mat A,PetscInt nr,const PetscInt *rows,PetscInt nc,const PetscInt *cols,const PetscScalar *vals,InsertMode imode)
{
  PetscGinkgoScalar *a = (Mat_GinkgoCSR*)A->get_values();
  PetscGinkgoInt    *row_ptr = (Mat_GinkgoCSR*)A->get_row_ptrs();
  PetscGinkgoInt    *col_idx = (Mat_GinkgoCSR*)A->get_col_idxs();
  PetscInt          i,j,rrank,ridx,crank,cidx;

  PetscFunctionBegin;

  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultTranspose_Ginkgo(Mat A,Vec X,Vec Y)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCopy_Ginkgo(Mat A,Mat B,MatStructure str)
{
  Mat_GinkgoCSR *a=(Mat_GinkgoCSR*)A->data;
  Mat_GinkgoCSR *b=(Mat_GinkgoCSR*)B->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  El::Copy(*a->emat,*b->emat);
  ierr = PetscObjectStateIncrease((PetscObject)B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDuplicate_Ginkgo(Mat A,MatDuplicateOption op,Mat *B)
{
  Mat            Be;
  MPI_Comm       comm;
  Mat_GinkgoCSR  *a=(Mat_GinkgoCSR*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)A,&comm);CHKERRQ(ierr);
  ierr = MatCreate(comm,&Be);CHKERRQ(ierr);
  ierr = MatSetSizes(Be,A->rmap->n,A->cmap->n,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = MatSetType(Be,MATGINKGO);CHKERRQ(ierr);
  ierr = MatSetUp(Be);CHKERRQ(ierr);
  *B = Be;
  if (op == MAT_COPY_VALUES) {
    Mat_GinkgoCSR *b=(Mat_GinkgoCSR*)Be->data;
    El::Copy(*a->emat,*b->emat);
  }
  Be->assembled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatTranspose_Ginkgo(Mat A,MatReuse reuse,Mat *B)
{
  Mat            Be = *B;
  PetscErrorCode ierr;
  MPI_Comm       comm;
  Mat_GinkgoCSR  *a = (Mat_GinkgoCSR*)A->data, *b;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)A,&comm);CHKERRQ(ierr);
  /* Only out-of-place supported */
  if (reuse == MAT_INPLACE_MATRIX) SETERRQ(comm,PETSC_ERR_SUP,"Only out-of-place supported");
  if (reuse == MAT_INITIAL_MATRIX) {
    ierr = MatCreate(comm,&Be);CHKERRQ(ierr);
    ierr = MatSetSizes(Be,A->cmap->n,A->rmap->n,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
    ierr = MatSetType(Be,MATGINKGO);CHKERRQ(ierr);
    ierr = MatSetUp(Be);CHKERRQ(ierr);
    *B = Be;
  }
  b = (Mat_GinkgoCSR*)Be->data;
  /* El::Transpose(*a->emat,*b->emat); */
  Be->assembled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatConjugate_Ginkgo(Mat A)
{
  Mat_GinkgoCSR  *a = (Mat_GinkgoCSR*)A->data;

  PetscFunctionBegin;
  /* El::Conjugate(*a->emat); */
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolve_Ginkgo(Mat A,Vec B,Vec X)
{
  Mat_GinkgoCSR     *a = (Mat_GinkgoCSR*)A->data;
  PetscErrorCode    ierr;
  PetscGinkgoScalar   *x;
  PetscInt          pivoting = a->pivoting;

  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolveAdd_Ginkgo(Mat A,Vec B,Vec Y,Vec X)
{
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = MatSolve_Ginkgo(A,B,X);CHKERRQ(ierr);
  ierr = VecAXPY(X,1,Y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatSolverTypeRegister_Ginkgo(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSolverTypeRegister(MATSOLVERGINKGO,MATGINKGO,        MAT_FACTOR_LU,MatGetFactor_Ginkgo_Ginkgo);CHKERRQ(ierr);
  ierr = MatSolverTypeRegister(MATSOLVERGINKGO,MATGINKGO,        MAT_FACTOR_CHOLESKY,MatGetFactor_Ginkgo_Ginkgo);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatNorm_Ginkgo(Mat A,NormType type,PetscReal *nrm)
{
  Mat_GinkgoCSR *a=(Mat_GinkgoCSR*)A->data;

  PetscFunctionBegin;
  switch (type) {
  case NORM_1:
    /* *nrm = El::OneNorm(*a->emat); */
    break;
  case NORM_FROBENIUS:
    /* *nrm = El::FrobeniusNorm(*a->emat); */
    break;
  case NORM_INFINITY:
    /* *nrm = El::InfinityNorm(*a->emat); */
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Unsupported norm type");
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatZeroEntries_Ginkgo(Mat A)
{
  Mat_GinkgoCSR *a=(Mat_GinkgoCSR*)A->data;

  PetscFunctionBegin;
  El::Zero(*a->emat);
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatConvert_SeqAIJ_Ginkgo(Mat A, MatType newtype,MatReuse reuse,Mat *newmat)
{
  Mat               mat_ginkgo;
  PetscErrorCode    ierr;
  PetscInt          M=A->rmap->N,N=A->cmap->N,row,ncols;
  const PetscInt    *cols;
  const PetscScalar *vals;

  PetscFunctionBegin;
  if (reuse == MAT_REUSE_MATRIX) {
    mat_ginkgo = *newmat;
    ierr = MatZeroEntries(mat_ginkgo);CHKERRQ(ierr);
  } else {
    ierr = MatCreate(PetscObjectComm((PetscObject)A), &mat_ginkgo);CHKERRQ(ierr);
    ierr = MatSetSizes(mat_ginkgo,PETSC_DECIDE,PETSC_DECIDE,M,N);CHKERRQ(ierr);
    ierr = MatSetType(mat_ginkgo,MATGINKGO);CHKERRQ(ierr);
    ierr = MatSetUp(mat_ginkgo);CHKERRQ(ierr);
  }
  for (row=0; row<M; row++) {
    ierr = MatGetRow(A,row,&ncols,&cols,&vals);CHKERRQ(ierr);
    /* PETSc-Ginkgo interface uses axpy for setting off-processor entries, only ADD_VALUES is allowed */
    ierr = MatSetValues(mat_ginkgo,1,&row,ncols,cols,vals,ADD_VALUES);CHKERRQ(ierr);
    ierr = MatRestoreRow(A,row,&ncols,&cols,&vals);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(mat_ginkgo, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat_ginkgo, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  if (reuse == MAT_INPLACE_MATRIX) {
    ierr = MatHeaderReplace(A,&mat_ginkgo);CHKERRQ(ierr);
  } else {
    *newmat = mat_ginkgo;
  }
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatConvert_MPIAIJ_Ginkgo(Mat A, MatType newtype,MatReuse reuse,Mat *newmat)
{
  Mat               mat_ginkgo;
  PetscErrorCode    ierr;
  PetscInt          row,ncols,rstart=A->rmap->rstart,rend=A->rmap->rend,j;
  const PetscInt    *cols;
  const PetscScalar *vals;

  PetscFunctionBegin;
  if (reuse == MAT_REUSE_MATRIX) {
    mat_ginkgo = *newmat;
    ierr = MatZeroEntries(mat_ginkgo);CHKERRQ(ierr);
  } else {
    ierr = MatCreate(PetscObjectComm((PetscObject)A), &mat_ginkgo);CHKERRQ(ierr);
    ierr = MatSetSizes(mat_ginkgo,PETSC_DECIDE,PETSC_DECIDE,A->rmap->N,A->cmap->N);CHKERRQ(ierr);
    ierr = MatSetType(mat_ginkgo,MATGINKGO);CHKERRQ(ierr);
    ierr = MatSetUp(mat_ginkgo);CHKERRQ(ierr);
  }
  for (row=rstart; row<rend; row++) {
    ierr = MatGetRow(A,row,&ncols,&cols,&vals);CHKERRQ(ierr);
    for (j=0; j<ncols; j++) {
      /* PETSc-Ginkgo interface uses axpy for setting off-processor entries, only ADD_VALUES is allowed */
      ierr = MatSetValues(mat_ginkgo,1,&row,1,&cols[j],&vals[j],ADD_VALUES);CHKERRQ(ierr);
    }
    ierr = MatRestoreRow(A,row,&ncols,&cols,&vals);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(mat_ginkgo, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat_ginkgo, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  if (reuse == MAT_INPLACE_MATRIX) {
    ierr = MatHeaderReplace(A,&mat_ginkgo);CHKERRQ(ierr);
  } else {
    *newmat = mat_ginkgo;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatSetUp_Ginkgo(Mat A)
{
  Mat_GinkgoCSR  *a = (Mat_GinkgoCSR*)A->data;
  PetscErrorCode ierr;
  MPI_Comm       comm;
  PetscMPIInt    rsize,csize;
  PetscInt       n;

  PetscFunctionBegin;
  ierr = PetscLayoutSetUp(A->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(A->cmap);CHKERRQ(ierr);

  /* Check if local row and column sizes are equally distributed.
     Jed: Elemental uses "element" cyclic ordering so the sizes need to match that
     exactly.  The strategy in MatElemental is for PETSc to implicitly permute to block ordering (like would be returned by
     PetscSplitOwnership(comm,&n,&N), at which point Elemental matrices can act on PETSc vectors without redistributing the vectors. */
  ierr = PetscObjectGetComm((PetscObject)A,&comm);CHKERRQ(ierr);
  n = PETSC_DECIDE;
  ierr = PetscSplitOwnership(comm,&n,&A->rmap->N);CHKERRQ(ierr);
  if (n != A->rmap->n) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Local row size %D of GINKGO matrix must be equally distributed",A->rmap->n);

  n = PETSC_DECIDE;
  ierr = PetscSplitOwnership(comm,&n,&A->cmap->N);CHKERRQ(ierr);
  if (n != A->cmap->n) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Local column size %D of GINKGO matrix must be equally distributed",A->cmap->n);

  a->emat->Resize(A->rmap->N,A->cmap->N);CHKERRQ(ierr);
  El::Zero(*a->emat);

  ierr = MPI_Comm_size(A->rmap->comm,&rsize);CHKERRMPI(ierr);
  ierr = MPI_Comm_size(A->cmap->comm,&csize);CHKERRMPI(ierr);
  if (csize != rsize) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_INCOMP,"Cannot use row and column communicators of different sizes");
  a->commsize = rsize;
  a->mr[0] = A->rmap->N % rsize; if (!a->mr[0]) a->mr[0] = rsize;
  a->mr[1] = A->cmap->N % csize; if (!a->mr[1]) a->mr[1] = csize;
  a->m[0]  = A->rmap->N / rsize + (a->mr[0] != rsize);
  a->m[1]  = A->cmap->N / csize + (a->mr[1] != csize);
  PetscFunctionReturn(0);
}

PetscErrorCode MatAssemblyBegin_Ginkgo(Mat A, MatAssemblyType type)
{
  Mat_GinkgoCSR  *a = (Mat_GinkgoCSR*)A->data;

  PetscFunctionBegin;
  /* printf("Calling ProcessQueues\n"); */
  a->emat->ProcessQueues();
  /* printf("Finished ProcessQueues\n"); */
  PetscFunctionReturn(0);
}

PetscErrorCode MatAssemblyEnd_Ginkgo(Mat A, MatAssemblyType type)
{
  PetscFunctionBegin;
  /* Currently does nothing */
  PetscFunctionReturn(0);
}

PetscErrorCode MatLoad_Ginkgo(Mat newMat, PetscViewer viewer)
{
  PetscErrorCode ierr;
  Mat            Adense,Ae;
  MPI_Comm       comm;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)newMat,&comm);CHKERRQ(ierr);
  ierr = MatCreate(comm,&Adense);CHKERRQ(ierr);
  ierr = MatSetType(Adense,MATDENSE);CHKERRQ(ierr);
  ierr = MatLoad(Adense,viewer);CHKERRQ(ierr);
  ierr = MatConvert(Adense, MATGINKGO, MAT_INITIAL_MATRIX,&Ae);CHKERRQ(ierr);
  ierr = MatDestroy(&Adense);CHKERRQ(ierr);
  ierr = MatHeaderReplace(newMat,&Ae);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDestroy_Ginkgo(Mat A)
{
  Mat_GinkgoCSR      *a = (Mat_GinkgoCSR*)A->data;
  PetscErrorCode     ierr;
  Mat_GinkgoCSR_Grid *commgrid;
  PetscBool          flg;
  MPI_Comm           icomm;

  PetscFunctionBegin;
  delete a->emat;
  delete a->P;
  delete a->Q;

  El::mpi::Comm cxxcomm(PetscObjectComm((PetscObject)A));
  ierr = PetscCommDuplicate(cxxcomm.comm,&icomm,NULL);CHKERRQ(ierr);
  ierr = MPI_Comm_get_attr(icomm,Petsc_Ginkgo_keyval,(void**)&commgrid,(int*)&flg);CHKERRMPI(ierr);
  if (--commgrid->grid_refct == 0) {
    delete commgrid->grid;
    ierr = PetscFree(commgrid);CHKERRQ(ierr);
    ierr = MPI_Comm_free_keyval(&Petsc_Ginkgo_keyval);CHKERRMPI(ierr);
  }
  ierr = PetscCommDestroy(&icomm);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatGetOwnershipIS_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatFactorGetSolverType_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatProductSetFromOptions_ginkgo_mpidense_C",NULL);CHKERRQ(ierr);
  ierr = PetscFree(A->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------*/
static struct _MatOps MatOps_Values = {
       MatSetValues_Ginkgo,
       0,
       0,
       MatMult_Ginkgo,
/* 4*/ MatMultAdd_Ginkgo,
       MatMultTranspose_Ginkgo,
       MatMultTransposeAdd_Ginkgo,
       MatSolve_Ginkgo,
       MatSolveAdd_Ginkgo,
       0,
/*10*/ 0,
       MatLUFactor_Ginkgo,
       MatCholeskyFactor_Ginkgo,
       0,
       MatTranspose_Ginkgo,
/*15*/ MatGetInfo_Ginkgo,
       0,
       MatGetDiagonal_Ginkgo,
       MatDiagonalScale_Ginkgo,
       MatNorm_Ginkgo,
/*20*/ MatAssemblyBegin_Ginkgo,
       MatAssemblyEnd_Ginkgo,
       MatSetOption_Ginkgo,
       MatZeroEntries_Ginkgo,
/*24*/ 0,
       MatLUFactorSymbolic_Ginkgo,
       MatLUFactorNumeric_Ginkgo,
       MatCholeskyFactorSymbolic_Ginkgo,
       MatCholeskyFactorNumeric_Ginkgo,
/*29*/ MatSetUp_Ginkgo,
       0,
       0,
       0,
       0,
/*34*/ MatDuplicate_Ginkgo,
       0,
       0,
       0,
       0,
/*39*/ MatAXPY_Ginkgo,
       0,
       0,
       0,
       MatCopy_Ginkgo,
/*44*/ 0,
       MatScale_Ginkgo,
       MatShift_Basic,
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
       MatDestroy_Ginkgo,
       MatView_Ginkgo,
       0,
       0,
/*64*/ 0,
       0,
       0,
       0,
       0,
/*69*/ 0,
       0,
       MatConvert_Ginkgo_Dense,
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
       MatLoad_Ginkgo,
/*84*/ 0,
       0,
       0,
       0,
       0,
/*89*/ 0,
       0,
       MatMatMultNumeric_Ginkgo,
       0,
       0,
/*94*/ 0,
       0,
       0,
       MatMatTransposeMultNumeric_Ginkgo,
       0,
/*99*/ MatProductSetFromOptions_Ginkgo,
       0,
       0,
       MatConjugate_Ginkgo,
       0,
/*104*/0,
       0,
       0,
       0,
       0,
/*109*/MatMatSolve_Ginkgo,
       0,
       0,
       0,
       MatMissingDiagonal_Ginkgo,
/*114*/0,
       0,
       0,
       0,
       0,
/*119*/0,
       MatHermitianTranspose_Ginkgo,
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
       0,
       0,
/*140*/0,
       0,
       0,
       0,
       0,
/*145*/0,
       0,
       0
};

