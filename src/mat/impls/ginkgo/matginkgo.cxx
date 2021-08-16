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
  Mat_Ginkgo  *a = (Mat_Ginkgo*)A->data;

  PetscFunctionBegin;
  switch (op) {
  case MAT_NEW_NONZERO_LOCATIONS:
  case MAT_NEW_NONZERO_LOCATION_ERR:
  case MAT_NEW_NONZERO_ALLOCATION_ERR:
  case MAT_SYMMETRIC:
  case MAT_SORTED_FULL:
  case MAT_HERMITIAN:
    break;
  case MAT_ROW_ORIENTED:
    a->roworiented = flg;
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"unknown option %s",MatOptions[op]);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSetValues_Ginkgo(Mat A,PetscInt nr,const PetscInt *rows,PetscInt nc,const PetscInt *cols,const PetscScalar *vals,InsertMode imode)
{
  Mat_Ginkgo  *a = (Mat_Ginkgo*)A->data;
  PetscInt       i,j,rrank,ridx,crank,cidx,erow,ecol,numQueues=0;

  PetscFunctionBegin;
  // TODO: Initialize matrix to all zeros?

  // Count the number of queues from this process
  if (a->roworiented) {
    for (i=0; i<nr; i++) {
      if (rows[i] < 0) continue;
      P2RO(A,0,rows[i],&rrank,&ridx);
      RO2E(A,0,rrank,ridx,&erow);
      if (rrank < 0 || ridx < 0 || erow < 0) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Incorrect row translation");
      for (j=0; j<nc; j++) {
        if (cols[j] < 0) continue;
        P2RO(A,1,cols[j],&crank,&cidx);
        RO2E(A,1,crank,cidx,&ecol);
        if (crank < 0 || cidx < 0 || ecol < 0) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Incorrect col translation");
        if (!a->emat->IsLocal(erow,ecol)) { /* off-proc entry */
          /* printf("Will later remotely update (%d,%d)\n",erow,ecol); */
          if (imode != ADD_VALUES) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Only ADD_VALUES to off-processor entry is supported");
          ++numQueues;
          continue;
        }
        /* printf("Locally updating (%d,%d)\n",erow,ecol); */
        switch (imode) {
        case INSERT_VALUES: a->emat->Set(erow,ecol,(PetscGinkgoScalar)vals[i*nc+j]); break;
        case ADD_VALUES: a->emat->Update(erow,ecol,(PetscGinkgoScalar)vals[i*nc+j]); break;
        default: SETERRQ1(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"No support for InsertMode %d",(int)imode);
        }
      }
    }

    /* printf("numQueues=%d\n",numQueues); */
    a->emat->Reserve( numQueues);
    for (i=0; i<nr; i++) {
      if (rows[i] < 0) continue;
      P2RO(A,0,rows[i],&rrank,&ridx);
      RO2E(A,0,rrank,ridx,&erow);
      for (j=0; j<nc; j++) {
        if (cols[j] < 0) continue;
        P2RO(A,1,cols[j],&crank,&cidx);
        RO2E(A,1,crank,cidx,&ecol);
        if (!a->emat->IsLocal(erow,ecol)) { /*off-proc entry*/
          /* printf("Queueing remotely update of (%d,%d)\n",erow,ecol); */
          a->emat->QueueUpdate( erow, ecol, vals[i*nc+j]);
        }
      }
    }
  } else { /* columnoriented */
    for (j=0; j<nc; j++) {
      if (cols[j] < 0) continue;
      P2RO(A,1,cols[j],&crank,&cidx);
      RO2E(A,1,crank,cidx,&ecol);
      if (crank < 0 || cidx < 0 || ecol < 0) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Incorrect col translation");
      for (i=0; i<nr; i++) {
        if (rows[i] < 0) continue;
        P2RO(A,0,rows[i],&rrank,&ridx);
        RO2E(A,0,rrank,ridx,&erow);
        if (rrank < 0 || ridx < 0 || erow < 0) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Incorrect row translation");
        if (!a->emat->IsLocal(erow,ecol)) { /* off-proc entry */
          /* printf("Will later remotely update (%d,%d)\n",erow,ecol); */
          if (imode != ADD_VALUES) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Only ADD_VALUES to off-processor entry is supported");
          ++numQueues;
          continue;
        }
        /* printf("Locally updating (%d,%d)\n",erow,ecol); */
        switch (imode) {
        case INSERT_VALUES: a->emat->Set(erow,ecol,(PetscGinkgoScalar)vals[i+j*nr]); break;
        case ADD_VALUES: a->emat->Update(erow,ecol,(PetscGinkgoScalar)vals[i+j*nr]); break;
        default: SETERRQ1(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"No support for InsertMode %d",(int)imode);
        }
      }
    }

    /* printf("numQueues=%d\n",numQueues); */
    a->emat->Reserve( numQueues);
    for (j=0; j<nc; j++) {
      if (cols[j] < 0) continue;
      P2RO(A,1,cols[j],&crank,&cidx);
      RO2E(A,1,crank,cidx,&ecol);

      for (i=0; i<nr; i++) {
        if (rows[i] < 0) continue;
        P2RO(A,0,rows[i],&rrank,&ridx);
        RO2E(A,0,rrank,ridx,&erow);
        if (!a->emat->IsLocal(erow,ecol)) { /*off-proc entry*/
          /* printf("Queueing remotely update of (%d,%d)\n",erow,ecol); */
          a->emat->QueueUpdate( erow, ecol, vals[i+j*nr]);
        }
      }
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMult_Ginkgo(Mat A,Vec X,Vec Y)
{
  Mat_Ginkgo            *a = (Mat_Ginkgo*)A->data;
  PetscErrorCode        ierr;
  const PetscGinkgoScalar *x;
  PetscGinkgoScalar       *y;
  PetscGinkgoScalar       one = 1,zero = 0;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(X,(const PetscScalar **)&x);CHKERRQ(ierr);
  ierr = VecGetArray(Y,(PetscScalar **)&y);CHKERRQ(ierr);
  { /* Scoping so that constructor is called before pointer is returned */
    /* TODO: SEK */
    El::DistMatrix<PetscGinkgoScalar,El::VC,El::STAR> xe, ye;
    xe.LockedAttach(A->cmap->N,1,*a->grid,0,0,x,A->cmap->n);
    ye.Attach(A->rmap->N,1,*a->grid,0,0,y,A->rmap->n);
    /* TODO: SEK */
    El::Gemv(El::NORMAL,one,*a->emat,xe,zero,ye);
  }
  ierr = VecRestoreArrayRead(X,(const PetscScalar **)&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(Y,(PetscScalar **)&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultTranspose_Ginkgo(Mat A,Vec X,Vec Y)
{
  Mat_Ginkgo            *a = (Mat_Ginkgo*)A->data;
  PetscErrorCode        ierr;
  const PetscGinkgoScalar *x;
  PetscGinkgoScalar       *y;
  PetscGinkgoScalar       one = 1,zero = 0;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(X,(const PetscScalar **)&x);CHKERRQ(ierr);
  ierr = VecGetArray(Y,(PetscScalar **)&y);CHKERRQ(ierr);
  { /* Scoping so that constructor is called before pointer is returned */
    /* TODO: SEK */
    El::DistMatrix<PetscGinkgoScalar,El::VC,El::STAR> xe, ye;
    xe.LockedAttach(A->rmap->N,1,*a->grid,0,0,x,A->rmap->n);
    ye.Attach(A->cmap->N,1,*a->grid,0,0,y,A->cmap->n);
    /* TODO: SEK */
    El::Gemv(El::TRANSPOSE,one,*a->emat,xe,zero,ye);
  }
  ierr = VecRestoreArrayRead(X,(const PetscScalar **)&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(Y,(PetscScalar **)&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultAdd_Ginkgo(Mat A,Vec X,Vec Y,Vec Z)
{
  Mat_Ginkgo         *a = (Mat_Ginkgo*)A->data;
  PetscErrorCode        ierr;
  const PetscGinkgoScalar *x;
  PetscGinkgoScalar       *z;
  PetscGinkgoScalar       one = 1;

  PetscFunctionBegin;
  if (Y != Z) {ierr = VecCopy(Y,Z);CHKERRQ(ierr);}
  ierr = VecGetArrayRead(X,(const PetscScalar **)&x);CHKERRQ(ierr);
  ierr = VecGetArray(Z,(PetscScalar **)&z);CHKERRQ(ierr);
  { /* Scoping so that constructor is called before pointer is returned */
    /* TODO: SEK */
    El::DistMatrix<PetscGinkgoScalar,El::VC,El::STAR> xe, ze;
    xe.LockedAttach(A->cmap->N,1,*a->grid,0,0,x,A->cmap->n);
    ze.Attach(A->rmap->N,1,*a->grid,0,0,z,A->rmap->n);
    /* TODO: SEK */
    El::Gemv(El::NORMAL,one,*a->emat,xe,one,ze);
  }
  ierr = VecRestoreArrayRead(X,(const PetscScalar **)&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(Z,(PetscScalar **)&z);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultTransposeAdd_Ginkgo(Mat A,Vec X,Vec Y,Vec Z)
{
  Mat_Ginkgo         *a = (Mat_Ginkgo*)A->data;
  PetscErrorCode        ierr;
  const PetscGinkgoScalar *x;
  PetscGinkgoScalar       *z;
  PetscGinkgoScalar       one = 1;

  PetscFunctionBegin;
  if (Y != Z) {ierr = VecCopy(Y,Z);CHKERRQ(ierr);}
  ierr = VecGetArrayRead(X,(const PetscScalar **)&x);CHKERRQ(ierr);
  ierr = VecGetArray(Z,(PetscScalar **)&z);CHKERRQ(ierr);
  { /* Scoping so that constructor is called before pointer is returned */
    El::DistMatrix<PetscGinkgoScalar,El::VC,El::STAR> xe, ze;
    xe.LockedAttach(A->rmap->N,1,*a->grid,0,0,x,A->rmap->n);
    ze.Attach(A->cmap->N,1,*a->grid,0,0,z,A->cmap->n);
    El::Gemv(El::TRANSPOSE,one,*a->emat,xe,one,ze);
  }
  ierr = VecRestoreArrayRead(X,(const PetscScalar **)&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(Z,(PetscScalar **)&z);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMatMultNumeric_Ginkgo(Mat A,Mat B,Mat C)
{
  Mat_Ginkgo    *a = (Mat_Ginkgo*)A->data;
  Mat_Ginkgo    *b = (Mat_Ginkgo*)B->data;
  Mat_Ginkgo    *c = (Mat_Ginkgo*)C->data;
  PetscGinkgoScalar  one = 1,zero = 0;

  PetscFunctionBegin;
  { /* Scoping so that constructor is called before pointer is returned */
    El::Gemm(El::NORMAL,El::NORMAL,one,*a->emat,*b->emat,zero,*c->emat);
  }
  C->assembled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

PetscErrorCode MatMatMultSymbolic_Ginkgo(Mat A,Mat B,PetscReal fill,Mat Ce)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSetSizes(Ce,A->rmap->n,B->cmap->n,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = MatSetType(Ce,MATGINKGO);CHKERRQ(ierr);
  ierr = MatSetUp(Ce);CHKERRQ(ierr);
  Ce->ops->matmultnumeric = MatMatMultNumeric_Ginkgo;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMatTransposeMultNumeric_Ginkgo(Mat A,Mat B,Mat C)
{
  Mat_Ginkgo      *a = (Mat_Ginkgo*)A->data;
  Mat_Ginkgo      *b = (Mat_Ginkgo*)B->data;
  Mat_Ginkgo      *c = (Mat_Ginkgo*)C->data;
  PetscGinkgoScalar    one = 1,zero = 0;

  PetscFunctionBegin;
  { /* Scoping so that constructor is called before pointer is returned */
    El::Gemm(El::NORMAL,El::TRANSPOSE,one,*a->emat,*b->emat,zero,*c->emat);
  }
  C->assembled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMatTransposeMultSymbolic_Ginkgo(Mat A,Mat B,PetscReal fill,Mat C)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSetSizes(C,A->rmap->n,B->rmap->n,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = MatSetType(C,MATGINKGO);CHKERRQ(ierr);
  ierr = MatSetUp(C);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* --------------------------------------- */
static PetscErrorCode MatProductSetFromOptions_Ginkgo_AB(Mat C)
{


  PetscFunctionBegin;
  C->ops->matmultsymbolic = MatMatMultSymbolic_Ginkgo;
  C->ops->productsymbolic = MatProductSymbolic_AB;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatProductSetFromOptions_Ginkgo_ABt(Mat C)
{
  PetscFunctionBegin;
  C->ops->mattransposemultsymbolic = MatMatTransposeMultSymbolic_Ginkgo;
  C->ops->productsymbolic          = MatProductSymbolic_ABt;
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatProductSetFromOptions_Ginkgo(Mat C)
{
  PetscErrorCode ierr;
  Mat_Product    *product = C->product;

  PetscFunctionBegin;
  switch (product->type) {
  case MATPRODUCT_AB:
    ierr = MatProductSetFromOptions_Ginkgo_AB(C);CHKERRQ(ierr);
    break;
  case MATPRODUCT_ABt:
    ierr = MatProductSetFromOptions_Ginkgo_ABt(C);CHKERRQ(ierr);
    break;
  default:
    break;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatMatMultNumeric_Ginkgo_MPIDense(Mat A,Mat B,Mat C)
{
  Mat            Be,Ce;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatConvert(B,MATGINKGO,MAT_INITIAL_MATRIX,&Be);CHKERRQ(ierr);
  ierr = MatMatMult(A,Be,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&Ce);CHKERRQ(ierr);
  ierr = MatConvert(Ce,MATMPIDENSE,MAT_REUSE_MATRIX,&C);CHKERRQ(ierr);
  ierr = MatDestroy(&Be);CHKERRQ(ierr);
  ierr = MatDestroy(&Ce);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMatMultSymbolic_Ginkgo_MPIDense(Mat A,Mat B,PetscReal fill,Mat C)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSetSizes(C,A->rmap->n,B->cmap->n,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = MatSetType(C,MATMPIDENSE);CHKERRQ(ierr);
  ierr = MatSetUp(C);CHKERRQ(ierr);
  C->ops->matmultnumeric = MatMatMultNumeric_Ginkgo_MPIDense;
  PetscFunctionReturn(0);
}

PetscErrorCode MatProductSetFromOptions_Ginkgo_MPIDense_AB(Mat C)
{
  PetscFunctionBegin;
  C->ops->matmultsymbolic = MatMatMultSymbolic_Ginkgo_MPIDense;
  C->ops->productsymbolic = MatProductSymbolic_AB;
  PetscFunctionReturn(0);
}

PetscErrorCode MatProductSetFromOptions_Ginkgo_MPIDense(Mat C)
{
  PetscErrorCode ierr;
  Mat_Product    *product = C->product;

  PetscFunctionBegin;
  if (product->type == MATPRODUCT_AB) {
    ierr = MatProductSetFromOptions_Ginkgo_MPIDense_AB(C);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
/* --------------------------------------- */

static PetscErrorCode MatGetDiagonal_Ginkgo(Mat A,Vec D)
{
  PetscInt        i,nrows,ncols,nD,rrank,ridx,crank,cidx;
  Mat_Ginkgo   *a = (Mat_Ginkgo*)A->data;
  PetscErrorCode  ierr;
  PetscGinkgoScalar v;
  MPI_Comm        comm;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)A,&comm);CHKERRQ(ierr);
  ierr = MatGetSize(A,&nrows,&ncols);CHKERRQ(ierr);
  nD = nrows>ncols ? ncols : nrows;
  for (i=0; i<nD; i++) {
    PetscInt erow,ecol;
    P2RO(A,0,i,&rrank,&ridx);
    RO2E(A,0,rrank,ridx,&erow);
    if (rrank < 0 || ridx < 0 || erow < 0) SETERRQ(comm,PETSC_ERR_PLIB,"Incorrect row translation");
    P2RO(A,1,i,&crank,&cidx);
    RO2E(A,1,crank,cidx,&ecol);
    if (crank < 0 || cidx < 0 || ecol < 0) SETERRQ(comm,PETSC_ERR_PLIB,"Incorrect col translation");
    v = a->emat->Get(erow,ecol);
    ierr = VecSetValues(D,1,&i,(PetscScalar*)&v,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(D);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(D);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDiagonalScale_Ginkgo(Mat X,Vec L,Vec R)
{
  Mat_Ginkgo         *x = (Mat_Ginkgo*)X->data;
  const PetscGinkgoScalar *d;
  PetscErrorCode        ierr;

  PetscFunctionBegin;
  if (R) {
    ierr = VecGetArrayRead(R,(const PetscScalar **)&d);CHKERRQ(ierr);
    El::DistMatrix<PetscGinkgoScalar,El::VC,El::STAR> de;
    de.LockedAttach(X->cmap->N,1,*x->grid,0,0,d,X->cmap->n);
    El::DiagonalScale(El::RIGHT,El::NORMAL,de,*x->emat);
    ierr = VecRestoreArrayRead(R,(const PetscScalar **)&d);CHKERRQ(ierr);
  }
  if (L) {
    ierr = VecGetArrayRead(L,(const PetscScalar **)&d);CHKERRQ(ierr);
    El::DistMatrix<PetscGinkgoScalar,El::VC,El::STAR> de;
    de.LockedAttach(X->rmap->N,1,*x->grid,0,0,d,X->rmap->n);
    El::DiagonalScale(El::LEFT,El::NORMAL,de,*x->emat);
    ierr = VecRestoreArrayRead(L,(const PetscScalar **)&d);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMissingDiagonal_Ginkgo(Mat A,PetscBool *missing,PetscInt *d)
{
  PetscFunctionBegin;
  *missing = PETSC_FALSE;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatScale_Ginkgo(Mat X,PetscScalar a)
{
  Mat_Ginkgo  *x = (Mat_Ginkgo*)X->data;

  PetscFunctionBegin;
  El::Scale((PetscGinkgoScalar)a,*x->emat);
  PetscFunctionReturn(0);
}

/*
  MatAXPY - Computes Y = a*X + Y.
*/
static PetscErrorCode MatAXPY_Ginkgo(Mat Y,PetscScalar a,Mat X,MatStructure str)
{
  Mat_Ginkgo  *x = (Mat_Ginkgo*)X->data;
  Mat_Ginkgo  *y = (Mat_Ginkgo*)Y->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  El::Axpy((PetscGinkgoScalar)a,*x->emat,*y->emat);
  ierr = PetscObjectStateIncrease((PetscObject)Y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCopy_Ginkgo(Mat A,Mat B,MatStructure str)
{
  Mat_Ginkgo *a=(Mat_Ginkgo*)A->data;
  Mat_Ginkgo *b=(Mat_Ginkgo*)B->data;
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
  Mat_Ginkgo  *a=(Mat_Ginkgo*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)A,&comm);CHKERRQ(ierr);
  ierr = MatCreate(comm,&Be);CHKERRQ(ierr);
  ierr = MatSetSizes(Be,A->rmap->n,A->cmap->n,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = MatSetType(Be,MATGINKGO);CHKERRQ(ierr);
  ierr = MatSetUp(Be);CHKERRQ(ierr);
  *B = Be;
  if (op == MAT_COPY_VALUES) {
    Mat_Ginkgo *b=(Mat_Ginkgo*)Be->data;
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
  Mat_Ginkgo  *a = (Mat_Ginkgo*)A->data, *b;

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
  b = (Mat_Ginkgo*)Be->data;
  El::Transpose(*a->emat,*b->emat);
  Be->assembled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatConjugate_Ginkgo(Mat A)
{
  Mat_Ginkgo  *a = (Mat_Ginkgo*)A->data;

  PetscFunctionBegin;
  El::Conjugate(*a->emat);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatHermitianTranspose_Ginkgo(Mat A,MatReuse reuse,Mat *B)
{
  Mat            Be = *B;
  PetscErrorCode ierr;
  MPI_Comm       comm;
  Mat_Ginkgo  *a = (Mat_Ginkgo*)A->data, *b;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)A,&comm);CHKERRQ(ierr);
  /* Only out-of-place supported */
  if (reuse == MAT_INITIAL_MATRIX) {
    ierr = MatCreate(comm,&Be);CHKERRQ(ierr);
    ierr = MatSetSizes(Be,A->cmap->n,A->rmap->n,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
    ierr = MatSetType(Be,MATGINKGO);CHKERRQ(ierr);
    ierr = MatSetUp(Be);CHKERRQ(ierr);
    *B = Be;
  }
  b = (Mat_Ginkgo*)Be->data;
  El::Adjoint(*a->emat,*b->emat);
  Be->assembled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolve_Ginkgo(Mat A,Vec B,Vec X)
{
  Mat_Ginkgo     *a = (Mat_Ginkgo*)A->data;
  PetscErrorCode    ierr;
  PetscGinkgoScalar   *x;
  PetscInt          pivoting = a->pivoting;

  PetscFunctionBegin;
  ierr = VecCopy(B,X);CHKERRQ(ierr);
  ierr = VecGetArray(X,(PetscScalar **)&x);CHKERRQ(ierr);

  El::DistMatrix<PetscGinkgoScalar,El::VC,El::STAR> xe;
  xe.Attach(A->rmap->N,1,*a->grid,0,0,x,A->rmap->n);
  El::DistMatrix<PetscGinkgoScalar,El::MC,El::MR> xer(xe);
  switch (A->factortype) {
  case MAT_FACTOR_LU:
    if (pivoting == 0) {
      El::lu::SolveAfter(El::NORMAL,*a->emat,xer);
    } else if (pivoting == 1) {
      El::lu::SolveAfter(El::NORMAL,*a->emat,*a->P,xer);
    } else { /* pivoting == 2 */
      El::lu::SolveAfter(El::NORMAL,*a->emat,*a->P,*a->Q,xer);
    }
    break;
  case MAT_FACTOR_CHOLESKY:
    El::cholesky::SolveAfter(El::UPPER,El::NORMAL,*a->emat,xer);
    break;
  default:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Unfactored Matrix or Unsupported MatFactorType");
    break;
  }
  El::Copy(xer,xe);

  ierr = VecRestoreArray(X,(PetscScalar **)&x);CHKERRQ(ierr);
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

static PetscErrorCode MatMatSolve_Ginkgo(Mat A,Mat B,Mat X)
{
  Mat_Ginkgo  *a = (Mat_Ginkgo*)A->data;
  Mat_Ginkgo  *x;
  Mat            C;
  PetscInt       pivoting = a->pivoting;
  PetscBool      flg;
  MatType        type;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatGetType(X,&type);CHKERRQ(ierr);
  ierr = PetscStrcmp(type,MATGINKGO,&flg);CHKERRQ(ierr);
  if (!flg) {
    ierr = MatConvert(B,MATGINKGO,MAT_INITIAL_MATRIX,&C);CHKERRQ(ierr);
    x = (Mat_Ginkgo*)C->data;
  } else {
    x = (Mat_Ginkgo*)X->data;
    El::Copy(*((Mat_Ginkgo*)B->data)->emat,*x->emat);
  }
  switch (A->factortype) {
  case MAT_FACTOR_LU:
    if (pivoting == 0) {
      El::lu::SolveAfter(El::NORMAL,*a->emat,*x->emat);
    } else if (pivoting == 1) {
      El::lu::SolveAfter(El::NORMAL,*a->emat,*a->P,*x->emat);
    } else {
      El::lu::SolveAfter(El::NORMAL,*a->emat,*a->P,*a->Q,*x->emat);
    }
    break;
  case MAT_FACTOR_CHOLESKY:
    El::cholesky::SolveAfter(El::UPPER,El::NORMAL,*a->emat,*x->emat);
    break;
  default:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Unfactored Matrix or Unsupported MatFactorType");
    break;
  }
  if (!flg) {
    ierr = MatConvert(C,type,MAT_REUSE_MATRIX,&X);CHKERRQ(ierr);
    ierr = MatDestroy(&C);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatLUFactor_Ginkgo(Mat A,IS row,IS col,const MatFactorInfo *info)
{
  Mat_Ginkgo  *a = (Mat_Ginkgo*)A->data;
  PetscErrorCode ierr;
  PetscInt       pivoting = a->pivoting;

  PetscFunctionBegin;
  if (pivoting == 0) {
    El::LU(*a->emat);
  } else if (pivoting == 1) {
    El::LU(*a->emat,*a->P);
  } else {
    El::LU(*a->emat,*a->P,*a->Q);
  }
  A->factortype = MAT_FACTOR_LU;
  A->assembled  = PETSC_TRUE;

  ierr = PetscFree(A->solvertype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(MATSOLVERGINKGO,&A->solvertype);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode  MatLUFactorNumeric_Ginkgo(Mat F,Mat A,const MatFactorInfo *info)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCopy(A,F,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatLUFactor_Ginkgo(F,0,0,info);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode  MatLUFactorSymbolic_Ginkgo(Mat F,Mat A,IS r,IS c,const MatFactorInfo *info)
{
  PetscFunctionBegin;
  /* F is created and allocated by MatGetFactor_ginkgo_petsc(), skip this routine. */
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCholeskyFactor_Ginkgo(Mat A,IS perm,const MatFactorInfo *info)
{
  Mat_Ginkgo  *a = (Mat_Ginkgo*)A->data;
  El::DistMatrix<PetscGinkgoScalar,El::MC,El::STAR> d;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  El::Cholesky(El::UPPER,*a->emat);
  A->factortype = MAT_FACTOR_CHOLESKY;
  A->assembled  = PETSC_TRUE;

  ierr = PetscFree(A->solvertype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(MATSOLVERGINKGO,&A->solvertype);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCholeskyFactorNumeric_Ginkgo(Mat F,Mat A,const MatFactorInfo *info)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCopy(A,F,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatCholeskyFactor_Ginkgo(F,0,info);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCholeskyFactorSymbolic_Ginkgo(Mat F,Mat A,IS perm,const MatFactorInfo *info)
{
  PetscFunctionBegin;
  /* F is created and allocated by MatGetFactor_ginkgo_petsc(), skip this routine. */
  PetscFunctionReturn(0);
}

PetscErrorCode MatFactorGetSolverType_Ginkgo_Ginkgo(Mat A,MatSolverType *type)
{
  PetscFunctionBegin;
  *type = MATSOLVERGINKGO;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatGetFactor_Ginkgo_Ginkgo(Mat A,MatFactorType ftype,Mat *F)
{
  Mat            B;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Create the factorization matrix */
  ierr = MatCreate(PetscObjectComm((PetscObject)A),&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,A->rmap->n,A->cmap->n,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = MatSetType(B,MATGINKGO);CHKERRQ(ierr);
  ierr = MatSetUp(B);CHKERRQ(ierr);
  B->factortype = ftype;
  B->trivialsymbolic = PETSC_TRUE;
  ierr = PetscFree(B->solvertype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(MATSOLVERGINKGO,&B->solvertype);CHKERRQ(ierr);

  ierr = PetscObjectComposeFunction((PetscObject)B,"MatFactorGetSolverType_C",MatFactorGetSolverType_Ginkgo_Ginkgo);CHKERRQ(ierr);
  *F            = B;
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
  Mat_Ginkgo *a=(Mat_Ginkgo*)A->data;

  PetscFunctionBegin;
  switch (type) {
  case NORM_1:
    *nrm = El::OneNorm(*a->emat);
    break;
  case NORM_FROBENIUS:
    *nrm = El::FrobeniusNorm(*a->emat);
    break;
  case NORM_INFINITY:
    *nrm = El::InfinityNorm(*a->emat);
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Unsupported norm type");
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatZeroEntries_Ginkgo(Mat A)
{
  Mat_Ginkgo *a=(Mat_Ginkgo*)A->data;

  PetscFunctionBegin;
  El::Zero(*a->emat);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatGetOwnershipIS_Ginkgo(Mat A,IS *rows,IS *cols)
{
  Mat_Ginkgo  *a = (Mat_Ginkgo*)A->data;
  PetscErrorCode ierr;
  PetscInt       i,m,shift,stride,*idx;

  PetscFunctionBegin;
  if (rows) {
    m = a->emat->LocalHeight();
    shift = a->emat->ColShift();
    stride = a->emat->ColStride();
    ierr = PetscMalloc1(m,&idx);CHKERRQ(ierr);
    for (i=0; i<m; i++) {
      PetscInt rank,offset;
      E2RO(A,0,shift+i*stride,&rank,&offset);
      RO2P(A,0,rank,offset,&idx[i]);
    }
    ierr = ISCreateGeneral(PETSC_COMM_SELF,m,idx,PETSC_OWN_POINTER,rows);CHKERRQ(ierr);
  }
  if (cols) {
    m = a->emat->LocalWidth();
    shift = a->emat->RowShift();
    stride = a->emat->RowStride();
    ierr = PetscMalloc1(m,&idx);CHKERRQ(ierr);
    for (i=0; i<m; i++) {
      PetscInt rank,offset;
      E2RO(A,1,shift+i*stride,&rank,&offset);
      RO2P(A,1,rank,offset,&idx[i]);
    }
    ierr = ISCreateGeneral(PETSC_COMM_SELF,m,idx,PETSC_OWN_POINTER,cols);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatConvert_Ginkgo_Dense(Mat A,MatType newtype,MatReuse reuse,Mat *B)
{
  Mat                Bmpi;
  Mat_Ginkgo      *a = (Mat_Ginkgo*)A->data;
  MPI_Comm           comm;
  PetscErrorCode     ierr;
  IS                 isrows,iscols;
  PetscInt           rrank,ridx,crank,cidx,nrows,ncols,i,j,erow,ecol,elrow,elcol;
  const PetscInt     *rows,*cols;
  PetscGinkgoScalar    v;
  const El::Grid     &grid = a->emat->Grid();

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)A,&comm);CHKERRQ(ierr);

  if (reuse == MAT_REUSE_MATRIX) {
    Bmpi = *B;
  } else {
    ierr = MatCreate(comm,&Bmpi);CHKERRQ(ierr);
    ierr = MatSetSizes(Bmpi,A->rmap->n,A->cmap->n,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
    ierr = MatSetType(Bmpi,MATDENSE);CHKERRQ(ierr);
    ierr = MatSetUp(Bmpi);CHKERRQ(ierr);
  }

  /* Get local entries of A */
  ierr = MatGetOwnershipIS(A,&isrows,&iscols);CHKERRQ(ierr);
  ierr = ISGetLocalSize(isrows,&nrows);CHKERRQ(ierr);
  ierr = ISGetIndices(isrows,&rows);CHKERRQ(ierr);
  ierr = ISGetLocalSize(iscols,&ncols);CHKERRQ(ierr);
  ierr = ISGetIndices(iscols,&cols);CHKERRQ(ierr);

  if (a->roworiented) {
    for (i=0; i<nrows; i++) {
      P2RO(A,0,rows[i],&rrank,&ridx); /* convert indices between PETSc <-> (Rank,Offset) <-> Ginkgo */
      RO2E(A,0,rrank,ridx,&erow);
      if (rrank < 0 || ridx < 0 || erow < 0) SETERRQ(comm,PETSC_ERR_PLIB,"Incorrect row translation");
      for (j=0; j<ncols; j++) {
        P2RO(A,1,cols[j],&crank,&cidx);
        RO2E(A,1,crank,cidx,&ecol);
        if (crank < 0 || cidx < 0 || ecol < 0) SETERRQ(comm,PETSC_ERR_PLIB,"Incorrect col translation");

        elrow = erow / grid.MCSize(); /* Ginkgo local row index */
        elcol = ecol / grid.MRSize(); /* Ginkgo local column index */
        v = a->emat->GetLocal(elrow,elcol);
        ierr = MatSetValues(Bmpi,1,&rows[i],1,&cols[j],(PetscScalar *)&v,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  } else { /* column-oriented */
    for (j=0; j<ncols; j++) {
      P2RO(A,1,cols[j],&crank,&cidx);
      RO2E(A,1,crank,cidx,&ecol);
      if (crank < 0 || cidx < 0 || ecol < 0) SETERRQ(comm,PETSC_ERR_PLIB,"Incorrect col translation");
      for (i=0; i<nrows; i++) {
        P2RO(A,0,rows[i],&rrank,&ridx); /* convert indices between PETSc <-> (Rank,Offset) <-> Ginkgo */
        RO2E(A,0,rrank,ridx,&erow);
        if (rrank < 0 || ridx < 0 || erow < 0) SETERRQ(comm,PETSC_ERR_PLIB,"Incorrect row translation");

        elrow = erow / grid.MCSize(); /* Ginkgo local row index */
        elcol = ecol / grid.MRSize(); /* Ginkgo local column index */
        v = a->emat->GetLocal(elrow,elcol);
        ierr = MatSetValues(Bmpi,1,&rows[i],1,&cols[j],(PetscScalar *)&v,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  }
  ierr = MatAssemblyBegin(Bmpi,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Bmpi,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (reuse == MAT_INPLACE_MATRIX) {
    ierr = MatHeaderReplace(A,&Bmpi);CHKERRQ(ierr);
  } else {
    *B = Bmpi;
  }
  ierr = ISDestroy(&isrows);CHKERRQ(ierr);
  ierr = ISDestroy(&iscols);CHKERRQ(ierr);
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

PETSC_INTERN PetscErrorCode MatConvert_SeqSBAIJ_Ginkgo(Mat A, MatType newtype,MatReuse reuse,Mat *newmat)
{
  Mat               mat_ginkgo;
  PetscErrorCode    ierr;
  PetscInt          M=A->rmap->N,N=A->cmap->N,row,ncols,j;
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
  ierr = MatGetRowUpperTriangular(A);CHKERRQ(ierr);
  for (row=0; row<M; row++) {
    ierr = MatGetRow(A,row,&ncols,&cols,&vals);CHKERRQ(ierr);
    /* PETSc-Ginkgo interface uses axpy for setting off-processor entries, only ADD_VALUES is allowed */
    ierr = MatSetValues(mat_ginkgo,1,&row,ncols,cols,vals,ADD_VALUES);CHKERRQ(ierr);
    for (j=0; j<ncols; j++) { /* lower triangular part */
      PetscScalar v;
      if (cols[j] == row) continue;
      v    = A->hermitian ? PetscConj(vals[j]) : vals[j];
      ierr = MatSetValues(mat_ginkgo,1,&cols[j],1,&row,&v,ADD_VALUES);CHKERRQ(ierr);
    }
    ierr = MatRestoreRow(A,row,&ncols,&cols,&vals);CHKERRQ(ierr);
  }
  ierr = MatRestoreRowUpperTriangular(A);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(mat_ginkgo, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat_ginkgo, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  if (reuse == MAT_INPLACE_MATRIX) {
    ierr = MatHeaderReplace(A,&mat_ginkgo);CHKERRQ(ierr);
  } else {
    *newmat = mat_ginkgo;
  }
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatConvert_MPISBAIJ_Ginkgo(Mat A, MatType newtype,MatReuse reuse,Mat *newmat)
{
  Mat               mat_ginkgo;
  PetscErrorCode    ierr;
  PetscInt          M=A->rmap->N,N=A->cmap->N,row,ncols,j,rstart=A->rmap->rstart,rend=A->rmap->rend;
  const PetscInt    *cols;
  const PetscScalar *vals;

  PetscFunctionBegin;
  if (reuse == MAT_REUSE_MATRIX) {
    mat_ginkgo = *newmat;
    ierr = MatZeroEntries(mat_ginkgo);CHKERRQ(ierr);
  } else {
    ierr = MatCreate(PetscObjectComm((PetscObject)A), &mat_ginkgo);CHKERRQ(ierr);
    ierr = MatSetSizes(mat_ginkgo,PETSC_DECIDE,PETSC_DECIDE,M,N);CHKERRQ(ierr);
    ierr = MatSetType(mat_ginkgo,MATginkgo);CHKERRQ(ierr);
    ierr = MatSetUp(mat_ginkgo);CHKERRQ(ierr);
  }
  ierr = MatGetRowUpperTriangular(A);CHKERRQ(ierr);
  for (row=rstart; row<rend; row++) {
    ierr = MatGetRow(A,row,&ncols,&cols,&vals);CHKERRQ(ierr);
    /* PETSc-Ginkgo interface uses axpy for setting off-processor entries, only ADD_VALUES is allowed */
    ierr = MatSetValues(mat_ginkgo,1,&row,ncols,cols,vals,ADD_VALUES);CHKERRQ(ierr);
    for (j=0; j<ncols; j++) { /* lower triangular part */
      PetscScalar v;
      if (cols[j] == row) continue;
      v    = A->hermitian ? PetscConj(vals[j]) : vals[j];
      ierr = MatSetValues(mat_ginkgo,1,&cols[j],1,&row,&v,ADD_VALUES);CHKERRQ(ierr);
    }
    ierr = MatRestoreRow(A,row,&ncols,&cols,&vals);CHKERRQ(ierr);
  }
  ierr = MatRestoreRowUpperTriangular(A);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(mat_ginkgo, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat_ginkgo, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  if (reuse == MAT_INPLACE_MATRIX) {
    ierr = MatHeaderReplace(A,&mat_ginkgo);CHKERRQ(ierr);
  } else {
    *newmat = mat_ginkgo;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDestroy_Ginkgo(Mat A)
{
  Mat_Ginkgo      *a = (Mat_Ginkgo*)A->data;
  PetscErrorCode     ierr;
  Mat_Ginkgo_Grid *commgrid;
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

PetscErrorCode MatSetUp_Ginkgo(Mat A)
{
  Mat_Ginkgo  *a = (Mat_Ginkgo*)A->data;
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
  Mat_Ginkgo  *a = (Mat_Ginkgo*)A->data;

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

/*MC
   MATGINKGO = "ginkgo" - A matrix type for dense matrices using the Ginkgo package

  Use ./configure --download-ginkgo to install PETSc to use Ginkgo

  Use -pc_type lu -pc_factor_mat_solver_type ginkgo to use this direct solver

   Options Database Keys:
+ -mat_type ginkgo - sets the matrix type to "ginkgo" during a call to MatSetFromOptions()
- -mat_ginkgo_grid_height - sets Grid Height for 2D cyclic ordering of internal matrix

  Level: beginner

.seealso: MATDENSE
M*/

PETSC_EXTERN PetscErrorCode MatCreate_Ginkgo(Mat A)
{
  Mat_Ginkgo      *a;
  PetscErrorCode     ierr;
  PetscBool          flg,flg1;
  Mat_Ginkgo_Grid *commgrid;
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
}
