#include <petsc/private/petscginkgo.hpp>

/* Add the STL map header for the executor selection */
#include <map>

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
  /* PetscGinkgoScalar *a = A->get_values(); */

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

PETSC_EXTERN PetscErrorCode MatCreate_GinkgoCSR(Mat A)
{
  Mat_GinkgoCSR      *a;
  PetscErrorCode     ierr;
  PetscBool          flg;
  PetscInt           gexec=0;
  const char         *execTypes[3] = {"omp","cuda","hip"};
  PetscInt           ngexec=3;
  PetscInt           rs,re,bs;


  PetscFunctionBegin;

  ierr = PetscNewLog(A,&a);CHKERRQ(ierr);
  A->data = (void*)a;

  /* Map petsc options to ginkgo setup */
  ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)A),((PetscObject)A)->prefix,"Ginkgo Options","Mat");CHKERRQ(ierr);
  ierr = PetscCitationsRegister(GinkgoCitation,&GinkgoCite);CHKERRQ(ierr);

  ierr = PetscOptionsEList("-mat_ginkgo_executor","Executor: OMP/Cuda/Hip","None",execTypes,ngexec,execTypes[gexec],&gexec,&flg);CHKERRQ(ierr);

  /* CPU vs. device handled using executor design pattern.  
   * Including everything even though our petsc arg is a subset */
  std::map<std::string, std::function<std::shared_ptr<gko::Executor>()>>
      exec_map{
          {"omp", [] { return gko::OmpExecutor::create(); }},
          {"cuda",
           [] {
               return gko::CudaExecutor::create(0, gko::OmpExecutor::create(),
                                                true);
           }},
          {"hip",
           [] {
               return gko::HipExecutor::create(0, gko::OmpExecutor::create(),
                                               true);
           }},
          {"dpcpp",
           [] {
               return gko::DpcppExecutor::create(0,
                                                 gko::OmpExecutor::create());
           }},
          {"reference", [] { return gko::ReferenceExecutor::create(); }}};

  /* the actual executor */
  const auto exectr = exec_map.at(std::string(execTypes[gexec]))();

  ierr = MatGetBlockSize(A,&bs);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(A->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(A->cmap);CHKERRQ(ierr);
  rs   = A->rmap->rstart;
  re   = A->rmap->rend;
  PetscGinkgoInt  nrows = re - rs;
  PetscGinkgoInt  nnz = nrows; /* SEK: Need to figure this out */

  ierr = PetscFPrintf(PETSC_COMM_SELF,stdout,"#--- nrows: %d \n\n",nrows);CHKERRQ(ierr);
  ierr = PetscFPrintf(PETSC_COMM_SELF,stdout,"#--- nrows: %d \n\n",nnz);CHKERRQ(ierr);


  /* Now create the ginkgoCSR matrix */
  a->A_csr        = gko::share(Csr::create(exectr, gko::dim<2>(nrows), nnz));

  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  ierr = PetscObjectChangeTypeName((PetscObject)A,MATGINKGOCSR);CHKERRQ(ierr);
  ierr = PetscFPrintf(PETSC_COMM_SELF,stdout,"#--- EEE created ---#\n\n");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatSolverTypeRegister_Ginkgo(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /*  SEK: Not clear whether we want to register the solver
ierr = MatSolverTypeRegister(MATSOLVERGINKGO,MATGINKGO,        MAT_FACTOR_CHOLESKY,MatGetFactor_Ginkgo_Ginkgo);CHKERRQ(ierr); */
  PetscFunctionReturn(0);
}



