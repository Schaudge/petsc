#include <../src/ksp/ksp/utils/lmvm/diagbb/diagbb.h> /*I "petscksp.h" I*/

/*------------------------------------------------------------*/

static PetscErrorCode MatSolve_DiagBB(Mat B, Vec F, Vec dX)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_DiagBB        *lbb = (Mat_DiagBB*)lmvm->ctx;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  VecCheckSameSize(F, 2, dX, 3);
  VecCheckMatCompatible(B, dX, 3, F, 2);
  ierr = VecPointwiseMult(dX, lbb->invD, F);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatMult_DiagBB(Mat B, Vec X, Vec Z)
{
  Mat_LMVM        *lmvm = (Mat_LMVM*)B->data;
  Mat_DiagBB      *lbb  = (Mat_DiagBB*)lmvm->ctx;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  VecCheckSameSize(X, 2, Z, 3);
  VecCheckMatCompatible(B, X, 2, Z, 3);
  ierr = VecPointwiseDivide(Z, X, lbb->invD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatUpdate_DiagBB(Mat B, Vec X, Vec F)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_DiagBB        *lbb = (Mat_DiagBB*)lmvm->ctx;
  PetscErrorCode    ierr;
  PetscInt          i, nlocal;
  PetscReal         ysdot,skdot,ykdot,a_bb1,a_bb2;
  PetscScalar       *inarray, *outarray;

  PetscFunctionBegin;
  if (!lmvm->m) PetscFunctionReturn(0);
  if (lmvm->prev_set) {
    /* Compute the new (S = X - Xprev) and (Y = F - Fprev) vectors */
    ierr = VecAYPX(lmvm->Xprev, -1.0, X);CHKERRQ(ierr);
    ierr = VecAYPX(lmvm->Fprev, -1.0, F);CHKERRQ(ierr);
    /* Compute tolerance for accepting the update */
    ierr = VecDotBegin(lmvm->Xprev, lmvm->Fprev, &ysdot);CHKERRQ(ierr);
    ierr = VecDotBegin(lmvm->Xprev, lmvm->Xprev, &skdot);CHKERRQ(ierr);
    ierr = VecDotBegin(lmvm->Fprev, lmvm->Fprev, &ykdot);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->Xprev, lmvm->Fprev, &ysdot);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->Xprev, lmvm->Xprev, &skdot);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->Fprev, lmvm->Fprev, &ykdot);CHKERRQ(ierr);

    a_bb1 = ykdot/(2*skdot);
    a_bb2 = ykdot/ysdot;


    ierr = VecCopy(lbb->invD,lbb->invDnew);CHKERRQ(ierr);
    ierr = VecPointwiseMult(lbb->temp,lmvm->Xprev,lmvm->Fprev);CHKERRQ(ierr);
    ierr = VecAXPY(lbb->temp,lbb->mu,lbb->invD);CHKERRQ(ierr);
    ierr = VecPointwiseDivide(lbb->temp2,lbb->temp,lmvm->Fprev);CHKERRQ(ierr);
    ierr = VecPointwiseDivide(lbb->temp,lbb->temp2,lmvm->Fprev);CHKERRQ(ierr); // ((sk*yk) +mu/D)/ (yk*yk)
  
    ierr = VecGetArrayPair(lbb->temp, lbb->invDnew, &inarray, &outarray);CHKERRQ(ierr);
    ierr = VecGetLocalSize(lbb->invDnew, &nlocal);CHKERRQ(ierr);
  
    for (i=0; i<nlocal; i++) {
      if (inarray[i] < a_bb1) {
        outarray[i] = a_bb1;    
      } else if (inarray[i] > a_bb2) {
        outarray[i] = a_bb2;
      } else {
        outarray[i] = inarray[i];
      }
    }

    if (!lbb->lip) {
      for (i=0; i<nlocal; i++) {
        if (outarray[i] < lbb->lip) {
          outarray[i] = lbb->lip;      
        }
      }
    }
    ierr = VecRestoreArrayPair(lbb->temp, lbb->invDnew, &inarray, &outarray);CHKERRQ(ierr);
    /* End DiagBB update */
  }
  /* Save the solution and function to be used in the next update */
  ierr = VecCopy(X, lmvm->Xprev);CHKERRQ(ierr);
  ierr = VecCopy(F, lmvm->Fprev);CHKERRQ(ierr);
  lmvm->prev_set = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatCopy_DiagBB(Mat B, Mat M, MatStructure str)
{
  Mat_LMVM        *bdata = (Mat_LMVM*)B->data;
  Mat_DiagBB      *bctx  = (Mat_DiagBB*)bdata->ctx;
  Mat_LMVM        *mdata = (Mat_LMVM*)M->data;
  Mat_DiagBB      *mctx  = (Mat_DiagBB*)mdata->ctx;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  mctx->lip = bctx->lip;
  mctx->mu  = bctx->mu;
  mctx->tol = bctx->tol;
  ierr = VecCopy(bctx->invD, mctx->invD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatView_DiagBB(Mat B, PetscViewer pv)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_DiagBB        *lbb  = (Mat_DiagBB*)lmvm->ctx;
  PetscErrorCode    ierr;
  PetscBool         isascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)pv,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerASCIIPrintf(pv,"Lipschitz Constant =%g\n", (double)lbb->lip);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(pv,"Mu weight for Hessian =%g\n", (double)lbb->mu);CHKERRQ(ierr);
  }
  ierr = MatView_LMVM(B, pv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatSetFromOptions_DiagBB(PetscOptionItems *PetscOptionsObject, Mat B)
{
  Mat_LMVM        *lmvm = (Mat_LMVM*)B->data;
  Mat_DiagBB      *lbb = (Mat_DiagBB*)lmvm->ctx;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = MatSetFromOptions_LMVM(PetscOptionsObject, B);CHKERRQ(ierr);
  ierr = PetscOptionsHead(PetscOptionsObject,"Barzilai-Borwein method for approximating Diagonal SPD Hessian estimate (MATLMVMDIAGBB)");CHKERRQ(ierr);
  ierr = PetscOptionsReal("-mat_lmvm_lip","(developer) Lipschitz constant threshold","",lbb->lip,&lbb->lip,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-mat_lmvm_mu","(developer) mu weight for changing tendency of Hessian ","",lbb->mu,&lbb->mu,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-mat_lmvm_tol","(developer) tolerance for bounding rescaling denominator","",lbb->tol,&lbb->tol,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  if ((lbb->mu <= 0.0)) SETERRQ(PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_OUTOFRANGE, "mu weight for changing tendency cannot be negative.");
  if ((lbb->lip <= 0.0)) SETERRQ(PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_OUTOFRANGE, "Lipschitz constant cannot be negative.");
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatReset_DiagBB(Mat B, PetscBool destructive)
{
  Mat_LMVM        *lmvm = (Mat_LMVM*)B->data;
  Mat_DiagBB      *lbb  = (Mat_DiagBB*)lmvm->ctx;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = VecSet(lbb->invD, 0.);CHKERRQ(ierr);
  if (destructive && lbb->allocated) {
    ierr = VecDestroy(&lbb->invDnew);CHKERRQ(ierr);
    ierr = VecDestroy(&lbb->invD);CHKERRQ(ierr);
    ierr = VecDestroy(&lbb->temp);CHKERRQ(ierr);
    ierr = VecDestroy(&lbb->temp2);CHKERRQ(ierr);
    lbb->allocated = PETSC_FALSE;
  }
  ierr = MatReset_LMVM(B, destructive);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatAllocate_DiagBB(Mat B, Vec X, Vec F)
{
  Mat_LMVM        *lmvm = (Mat_LMVM*)B->data;
  Mat_DiagBB      *lbb  = (Mat_DiagBB*)lmvm->ctx;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = MatAllocate_LMVM(B, X, F);CHKERRQ(ierr);
  if (!lbb->allocated) {
    ierr = VecDuplicate(lmvm->Xprev, &lbb->invDnew);CHKERRQ(ierr);
    ierr = VecDuplicate(lmvm->Xprev, &lbb->invD);CHKERRQ(ierr);
    ierr = VecDuplicate(lmvm->Xprev, &lbb->temp);CHKERRQ(ierr);
    ierr = VecDuplicate(lmvm->Xprev, &lbb->temp2);CHKERRQ(ierr);
    lbb->allocated = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatDestroy_DiagBB(Mat B)
{
  Mat_LMVM        *lmvm = (Mat_LMVM*)B->data;
  Mat_DiagBB      *lbb  = (Mat_DiagBB*)lmvm->ctx;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  if (lbb->allocated) {
    ierr = VecDestroy(&lbb->invDnew);CHKERRQ(ierr);
    ierr = VecDestroy(&lbb->invD);CHKERRQ(ierr);
    ierr = VecDestroy(&lbb->temp);CHKERRQ(ierr);
    ierr = VecDestroy(&lbb->temp);CHKERRQ(ierr);
    lbb->allocated = PETSC_FALSE;
  }
  ierr = PetscFree(lmvm->ctx);CHKERRQ(ierr);
  ierr = MatDestroy_LMVM(B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatSetUp_DiagBB(Mat B)
{
  Mat_LMVM        *lmvm = (Mat_LMVM*)B->data;
  Mat_DiagBB      *lbb  = (Mat_DiagBB*)lmvm->ctx;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = MatSetUp_LMVM(B);CHKERRQ(ierr);
  if (!lbb->allocated) {
    ierr = VecDuplicate(lmvm->Xprev, &lbb->invDnew);CHKERRQ(ierr);
    ierr = VecDuplicate(lmvm->Xprev, &lbb->invD);CHKERRQ(ierr);
    ierr = VecDuplicate(lmvm->Xprev, &lbb->temp);CHKERRQ(ierr);
    ierr = VecDuplicate(lmvm->Xprev, &lbb->temp);CHKERRQ(ierr);
    ierr = VecDuplicate(lmvm->Xprev, &lbb->temp2);CHKERRQ(ierr);
    lbb->allocated = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PetscErrorCode MatCreate_LMVMDiagBB(Mat B)
{
  Mat_LMVM        *lmvm;
  Mat_DiagBB      *lbb;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = MatCreate_LMVM(B);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)B, MATLMVMDIAGBB);CHKERRQ(ierr);
  B->ops->setup = MatSetUp_DiagBB;
  B->ops->setfromoptions = MatSetFromOptions_DiagBB;
  B->ops->destroy = MatDestroy_DiagBB;
  B->ops->solve = MatSolve_DiagBB;
  B->ops->view = MatView_DiagBB;

  lmvm = (Mat_LMVM*)B->data;
  lmvm->square = PETSC_TRUE;
  lmvm->m = 1;
  lmvm->ops->allocate = MatAllocate_DiagBB;
  lmvm->ops->reset = MatReset_DiagBB;
  lmvm->ops->mult = MatMult_DiagBB;
  lmvm->ops->update = MatUpdate_DiagBB;
  lmvm->ops->copy = MatCopy_DiagBB;

  ierr = PetscNewLog(B, &lbb);CHKERRQ(ierr);
  lmvm->ctx = (void*)lbb;
  lbb->tol = 1e-8;
  lbb->mu  = 1000.;
  lbb->allocated = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

/*@
   MatCreateLMVMDiagBB - DiagBB creates a symmetric Barzilai-Borwein type diagonal matrix used
   for approximating Hessians. It consists of a convex combination of DFP and BFGS
   diagonal approximation schemes, such that DiagBB = (1-theta)*BFGS + theta*DFP.
   To preserve symmetric positive-definiteness, we restrict theta to be in [0, 1].
   We also ensure positive definiteness by taking the VecAbs() of the final vector.

   There are two ways of approximating the diagonal: using the forward (B) update
   schemes for BFGS and DFP and then taking the inverse, or directly working with
   the inverse (H) update schemes for the BFGS and DFP updates, derived using the
   Sherman-Morrison-Woodbury formula. We have implemented both, controlled by a
   parameter below.

   In order to use the DiagBB matrix with other vector types, i.e. doing MatMults
   and MatSolves, the matrix must first be created using MatCreate() and MatSetType(),
   followed by MatLMVMAllocate(). Then it will be available for updating
   (via MatLMVMUpdate) in one's favored solver implementation.
   This allows for MPI compatibility.

   Collective

   Input Parameters:
+  comm - MPI communicator, set to PETSC_COMM_SELF
.  n - number of local rows for storage vectors
-  N - global size of the storage vectors

   Output Parameter:
.  B - the matrix

   It is recommended that one use the MatCreate(), MatSetType() and/or MatSetFromOptions()
   paradigm instead of this routine directly.

   Options Database Keys:
+   -mat_lmvm_theta - (developer) convex ratio between BFGS and DFP components of the diagonal J0 scaling
.   -mat_lmvm_rho - (developer) update limiter for the J0 scaling
.   -mat_lmvm_alpha - (developer) coefficient factor for the quadratic subproblem in J0 scaling
.   -mat_lmvm_beta - (developer) exponential factor for the diagonal J0 scaling
.   -mat_lmvm_sigma_hist - (developer) number of past updates to use in J0 scaling.
.   -mat_lmvm_tol - (developer) tolerance for bounding the denominator of the rescaling away from 0.
-   -mat_lmvm_forward - (developer) whether or not to use the forward or backward Broyden update to the diagonal

   Level: intermediate

.seealso: MatCreate(), MATLMVM, MATLMVMDIAGBRDN, MatCreateLMVMDFP(), MatCreateLMVMSR1(),
          MatCreateLMVMBFGS(), MatCreateLMVMBB(), MatCreateLMVMSymBB()
@*/
PetscErrorCode MatCreateLMVMDiagBB(MPI_Comm comm, PetscInt n, PetscInt N, Mat *B)
{
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = MatCreate(comm, B);CHKERRQ(ierr);
  ierr = MatSetSizes(*B, n, n, N, N);CHKERRQ(ierr);
  ierr = MatSetType(*B, MATLMVMDIAGBB);CHKERRQ(ierr);
  ierr = MatSetUp(*B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
