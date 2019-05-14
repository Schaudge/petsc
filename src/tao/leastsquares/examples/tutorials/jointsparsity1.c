/* XH: todo add jointsparsity1.F90 and asjust makefile */
/* [petsc-users] Reshaping a vector into a matrix   https://lists.mcs.anl.gov/pipermail/petsc-users/2011-January/thread.html#7650
   You most definitely want to use the MAIJ.  MAIJ does not "repeat" X, it uses the original matrix passed in but does efficient multiple matrix-vector products at the same time.
   https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatCreateMAIJ.html
*/
/*
   Include "petsctao.h" so that we can use TAO solvers.  Note that this
   file automatically includes libraries such as:
     petsc.h       - base PETSc routines   petscvec.h - vectors
     petscsys.h    - sysem routines        petscmat.h - matrices
     petscis.h     - index sets            petscksp.h - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h  - preconditioners

*/

#include <petsctao.h>

/*
Description:   BRGN Joint-Sparsity reconstruction example 1.
Reference:     cs1.c, tomography.c
*/

#define BRGN_REGULARIZATION_L2  0
#define BRGN_REGULARIZATION_L1DICT  1
#define BRGN_REGULARIZATION_L1JOINT 2
#define BRGN_REGULARIZATION_USER  BRGN_REGULARIZATION_L1JOINT

#define DICT_CHOICE 0 /* DICT_CHOICE = 0/1/2/3, where 0:identity, 1:gradient1D, 2:gradient2D, 3:DCT etc. */
#define ALLOW_NULL_DICT PETSC_TRUE /* ALLOW_NULL_DICT = PETSC_TRUE/PETSC_FALSE, where PETSC_TRUE: user.D = NULL is equivalanent to NULL matrix. */


static char help[] = "Finds the least-squares solution to the under constraint linear model Ax = b, with L1-norm regularizer and multiple observations of same sparsity (joint-sparsity). \n\
            A is a M*N real matrix (M<N), x is sparse. In joint-sparsity, we have L observations: A*x^i = b^i for i=1, ... ,L. \n\
            We find the sparse solution by solving 0.5*sum_i{||Ax^i-b^i||^2} + lambda*sum_j{sum_i{((x^i)_j)^2}, where lambda (by default 1e-4) is a user specified weight.\n\
            In future, include a dictionary matrix if the sparsity is for transformed version y=Dx. D is the K*N transform matrix so that D*x is sparse. By default D is identity matrix, so that D*x = x.\n";
/*T
   Concepts: TAO^Solving a system of nonlinear equations, nonlinear least squares
   Routines: TaoCreate();
   Routines: TaoSetType();
   Routines: TaoSetSeparableObjectiveRoutine();
   Routines: TaoSetJacobianRoutine();
   Routines: TaoSetInitialVector();
   Routines: TaoSetFromOptions();
   Routines: TaoSetConvergenceHistory(); TaoGetConvergenceHistory();
   Routines: TaoSolve();
   Routines: TaoView(); TaoDestroy();
   Processors: 1
T*/

/* User-defined application context */
typedef struct {
  /* Working space. linear least square:  res(x) = A*x - b = reshape(Asub*reshape(x,N,L) - reshape(b,M,L), M*L,1) */
  PetscInt  M,N,K,L;          /* Problem dimension: A is ML*NL Matrix, D is KL*NL Matrix, L is the number of joint measurements*/
  Mat       A,D,Asub,Dsub,Hreg; /* Asub,Dsub: Coefficients, Dictionary Transform of size M*N and K*N respectively. For linear least square, Jacobian Matrix J = A. */
                               /* A,D are the block matrix where the block diagonal sub-matrices are Asub,Dsub repeated by L times*/
                               /* Hreg: regularizer Hessian matrix for user specified regularizer */
  Vec       b,xGT,xlb,xub;    /* observation b of M*L size, ground truth xGT, the lower bound and upper bound of x of N*L size*/
  /* Working scalars and vectors, borrowed from struct TAO_BRGN*/
  PetscReal epsilon; /* ||x||_1 is approximated with sum(sqrt(x.^2+epsilon^2)-epsilon)*/  
  Vec       x_old,x_work,r_work,diag,y,y_work;  /* x, r=J*x, and y=D*x have size N*L, M*L, and K*L respectively. */
  /* Working index sets, useful for joint sparsity to extract corresponding rows/cols of a matrix from the vectorized version of the matrix */  
  IS       *idxRowsX,*idxColsX,*idxColsB,*idxRowsY,*idxColsY; /* array of IS to extract rows and cols of X/B/Y from its column vectorized x/b/y, the array has size N,L,L,L; Note for b we only need idxCols*/
  /* Working vector only for joint sparsity */
  Vec       z,v1,v2,v3;  /* working vectors, z is K dimension vector for obj/gradient/hessian. v1, v2, v3 are KL dimention vectors precomputed for Hessian product. */
} AppCtx;

/* User provided Routines */
PetscErrorCode InitializeUserData(AppCtx *);
PetscErrorCode DestroyUserData(AppCtx *);
PetscErrorCode FormStartingPoint(Vec,AppCtx *);
PetscErrorCode EvaluateResidual(Tao,Vec,Vec,void *);
PetscErrorCode EvaluateJacobian(Tao,Vec,Mat,Mat,void *);
PetscErrorCode EvaluateRegularizerObjectiveAndGradient(Tao,Vec,PetscReal *,Vec,void*);
PetscErrorCode EvaluateRegularizerHessian(Tao,Vec,Mat,void*);
PetscErrorCode EvaluateRegularizerHessianProd(Mat,Vec,Vec);

/*--------------------------------------------------------------------*/
int main(int argc,char **argv)
{
  PetscErrorCode ierr;               /* used to check for functions returning nonzeros */
  Vec            x,res;              /* solution, function res(x) = A*x-b */
  Tao            tao;                /* Tao solver context */
  PetscReal      hist[100],resid[100],s1,s2;
  PetscInt       lits[100];
  AppCtx         user;               /* user-defined work context */
  PetscViewer    fd;   /* used to save result to file */
  char           resultFile[] = "jointsparsityResult_x";  /* save result x as a file */

  ierr = PetscInitialize(&argc,&argv,(char *)0,help);if (ierr) return ierr;

  /* Create TAO solver and set desired solution method */
  ierr = TaoCreate(PETSC_COMM_SELF,&tao);CHKERRQ(ierr);
  ierr = TaoSetType(tao,TAOBRGN);CHKERRQ(ierr);

  /* User set work context, where we specify the */
  ierr = InitializeUserData(&user);CHKERRQ(ierr);
  /********* User set regularizer function objective, gradient, and heassian ******/
  /* (0) Get espilon parameter from command line */
  ierr = PetscOptionsBegin(PETSC_COMM_SELF, "", "Need Extract Opitions to get epsilon", "");CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_brgn_l1_smooth_epsilon","L1-norm smooth approximation parameter: ||x||_1 = sum(sqrt(x.^2+epsilon^2)-epsilon) (default 1e-6)","",user.epsilon,&user.epsilon,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"user.epsilon = %e. -------- \n",user.epsilon);CHKERRQ(ierr);
  /* (1) User set the regularizer objective and gradient */
  ierr = TaoBRGNSetRegularizerObjectiveAndGradientRoutine(tao,EvaluateRegularizerObjectiveAndGradient,(void*)&user);CHKERRQ(ierr); /* TODO for joint sparsity */  
  /* (2) User set the Hessian, as a shell matrix with MatMult Operations*/  /* TODO for joint sparsity */      
  ierr = MatCreateShell(PETSC_COMM_SELF,PETSC_DECIDE,PETSC_DECIDE,user.N*user.L,user.N*user.L,(void*)&user,&user.Hreg);CHKERRQ(ierr);
  ierr = MatShellSetOperation(user.Hreg,MATOP_MULT,(void (*)(void))EvaluateRegularizerHessianProd);CHKERRQ(ierr);
  ierr = TaoBRGNSetRegularizerHessianRoutine(tao,user.Hreg,EvaluateRegularizerHessian,(void*)&user);CHKERRQ(ierr);

  /* Allocate solution vector x, and residual vectors Ax-b.*/  
  ierr = VecDuplicate(user.xGT,&x);CHKERRQ(ierr); 
  ierr = VecDuplicate(user.b,&res);CHKERRQ(ierr);

  /* Set initial guess */
  ierr = FormStartingPoint(x,&user);CHKERRQ(ierr);
  /* Bind x to tao->solution. */
  ierr = TaoSetInitialVector(tao,x);CHKERRQ(ierr);
  /* Sets the upper and lower bounds of x for tao */
  ierr = TaoSetVariableBounds(tao,user.xlb,user.xub);CHKERRQ(ierr);
  
  /* Bind user.D to tao->data->D. This is only used for "-tao_brgn_regularization_type l1dict" */
  ierr = TaoBRGNSetDictionaryMatrix(tao,user.D);CHKERRQ(ierr);

  /* Set the residual function and Jacobian routines for least squares. */
  ierr = TaoSetResidualRoutine(tao,res,EvaluateResidual,(void*)&user);CHKERRQ(ierr); /* TODO for joint sparsity */
  /* Jacobian matrix fixed as user.A (block matrix from user.A) for Linear least square problem (joint sparsity problem). */
  ierr = TaoSetJacobianResidualRoutine(tao,user.A,user.A,EvaluateJacobian,(void*)&user);CHKERRQ(ierr); /* TODO for joint sparsity */

  /* Check for any TAO command line arguments */
  ierr = TaoSetFromOptions(tao);CHKERRQ(ierr);  
  ierr = TaoSetConvergenceHistory(tao,hist,resid,NULL,lits,100,PETSC_TRUE);CHKERRQ(ierr);

  /* Perform the Solve */
  ierr = TaoSolve(tao);CHKERRQ(ierr);

  /* Save x (reconstruction of object) vector to a binary file, which maybe read from Matlab and convert to a 2D image for comparison. */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_SELF,resultFile,FILE_MODE_WRITE,&fd);CHKERRQ(ierr);
  ierr = VecView(x,fd);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);

    /* XH: Debug: View the result, function and Jacobian.  */
/*    
  ierr = PetscPrintf(PETSC_COMM_SELF, "-------- result x, residual res =A*x-b. -------- \n");CHKERRQ(ierr);
  ierr = VecView(x,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  ierr = VecView(res,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  */
    /* compute the error */
  ierr = VecAXPY(x,-1,user.xGT);CHKERRQ(ierr);
  ierr = VecNorm(x,NORM_2,&s1);CHKERRQ(ierr);
  ierr = VecNorm(user.xGT,NORM_2,&s2);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF, "relative reconstruction error: ||x-xGT||/||xGT|| = %6.4e.\n", (double)(s1/s2));CHKERRQ(ierr);


  /* Free TAO data structures */
  ierr = TaoDestroy(&tao);CHKERRQ(ierr);

   /* Free PETSc data structures declared in main funtion */
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&res);CHKERRQ(ierr);
  /* Free user data structures */
  ierr = DestroyUserData(&user);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/* ---------------------------------------------------------------------- */
PetscErrorCode BlockMatrixOperation(Mat Asub,Vec in,Vec out,IS *idxColsIN,IS *idxColsOUT,PetscInt L,PetscErrorCode (*fun_ptr)())
{
  /* Block matrix computation, includes two cases specified by *fun_ptr() = MatMult()/MatMultTranspose()
     out = diag(Asub,...,Asub)*in  = A*in,   If we use Matlab, out = rehape(Asub*IN, [], 1), where IN = reshape(in, [], L)
     out = diag(Asub,...,Asub)'*in = A'*in.      
     L is size of blocks, 
     idxColsIN[l] represent the lth subvector of in, which is lth column of matrix IN.
     idxColsOUT[l] represent the lth subvector of out.            
  */  
  PetscErrorCode ierr;
  Vec            inSub,outSub;
  PetscInt       l;
  
  PetscFunctionBegin; 
  
  if (Asub) {
    for (l=0;l<L;l++) {
      ierr = VecGetSubVector(in,idxColsIN[l],&inSub);CHKERRQ(ierr);
      ierr = VecGetSubVector(out,idxColsOUT[l],&outSub);CHKERRQ(ierr);
      ierr = (*fun_ptr)(Asub,inSub,outSub);CHKERRQ(ierr);
      ierr = VecRestoreSubVector(in,idxColsIN[l],&inSub);CHKERRQ(ierr);
      ierr = VecRestoreSubVector(out,idxColsOUT[l],&outSub);CHKERRQ(ierr);
    }
  } else {
    /* We treat NULL matrix as identity matrix here as shortcut to specify identity matrix multiplication */
    ierr = VecCopy(in,out);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


/* ---------------------------------------------------------------------- */
PetscErrorCode BlockMatrixMultiplyA(Mat A,Vec in,Vec out)
{
  /* b = A*in = diag(Asub,...,Asub)*x = reshape(Asub*reshape(x,[],L),M*L,1) */
  AppCtx         *user;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(A,&user);CHKERRQ(ierr);
  ierr = BlockMatrixOperation(user->Asub,in,out,user->idxColsX,user->idxColsB,user->L,&MatMult);CHKERRQ(ierr);
  PetscFunctionReturn(0);  
}

/* ---------------------------------------------------------------------- */
PetscErrorCode BlockMatrixMultiplyTransposeA(Mat A,Vec in,Vec out)
{
  /* x = A'*in = diag(Asub',...,Asub')*b = reshape(Asub'*reshape(b,[],L),N*L,1) */
  AppCtx         *user;
  PetscErrorCode ierr;

  PetscFunctionBegin;  
  ierr = MatShellGetContext(A,&user);CHKERRQ(ierr);
  ierr = BlockMatrixOperation(user->Asub,in,out,user->idxColsB,user->idxColsX,user->L,&MatMultTranspose);CHKERRQ(ierr);
  PetscFunctionReturn(0);  
}

/* ---------------------------------------------------------------------- */
PetscErrorCode BlockMatrixMultiplyD(Mat D,Vec in,Vec out)
{
  /* y = D*in = diag(Dsub,...,Dsub)*x = reshape(Dsub*reshape(x,[],L),K*L,1) */
  AppCtx         *user;
  PetscErrorCode ierr;
  
  PetscFunctionBegin; 
  ierr = MatShellGetContext(D,&user);CHKERRQ(ierr);
  ierr = BlockMatrixOperation(user->Dsub,in,out,user->idxColsX,user->idxColsY,user->L,&MatMult);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------- */
PetscErrorCode BlockMatrixMultiplyTransposeD(Mat D,Vec in,Vec out)
{
  /* x = D'*in = diag(Dsub',...,Dsub')*y = reshape(Dsub'*reshape(y,[],L),N*L,1) */
  AppCtx         *user;
  PetscErrorCode ierr;

  PetscFunctionBegin;  
  ierr = MatShellGetContext(D,&user);CHKERRQ(ierr);
  ierr = BlockMatrixOperation(user->Dsub,in,out,user->idxColsY,user->idxColsX,user->L,&MatMultTranspose);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------- */
PetscErrorCode InitializeUserData(AppCtx *user)
{
  PetscInt       n,k,l; /* (m,n), (k,n) and (n,l) indices for row and columns of Asub, D and X */
  /* make jointsparsity1; ./jointsparsity1 -tao_monitor -tao_max_it 10 -tao_brgn_regularization_type l1dict -tao_brgn_regularizer_weight 1e-8 -tao_brgn_l1_smooth_epsilon 1e-6 -tao_gatol 1.e-8
     relative reconstruction error: ||x-xGT||/||xGT|| = 4.1366e-01/2.3728e-01 for DICT_CHOICE 0/1. */  
  char           dataFile[] = "jointsparsity1Data_A_b_xGT_L";   /* Matrix A and vectors b, xGT(ground truth) binary files generated by Matlab. e.g., "tomographyData_A_b_xGT", "cs1Data_A_b_xGT", "jointsparsity1Data_A_b_xGT_L". */
  PetscViewer    fd;   /* used to load data from file */
  PetscErrorCode ierr;
  PetscReal      v;
  Vec            dummyVec; 

  PetscFunctionBegin;
  /*
  Matrix Vector read and write refer to:
  https://www.mcs.anl.gov/petsc/petsc-current/src/mat/examples/tutorials/ex10.c
  https://www.mcs.anl.gov/petsc/petsc-current/src/mat/examples/tutorials/ex12.c
 */
  /* Load the A matrix, b vector, and xGT vector from a binary file. */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_SELF,dataFile,FILE_MODE_READ,&fd);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_SELF,&user->Asub);CHKERRQ(ierr);
  ierr = MatSetType(user->Asub,MATSEQAIJ);CHKERRQ(ierr);
  ierr = MatLoad(user->Asub,fd);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_SELF,&user->b);CHKERRQ(ierr);
  ierr = VecLoad(user->b,fd);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_SELF,&user->xGT);CHKERRQ(ierr);
  ierr = VecLoad(user->xGT,fd);CHKERRQ(ierr);
  /* user->L is a scalar, but we have to first load it as a vector, then convert to a scalar, because data generated using matlab PetscBinaryWrite() only write Mat and Vec*/
  ierr = VecCreate(PETSC_COMM_SELF,&dummyVec);CHKERRQ(ierr); /*temporarily use user->xlb*/
  ierr = VecLoad(dummyVec,fd);CHKERRQ(ierr);
  k = 0;
  ierr = VecGetValues(dummyVec,1,&k,&v);CHKERRQ(ierr); 
  user->L = v;
  ierr = VecDestroy(&dummyVec);CHKERRQ(ierr);

  /* Template of setting the bounds of x. We allow -inf < x < +inf. But User may specify the bounds, e.g. x >= 0 */
  ierr = VecDuplicate(user->xGT,&(user->xlb));CHKERRQ(ierr);
  ierr = VecSet(user->xlb,PETSC_NINFINITY);CHKERRQ(ierr); /* xlb = PETSC_NINFINITY/0.0, where 0.0 generate more accurate result if x>=0.0 is really true*/
  ierr = VecDuplicate(user->xGT,&(user->xub));CHKERRQ(ierr);
  ierr = VecSet(user->xub,PETSC_INFINITY);CHKERRQ(ierr);

  /* Specify the size */
  ierr = MatGetSize(user->Asub,&user->M,&user->N);CHKERRQ(ierr);

  /******* Speficy D *******/
  /* (1) Specify D Size */
  switch (DICT_CHOICE) {
    case 0: /* 0:identity */
      user->K = user->N;
      break;
    case 1: /* 1:gradient1D */
      user->K = user->N-1;
      break;
  }

  /* (2) Specify D Content */
  if (DICT_CHOICE==0 && ALLOW_NULL_DICT) {
    user->Dsub = NULL;  /* shortcut, when DICT_CHOICE == 0, D is identity matrix, we may just specify it as NULL, and brgn will treat D*x as x without actually computing D*x */    
  } else {
    ierr = MatCreate(PETSC_COMM_SELF,&user->Dsub);CHKERRQ(ierr);
    ierr = MatSetSizes(user->Dsub,PETSC_DECIDE,PETSC_DECIDE,user->K,user->N);CHKERRQ(ierr);
    ierr = MatSetFromOptions(user->Dsub);CHKERRQ(ierr);
    ierr = MatSetUp(user->Dsub);CHKERRQ(ierr);
    switch (DICT_CHOICE) {
      case 0: /* 0:identity */
        /* Old way to actually set up a indentity matrix. */
        for (k=0; k<user->K; k++) {
          v = 1.0;
          ierr = MatSetValues(user->Dsub,1,&k,1,&k,&v,INSERT_VALUES);CHKERRQ(ierr);
        }      
        break;
      case 1: /* 1:gradient1D.  [-1, 1, 0,...; 0, -1, 1, 0, ...] */
        for (k=0; k<user->K; k++) {
          v = 1.0;
          n = k+1;
          ierr = MatSetValues(user->Dsub,1,&k,1,&n,&v,INSERT_VALUES);CHKERRQ(ierr);
          v = -1.0;
          ierr = MatSetValues(user->Dsub,1,&k,1,&k,&v,INSERT_VALUES);CHKERRQ(ierr);
        }
        break;
    }
    ierr = MatAssemblyBegin(user->Dsub,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(user->Dsub,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }

  /* Generate A,D by blocking of Asub,Dsub: Set the A, D Shell Matrix with MatMult(), MatMultTranspose() Operation*/  
  /* in future, may get localsize from VecGetLocalSize(gn->solution,&n); */
  ierr = MatCreateShell(PETSC_COMM_SELF,PETSC_DECIDE,PETSC_DECIDE,user->M*user->L,user->N*user->L,(void*)user,&user->A);CHKERRQ(ierr);
  ierr = MatShellSetOperation(user->A,MATOP_MULT,(void (*)(void))BlockMatrixMultiplyA);CHKERRQ(ierr);
  ierr = MatShellSetOperation(user->A,MATOP_MULT_TRANSPOSE,(void (*)(void))BlockMatrixMultiplyTransposeA);CHKERRQ(ierr);

  ierr = MatCreateShell(PETSC_COMM_SELF,PETSC_DECIDE,PETSC_DECIDE,user->K*user->L,user->N*user->L,(void*)user,&user->D);CHKERRQ(ierr);
  ierr = MatShellSetOperation(user->D,MATOP_MULT,(void (*)(void))BlockMatrixMultiplyD);CHKERRQ(ierr);
  ierr = MatShellSetOperation(user->D,MATOP_MULT_TRANSPOSE,(void (*)(void))BlockMatrixMultiplyTransposeD);CHKERRQ(ierr);

  /*************************Setup working vectors: Vec x_old,x_work,r_work,diag,y,y_work; Refer to TaoSetFromOptions_BRGN() ***************************/
  /* Vectors of size N*L */
  ierr = VecDuplicate(user->xGT,&user->x_old);CHKERRQ(ierr);
  ierr = VecDuplicate(user->xGT,&user->x_work);CHKERRQ(ierr);

  /* Vectors of size M*L */
  ierr = VecDuplicate(user->b,&user->r_work);CHKERRQ(ierr);
  
  /* Vectors of size K*L */
  ierr = VecCreate(PETSC_COMM_SELF,&user->y);CHKERRQ(ierr);
  ierr = VecSetSizes(user->y,PETSC_DECIDE,user->K*user->L);CHKERRQ(ierr);
  ierr = VecSetFromOptions(user->y);CHKERRQ(ierr);
  ierr = VecDuplicate(user->y,&user->y_work);CHKERRQ(ierr);
  ierr = VecDuplicate(user->y,&user->diag);CHKERRQ(ierr);
  ierr = VecDuplicate(user->y,&user->v1);CHKERRQ(ierr);
  ierr = VecDuplicate(user->y,&user->v2);CHKERRQ(ierr);
  ierr = VecDuplicate(user->y,&user->v3);CHKERRQ(ierr);

  /* Vectors of size K */
  ierr = VecCreate(PETSC_COMM_SELF,&user->z);CHKERRQ(ierr);
  ierr = VecSetSizes(user->z,PETSC_DECIDE,user->K);CHKERRQ(ierr);
  ierr = VecSetFromOptions(user->z);CHKERRQ(ierr);


  /*************************Precompute row/col indices to exact rows/cols from a vectorized version x,b,y of matrices X,B,D*X ***************************/  
  ierr = PetscMalloc1(user->N,&user->idxRowsX);CHKERRQ(ierr);
  ierr = PetscMalloc1(user->K,&user->idxRowsY);CHKERRQ(ierr);

  ierr = PetscMalloc1(user->L,&user->idxColsX);CHKERRQ(ierr);
  ierr = PetscMalloc1(user->L,&user->idxColsB);CHKERRQ(ierr);
  ierr = PetscMalloc1(user->L,&user->idxColsY);CHKERRQ(ierr);
  
  /* Indices of rows. e.g., first row of X in x is 0:N:(N*L-1) which has L elements,  where X is an N*L matrix and x=X(:) is X vectorized along column */
  for (n=0;n<user->N;++n) {
    ierr = ISCreateStride(PETSC_COMM_SELF,user->L,n,user->N,&user->idxRowsX[n]);CHKERRQ(ierr);  /* len=L,first=n,step=N, the first element in nth row is n */  
  }
  for (n=0;n<user->K;++n) {
    ierr = ISCreateStride(PETSC_COMM_SELF,user->L,n,user->K,&user->idxRowsY[n]);CHKERRQ(ierr);  /* len=L,first=n,step=K, the first element in nth row is n */  
  }  

  /* Indices of cols. e.g., first col of X in x is 0:1:(N-1) which has N elements,  where X is an N*L matrix and x=X(:) is X vectorized along column */  
  for (l=0;l<user->L;++l) {
    ierr = ISCreateStride(PETSC_COMM_SELF,user->N,l*user->N,1,&user->idxColsX[l]);CHKERRQ(ierr);  /* len=N,first=l*N,step=1, the first element in lth col is l*N */
  }
  for (l=0;l<user->L;++l) {
    ierr = ISCreateStride(PETSC_COMM_SELF,user->M,l*user->M,1,&user->idxColsB[l]);CHKERRQ(ierr);  /* len=M,first=l*M,step=1, the first element in lth col is l*M */
  }
  for (l=0;l<user->L;++l) {
    ierr = ISCreateStride(PETSC_COMM_SELF,user->K,l*user->K,1,&user->idxColsY[l]);CHKERRQ(ierr);  /* len=K,first=l*K,step=1, the first element in lth col is l*K */
  }  
  
#if 0  
  PetscInt       len,first,step;
  ierr = ISView(user->idxRowsX[0],PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  ierr = ISView(user->idxColsX[user->L-1],PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  ierr = ISView(user->idxColsB[user->L-1],PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);

  /* Determine information on stride */
  ierr = ISStrideGetInfo(user->idxRowsX[user->N-1],&first,&step);CHKERRQ(ierr);
  ierr = ISGetSize(user->idxRowsX[user->N-1],&len);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"user->idxRowsX[N-1]: first: %d, step:%d, len:%d\n", first, step, len);CHKERRQ(ierr);

  ierr = ISStrideGetInfo(user->idxColsX[user->L-1],&first,&step);CHKERRQ(ierr);
  ierr = ISGetSize(user->idxColsX[user->L-1],&len);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"user->idxColsX[L-1]: first: %d, step:%d, len:%d\n", first, step, len);CHKERRQ(ierr);
#endif

  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------- */
PetscErrorCode DestroyUserData(AppCtx *user)
{  
  PetscErrorCode ierr; 
  PetscInt       n,l;

  PetscFunctionBegin;  

  ierr = MatDestroy(&user->Asub);CHKERRQ(ierr);
  ierr = MatDestroy(&user->Dsub);CHKERRQ(ierr);
  ierr = MatDestroy(&user->A);CHKERRQ(ierr);
  ierr = MatDestroy(&user->D);CHKERRQ(ierr);
  ierr = MatDestroy(&user->Hreg);CHKERRQ(ierr);
  ierr = VecDestroy(&user->b);CHKERRQ(ierr);
  ierr = VecDestroy(&user->xGT);CHKERRQ(ierr);
  ierr = VecDestroy(&user->xlb);CHKERRQ(ierr);
  ierr = VecDestroy(&user->xub);CHKERRQ(ierr);
  
  ierr = VecDestroy(&user->x_old);CHKERRQ(ierr);
  ierr = VecDestroy(&user->x_work);CHKERRQ(ierr);
  ierr = VecDestroy(&user->r_work);CHKERRQ(ierr);
  ierr = VecDestroy(&user->diag);CHKERRQ(ierr);
  ierr = VecDestroy(&user->y);CHKERRQ(ierr);
  ierr = VecDestroy(&user->y_work);CHKERRQ(ierr);
  ierr = VecDestroy(&user->v1);CHKERRQ(ierr);
  ierr = VecDestroy(&user->v2);CHKERRQ(ierr);
  ierr = VecDestroy(&user->v3);CHKERRQ(ierr);

  ierr = VecDestroy(&user->z);CHKERRQ(ierr);

  /* Destroy array of index sets */
  for (n=0;n<user->N;++n) {
    ierr = ISDestroy(&user->idxRowsX[n]);CHKERRQ(ierr);
  }
  for (n=0;n<user->K;++n) {
    ierr = ISDestroy(&user->idxRowsY[n]);CHKERRQ(ierr);
  }  
  for (l=0;l<user->L;++l) {
    ierr = ISDestroy(&user->idxColsX[l]);CHKERRQ(ierr);
    ierr = ISDestroy(&user->idxColsB[l]);CHKERRQ(ierr);
    ierr = ISDestroy(&user->idxColsY[l]);CHKERRQ(ierr);
  }
  /* Free the arrays theirselves */
  ierr = PetscFree(user->idxRowsX);CHKERRQ(ierr);
  ierr = PetscFree(user->idxRowsY);CHKERRQ(ierr);
  ierr = PetscFree(user->idxColsX);CHKERRQ(ierr);
  ierr = PetscFree(user->idxColsB);CHKERRQ(ierr);
  ierr = PetscFree(user->idxColsY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}  

/* ------------------------------------------------------------ */
PetscErrorCode FormStartingPoint(Vec X,AppCtx *user)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = VecSet(X,0.0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*--------------------------------------------------------------------*/
/* Evaluate residual function F = A(x)-b in least square problem ||A(x)-b||^2 */
PetscErrorCode EvaluateResidual(Tao tao,Vec X,Vec F,void *ctx)
{
  AppCtx         *user = (AppCtx *)ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Compute Ax - b */
  ierr = MatMult(user->A,X,F);CHKERRQ(ierr);   /* F = Ax */
  ierr = VecAXPY(F,-1,user->b);CHKERRQ(ierr);  /* F = -b + F = -b + Ax */
  PetscLogFlops(user->M*user->N*user->L*2);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
PetscErrorCode EvaluateJacobian(Tao tao,Vec X,Mat J,Mat Jpre,void *ctx)
{  
  /* Jacobian is not changing here, so use a empty dummy function here.  J[m][n] = df[m]/dx[n] = A[m][n] for linear least square */
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------ */
PetscErrorCode EvaluateRegularizerObjectiveAndGradient(Tao tao,Vec X,PetscReal *f_reg,Vec G_reg,void *ctx)
{
  /* Compute the regularizer objective function value as *f_reg, and its gradient as G_reg */
  AppCtx         *user = (AppCtx *)ctx;
  PetscInt       sz,k,l;    /* sz = dimension of D*X, k,l loop indices for rows and cols of Y matrix */
  PetscScalar    vecSum;
  PetscErrorCode ierr;  
  PetscReal      *zArray; /* only used for joint sparsity */
  Vec            subVec; /* only used for joint sparsity */

  PetscFunctionBegin;

  switch (BRGN_REGULARIZATION_USER) {
  case BRGN_REGULARIZATION_L2:
    /************ Regularizer: 0.5*x'*x ******************/    
    ierr = VecDot(X,X,f_reg);CHKERRQ(ierr);
    *f_reg *= 0.5;
    /* compute regularizer gradient = x */
    ierr = VecCopy(X,G_reg);CHKERRQ(ierr);
    break;
  case BRGN_REGULARIZATION_L1DICT:
    /************ Regularizer: L1 Dict ******************/
    /* compute regularizer objective *f_reg = sum(sqrt(y.^2+epsilon^2) - epsilon), where y = D*x*/
    if (user->D) {
      ierr = MatMult(user->D,X,user->y);CHKERRQ(ierr);/* y = D*x */
    } else {    
      ierr = VecCopy(X,user->y);CHKERRQ(ierr);
    }
    ierr = VecPointwiseMult(user->y_work,user->y,user->y);CHKERRQ(ierr);
    ierr = VecShift(user->y_work,user->epsilon*user->epsilon);CHKERRQ(ierr);
    ierr = VecSqrtAbs(user->y_work);CHKERRQ(ierr);  /* y_work = sqrt(y.^2+epsilon^2) */ 
    ierr = VecSum(user->y_work,&vecSum);CHKERRQ(ierr);CHKERRQ(ierr);
    ierr = VecGetSize(user->y,&sz);CHKERRQ(ierr);
    *f_reg = (vecSum - sz*user->epsilon);
    
    /* compute regularizer gradient G_reg = D'*(y./sqrt(y.^2+epsilon^2)),where y = D*x */  
    ierr = VecPointwiseDivide(user->y,user->y,user->y_work);CHKERRQ(ierr); /* reuse already computed y = D*x, y_work = sqrt(y.^2+epsilon^2) */
    if (user->D) {
      ierr = MatMultTranspose(user->D,user->y,G_reg);CHKERRQ(ierr);
    } else {
      ierr = VecCopy(user->y,G_reg);CHKERRQ(ierr);
    }
    break;
  case BRGN_REGULARIZATION_L1JOINT:
    /************ Regularizer: joint-L1-sparsity with Dict ******************/
    /* compute regularizer objective *f_reg = sum(sqrt(z.^2+epsilon^2) - epsilon), where z = ||Y||_{2,1},  Y = Dsub*X matrix or y=D*x vector */
    /* Compute y = D*x */
    if (user->D) {
      ierr = MatMult(user->D,X,user->y);CHKERRQ(ierr);/* y = D*x */
    } else {    
      ierr = VecCopy(X,user->y);CHKERRQ(ierr);
    }

    /* Compute z = sum(reshape(y_work, [K,L]), 2) = sum(reshape((D*x).^2, [K,L]), 2) which is the row sum */
    ierr = VecPointwiseMult(user->y_work,user->y,user->y);CHKERRQ(ierr); /* y_work = y.^2 = (D*x).^2 */
    ierr = VecGetArray(user->z,&zArray);CHKERRQ(ierr);
    for (k=0;k<user->K;++k) {        
      ierr = VecGetSubVector(user->y_work,user->idxRowsY[k],&subVec);CHKERRQ(ierr);
      ierr = VecSum(subVec,&zArray[k]);CHKERRQ(ierr);
      ierr = VecRestoreSubVector(user->y_work,user->idxRowsY[k],&subVec);CHKERRQ(ierr);    
    }
    ierr = VecRestoreArray(user->z,&zArray);CHKERRQ(ierr);    
    /* Compute z = sqrt(z + epsilon^2) = sqrt(||Dx||_{2,1}^2 + epsilon^2) */
    ierr = VecShift(user->z,user->epsilon*user->epsilon);CHKERRQ(ierr);
    ierr = VecSqrtAbs(user->z);CHKERRQ(ierr);  /* z = sqrt(||Dx||_{2,1}^2 + epsilon^2) */ 

    ierr = VecSum(user->z,&vecSum);CHKERRQ(ierr);CHKERRQ(ierr);
    ierr = VecGetSize(user->z,&sz);CHKERRQ(ierr); /* sz = user->K, or just use user->K */
    *f_reg = (vecSum - sz*user->epsilon);
    
    /* compute regularizer gradient G_reg = D'*(y./zExtend)),where y = D*x (already computed as user->y), zExtend = repmat(z, [L,1]), z = sqrt(||Dx||_{2,1}^2 + epsilon^2)*/
    /* Compute y = y./repmat(z, [L,1]) */
    for (l=0;l<user->L;++l) {        
      ierr = VecGetSubVector(user->y,user->idxColsY[l],&subVec);CHKERRQ(ierr);
      ierr = VecPointwiseDivide(subVec,subVec,user->z);CHKERRQ(ierr);
      ierr = VecRestoreSubVector(user->y,user->idxColsY[l],&subVec);CHKERRQ(ierr);
    }        
    if (user->D) {
      ierr = MatMultTranspose(user->D,user->y,G_reg);CHKERRQ(ierr);
    } else {
      ierr = VecCopy(user->y,G_reg);CHKERRQ(ierr);
    }
    break;
  }

  PetscFunctionReturn(0);
}

PetscErrorCode EvaluateRegularizerHessianProd(Mat Hreg,Vec in,Vec out)
{
  AppCtx         *user;
  PetscErrorCode ierr;
  PetscReal      *zArray; /* only used for joint sparsity */
  Vec            subVec;  /* only used for joint sparsity */
  PetscInt       k,l;    /* k,l loop indices for rows and cols of Y matrix */

  PetscFunctionBegin;
  switch (BRGN_REGULARIZATION_USER) {
  case BRGN_REGULARIZATION_L2:
    /************ Regularizer: 0.5*x'*x, Hessian is just an identity matrix which doesn't change ******************/    
    ierr = VecCopy(in,out);CHKERRQ(ierr);
    break;
  case BRGN_REGULARIZATION_L1DICT:
    /************ Regularizer: L1 Dict ******************/
    /* out = D'*(diag.*(D*in)) */
    ierr = MatShellGetContext(Hreg,&user);CHKERRQ(ierr);
    if (user->D) {
      ierr = MatMult(user->D,in,user->y);CHKERRQ(ierr); /* y = D*in */
    } else {
      ierr = VecCopy(in,user->y);CHKERRQ(ierr); /* y = in */
    }
    ierr = VecPointwiseMult(user->y_work,user->diag,user->y);CHKERRQ(ierr);   /* y_work = diag.*(D*in), where diag = epsilon^2 ./ sqrt(x.^2+epsilon^2).^3 */
    if (user->D) {
      ierr = MatMultTranspose(user->D,user->y_work,out);CHKERRQ(ierr); /* out = D'*(diag.*(D*in)) */
    } else {
      ierr = VecCopy(user->y_work,out);CHKERRQ(ierr); /* out = diag.*in */
    }  
    break;
  case BRGN_REGULARIZATION_L1JOINT:
    /************ Regularizer: joint-L1-sparsity with Dict ******************/    
    /* out = D'*(Hy*(D*in)), Let y = D*in
       Hy*y = 1./zHat0.*y - y0./zHat0^2 .* repmat(sum(reshape((y0./zHat0).*y, K,L), 2), [L, 1])
            = v1.*y - v2.*repmat(sum(reshape(v3.*y, K,L), 2), [L, 1])
     */
    /* 1. Computer y = D*in */
    ierr = MatShellGetContext(Hreg,&user);CHKERRQ(ierr);
    if (user->D) {
      ierr = MatMult(user->D,in,user->y);CHKERRQ(ierr); /* y = D*in */
    } else {
      ierr = VecCopy(in,user->y);CHKERRQ(ierr); /* y = in */
    }

    /* 2. Compute Hy*y = Hy*(D*in) */
    /* Compute y_work = v3.*y */
    ierr = VecPointwiseMult(user->y_work,user->v3,user->y);CHKERRQ(ierr); 
    /* Compute z = sum(reshape(y_work, [K,L]), 2) = sum(reshape(v3.*y, [K,L]), 2)which is the row sum, using user->idxRowsY */
    ierr = VecGetArray(user->z,&zArray);CHKERRQ(ierr);
    for (k=0;k<user->K;++k) {        
      ierr = VecGetSubVector(user->y_work,user->idxRowsY[k],&subVec);CHKERRQ(ierr);
      ierr = VecSum(subVec,&zArray[k]);CHKERRQ(ierr);
      ierr = VecRestoreSubVector(user->y_work,user->idxRowsY[k],&subVec);CHKERRQ(ierr);    
    }
    ierr = VecRestoreArray(user->z,&zArray);CHKERRQ(ierr);      
    /* Compute y_work = v2.*repmat(z, [L,1]) =  v2.*repmat(sum(reshape(v3.*y, K,L), 2), [L, 1]) */
    ierr = VecCopy(user->v2,user->y_work);CHKERRQ(ierr);
    for (l=0;l<user->L;++l) {        
      ierr = VecGetSubVector(user->y_work,user->idxColsY[l],&subVec);CHKERRQ(ierr);
      ierr = VecPointwiseMult(subVec,subVec,user->z);CHKERRQ(ierr);
      ierr = VecRestoreSubVector(user->y_work,user->idxColsY[l],&subVec);CHKERRQ(ierr);
    } 
    /* Compute y_work = v1.*y - y_work = v1.*y - v2.*repmat(sum(reshape(v3.*y, K,L), 2), [L, 1]) */
    ierr = VecPointwiseMult(user->y,user->v1,user->y);CHKERRQ(ierr); /* y = v1.*y. We don't need y any more */
    ierr = VecAYPX(user->y_work,-1,user->y);CHKERRQ(ierr);           /* y_work = y + (-1)*y_work = v1.*y - y_work */

    /* 3. Compute D'*y_work = D'*(Hy*(D*in)) */
    if (user->D) {
      ierr = MatMultTranspose(user->D,user->y_work,out);CHKERRQ(ierr); /* out = D'*(Hy.*(D*in)) */
    } else {
      ierr = VecCopy(user->y_work,out);CHKERRQ(ierr); /* out = diag.*in */
    }      
    break;
  }

  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------ */
PetscErrorCode EvaluateRegularizerHessian(Tao tao,Vec X,Mat Hreg,void *ctx)
{
  AppCtx         *user = (AppCtx *)ctx;
  PetscErrorCode ierr;
  PetscInt       k,l;    /* k,l loop indices for rows and cols of Y matrix */
  PetscReal      *zArray; /* only used for joint sparsity */
  Vec            subVec; /* only used for joint sparsity */


  PetscFunctionBegin;
  switch (BRGN_REGULARIZATION_USER) {
  case BRGN_REGULARIZATION_L2:
    /************ Regularizer: 0.5*x'*x, Hessian is just an identity matrix which doesn't change ******************/
    break;
  case BRGN_REGULARIZATION_L1DICT:
    /************ Regularizer: L1 Dict ******************/
    /* calculate and store diagonal matrix as a vector: diag = epsilon^2 ./ sqrt(x.^2+epsilon^2).^3* --> diag = epsilon^2 ./ sqrt(y.^2+epsilon^2).^3,where y = D*x */  
    if (user->D) {
      ierr = MatMult(user->D,X,user->y);CHKERRQ(ierr);/* y = D*x */
    } else {
      ierr = VecCopy(X,user->y);CHKERRQ(ierr);
    }
    ierr = VecPointwiseMult(user->y_work,user->y,user->y);CHKERRQ(ierr);
    ierr = VecShift(user->y_work,user->epsilon*user->epsilon);CHKERRQ(ierr);
    ierr = VecCopy(user->y_work,user->diag);CHKERRQ(ierr);                    /* user->diag = y.^2+epsilon^2 */
    ierr = VecSqrtAbs(user->y_work);CHKERRQ(ierr);                            /* user->y_work = sqrt(y.^2+epsilon^2) */ 
    ierr = VecPointwiseMult(user->diag,user->y_work,user->diag);CHKERRQ(ierr);/* user->diag = sqrt(y.^2+epsilon^2).^3 */
    ierr = VecReciprocal(user->diag);CHKERRQ(ierr);
    ierr = VecScale(user->diag,user->epsilon*user->epsilon);CHKERRQ(ierr);
    break;
  case BRGN_REGULARIZATION_L1JOINT:
    /************ Regularizer: joint-L1-sparsity with Dict ******************/
    /* Pre-compute v1, v2, v3 for computing the Heassian
       Hy*y = 1./zHat0.*y - y0./zHat0^2 .* repmat(sum(reshape((y0./zHat0).*y, K,L), 2), [L, 1])
            = v1.*y - v2.*repmat(sum(reshape(v3.*y, K,L), 2), [L, 1])
       v1 = 1./zHat0
       v2 = y0./zHat0^2 = y0.*v1.*v1 = v3.*v1
       v3 = y0./zHat0 = y0.*v1
    */          
    /***************************Copied from EvaluateRegularizerObjectiveAndGradient()*************************************/
    /* Compute y = D*x */
    if (user->D) {
      ierr = MatMult(user->D,X,user->y);CHKERRQ(ierr);/* y = D*x */
    } else {    
      ierr = VecCopy(X,user->y);CHKERRQ(ierr);
    }

    /* Compute z = sum(reshape(y_work, [K,L]), 2) = sum(reshape((D*x).^2, [K,L]), 2) which is the row sum */
    ierr = VecPointwiseMult(user->y_work,user->y,user->y);CHKERRQ(ierr); /* y_work = y.^2 = (D*x).^2 */
    ierr = VecGetArray(user->z,&zArray);CHKERRQ(ierr);
    for (k=0;k<user->K;++k) {        
      ierr = VecGetSubVector(user->y_work,user->idxRowsY[k],&subVec);CHKERRQ(ierr);
      ierr = VecSum(subVec,&zArray[k]);CHKERRQ(ierr);
      ierr = VecRestoreSubVector(user->y_work,user->idxRowsY[k],&subVec);CHKERRQ(ierr);    
    }
    ierr = VecRestoreArray(user->z,&zArray);CHKERRQ(ierr);    
    /* Compute z = sqrt(z + epsilon^2) = sqrt(||Dx||_{2,1}^2 + epsilon^2) */
    ierr = VecShift(user->z,user->epsilon*user->epsilon);CHKERRQ(ierr);
    ierr = VecSqrtAbs(user->z);CHKERRQ(ierr);  /* z = sqrt(||Dx||_{2,1}^2 + epsilon^2) */ 
    /***************************End of Copied from EvaluateRegularizerObjectiveAndGradient(). *************************************/    

    /* Compute v1 = 1./zHat0 = 1./repmat(z, [L,1]). Modified from EvaluateRegularizerObjectiveAndGradient() code of Compute y = y./repmat(z, [L,1])*/
    ierr = VecSet(user->v1,1.0);CHKERRQ(ierr);
    for (l=0;l<user->L;++l) {        
      ierr = VecGetSubVector(user->v1,user->idxColsY[l],&subVec);CHKERRQ(ierr);
      ierr = VecPointwiseDivide(subVec,subVec,user->z);CHKERRQ(ierr);
      ierr = VecRestoreSubVector(user->v1,user->idxColsY[l],&subVec);CHKERRQ(ierr);
    }  

    /* Compute v3 = y0.*v1 = y0./zHat0 = (D*x)./repmat(z, [L,1]), which actually is the gradient */
    ierr = VecPointwiseMult(user->v3,user->y,user->v1);CHKERRQ(ierr);  /* Compare with Compute y = y./repmat(z, [L,1]) in EvaluateRegularizerObjectiveAndGradient(), which way is faster?*/
    
    /* v2 = v3.*v1 = y0./zHat0^2 */
    ierr = VecPointwiseMult(user->v2,user->v3,user->v1);CHKERRQ(ierr);  

    break;
  }
  PetscFunctionReturn(0);
}

/*TEST

   build:
      requires: !complex !single !__float128 !define(PETSC_USE_64BIT_INDICES)

   test:
      localrunfiles: jointsparsity1Data_A_b_xGT_L
      args: -tao_max_it 1000 -tao_brgn_regularization_type l1dict -tao_brgn_regularizer_weight 1e-8 -tao_brgn_l1_smooth_epsilon 1e-6 -tao_gatol 1.e-8

   test:
      suffix: 2
      localrunfiles: jointsparsity1Data_A_b_xGT_L
      args: -tao_monitor -tao_max_it 1000 -tao_brgn_regularization_type l2prox -tao_brgn_regularizer_weight 1e-8 -tao_gatol 1.e-6

   test:
      suffix: 3
      localrunfiles: jointsparsity1Data_A_b_xGT_L
      args: -tao_monitor -tao_max_it 1000 -tao_brgn_regularization_type user -tao_brgn_regularizer_weight 1e-8 -tao_gatol 1.e-6

TEST*/
