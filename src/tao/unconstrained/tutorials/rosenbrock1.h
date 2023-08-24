#ifndef ROSENBROCK1_H_
#define ROSENBROCK1_H_

#include <petsctao.h>
#include <petscsf.h>
#include <petscdevice_cuda.h>

/*
   User-defined application context - contains data needed by the
   application-provided call-back routines that evaluate the function,
   gradient, and hessian.
*/

typedef struct _AppCtx *AppCtx;
struct _AppCtx {
  MPI_Comm  comm;
  PetscInt  n;     /* dimension */
  PetscInt  n_local;
  PetscInt  n_local_comp;
  PetscReal alpha; /* condition parameter */
  PetscBool chained, is_cuda, test_lmvm, J0_scale;
  Vec       Hvalues; /* vector for writing COO values of this MPI process */
  Vec       gvalues; /* vector for writing COO values of this MPI process */
  Vec       fvector;
  PetscSF   off_process_scatter;
  Vec       off_process_values; /* buffer for off-process values if chained */
};

/* -------------- User-defined routines ---------- */

static PETSC_HOSTDEVICE_INLINE_DECL PetscReal RosenbrockFunctionGradient(PetscScalar alpha, PetscScalar x_1, PetscScalar x_2, PetscScalar g[2])
{
  PetscScalar d   = x_2 - x_1 * x_1;
  PetscScalar e   = 1.0 - x_1;
  PetscScalar g2  = alpha * d * 2.0;

  g[0] = -2.0 * x_1 * g2 - 2.0 * e;
  g[1] = g2;
  return alpha * d * d + e * e;
}

static PETSC_HOSTDEVICE_INLINE_DECL void RosenbrockHessian(PetscScalar alpha, PetscScalar x_1, PetscScalar x_2, PetscScalar h[4])
{
  PetscScalar d   = x_2 - x_1 * x_1;
  PetscScalar g2  = alpha * d * 2.0;
  PetscScalar h2  = -4.0 * alpha * x_1;

  h[0] = -2.0 * (g2 + x_1 * h2) + 2.0;
  h[1] = h[2] = h2;
  h[3] = 2.0 * alpha;
}

static PetscErrorCode AppCtxCreate(MPI_Comm comm, AppCtx *ctx)
{
  AppCtx user;
  PetscBool flg;
  PetscLogStage warmup;

  PetscFunctionBegin;
  PetscCall(PetscNew(ctx));
  user = *ctx;
  user->comm = PETSC_COMM_WORLD;

  /* Initialize problem parameters */
  user->n       = 2;
  user->alpha   = 99.0;
  /* Check for command line arguments to override defaults */
  PetscOptionsBegin(user->comm, NULL, "Rosenbrock example", NULL);
  PetscCall(PetscOptionsInt("-n", "Rosenbrock problem size", NULL, user->n, &user->n, &flg));
  PetscCall(PetscOptionsReal("-alpha", "Rosenbrock off-diagonal coefficient", NULL, user->alpha, &user->alpha, &flg));
  PetscCall(PetscOptionsBool("-chained", "Whether to solve the chained multidimensional Rosenbrock problem", NULL, user->chained, &user->chained, &flg));
  PetscOptionsEnd();

  PetscCall(PetscLogStageRegister("CUDA Warmup", &warmup));
  PetscCall(PetscLogStageSetActive(warmup, PETSC_FALSE));

#if PetscDefined(HAVE_CUDA)
  PetscLogStagePush(warmup);
  {
    Mat A, AtA;
    Vec x, b;
    PetscInt warmup_size = 1000;

    PetscCall(MatCreateDenseCUDA(PETSC_COMM_SELF, warmup_size,warmup_size,warmup_size,warmup_size, NULL, &A));
    PetscCall(MatSetRandom(A, NULL));
    PetscCall(MatCreateVecs(A, &x, &b));
    PetscCall(VecSetRandom(x, NULL));

    PetscCall(MatMult(A, x, b));
    PetscCall(MatTransposeMatMult(A, A, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &AtA));
    PetscCall(MatShift(AtA, (PetscScalar) warmup_size));
    PetscCall(MatSetOption(AtA, MAT_SPD, PETSC_TRUE));
    PetscCall(MatCholeskyFactor(AtA, NULL,NULL));
    PetscCall(MatDestroy(&AtA));
    PetscCall(VecDestroy(&b));
    PetscCall(VecDestroy(&x));
    PetscCall(MatDestroy(&A));
    cudaDeviceSynchronize();
  }
  PetscLogStagePop();
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode AppCtxDestroy(AppCtx *ctx)
{
  AppCtx user;

  PetscFunctionBegin;
  user = *ctx;
  *ctx = NULL;
  PetscCall(VecDestroy(&user->Hvalues));
  PetscCall(VecDestroy(&user->gvalues));
  PetscCall(VecDestroy(&user->fvector));
  PetscCall(VecDestroy(&user->off_process_values));
  PetscCall(PetscSFDestroy(&user->off_process_scatter));
  PetscCall(PetscFree(user));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateHessian(AppCtx user, Mat *Hessian)
{
  Mat         H;
  PetscLayout layout;
  PetscInt    i_start, i_end, i_stride, n_local, n_local_comp, nnz_local;
  PetscInt   *coo_i;
  PetscInt   *coo_j;
  VecType     vec_type;

  PetscFunctionBegin;
  /* Divide of the optimization variables.  If we are solving the chained variant, they can be divided between
     MPI processes in any way; otherwise, the variables are organized into a block size of 2 */
  PetscCall(PetscLayoutCreateFromSizes(user->comm, PETSC_DECIDE, user->n, user->chained ? 1 : 2, &layout));
  PetscCall(PetscLayoutSetUp(layout));
  PetscCall(PetscLayoutGetRange(layout, &i_start, &i_end));
  user->n_local = n_local = i_end - i_start;
  n_local_comp = user->chained ? n_local : n_local / 2;
  if (user->chained && n_local > 0 && i_end == user->n) n_local_comp--;
  user->n_local_comp = n_local_comp;

  PetscCall(MatCreate(user->comm, Hessian));
  H = *Hessian;
  PetscCall(MatSetLayouts(H, layout, layout));
  PetscCall(PetscLayoutDestroy(&layout));
  PetscCall(MatSetType(H, MATAIJ));
  PetscCall(MatSetOption(H, MAT_HERMITIAN, PETSC_TRUE));
  PetscCall(MatSetOption(H, MAT_SYMMETRIC, PETSC_TRUE));
  PetscCall(MatSetOption(H, MAT_SYMMETRY_ETERNAL, PETSC_TRUE));
  PetscCall(MatSetOption(H, MAT_STRUCTURALLY_SYMMETRIC, PETSC_TRUE));
  PetscCall(MatSetOption(H, MAT_STRUCTURAL_SYMMETRY_ETERNAL, PETSC_TRUE));
  PetscCall(MatSetFromOptions(H)); /* set from options so that we can change the underlying matrix type */

  nnz_local = n_local_comp * 4;
  PetscCall(PetscMalloc2(nnz_local, &coo_i, nnz_local, &coo_j));
  i_stride = user->chained ? 1 : 2;
  for (PetscInt c = 0, k = 0, i = i_start; c < n_local_comp; c++, k += 4, i += i_stride) {
    coo_i[k+0] = i;
    coo_i[k+1] = i;
    coo_i[k+2] = i + 1;
    coo_i[k+3] = i + 1;

    coo_j[k+0] = i;
    coo_j[k+1] = i + 1;
    coo_j[k+2] = i;
    coo_j[k+3] = i + 1;
  }
  PetscCall(MatSetPreallocationCOO(H, nnz_local, coo_i, coo_j));
  PetscCall(PetscFree2(coo_i, coo_j));

  PetscCall(MatGetVecType(H, &vec_type));
  PetscCall(VecCreate(user->comm, &user->Hvalues));
  PetscCall(VecSetSizes(user->Hvalues, nnz_local, PETSC_DETERMINE));
  PetscCall(VecSetType(user->Hvalues, vec_type));

  // vector to collect contributions to the objective
  PetscCall(VecCreate(user->comm, &user->fvector));
  PetscCall(VecSetSizes(user->fvector, user->n_local_comp, PETSC_DETERMINE));
  PetscCall(VecSetType(user->fvector, vec_type));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateVectors(AppCtx user, Mat H, Vec *solution, Vec *gradient)
{
  VecType vec_type;
  PetscInt n_coo, *coo_i, i_start, i_end, i_stride;

  PetscFunctionBegin;
  PetscCall(MatCreateVecs(H, solution, gradient));
  PetscCall(VecGetOwnershipRange(*solution, &i_start, &i_end));
  PetscCall(VecGetType(*solution, &vec_type));
  if (user->chained) {
    // create scatter for communicating values
    Vec     x = *solution;
    PetscInt     n_recv;
    PetscSFNode  recv;
    PetscLayout layout;

    PetscCall(VecGetLayout(x, &layout));
    n_recv = 0;
    if (user->n_local_comp && i_end < user->n) {
      PetscMPIInt rank;
      PetscInt    index;

      n_recv = 1;
      PetscCall(PetscLayoutFindOwnerIndex(layout, i_end, &rank, &index));
      recv.rank = rank;
      recv.index = index;
    }
    PetscCall(PetscSFCreate(user->comm, &user->off_process_scatter));
    PetscCall(PetscSFSetGraph(user->off_process_scatter, user->n_local, n_recv, NULL, PETSC_USE_POINTER, &recv, PETSC_COPY_VALUES));
    PetscCall(VecCreate(user->comm, &user->off_process_values));
    PetscCall(VecSetSizes(user->off_process_values, 1, PETSC_DETERMINE));
    PetscCall(VecSetType(user->off_process_values, vec_type));
    PetscCall(VecZeroEntries(user->off_process_values));
  }

  // create COO data for writing the gradient
  n_coo = user->n_local_comp * 2;
  PetscCall(PetscMalloc1(n_coo, &coo_i));
  i_stride = user->chained ? 1 : 2;
  for (PetscInt c = 0, k = 0, i = i_start; c < user->n_local_comp; c++, k += 2, i += i_stride) {
    coo_i[k + 0] = i;
    coo_i[k + 1] = i + 1;
  }
  PetscCall(VecSetPreallocationCOO(*gradient, n_coo, coo_i));
  PetscCall(PetscFree(coo_i));
  PetscCall(VecCreate(user->comm, &user->gvalues));
  PetscCall(VecSetSizes(user->gvalues, n_coo, PETSC_DETERMINE));
  PetscCall(VecSetType(user->gvalues, vec_type));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if PetscDefined(USING_NVCC)
PETSC_KERNEL_DECL void RosenbrockFunctionGradient_CUDA_Kernel(PetscInt n_comp, PetscInt n, PetscInt stride, PetscReal alpha, const PetscScalar x[], const PetscScalar o[], PetscScalar f_vec[], PetscScalar g[])
{
  int idx         = blockIdx.x * blockDim.x + threadIdx.x; // 1D grid
  int num_threads = gridDim.x * blockDim.x;

  for (int c = idx, i = c * stride, k = c * 2; c < n_comp; c += num_threads, i += num_threads * stride, k += num_threads * 2) {
    PetscScalar x_a = x[i + 0];
    PetscScalar x_b = ((i + 1) < n) ? x[i + 1] : o[0];

    f_vec[c] = RosenbrockFunctionGradient(alpha, x_a, x_b, &g[k]);
  }
}

PETSC_KERNEL_DECL void RosenbrockHessian_CUDA_Kernel(PetscInt n_comp, PetscInt n, PetscInt stride, PetscReal alpha, const PetscScalar x[], const PetscScalar o[], PetscScalar h[])
{
  int idx         = blockIdx.x * blockDim.x + threadIdx.x; // 1D grid
  int num_threads = gridDim.x * blockDim.x;

  for (int c = idx, i = c * stride, k = c * 4; c < n_comp; c += num_threads, i += num_threads * stride, k += num_threads * 4) {
    PetscScalar x_a = x[i + 0];
    PetscScalar x_b = ((i + 1) < n) ? x[i + 1] : o[0];

    RosenbrockHessian(alpha, x_a, x_b, &h[k]);
  }
}

static PetscErrorCode RosenbrockFunctionGradient_CUDA(PetscInt n_comp, PetscInt n, PetscInt stride, PetscReal alpha, const PetscScalar x[], const PetscScalar o[], PetscScalar f_vec[], PetscScalar g[])
{
  PetscFunctionBegin;
  RosenbrockFunctionGradient_CUDA_Kernel<<<(n_comp + 255) / 256, 256>>>(n_comp, n, stride, alpha, x, o, f_vec, g);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode RosenbrockHessian_CUDA(PetscInt n_comp, PetscInt n, PetscInt stride, PetscReal alpha, const PetscScalar x[], const PetscScalar o[], PetscScalar h[])
{
  PetscFunctionBegin;
  RosenbrockHessian_CUDA_Kernel<<<(n_comp + 255) / 256, 256>>>(n_comp, n, stride, alpha, x, o, h);
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

static PetscErrorCode RosenbrockFunctionGradient_Host(PetscInt n_comp, PetscInt n, PetscInt stride, PetscReal alpha, const PetscScalar x[], const PetscScalar o[], PetscReal *f, PetscScalar g[])
{
  PetscReal _f = 0.0;

  PetscFunctionBegin;
  for (PetscInt c = 0, i = 0, k = 0; c < n_comp; c++, k += 2, i += stride) {
    PetscScalar x_a = x[i + 0];
    PetscScalar x_b = ((i + 1) < n) ? x[i + 1] : o[0];

    _f += RosenbrockFunctionGradient(alpha, x_a, x_b, &g[k]);
  }
  *f = _f;
  PetscCall(PetscLogFlops(14.0 * n_comp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode RosenbrockHessian_Host(PetscInt n_comp, PetscInt n, PetscInt stride, PetscReal alpha, const PetscScalar x[], const PetscScalar o[], PetscScalar h[])
{
  PetscFunctionBegin;
  for (PetscInt c = 0, i = 0, k = 0; c < n_comp; c++, k += 4, i += stride) {
    PetscScalar x_a = x[i + 0];
    PetscScalar x_b = ((i + 1) < n) ? x[i + 1] : o[0];

    RosenbrockHessian(alpha, x_a, x_b, &h[k]);
  }
  PetscCall(PetscLogFlops(11.0 * n_comp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* -------------------------------------------------------------------- */
/*
    FormFunctionGradient - Evaluates the function, f(X), and gradient, G(X).

    Input Parameters:
.   tao  - the Tao context
.   X    - input vector
.   ptr  - optional user-defined context, as set by TaoSetFunctionGradient()

    Output Parameters:
.   G - vector containing the newly evaluated gradient
.   f - function value

    Note:
    Some optimization methods ask for the function and the gradient evaluation
    at the same time.  Evaluating both at once may be more efficient that
    evaluating each separately.
*/
static PetscErrorCode FormFunctionGradient(Tao tao, Vec X, PetscReal *f, Vec G, void *ptr)
{
  AppCtx             user = (AppCtx)ptr;
  PetscReal          f_local = 0.0;
  PetscScalar       *g;
  const PetscScalar *x;
  const PetscScalar *o = NULL;
  PetscMemType       memtype_x, memtype_g;

  PetscFunctionBeginUser;
  if (user->chained) {
    PetscCall(VecScatterBegin(user->off_process_scatter, X, user->off_process_values, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterEnd(user->off_process_scatter, X, user->off_process_values, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecGetArrayReadAndMemType(user->off_process_values, &o, NULL));
  }
  PetscCall(VecGetArrayReadAndMemType(X, &x, &memtype_x));
  PetscCall(VecGetArrayWriteAndMemType(user->chained ? user->gvalues : G, &g, &memtype_g));
  PetscAssert(memtype_x == memtype_g, user->comm, PETSC_ERR_ARG_INCOMP, "solution vector and gradient must have save memtype");
  if (memtype_x == PETSC_MEMTYPE_HOST) {
    PetscCall(RosenbrockFunctionGradient_Host(user->n_local_comp, user->n_local, user->chained ? 1 : 2, user->alpha, x, o, &f_local, g));
    PetscCallMPI(MPI_Allreduce(&f_local, f, 1, MPIU_REAL, MPI_SUM, user->comm));
#if PetscDefined(HAVE_CUDA) && PetscDefined(USING_NVCC)
  } else if (memtype_x == PETSC_MEMTYPE_CUDA) {
    PetscScalar *_fvec;
    PetscScalar f_scalar;

    PetscCall(VecGetArrayWriteAndMemType(user->fvector, &_fvec, NULL));
    PetscCall(RosenbrockFunctionGradient_CUDA(user->n_local_comp, user->n_local, user->chained ? 1 : 2, user->alpha, x, o, _fvec, g));
    PetscCall(VecRestoreArrayWriteAndMemType(user->fvector, &_fvec));
    PetscCall(VecSum(user->fvector, &f_scalar));
    *f = PetscRealPart(f_scalar);
#endif
  } else SETERRQ(user->comm, PETSC_ERR_SUP, "Unsuported memtype %d", (int) memtype_x);

  if (user->chained) {
    PetscCall(VecSetValuesCOO(G, g, INSERT_VALUES));
    PetscCall(VecRestoreArrayWriteAndMemType(user->gvalues, &g));
  } else {
    PetscCall(VecRestoreArrayWriteAndMemType(G, &g));
  }

  PetscCall(VecRestoreArrayReadAndMemType(X, &x));
  if (user->chained) PetscCall(VecRestoreArrayReadAndMemType(user->off_process_values, &o));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ------------------------------------------------------------------- */
/*
   FormHessian - Evaluates Hessian matrix.

   Input Parameters:
.  tao   - the Tao context
.  x     - input vector
.  ptr   - optional user-defined context, as set by TaoSetHessian()

   Output Parameters:
.  H     - Hessian matrix

   Note:  Providing the Hessian may not be necessary.  Only some solvers
   require this matrix.
*/
static PetscErrorCode FormHessian(Tao tao, Vec X, Mat H, Mat Hpre, void *ptr)
{
  AppCtx             user = (AppCtx)ptr;
  PetscScalar       *h;
  const PetscScalar *x;
  const PetscScalar *o = NULL;
  PetscMemType       memtype_x, memtype_h;

  PetscFunctionBeginUser;
  if (user->chained) {
    PetscCall(VecScatterBegin(user->off_process_scatter, X, user->off_process_values, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterEnd(user->off_process_scatter, X, user->off_process_values, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecGetArrayReadAndMemType(user->off_process_values, &o, NULL));
  }
  PetscCall(VecGetArrayReadAndMemType(X, &x, &memtype_x));
  PetscCall(VecGetArrayWriteAndMemType(user->Hvalues, &h, &memtype_h));
  PetscAssert(memtype_x == memtype_h, user->comm, PETSC_ERR_ARG_INCOMP, "solution vector and hessian must have save memtype");
  if (memtype_x == PETSC_MEMTYPE_HOST) {
    PetscCall(RosenbrockHessian_Host(user->n_local_comp, user->n_local, user->chained ? 1 : 2, user->alpha, x, o, h));
#if PetscDefined(HAVE_CUDA) && PetscDefined(USING_NVCC)
  } else if (memtype_x == PETSC_MEMTYPE_CUDA) {
    PetscCall(RosenbrockHessian_CUDA(user->n_local_comp, user->n_local, user->chained ? 1 : 2, user->alpha, x, o, h));
#endif
  } else SETERRQ(user->comm, PETSC_ERR_SUP, "Unsuported memtype %d", (int) memtype_x);

  PetscCall(MatSetValuesCOO(H, h, INSERT_VALUES));
  PetscCall(VecRestoreArrayWriteAndMemType(user->Hvalues, &h));

  PetscCall(VecRestoreArrayReadAndMemType(X, &x));
  if (user->chained) PetscCall(VecRestoreArrayReadAndMemType(user->off_process_values, &o));

  if (Hpre != H) {
    PetscCall(MatCopy(H, Hpre, SAME_NONZERO_PATTERN));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestLMVM(Tao tao)
{
  KSP       ksp;
  PC        pc;
  PetscBool is_lmvm;

  PetscFunctionBegin;
  PetscCall(TaoGetKSP(tao, &ksp));
  if (!ksp) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PetscObjectTypeCompare((PetscObject)pc, PCLMVM, &is_lmvm));
  if (is_lmvm) {
    Mat       M;
    Vec       in, out, out2;
    PetscReal mult_solve_dist;
    Vec       x;

    PetscCall(PCLMVMGetMatLMVM(pc, &M));
    PetscCall(TaoGetSolution(tao, &x));
    PetscCall(VecDuplicate(x, &in));
    PetscCall(VecDuplicate(x, &out));
    PetscCall(VecDuplicate(x, &out2));
    PetscCall(VecSet(in, 1.0));
    PetscCall(MatMult(M, in, out));
    PetscCall(MatSolve(M, out, out2));

    PetscCall(VecAXPY(out2, -1.0, in));
    PetscCall(VecNorm(out2, NORM_2, &mult_solve_dist));
    if (mult_solve_dist < 1.e-11) {
      PetscCall(PetscPrintf(PetscObjectComm((PetscObject)tao), "error between LMVM MatMult and MatSolve: < 1.e-11\n"));
    } else if (mult_solve_dist < 1.e-6) {
      PetscCall(PetscPrintf(PetscObjectComm((PetscObject)tao), "error between LMVM MatMult and MatSolve: < 1.e-6\n"));
    } else {
      PetscCall(PetscPrintf(PetscObjectComm((PetscObject)tao), "error between LMVM MatMult and MatSolve: %e\n", (double)mult_solve_dist));
    }
    PetscCall(VecDestroy(&in));
    PetscCall(VecDestroy(&out));
    PetscCall(VecDestroy(&out2));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

#endif
