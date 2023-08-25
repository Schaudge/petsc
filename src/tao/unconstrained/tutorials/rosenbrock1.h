#ifndef ROSENBROCK1_H_
#define ROSENBROCK1_H_

#include <petsctao.h>
#include <petscsf.h>
#include <petscdevice_cuda.h>

#if PetscDefined(USING_NVCC)
  #include <cuda_profiler_api.h>
#endif

/*
   User-defined application context - contains data needed by the
   application-provided call-back routines that evaluate the function,
   gradient, and hessian.
*/

typedef struct _Rosenbrock
{
  PetscInt  bs;
  PetscInt  i_start, i_end;
  PetscInt  c_start, c_end;
  PetscReal alpha; /* condition parameter */
}
Rosenbrock;

typedef struct _AppCtx *AppCtx;
struct _AppCtx {
  MPI_Comm   comm;
  PetscInt   n;     /* dimension */
  PetscInt   n_local;
  PetscInt   n_local_comp;
  Rosenbrock problem;
  Vec        Hvalues; /* vector for writing COO values of this MPI process */
  Vec        gvalues; /* vector for writing COO values of this MPI process */
  Vec        fvector;
  Vec        gtemplate;
  PetscSF    off_process_scatter;
  Vec        off_process_values; /* buffer for off-process values if chained */
};

/* -------------- User-defined routines ---------- */

static PETSC_HOSTDEVICE_INLINE_DECL PetscReal RosenbrockObjective(PetscScalar alpha, PetscScalar x_1, PetscScalar x_2)
{
  PetscScalar d   = x_2 - x_1 * x_1;
  PetscScalar e   = 1.0 - x_1;
  return alpha * d * d + e * e;
}

static const PetscLogDouble RosenbrockObjectiveFlops = 7.0;

static PETSC_HOSTDEVICE_INLINE_DECL void RosenbrockGradient(PetscScalar alpha, PetscScalar x_1, PetscScalar x_2, PetscScalar g[2])
{
  PetscScalar d   = x_2 - x_1 * x_1;
  PetscScalar e   = 1.0 - x_1;
  PetscScalar g2  = alpha * d * 2.0;

  g[0] = -2.0 * x_1 * g2 - 2.0 * e;
  g[1] = g2;
}

static const PetscInt RosenbrockGradientFlops = 9.0;

static PETSC_HOSTDEVICE_INLINE_DECL PetscReal RosenbrockObjectiveGradient(PetscScalar alpha, PetscScalar x_1, PetscScalar x_2, PetscScalar g[2])
{
  PetscScalar d   = x_2 - x_1 * x_1;
  PetscScalar e   = 1.0 - x_1;
  PetscScalar ad  = alpha * d;
  PetscScalar g2  = ad * 2.0;

  g[0] = -2.0 * x_1 * g2 - 2.0 * e;
  g[1] = g2;
  return ad * d + e * e;
}

static const PetscLogDouble RosenbrockObjectiveGradientFlops = 12.0;

static PETSC_HOSTDEVICE_INLINE_DECL void RosenbrockHessian(PetscScalar alpha, PetscScalar x_1, PetscScalar x_2, PetscScalar h[4])
{
  PetscScalar d   = x_2 - x_1 * x_1;
  PetscScalar g2  = alpha * d * 2.0;
  PetscScalar h2  = -4.0 * alpha * x_1;

  h[0] = -2.0 * (g2 + x_1 * h2) + 2.0;
  h[1] = h[2] = h2;
  h[3] = 2.0 * alpha;
}

static const PetscLogDouble RosenbrockHessianFlops = 11.0;

static PetscErrorCode AppCtxCreate(MPI_Comm comm, AppCtx *ctx)
{
  AppCtx user;

  PetscFunctionBegin;
  PetscCall(PetscNew(ctx));
  user = *ctx;
  user->comm = PETSC_COMM_WORLD;

  /* Initialize problem parameters */
  user->n             = 2;
  user->problem.alpha = 99.0;
  user->problem.bs    = 2; // bs = 2 is block Rosenbrock, bs = n is chained Rosenbrock
  /* Check for command line arguments to override defaults */
  PetscOptionsBegin(user->comm, NULL, "Rosenbrock example", NULL);
  PetscCall(PetscOptionsInt("-n", "Rosenbrock problem size", NULL, user->n, &user->n, NULL));
  PetscCall(PetscOptionsInt("-bs", "Rosenbrock block size (2 <= bs <= n)", NULL, user->problem.bs, &user->problem.bs, NULL));
  PetscCall(PetscOptionsReal("-alpha", "Rosenbrock off-diagonal coefficient", NULL, user->problem.alpha, &user->problem.alpha, NULL));
  PetscOptionsEnd();
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
  PetscCall(VecDestroy(&user->gtemplate));
  PetscCall(PetscFree(user));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateHessian(AppCtx user, Mat *Hessian)
{
  Mat         H;
  PetscLayout layout;
  PetscInt    i_start, i_end, n_local_comp, nnz_local;
  PetscInt    c_start, c_end;
  PetscInt   *coo_i;
  PetscInt   *coo_j;
  PetscInt    bs = user->problem.bs;
  VecType     vec_type;


  PetscFunctionBegin;
  /* Divide of the optimization variables.  If we are solving the chained variant, they can be divided between
     MPI processes in any way; otherwise, the variables are organized into a block size of 2 */
  PetscCall(PetscLayoutCreateFromSizes(user->comm, PETSC_DECIDE, user->n, 1, &layout));
  PetscCall(PetscLayoutSetUp(layout));
  PetscCall(PetscLayoutGetRange(layout, &i_start, &i_end));
  user->problem.i_start = i_start;
  user->problem.i_end = i_end;
  user->n_local = i_end - i_start;
  user->problem.c_start = c_start = (i_start / bs) * (bs - 1) + (i_start % bs);
  user->problem.c_end = c_end = (i_end / bs) * (bs - 1) + (i_end % bs);
  user->n_local_comp = n_local_comp = c_end - c_start;

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
  for (PetscInt c = c_start, k = 0; c < c_end; c++, k += 4) {
    PetscInt i = (c / (bs - 1)) * bs + c % (bs - 1);

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

  {
    PetscMemType memtype;
    const PetscScalar *a;

    PetscCall(VecGetArrayReadAndMemType(user->fvector, &a, &memtype));
    PetscCall(VecRestoreArrayReadAndMemType(user->fvector, &a));

    if (memtype == PETSC_MEMTYPE_DEVICE) {
      PetscLogStage warmup;
      Mat A, AtA;
      Vec x, b;
      PetscInt warmup_size = 1000;

      PetscCall(PetscLogStageRegister("Device Warmup", &warmup));
      PetscCall(PetscLogStageSetActive(warmup, PETSC_FALSE));

      PetscCall(PetscLogStagePush(warmup));
      PetscCall(MatCreateDenseFromVecType(PETSC_COMM_SELF, vec_type, warmup_size, warmup_size, warmup_size, warmup_size, PETSC_DEFAULT, NULL, &A));
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
      PetscCallCUDA(cudaDeviceSynchronize());
      PetscCall(PetscLogStagePop());
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateVectors(AppCtx user, Mat H, Vec *solution, Vec *gradient)
{
  VecType vec_type;
  PetscInt n_coo, *coo_i, i_start, i_end;
  Vec     x;
  PetscInt     n_recv;
  PetscSFNode  recv;
  PetscLayout layout;
  PetscInt c_start = user->problem.c_start, c_end = user->problem.c_end, bs = user->problem.bs;

  PetscFunctionBegin;
  PetscCall(MatCreateVecs(H, solution, gradient));
  x = *solution;
  PetscCall(VecGetOwnershipRange(x, &i_start, &i_end));
  PetscCall(VecGetType(x, &vec_type));
  // create scatter for communicating values
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

  // create COO data for writing the gradient
  PetscCall(VecDuplicate(*gradient, &user->gtemplate));
  n_coo = user->n_local_comp * 2;
  PetscCall(PetscMalloc1(n_coo, &coo_i));
  for (PetscInt c = c_start, k = 0; c < c_end; c++, k += 2) {
    PetscInt i = (c / (bs - 1)) * bs + (c % (bs - 1));

    coo_i[k + 0] = i;
    coo_i[k + 1] = i + 1;
  }
  PetscCall(VecSetPreallocationCOO(user->gtemplate, n_coo, coo_i));
  PetscCall(PetscFree(coo_i));
  PetscCall(VecCreate(user->comm, &user->gvalues));
  PetscCall(VecSetSizes(user->gvalues, n_coo, PETSC_DETERMINE));
  PetscCall(VecSetType(user->gvalues, vec_type));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if PetscDefined(USING_NVCC)
PETSC_KERNEL_DECL void RosenbrockObjective_CUDA_Kernel(Rosenbrock r, const PetscScalar x[], const PetscScalar o[], PetscScalar f_vec[])
{
  int idx         = blockIdx.x * blockDim.x + threadIdx.x; // 1D grid
  int num_threads = gridDim.x * blockDim.x;

  for (PetscInt c = r.c_start + idx, k = c - r.c_start; c < r.c_end; c += num_threads, k += num_threads) {
    PetscInt    i = (c / (r.bs - 1)) * r.bs + (c % (r.bs - 1));
    PetscScalar x_a = x[i - r.i_start];
    PetscScalar x_b = ((i + 1) < r.i_end) ? x[i + 1 - r.i_start] : o[0];

    f_vec[k] = RosenbrockObjective(r.alpha, x_a, x_b);
  }
}

PETSC_KERNEL_DECL void RosenbrockGradient_CUDA_Kernel(Rosenbrock r, const PetscScalar x[], const PetscScalar o[], PetscScalar g[])
{
  int idx         = blockIdx.x * blockDim.x + threadIdx.x; // 1D grid
  int num_threads = gridDim.x * blockDim.x;

  for (PetscInt c = r.c_start + idx, k = c - r.c_start; c < r.c_end; c += num_threads, k += num_threads) {
    PetscInt    i = (c / (r.bs - 1)) * r.bs + (c % (r.bs - 1));
    PetscScalar x_a = x[i - r.i_start];
    PetscScalar x_b = ((i + 1) < r.i_end) ? x[i + 1 - r.i_start] : o[0];

    RosenbrockGradient(r.alpha, x_a, x_b, &g[2 * k]);
  }
}

PETSC_KERNEL_DECL void RosenbrockObjectiveGradient_CUDA_Kernel(Rosenbrock r, const PetscScalar x[], const PetscScalar o[], PetscScalar f_vec[], PetscScalar g[])
{
  int idx         = blockIdx.x * blockDim.x + threadIdx.x; // 1D grid
  int num_threads = gridDim.x * blockDim.x;

  for (PetscInt c = r.c_start + idx, k = c - r.c_start; c < r.c_end; c += num_threads, k += num_threads) {
    PetscInt    i = (c / (r.bs - 1)) * r.bs + (c % (r.bs - 1));
    PetscScalar x_a = x[i - r.i_start];
    PetscScalar x_b = ((i + 1) < r.i_end) ? x[i + 1 - r.i_start] : o[0];

    f_vec[k] = RosenbrockObjectiveGradient(r.alpha, x_a, x_b, &g[2 * k]);
  }
}

PETSC_KERNEL_DECL void RosenbrockHessian_CUDA_Kernel(Rosenbrock r, const PetscScalar x[], const PetscScalar o[], PetscScalar h[])
{
  int idx         = blockIdx.x * blockDim.x + threadIdx.x; // 1D grid
  int num_threads = gridDim.x * blockDim.x;

  for (PetscInt c = r.c_start + idx, k = c - r.c_start; c < r.c_end; c += num_threads, k += num_threads) {
    PetscInt    i = (c / (r.bs - 1)) * r.bs + (c % (r.bs - 1));
    PetscScalar x_a = x[i - r.i_start];
    PetscScalar x_b = ((i + 1) < r.i_end) ? x[i + 1 - r.i_start] : o[0];

    RosenbrockHessian(r.alpha, x_a, x_b, &h[4 * k]);
  }
}

static PetscErrorCode RosenbrockObjective_CUDA(cudaStream_t stream, Rosenbrock r, const PetscScalar x[], const PetscScalar o[], PetscScalar f_vec[])
{
  PetscInt n_comp = r.c_end - r.c_start;

  PetscFunctionBegin;
  RosenbrockObjective_CUDA_Kernel<<<(n_comp + 255) / 256, 256, 0, stream>>>(r, x, o, f_vec);
  PetscCall(PetscLogGpuFlops(RosenbrockObjectiveFlops * n_comp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode RosenbrockGradient_CUDA(cudaStream_t stream, Rosenbrock r, const PetscScalar x[], const PetscScalar o[], PetscScalar g[])
{
  PetscInt n_comp = r.c_end - r.c_start;

  PetscFunctionBegin;
  RosenbrockGradient_CUDA_Kernel<<<(n_comp + 255) / 256, 256, 0, stream>>>(r, x, o, g);
  PetscCall(PetscLogGpuFlops(RosenbrockGradientFlops * n_comp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode RosenbrockObjectiveGradient_CUDA(cudaStream_t stream, Rosenbrock r, const PetscScalar x[], const PetscScalar o[], PetscScalar f_vec[], PetscScalar g[])
{
  PetscInt n_comp = r.c_end - r.c_start;

  PetscFunctionBegin;
  RosenbrockObjectiveGradient_CUDA_Kernel<<<(n_comp + 255) / 256, 256, 0, stream>>>(r, x, o, f_vec, g);
  PetscCall(PetscLogGpuFlops(RosenbrockObjectiveGradientFlops * n_comp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode RosenbrockHessian_CUDA(cudaStream_t stream, Rosenbrock r, const PetscScalar x[], const PetscScalar o[], PetscScalar h[])
{
  PetscInt n_comp = r.c_end - r.c_start;

  PetscFunctionBegin;
  RosenbrockHessian_CUDA_Kernel<<<(n_comp + 255) / 256, 256, 0, stream>>>(r, x, o, h);
  PetscCall(PetscLogGpuFlops(RosenbrockHessianFlops * n_comp));
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

static PetscErrorCode RosenbrockObjective_Host(Rosenbrock r, const PetscScalar x[], const PetscScalar o[], PetscReal *f)
{
  PetscReal _f = 0.0;

  PetscFunctionBegin;
  for (PetscInt c = r.c_start; c < r.c_end; c++) {
    PetscInt i = (c / (r.bs - 1)) * r.bs + (c % (r.bs - 1));
    PetscScalar x_a = x[i - r.i_start];
    PetscScalar x_b = ((i + 1) < r.i_end) ? x[i + 1 - r.i_start] : o[0];

    _f += RosenbrockObjective(r.alpha, x_a, x_b);
  }
  *f = _f;
  PetscCall(PetscLogFlops((RosenbrockObjectiveFlops + 1.0) * (r.c_end - r.c_start)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode RosenbrockGradient_Host(Rosenbrock r, const PetscScalar x[], const PetscScalar o[], PetscScalar g[])
{
  PetscFunctionBegin;
  for (PetscInt c = r.c_start, k = 0; c < r.c_end; c++, k++) {
    PetscInt i = (c / (r.bs - 1)) * r.bs + (c % (r.bs - 1));
    PetscScalar x_a = x[i - r.i_start];
    PetscScalar x_b = ((i + 1) < r.i_end) ? x[i + 1 - r.i_start] : o[0];

    RosenbrockGradient(r.alpha, x_a, x_b, &g[2 * k]);
  }
  PetscCall(PetscLogFlops(RosenbrockGradientFlops * (r.c_end - r.c_start)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode RosenbrockObjectiveGradient_Host(Rosenbrock r, const PetscScalar x[], const PetscScalar o[], PetscReal *f, PetscScalar g[])
{
  PetscReal _f = 0.0;

  PetscFunctionBegin;
  for (PetscInt c = r.c_start, k = 0; c < r.c_end; c++, k++) {
    PetscInt i = (c / (r.bs - 1)) * r.bs + (c % (r.bs - 1));
    PetscScalar x_a = x[i - r.i_start];
    PetscScalar x_b = ((i + 1) < r.i_end) ? x[i + 1 - r.i_start] : o[0];

    _f += RosenbrockObjectiveGradient(r.alpha, x_a, x_b, &g[2 * k]);
  }
  *f = _f;
  PetscCall(PetscLogFlops(RosenbrockObjectiveGradientFlops * (r.c_end - r.c_start)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode RosenbrockHessian_Host(Rosenbrock r, const PetscScalar x[], const PetscScalar o[], PetscScalar h[])
{
  PetscFunctionBegin;
  for (PetscInt c = r.c_start, k = 0; c < r.c_end; c++, k++) {
    PetscInt i = (c / (r.bs - 1)) * r.bs + (c % (r.bs - 1));
    PetscScalar x_a = x[i - r.i_start];
    PetscScalar x_b = ((i + 1) < r.i_end) ? x[i + 1 - r.i_start] : o[0];

    RosenbrockHessian(r.alpha, x_a, x_b, &h[4 * k]);
  }
  PetscCall(PetscLogFlops(RosenbrockHessianFlops * (r.c_end - r.c_start)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* -------------------------------------------------------------------- */

static PetscErrorCode FormObjective(Tao tao, Vec X, PetscReal *f, void *ptr)
{
  AppCtx             user = (AppCtx)ptr;
  PetscReal          f_local = 0.0;
  const PetscScalar *x;
  const PetscScalar *o = NULL;
  PetscMemType       memtype_x;
  PetscDeviceContext dctx, current_dctx;

  PetscFunctionBeginUser;
  PetscCall(TaoGetCallbackDeviceContext(tao, &dctx));
  PetscCall(PetscDeviceContextGetCurrentContext(&current_dctx));
  PetscCall(PetscDeviceContextSetCurrentContext(dctx));
  PetscCall(VecScatterBegin(user->off_process_scatter, X, user->off_process_values, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(user->off_process_scatter, X, user->off_process_values, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecGetArrayReadAndMemType(user->off_process_values, &o, NULL));
  PetscCall(VecGetArrayReadAndMemType(X, &x, &memtype_x));
  if (memtype_x == PETSC_MEMTYPE_HOST) {
    PetscCall(RosenbrockObjective_Host(user->problem, x, o, &f_local));
    PetscCallMPI(MPI_Allreduce(&f_local, f, 1, MPIU_REAL, MPI_SUM, user->comm));
#if PetscDefined(HAVE_CUDA) && PetscDefined(USING_NVCC)
  } else if (memtype_x == PETSC_MEMTYPE_CUDA) {
    PetscScalar       *_fvec;
    PetscScalar        f_scalar;
    cudaStream_t      *stream;

    PetscCall(PetscDeviceContextGetStreamHandle(dctx, &stream));
    PetscCall(VecGetArrayWriteAndMemType(user->fvector, &_fvec, NULL));
    PetscCall(RosenbrockObjective_CUDA(*stream, user->problem, x, o, _fvec));
    PetscCall(VecRestoreArrayWriteAndMemType(user->fvector, &_fvec));
    PetscCall(VecSum(user->fvector, &f_scalar));
    *f = PetscRealPart(f_scalar);
#endif
  } else SETERRQ(user->comm, PETSC_ERR_SUP, "Unsuported memtype %d", (int) memtype_x);
  PetscCall(VecRestoreArrayReadAndMemType(X, &x));
  PetscCall(VecRestoreArrayReadAndMemType(user->off_process_values, &o));
  PetscCall(PetscDeviceContextSetCurrentContext(current_dctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FormGradient(Tao tao, Vec X, Vec G, void *ptr)
{
  AppCtx             user = (AppCtx)ptr;
  PetscScalar       *g;
  const PetscScalar *x;
  const PetscScalar *o = NULL;
  PetscMemType       memtype_x, memtype_g;
  PetscDeviceContext dctx, current_dctx;

  PetscFunctionBeginUser;
  PetscCall(TaoGetCallbackDeviceContext(tao, &dctx));
  PetscCall(PetscDeviceContextGetCurrentContext(&current_dctx));
  PetscCall(PetscDeviceContextSetCurrentContext(dctx));
  PetscCall(VecScatterBegin(user->off_process_scatter, X, user->off_process_values, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(user->off_process_scatter, X, user->off_process_values, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecGetArrayReadAndMemType(user->off_process_values, &o, NULL));
  PetscCall(VecGetArrayReadAndMemType(X, &x, &memtype_x));
  PetscCall(VecGetArrayWriteAndMemType(user->gvalues, &g, &memtype_g));
  PetscAssert(memtype_x == memtype_g, user->comm, PETSC_ERR_ARG_INCOMP, "solution vector and gradient must have save memtype");
  if (memtype_x == PETSC_MEMTYPE_HOST) {
    PetscCall(RosenbrockGradient_Host(user->problem, x, o, g));
#if PetscDefined(HAVE_CUDA) && PetscDefined(USING_NVCC)
  } else if (memtype_x == PETSC_MEMTYPE_CUDA) {
    cudaStream_t      *stream;

    PetscCall(PetscDeviceContextGetStreamHandle(dctx, &stream));
    PetscCall(RosenbrockGradient_CUDA(*stream, user->problem, x, o, g));
#endif
  } else SETERRQ(user->comm, PETSC_ERR_SUP, "Unsuported memtype %d", (int) memtype_x);
  PetscCall(VecSetValuesWithBorrowedCOO(G, user->gtemplate, g, INSERT_VALUES));
  PetscCall(VecRestoreArrayWriteAndMemType(user->gvalues, &g));
  PetscCall(VecRestoreArrayReadAndMemType(X, &x));
  PetscCall(VecRestoreArrayReadAndMemType(user->off_process_values, &o));
  PetscCall(PetscDeviceContextSetCurrentContext(current_dctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
    FormObjectiveGradient - Evaluates the function, f(X), and gradient, G(X).

    Input Parameters:
.   tao  - the Tao context
.   X    - input vector
.   ptr  - optional user-defined context, as set by TaoSetObjectiveGradient()

    Output Parameters:
.   G - vector containing the newly evaluated gradient
.   f - function value

    Note:
    Some optimization methods ask for the function and the gradient evaluation
    at the same time.  Evaluating both at once may be more efficient that
    evaluating each separately.
*/
static PetscErrorCode FormObjectiveGradient(Tao tao, Vec X, PetscReal *f, Vec G, void *ptr)
{
  AppCtx             user = (AppCtx)ptr;
  PetscReal          f_local = 0.0;
  PetscScalar       *g;
  const PetscScalar *x;
  const PetscScalar *o = NULL;
  PetscMemType       memtype_x, memtype_g;
  PetscDeviceContext dctx, current_dctx;

  PetscFunctionBeginUser;
  PetscCall(TaoGetCallbackDeviceContext(tao, &dctx));
  PetscCall(PetscDeviceContextGetCurrentContext(&current_dctx));
  PetscCall(PetscDeviceContextSetCurrentContext(dctx));
  PetscCall(VecScatterBegin(user->off_process_scatter, X, user->off_process_values, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(user->off_process_scatter, X, user->off_process_values, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecGetArrayReadAndMemType(user->off_process_values, &o, NULL));
  PetscCall(VecGetArrayReadAndMemType(X, &x, &memtype_x));
  PetscCall(VecGetArrayWriteAndMemType(user->gvalues, &g, &memtype_g));
  PetscAssert(memtype_x == memtype_g, user->comm, PETSC_ERR_ARG_INCOMP, "solution vector and gradient must have save memtype");
  if (memtype_x == PETSC_MEMTYPE_HOST) {
    PetscCall(RosenbrockObjectiveGradient_Host(user->problem, x, o, &f_local, g));
    PetscCallMPI(MPI_Allreduce(&f_local, f, 1, MPIU_REAL, MPI_SUM, user->comm));
#if PetscDefined(HAVE_CUDA) && PetscDefined(USING_NVCC)
  } else if (memtype_x == PETSC_MEMTYPE_CUDA) {
    PetscScalar *_fvec;
    PetscScalar f_scalar;
    cudaStream_t *stream;

    PetscCall(PetscDeviceContextGetStreamHandle(dctx, &stream));
    PetscCall(VecGetArrayWriteAndMemType(user->fvector, &_fvec, NULL));
    PetscCall(RosenbrockObjectiveGradient_CUDA(*stream, user->problem, x, o, _fvec, g));
    PetscCall(VecRestoreArrayWriteAndMemType(user->fvector, &_fvec));
    PetscCall(VecSum(user->fvector, &f_scalar));
    *f = PetscRealPart(f_scalar);
#endif
  } else SETERRQ(user->comm, PETSC_ERR_SUP, "Unsuported memtype %d", (int) memtype_x);

  PetscCall(VecSetValuesWithBorrowedCOO(G, user->gtemplate, g, INSERT_VALUES));
  PetscCall(VecRestoreArrayWriteAndMemType(user->gvalues, &g));

  PetscCall(VecRestoreArrayReadAndMemType(X, &x));
  PetscCall(VecRestoreArrayReadAndMemType(user->off_process_values, &o));
  PetscCall(PetscDeviceContextSetCurrentContext(current_dctx));
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
  PetscDeviceContext dctx, current_dctx;

  PetscFunctionBeginUser;
  PetscCall(TaoGetCallbackDeviceContext(tao, &dctx));
  PetscCall(PetscDeviceContextGetCurrentContext(&current_dctx));
  PetscCall(PetscDeviceContextSetCurrentContext(dctx));
  PetscCall(VecScatterBegin(user->off_process_scatter, X, user->off_process_values, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(user->off_process_scatter, X, user->off_process_values, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecGetArrayReadAndMemType(user->off_process_values, &o, NULL));
  PetscCall(VecGetArrayReadAndMemType(X, &x, &memtype_x));
  PetscCall(VecGetArrayWriteAndMemType(user->Hvalues, &h, &memtype_h));
  PetscAssert(memtype_x == memtype_h, user->comm, PETSC_ERR_ARG_INCOMP, "solution vector and hessian must have save memtype");
  if (memtype_x == PETSC_MEMTYPE_HOST) {
    PetscCall(RosenbrockHessian_Host(user->problem, x, o, h));
#if PetscDefined(HAVE_CUDA) && PetscDefined(USING_NVCC)
  } else if (memtype_x == PETSC_MEMTYPE_CUDA) {
    cudaStream_t      *stream;

    PetscCall(PetscDeviceContextGetStreamHandle(dctx, &stream));
    PetscCall(RosenbrockHessian_CUDA(*stream, user->problem, x, o, h));
#endif
  } else SETERRQ(user->comm, PETSC_ERR_SUP, "Unsuported memtype %d", (int) memtype_x);

  PetscCall(MatSetValuesCOO(H, h, INSERT_VALUES));
  PetscCall(VecRestoreArrayWriteAndMemType(user->Hvalues, &h));

  PetscCall(VecRestoreArrayReadAndMemType(X, &x));
  PetscCall(VecRestoreArrayReadAndMemType(user->off_process_values, &o));

  if (Hpre != H) PetscCall(MatCopy(H, Hpre, SAME_NONZERO_PATTERN));
  PetscCall(PetscDeviceContextSetCurrentContext(current_dctx));
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

static PetscErrorCode RosenbrockMain(void)
{
  Vec           x;    /* solution vector */
  Vec           g;    /* gradient vector */
  Mat           H;    /* Hessian matrix */
  Tao           tao;  /* Tao solver context */
  AppCtx        user; /* user-defined application context */
  PetscLogStage solve;

  /* Initialize TAO and PETSc */
  PetscFunctionBegin;
  PetscCall(PetscLogStageRegister("Rosenbrock solve", &solve));

  PetscCall(AppCtxCreate(PETSC_COMM_WORLD, &user));
  PetscCall(CreateHessian(user, &H));
  PetscCall(CreateVectors(user, H, &x, &g));

  /* The TAO code begins here */

  PetscCall(TaoCreate(user->comm, &tao));
  PetscCall(VecZeroEntries(x));
  PetscCall(TaoSetSolution(tao, x));

  /* Set routines for function, gradient, hessian evaluation */
  PetscCall(TaoSetObjective(tao, FormObjective, user));
  PetscCall(TaoSetObjectiveAndGradient(tao, g, FormObjectiveGradient, user));
  PetscCall(TaoSetGradient(tao, g, FormGradient, user));
  PetscCall(TaoSetHessian(tao, H, H, FormHessian, user));

  PetscCall(TaoSetFromOptions(tao));

  /* SOLVE THE APPLICATION */
#if PetscDefined(USING_NVCC)
  cudaProfilerStart();
#endif
  PetscCall(PetscLogStagePush(solve));
  PetscCall(TaoSolve(tao));
  PetscCall(PetscLogStagePop());
#if PetscDefined(USING_NVCC)
  cudaProfilerStop();
#endif

  PetscCall(TestLMVM(tao));

  PetscCall(TaoDestroy(&tao));
  PetscCall(VecDestroy(&g));
  PetscCall(VecDestroy(&x));
  PetscCall(MatDestroy(&H));
  PetscCall(AppCtxDestroy(&user));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#endif
