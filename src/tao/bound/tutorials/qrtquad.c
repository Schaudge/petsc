const char help[] = "Implementation of QRTQUAD bound constrained optimization problem";

/// QRTQUAD - The objective function is
//
// f(x) = \sum_{i = 1}^n x_i * (-10 * i) + \sum_{i = 1}^m (i / m) * (x_i * x_{i+1})^4 + \sum_{i=m+1}^{n-1} 4 x_i^2 + 2 * x_n^2 + x_i * x_n
//        \____________________________/   \________________________________________/   \________________________________________________/
//                       |                                     |                                                |
//                 linear terms     "quartic" terms (really degree 8, quartic in x_i * x_{i+1})          quadratic terms
//
// the variables (x_1, ..., x_m) are bounded in [0, 10]

#include <petsctao.h>
#include <petscsf.h>
#include <petscdevice.h>
#include <petscdevice_cupm.h>

typedef struct _AppCtx *AppCtx;

struct _AppCtx {
  MPI_Comm    comm;
  PetscInt    n;     // Total number of degrees of freedom
  PetscInt    m;     // Number of quartic & bounded degrees of freedeom
  PetscInt    k;     // Number of copies
  PetscInt    r_start;
  PetscInt    r_end;
  PetscInt    g_start;
  PetscInt    g_end;
  Vec         x;     // solution
  Vec         x_next; // k values from the next process
  VecScatter  x_next_scatter;
  Vec         x_end; // k values from the end of the vector
  PetscSF     x_end_bcast;
  Vec         xl;    // lower bound
  Vec         xu;    // upper bound
  Vec         g;     // gradient
  Vec         g_lin; // from the linear term
  Vec         vcopy; // local copies of the global v variables
  Vec         gvals; // gradient entries (excluding last k variables)
  Vec         arrowhead_ones;
  Mat         H;  // Hessian preconditioner
  Vec         Hvals; // Hessian entries
  PetscLayout layout;
  PetscBool   set_from_options_called;
};

static PetscErrorCode AppCtxCreate(MPI_Comm comm, AppCtx *ctx)
{
  PetscMPIInt size;
  AppCtx      c;

  PetscFunctionBegin;
  PetscCall(MPI_Comm_size(comm, &size));
  PetscCall(PetscNew(ctx));
  c = *ctx;
  c->comm = comm;
  c->n    = 5000;
  c->m    = 1100;
  c->k    = 1;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode AppCtxSetFromOptions(AppCtx ctx)
{
  PetscFunctionBegin;
  PetscOptionsBegin(ctx->comm, NULL, help, NULL);
  PetscCall(PetscOptionsBoundedInt("-n", "number of optimization variables", NULL, ctx->n, &ctx->n, NULL, 0));
  ctx->m = PetscMin(ctx->m, ctx->n);
  PetscCall(PetscOptionsRangeInt("-m", "number of quartic and bounded variables", NULL, ctx->m, &ctx->m, NULL, 0, ctx->n));
  PetscCall(PetscOptionsBoundedInt("-k", "number of copies", NULL, ctx->k, &ctx->k, NULL, 1));
  PetscOptionsEnd();
  ctx->set_from_options_called = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode AppCtxSetUp(AppCtx ctx)
{
  PetscInt     m, n, k, r_start, r_end;
  PetscScalar *ga, *la, *ua;
  PetscInt     g_start, g_end;
  PetscInt    *h_i, *h_j;

  PetscFunctionBegin;
  n = ctx->n;
  m = ctx->m;
  k = ctx->k;

  // Layout of solution degrees of freedom
  PetscCall(PetscLayoutCreate(ctx->comm, &ctx->layout));
  PetscCall(PetscLayoutSetSize(ctx->layout, n * k));
  PetscCall(PetscLayoutSetLocalSize(ctx->layout, PETSC_DETERMINE));
  PetscCall(PetscLayoutSetBlockSize(ctx->layout, k));
  PetscCall(PetscLayoutSetUp(ctx->layout));
  PetscCall(PetscLayoutGetRange(ctx->layout, &r_start, &r_end));
  ctx->r_start = r_start;
  ctx->r_end   = r_end;
  PetscCheck(r_start % k == 0, ctx->comm, PETSC_ERR_PLIB, "k boundary not respected");
  PetscCheck(r_end % k == 0, ctx->comm, PETSC_ERR_PLIB, "k boundary not respected");

  // Hessian: allocated even if it isn't used
  PetscCall(MatCreate(ctx->comm, &ctx->H));
  PetscCall(MatSetLayouts(ctx->H, ctx->layout, ctx->layout));
  PetscCall(MatSetBlockSize(ctx->H, ctx->k));
  PetscCall(MatSetType(ctx->H, MATAIJ));
  if (ctx->set_from_options_called) PetscCall(MatSetFromOptions(ctx->H));

  // COO Preallocation for H
  {
    ctx->g_start = g_start = r_start / k;
    ctx->g_end   = g_end   = PetscMax(g_start, PetscMin(n - 1, r_end / k));
    PetscCall(PetscMalloc2(4 * k * (g_end - g_start), &h_i, 4 * k * (g_end - g_start), &h_j));
    // Tridiagonal structure for the quartic terms
    for (PetscInt g = g_start; g < PetscMin(m, g_end); g++) {
      PetscInt *PETSC_RESTRICT g_i = &h_i[(g - g_start) * 4 * k];
      PetscInt *PETSC_RESTRICT g_j = &h_j[(g - g_start) * 4 * k];

      for (PetscInt i = 0; i < k; i++) {
        g_i[i + 0 * k] = i + k * g;
        g_j[i + 0 * k] = i + k * g;

        g_i[i + 1 * k] = i + k * g;
        g_j[i + 1 * k] = i + k * (g + 1);

        g_i[i + 2 * k] = i + k * (g + 1);
        g_j[i + 2 * k] = i + k * g;

        g_i[i + 3 * k] = i + k * (g + 1);
        g_j[i + 3 * k] = i + k * (g + 1);
      }
    }
    // Arrowhead structrure for the quadratic terms
    for (PetscInt g = PetscMax(g_start,PetscMin(m, g_end)); g < g_end; g++) {
      PetscInt *PETSC_RESTRICT g_i = &h_i[(g - g_start) * 4 * k];
      PetscInt *PETSC_RESTRICT g_j = &h_j[(g - g_start) * 4 * k];

      for (PetscInt i = 0; i < k; i++) {
        g_i[i + 0 * k] = i + k * g;
        g_j[i + 0 * k] = i + k * g;

        g_i[i + 1 * k] = i + k * g;
        g_j[i + 1 * k] = i + k * (n - 1);

        g_i[i + 2 * k] = i + k * (n - 1);
        g_j[i + 2 * k] = i + k * g;

        g_i[i + 3 * k] = i + k * (n - 1);
        g_j[i + 3 * k] = i + k * (n - 1);
      }
    }
    PetscCall(MatSetPreallocationCOO(ctx->H, 4 * (g_end - g_start), h_i, h_j));
    PetscCall(PetscFree2(h_i, h_j));
  }
  PetscCall(MatSetUp(ctx->H));
  PetscCall(MatZeroEntries(ctx->H));
  PetscCall(MatAssemblyBegin(ctx->H, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(ctx->H, MAT_FINAL_ASSEMBLY));
  PetscCall(MatViewFromOptions(ctx->H, NULL, "-hessian_structure_view"));

  PetscCall(MatCreateVecs(ctx->H, &ctx->x, &ctx->g));
  PetscCall(VecZeroEntries(ctx->x));

  // the quartic terms need ghost data
  {
    PetscInt n_next = 0;
    PetscInt *next_nodes = NULL;
    VecType  vec_type;
    IS       next_nodes_is;

    if (r_start < r_end && r_end <= k * m) {
      // This process had quartic terms
      n_next = k;
    }
    PetscCall(PetscMalloc1(n_next, &next_nodes));
    for (PetscInt i = 0; i < n_next; i++) next_nodes[i] = r_end + i;
    PetscCall(ISCreateGeneral(ctx->comm, n_next, next_nodes, PETSC_OWN_POINTER, &next_nodes_is));

    PetscCall(VecGetType(ctx->x, &vec_type));
    PetscCall(VecCreate(ctx->comm, &ctx->x_next));
    PetscCall(VecSetSizes(ctx->x_next, n_next, PETSC_DETERMINE));
    PetscCall(VecSetType(ctx->x_next, vec_type));
    PetscCall(VecSetUp(ctx->x_next));
    PetscCall(VecScatterCreate(ctx->x, next_nodes_is, ctx->x_next, NULL, &ctx->x_next_scatter));
    PetscCall(ISDestroy(&next_nodes_is));
    PetscCall(VecScatterSetUp(ctx->x_next_scatter));
    PetscCall(VecScatterViewFromOptions(ctx->x_next_scatter, NULL, "-x_next_scatter_view"));
  }

  // the last entries need to be broadcast for the quadratic terms
  {
    PetscLayout gather_layout;
    PetscBool   owns_end = PETSC_FALSE;
    VecType     vec_type;

    if (r_start < r_end && r_end == n) owns_end = PETSC_TRUE;

    PetscCall(VecGetType(ctx->x, &vec_type));
    PetscCall(VecCreate(ctx->comm, &ctx->x_end));
    PetscCall(VecSetSizes(ctx->x_end, k, PETSC_DETERMINE));
    PetscCall(VecSetType(ctx->x_end, vec_type));

    PetscCall(PetscSFCreate(ctx->comm, &ctx->x_end_bcast));
    PetscCall(PetscLayoutCreate(ctx->comm, &gather_layout));
    PetscCall(PetscLayoutSetSize(gather_layout, k));
    PetscCall(PetscLayoutSetLocalSize(gather_layout, owns_end ? k : 0));
    PetscCall(PetscLayoutSetUp(gather_layout));
    PetscCall(PetscSFSetGraphWithPattern(ctx->x_end_bcast, gather_layout, PETSCSF_PATTERN_ALLGATHER));
    PetscCall(PetscLayoutDestroy(&gather_layout));
    PetscCall(PetscSFSetUp(ctx->x_end_bcast));
    PetscCall(VecScatterViewFromOptions(ctx->x_end_bcast, NULL, "-x_end_bcast_view"));
  }

  // the vector for the linear portion of the gradient
  PetscCall(VecDuplicate(ctx->g, &ctx->g_lin));
  PetscCall(VecGetArray(ctx->g_lin, &ga));
  for (PetscInt i = r_start; i < r_end; i++) {
    ga[i - r_start] = -10.0 * (((PetscInt)(i / k)) + 1);
  }
  PetscCall(VecRestoreArray(ctx->g_lin, &ga));
  PetscCall(VecViewFromOptions(ctx->g_lin, NULL, "-linear_gradient_view"));

  // create bounds for the first m variables
  PetscCall(VecDuplicate(ctx->x, &ctx->xl));
  PetscCall(VecDuplicate(ctx->x, &ctx->xu));
  PetscCall(VecSet(ctx->xl, PETSC_NINFINITY));
  PetscCall(VecSet(ctx->xu, PETSC_INFINITY));
  PetscCall(VecGetArray(ctx->xl, &la));
  PetscCall(VecGetArray(ctx->xu, &ua));
  // variables in [0, m) are bounded in [0, 10]
  for (PetscInt i = r_start; i < PetscMin(r_end, m); i++) {
    la[i - r_start] = 0.0;
    ua[i - r_start] = 10.0;
  }
  PetscCall(VecRestoreArray(ctx->xu, &ua));
  PetscCall(VecRestoreArray(ctx->xl, &la));
  PetscCall(VecViewFromOptions(ctx->xl, NULL, "-xl_view"));
  PetscCall(VecViewFromOptions(ctx->xu, NULL, "-xu_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PETSC_HOSTDEVICE_INLINE_DECL PetscReal QrtObjective(PetscScalar x, PetscScalar y, PetscReal p)
{
  PetscScalar a  = x * y;
  PetscScalar a2 = a * a;
  PetscScalar a4 = a2 * a2;
  return p * a4;
}

static const PetscLogDouble QrtObjectiveFlops = 4.0;

static PETSC_HOSTDEVICE_INLINE_DECL void QrtGradient(PetscScalar x, PetscScalar y, PetscReal p, PetscScalar g[2])
{
  PetscScalar a  = x * y;
  PetscScalar a2 = a * a;
  PetscScalar a3 = a * a2;
  g[0] = y * (p * 4.0 * a3);
  g[1] = x * (p * 4.0 * a3);
}

static const PetscLogDouble QrtGradientFlops = 7.0;

static PETSC_HOSTDEVICE_INLINE_DECL PetscReal QrtObjectiveGradient(PetscScalar x, PetscScalar y, PetscReal p, PetscScalar g[2])
{
  PetscScalar a  = x * y;
  PetscScalar a2 = a * a;
  PetscScalar a3 = a * a2;
  PetscScalar a4 = a * a3;
  g[0] = y * (p * 4.0 * a3);
  g[1] = x * (p * 4.0 * a3);
  return p * a4;
}

static const PetscLogDouble QrtObjectiveGradientFlops = 9.0;

static PETSC_HOSTDEVICE_INLINE_DECL void QrtHessian(PetscScalar x, PetscScalar y, PetscReal p, PetscScalar *PETSC_RESTRICT h_xx, PetscScalar *PETSC_RESTRICT h_xy, PetscScalar *PETSC_RESTRICT h_yx, PetscScalar *PETSC_RESTRICT h_yy)
{
  PetscScalar a  = x * y;
  PetscScalar a2 = a * a;
  PetscScalar a3 = a * a2;
  *h_xx = (y*y) * (p * 12.0 * a2);
  *h_xy = *h_yx = (x*y) * (p * 12.0 * a2) + (p * 4.0 * a3);
  *h_yy = (x*x) * (p * 12.0 * a2);
}

static const PetscLogDouble QrtHessianFlops = 9.0;

static PETSC_HOSTDEVICE_INLINE_DECL PetscReal QuadObjective(PetscScalar x, PetscScalar y)
{
  return (4.0 * x * x + 2.0 * y * y + x * y);
}

static const PetscLogDouble QuadObjectiveFlops = 8.0;

static PETSC_HOSTDEVICE_INLINE_DECL void QuadGradient(PetscScalar x, PetscScalar y, PetscScalar g[2])
{
  g[0] = 8.0 * x + y;
  g[1] = 4.0 * y + x;
}

static const PetscLogDouble QuadGradientFlops = 4.0;

static PETSC_HOSTDEVICE_INLINE_DECL PetscReal QuadObjectiveGradient(PetscScalar x, PetscScalar y, PetscScalar g[2])
{
  g[0] = 8.0 * x + y;
  g[1] = 4.0 * y + x;
  return (4.0 * x * x + 2.0 * y * y + x * y);
}

static const PetscLogDouble QuadObjectiveGradientFlops = 12.0;

static PETSC_HOSTDEVICE_INLINE_DECL void QuadHessian(PetscScalar x, PetscScalar y, PetscScalar *PETSC_RESTRICT h_xx, PetscScalar *PETSC_RESTRICT h_xy, PetscScalar *PETSC_RESTRICT h_yx, PetscScalar *PETSC_RESTRICT h_yy)
{
  *h_xx = 8.0;
  *h_xy = *h_yx = 1.0;
  *h_yy = 4.0;
}

static const PetscLogDouble QuadHessianFlops = 9.0;

static PetscErrorCode QrtQuadObjective_Host(AppCtx ctx, const PetscScalar *PETSC_RESTRICT X, const PetscScalar *PETSC_RESTRICT X_next, const PetscScalar *PETSC_RESTRICT X_end, PetscReal *f)
{
  PetscReal _f = 0.0;
  PetscInt  m, k, g_start, g_end, g_mid, g_mid_local;

  PetscFunctionBegin;
  m           = ctx->m;
  k           = ctx->k;
  g_start     = ctx->g_start;
  g_end       = ctx->g_end;
  g_mid       = PetscMax(g_start,PetscMin(m, g_end));
  g_mid_local = g_mid;

  // quartic terms
  if (g_mid == g_end) g_mid_local = PetscMax(g_start, g_mid - 1);
  for (PetscInt g = g_start; g < g_mid_local; g++) {
    PetscPragmaSIMD
    for (PetscInt i = 0; i < k; i++) {
      PetscScalar x = X[i + k * (g - g_start)];
      PetscScalar y = X[i + k * (g + 1 - g_start)];
      PetscReal   p = ((PetscReal)i) * (1.0 / (PetscReal)m);

      _f += QrtObjective(x, y, p);
    }
  }
  if (g_mid_local + 1 == g_mid) {
    PetscPragmaSIMD
    for (PetscInt i = 0; i < k; i++) {
      PetscScalar x = X[i + k * (g_mid_local - g_start)];
      PetscScalar y = X_next[i];
      PetscReal   p = ((PetscReal)i) * (1.0 / (PetscReal)m);

      _f += QrtObjective(x, y, p);
    }
  }
  PetscCall(PetscLogFlops(QrtObjectiveFlops * (g_mid - g_end) * k));

  // quadratic terms
  for (PetscInt g = g_mid; g < g_end; g++) {
    PetscPragmaSIMD
    for (PetscInt i = 0; i < k; i++) {
      PetscScalar x = X[i + k * (g - g_start)];
      PetscScalar y = X_end[i];

      _f += QuadObjective(x, y);
    }
  }
  PetscCall(PetscLogFlops(QuadObjectiveFlops * (g_end - g_mid) * k));
  *f += _f;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode QrtQuadObjective(Tao tao, Vec x, PetscReal *f, void *data)
{
  AppCtx      ctx = (AppCtx) data;
  PetscScalar f_lin;
  PetscReal   _f = 0.0;
  PetscInt    n, k, r_start, r_end;
  const PetscScalar *xa, *xnexta, *xa_root = NULL;
  PetscScalar *xenda;

  PetscFunctionBegin;
  r_start = ctx->r_start;
  r_end = ctx->r_end;
  n = ctx->n;
  k = ctx->k;
  PetscCall(VecDot(x, ctx->g_lin, &f_lin));
  _f = PetscRealPart(f_lin);
  PetscCall(VecScatterBegin(ctx->x_next_scatter, x, ctx->x_next, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(ctx->x_next_scatter, x, ctx->x_next, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecGetArrayRead(x, &xa));
  PetscCall(VecGetArrayRead(ctx->x_next, &xnexta));
  PetscCall(VecGetArray(ctx->x_end, &xenda));
  if (ctx->r_start < ctx->r_end && ctx->r_end == n) xa_root = &xa[(r_end - r_start) - k];
  PetscCall(PetscSFBcastBegin(ctx->x_end_bcast, MPIU_SCALAR, xa_root, xenda, MPI_REPLACE));
  PetscCall(PetscSFBcastEnd(ctx->x_end_bcast, MPIU_SCALAR, xa_root, xenda, MPI_REPLACE));
  PetscCall(QrtQuadObjective_Host(ctx, xa, xnexta, xenda, &_f));
  PetscCall(VecRestoreArray(ctx->x_end, &xenda));
  PetscCall(VecRestoreArrayRead(ctx->x_next, &xnexta));
  PetscCall(VecRestoreArrayRead(x, &xa));
  *f = _f;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode QrtQuadObjectiveAndGradient(Tao tao, Vec x, PetscReal *f, Vec g, void *ctx)
{
  PetscReal _f = 0.0;
  PetscInt r_start, r_end;

  PetscFunctionBegin;
  *f = 0.0;
  PetscCall(VecZeroEntries(g));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode QrtQuadGradient(Tao tao, Vec x, Vec g, void *ctx)
{
  PetscReal _f = 0.0;

  PetscFunctionBegin;
  PetscCall(VecZeroEntries(g));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode AppCtxCreateTao(AppCtx ctx, Tao *tao)
{
  Tao t;

  PetscFunctionBegin;
  PetscCall(TaoCreate(ctx->comm, tao));
  t = *tao;
  PetscCall(TaoSetSolution(t, ctx->x));
  PetscCall(TaoSetVariableBounds(t, ctx->xl, ctx->xu));
  PetscCall(TaoSetType(t, TAOBLMVM));
  PetscCall(TaoSetObjective(t, QrtQuadObjective, (void *) ctx));
  PetscCall(TaoSetGradient(t, ctx->g, QrtQuadGradient, (void *) ctx));
  PetscCall(TaoSetObjectiveAndGradient(t, ctx->g,  QrtQuadObjectiveAndGradient, (void *) ctx));
  if (ctx->set_from_options_called) PetscCall(TaoSetFromOptions(t));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode AppCtxDestroy(AppCtx *ctx)
{
  AppCtx c;

  PetscFunctionBegin;
  c = *ctx;
  *ctx = NULL;
  PetscCall(VecDestroy(&c->xu));
  PetscCall(VecDestroy(&c->xl));
  PetscCall(VecDestroy(&c->g_lin));
  PetscCall(VecDestroy(&c->g));
  PetscCall(PetscSFDestroy(&c->x_end_bcast));
  PetscCall(VecDestroy(&c->x_end));
  PetscCall(VecScatterDestroy(&c->x_next_scatter));
  PetscCall(VecDestroy(&c->x_next));
  PetscCall(VecDestroy(&c->x));
  PetscCall(MatDestroy(&c->H));
  PetscCall(PetscLayoutDestroy(&c->layout));
  PetscCall(PetscFree(c));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  AppCtx   ctx;
  MPI_Comm comm;
  Tao      tao;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;
  PetscCall(AppCtxCreate(comm, &ctx));
  PetscCall(AppCtxSetFromOptions(ctx));
  PetscCall(AppCtxSetUp(ctx));
  PetscCall(AppCtxCreateTao(ctx, &tao));
  PetscCall(TaoSolve(tao));
  PetscCall(TaoDestroy(&tao));
  PetscCall(AppCtxDestroy(&ctx));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 0

TEST*/
