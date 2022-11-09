static char help[] = "Mixed element discretization of the Poisson equation.\n\n\n";

#include <petscdmplex.h>
#include <petscds.h>
#include <petscsnes.h>
#include <petscconvest.h>

/*
The Poisson equation

  -\Delta\phi = f

can be rewritten in first order form

  q - \nabla\phi  &= 0
  -\nabla \cdot q &= f
*/

typedef enum {SOL_CONST, SOL_LINEAR, SOL_QUADRATIC, SOL_TRIG, NUM_SOL_TYPES} SolType;
static const char *solTypes[] = {"const", "linear", "quadratic", "trig"};

typedef struct {
  SolType solType; // MMS solution type
} AppCtx;

// SOLUTION CONST: \phi = 1, q = 0, f = 0
static PetscErrorCode const_phi(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  *u = 1.0;
  return 0;
}

static PetscErrorCode const_q(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  for (PetscInt d = 0; d < dim; ++d) u[d] = 0.0;
  return 0;
}

// SOLUTION LINEAR: \phi = 2y, q = <0, 2>, f = 0
static PetscErrorCode linear_phi(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  u[0] = 2. * x[1];
  return 0;
}

static PetscErrorCode linear_q(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  u[0] = 0.;
  u[1] = 2.;
  return 0;
}

// SOLUTION QUADRATIC: \phi = x (2\pi - x) + (1 + y) (1 - y), q = <2\pi - 2 x, - 2 y> = <2\pi, 0> - 2 x, f = -4
static PetscErrorCode quadratic_phi(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  u[0] = x[0] * (6.283185307179586 - x[0]) + (1. + x[1]) * (1. - x[1]);
  return 0;
}

static PetscErrorCode quadratic_q(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  u[0] = 6.283185307179586 - 2. * x[0];
  u[1] = -2. * x[1];
  return 0;
}

static PetscErrorCode quadratic_q_bc(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  u[0] = x[1] > 0. ? -2. * x[1] : 2. * x[1];
  return 0;
}

static void f0_quadratic_phi(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  for (PetscInt d = 0; d < dim; ++d) f0[0] -= -2.0;
}

// SOLUTION TRIG: \phi = sin(x) + (1/3 - y^2), q = <cos(x), -2 y>, f = -sin(x) - 2
static PetscErrorCode trig_phi(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  u[0] = PetscSinReal(x[0]) + (1./3. - x[1]*x[1]);
  return 0;
}

static PetscErrorCode trig_q(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  u[0] = PetscCosReal(x[0]);
  u[1] = -2. * x[1];
  return 0;
}

static PetscErrorCode trig_q_bc(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  u[0] = x[1] > 0. ? -2. * x[1] : 2. * x[1];
  return 0;
}

static void f0_trig_phi(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] -= -PetscSinReal(x[0]);// - 2.;
}

static void f0_q(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  for (PetscInt d = 0; d < dim; ++d) f0[d] += u[uOff[0] + d];
}

static void f1_q(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  for (PetscInt d = 0; d < dim; ++d) f1[d * dim + d] = u[uOff[1]];
}

static void f0_phi(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  for (PetscInt d = 0; d < dim; ++d) f0[0] += u_x[uOff_x[0] + d * dim + d];
}

static void g0_qq(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  for (PetscInt d = 0; d < dim; ++d) g0[d * dim + d] = 1.0;
}

static void g2_qphi(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g2[])
{
  for (PetscInt d = 0; d < dim; ++d) g2[d * dim + d] = 1.0;
}

static void g1_phiq(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g1[])
{
  for (PetscInt d = 0; d < dim; ++d) g1[d * dim + d] = 1.0;
}

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscInt sol;

  PetscFunctionBeginUser;
  options->solType = SOL_CONST;

  PetscOptionsBegin(comm, "", "Mixed Poisson Options", "DMPLEX");
  sol = options->solType;
  PetscCall(PetscOptionsEList("-sol_type", "The MMS solution type", "ex12.c", solTypes, NUM_SOL_TYPES, solTypes[sol], &sol, NULL));
  options->solType = (SolType)sol;
  PetscOptionsEnd();
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscFunctionBeginUser;
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMSetApplicationContext(*dm, user));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupPrimalProblem(DM dm, AppCtx *user)
{
  PetscDS        ds;
  PetscWeakForm  wf;
  DMLabel        label;
  const PetscInt id = 1;

  PetscFunctionBeginUser;
  PetscCall(DMGetDS(dm, &ds));
  PetscCall(PetscDSGetWeakForm(ds, &wf));
  PetscCall(DMGetLabel(dm, "marker", &label));
  PetscCall(PetscDSSetResidual(ds, 0, f0_q, f1_q));
  PetscCall(PetscDSSetResidual(ds, 1, f0_phi, NULL));
  PetscCall(PetscDSSetJacobian(ds, 0, 0, g0_qq, NULL, NULL, NULL));
  PetscCall(PetscDSSetJacobian(ds, 0, 1, NULL, NULL, g2_qphi, NULL));
  PetscCall(PetscDSSetJacobian(ds, 1, 0, NULL, g1_phiq, NULL, NULL));
  switch (user->solType) {
    case SOL_CONST:
      PetscCall(PetscDSSetExactSolution(ds, 0, const_q, user));
      PetscCall(PetscDSSetExactSolution(ds, 1, const_phi, user));
      break;
    case SOL_LINEAR:
      PetscCall(PetscDSSetExactSolution(ds, 0, linear_q, user));
      PetscCall(PetscDSSetExactSolution(ds, 1, linear_phi, user));
      break;
    case SOL_QUADRATIC:
      PetscCall(PetscWeakFormAddResidual(wf, NULL, 0, 1, 0, f0_quadratic_phi, NULL));
      PetscCall(PetscDSSetExactSolution(ds, 0, quadratic_q, user));
      PetscCall(PetscDSSetExactSolution(ds, 1, quadratic_phi, user));
      PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", label, 1, &id, 0, 0, NULL, (void (*)(void))quadratic_q_bc, NULL, user, NULL));
      break;
    case SOL_TRIG:
      PetscCall(PetscWeakFormAddResidual(wf, NULL, 0, 1, 0, f0_trig_phi, NULL));
      PetscCall(PetscDSSetExactSolution(ds, 0, trig_q, user));
      PetscCall(PetscDSSetExactSolution(ds, 1, trig_phi, user));
      PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", label, 1, &id, 0, 0, NULL, (void (*)(void))trig_q_bc, NULL, user, NULL));
      break;
    default: SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Invalid solution type: %" PetscInt_FMT, user->solType);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupDiscretization(DM dm, PetscInt Nf, const char *names[], PetscErrorCode (*setup)(DM, AppCtx *), AppCtx *user)
{
  DM             cdm = dm;
  PetscFE        fe;
  DMPolytopeType ct;
  PetscInt       dim, cStart;
  char           prefix[PETSC_MAX_PATH_LEN];

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, NULL));
  PetscCall(DMPlexGetCellType(dm, cStart, &ct));
  for (PetscInt f = 0; f < Nf; ++f) {
    PetscCall(PetscSNPrintf(prefix, PETSC_MAX_PATH_LEN, "%s_", names[f]));
    PetscCall(PetscFECreateByCell(PETSC_COMM_SELF, dim, 1, ct, prefix, -1, &fe));
    PetscCall(PetscObjectSetName((PetscObject)fe, names[f]));
    if (f > 0) {
      PetscFE fe0;

      PetscCall(DMGetField(dm, 0, NULL, (PetscObject *)&fe0));
      PetscCall(PetscFECopyQuadrature(fe0, fe));
    }
    PetscCall(DMSetField(dm, f, NULL, (PetscObject)fe));
    PetscCall(PetscFEDestroy(&fe));
  }
  PetscCall(DMCreateDS(dm));
  PetscCall((*setup)(dm, user));
  while (cdm) {
    PetscCall(DMCopyDisc(dm, cdm));
    PetscCall(DMGetCoarseDM(cdm, &cdm));
  }
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM          dm;
  SNES        snes;
  Vec         u;
  AppCtx      user;
  const char *names[] = {"q", "phi"};

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &user));
  PetscCall(CreateMesh(PETSC_COMM_WORLD, &user, &dm));
  PetscCall(SNESCreate(PETSC_COMM_WORLD, &snes));
  PetscCall(SNESSetDM(snes, dm));
  PetscCall(SetupDiscretization(dm, 2, names, SetupPrimalProblem, &user));
  PetscCall(DMCreateGlobalVector(dm, &u));
  PetscCall(PetscObjectSetName((PetscObject)u, "solution"));
  PetscCall(SNESSetFromOptions(snes));
  PetscCall(DMPlexSetSNESLocalFEM(dm, &user, &user, &user));
  PetscCall(DMSNESCheckFromOptions(snes, u));
  PetscCall(SNESSolve(snes, NULL, u));
  PetscCall(VecDestroy(&u));
  PetscCall(SNESDestroy(&snes));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  # RT1-P0 on quads
  testset:
    args: -dm_plex_simplex 0 -dm_plex_box_bd periodic,none -dm_plex_box_faces 3,1 \
            -dm_plex_box_lower 0,-1 -dm_plex_box_upper 6.283185307179586,1 \
          -phi_petscspace_degree 0 \
          -phi_petscdualspace_lagrange_use_moments \
          -phi_petscdualspace_lagrange_moment_order 2 \
          -q_petscfe_default_quadrature_order 1 \
          -q_petscspace_type sum \
          -q_petscspace_variables 2 \
          -q_petscspace_components 2 \
          -q_petscspace_sum_spaces 2 \
          -q_petscspace_sum_concatenate true \
          -q_sumcomp_0_petscspace_variables 2 \
          -q_sumcomp_0_petscspace_type tensor \
          -q_sumcomp_0_petscspace_tensor_spaces 2 \
          -q_sumcomp_0_petscspace_tensor_uniform false \
          -q_sumcomp_0_tensorcomp_0_petscspace_degree 1 \
          -q_sumcomp_0_tensorcomp_1_petscspace_degree 0 \
          -q_sumcomp_1_petscspace_variables 2 \
          -q_sumcomp_1_petscspace_type tensor \
          -q_sumcomp_1_petscspace_tensor_spaces 2 \
          -q_sumcomp_1_petscspace_tensor_uniform false \
          -q_sumcomp_1_tensorcomp_0_petscspace_degree 0 \
          -q_sumcomp_1_tensorcomp_1_petscspace_degree 1 \
          -q_petscdualspace_form_degree -1 \
          -q_petscdualspace_order 1 \
          -q_petscdualspace_lagrange_trimmed true \
          -ksp_error_if_not_converged \
          -pc_type fieldsplit -pc_fieldsplit_type schur \
            -pc_fieldsplit_schur_fact_type full -pc_fieldsplit_schur_precondition full \
            -fieldsplit_q_pc_type lu -fieldsplit_phi_pc_type svd

    # The Jacobian test is meaningless here
    test:
          suffix: quad_hdiv_0
          args: -dmsnes_check
          filter: sed -e "s/Taylor approximation converging at order.*''//"

    # The Jacobian test is meaningless here
    test:
          suffix: quad_hdiv_1
          args: -sol_type linear -dmsnes_check
          filter: sed -e "s/Taylor approximation converging at order.*''//"

    test:
          suffix: quad_hdiv_2
          args: -sol_type quadratic -dmsnes_check

    test:
          suffix: quad_hdiv_3
          args: -sol_type trig

TEST*/
