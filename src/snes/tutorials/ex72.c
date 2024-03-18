static char help[] = "Stokes Problem with no-flow BC in an annulus in 2D and 3D,\n\
discretized with finite elements using a parallel unstructured mesh (DMPLEX) to represent the domain.\n\n\n";

#include<petscsnes.h>
#include<petscdmplex.h>
#include<petscds.h>

static void f1_u(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  const PetscReal mu = 1.0;//PetscRealPart(constants[0]);
  const PetscInt  Nc = uOff[1] - uOff[0];
  PetscInt        c, d;

  for (c = 0; c < Nc; ++c) {
    for (d = 0; d < dim; ++d) f1[c * dim + d] = mu * (u_x[c * dim + d] + u_x[d * dim + c]);
    f1[c * dim + c] -= u[uOff[1]];
  }
}

static void f0_p(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  PetscInt d;
  for (d = 0, f0[0] = 0.0; d < dim; ++d) f0[0] -= u_x[d * dim + d];
}

static void g1_pu(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g1[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) g1[d * dim + d] = -1.0; /* < q, -\nabla\cdot u > */
}

static void g2_up(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g2[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) g2[d * dim + d] = -1.0; /* -< \nabla\cdot v, p > */
}

static void g3_uu(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  const PetscReal mu = 1.0;//PetscRealPart(constants[0]);
  const PetscInt  Nc = uOff[1] - uOff[0];
  PetscInt        c, d;

  for (c = 0; c < Nc; ++c) {
    for (d = 0; d < dim; ++d) {
      g3[((c * Nc + c) * dim + d) * dim + d] += mu; /* < \nabla v, \nabla u > */
      g3[((c * Nc + d) * dim + d) * dim + c] += mu; /* < \nabla v, {\nabla u}^T > */
    }
  }
}

static void g0_pp(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  const PetscReal mu = 1.0;//PetscRealPart(constants[0]);

  g0[0] = 1.0 / mu;
}

/*
  We would like an exact solution without a rotaational null mode. Thus we want to fix the flow on the inner
  boundary and have free slip on the outer boundary. The flow law is isoviscous Stokes,

  -\Delta u + \nabla p = f
  \nabla\cdot u        = 0

  and in weak form

  < \nabla v, \nabla u + {\nabla u}^T > - < \nabla\cdot v, p > - < v, f > = 0
  < q, -\nabla\cdot u >                                                   = 0

  The velocity will only be a function of r, so we will write things in polar coordinates

  <-1/r d/dr (r du_r/dr), -1/r d/dr (r du_th/dr)> + <dp/dr, 0> = f
  du_r/dr + u_r/r                                              = 0

  We are led to

  u_r  = 0          u_x = c x \sqrt{x^2 + y^2}
  u_th = c r^2 - c  u_y = c y \sqrt{x^2 + y^2}
  p    = 0
  f_th = -4 c       f_x = -4 c x / r
                    f_y = -4 c y / r
  since

  <0, -4 c> + <0, 0> = <0, -4 c>
  0 + 0              = 0

*/

static PetscErrorCode zero_u(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  u[0] = 0.;
  u[1] = 0.;
  return PETSC_SUCCESS;
}

static PetscErrorCode swirl_u(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  const PetscReal r = PetscSqrtReal(x[0] * x[0] + x[1] * x[1]);

  //u[0] = x[0] * r;
  //u[1] = x[1] * r;
  u[0] = 0.0;
  u[1] = r * r - 1.;
  return PETSC_SUCCESS;
}

static PetscErrorCode swirl_p(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  u[0] = 0.;
  return PETSC_SUCCESS;
}

static void f0_swirl_u(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const PetscReal r = PetscSqrtReal(x[0] * x[0] + x[1] * x[1]);

  f0[0] = -4. * x[0] / r;
  f0[1] = -4. * x[1] / r;
}

static PetscErrorCode CreateMesh(MPI_Comm comm, DM *dm)
{
  PetscFunctionBeginUser;
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetupEqn(DM dm)
{
  PetscDS  ds;
  DMLabel  label;
  PetscInt id, comp;

  PetscFunctionBeginUser;
  PetscCall(DMGetDS(dm, &ds));

  PetscCall(PetscDSSetResidual(ds, 0, f0_swirl_u, f1_u));
  PetscCall(PetscDSSetResidual(ds, 1, f0_p, NULL));
  PetscCall(PetscDSSetJacobian(ds, 0, 0, NULL, NULL, NULL, g3_uu));
  PetscCall(PetscDSSetJacobian(ds, 0, 1, NULL, NULL, g2_up, NULL));
  PetscCall(PetscDSSetJacobian(ds, 1, 0, NULL, g1_pu, NULL, NULL));
  PetscCall(PetscDSSetJacobianPreconditioner(ds, 0, 0, NULL, NULL, NULL, g3_uu));
  PetscCall(PetscDSSetJacobianPreconditioner(ds, 1, 1, g0_pp, NULL, NULL, NULL));

  PetscCall(PetscDSSetExactSolution(ds, 0, swirl_u, NULL));
  PetscCall(PetscDSSetExactSolution(ds, 1, swirl_p, NULL));

  PetscCall(DMGetLabel(dm, "marker", &label));
  id = 1;
  PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "inner wall", label, 1, &id, 0, 0, NULL, (void (*)(void))zero_u, NULL, NULL, NULL));
  id   = 3;
  comp = 0;
  PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "outer wall", label, 1, &id, 0, 1, &comp, (void (*)(void))zero_u, NULL, NULL, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetupProblem(DM dm)
{
  DM              cdm = dm;
  PetscQuadrature q   = NULL;
  PetscFE         fe;
  DMPolytopeType  ct;
  PetscInt        dim, cStart;

  PetscFunctionBegin;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, NULL));
  PetscCall(DMPlexGetCellType(dm, cStart, &ct));

  PetscCall(PetscFECreateByCell(PETSC_COMM_SELF, dim, dim, ct, "vel_", -1, &fe));
  PetscCall(PetscObjectSetName((PetscObject)fe, "velocity"));
  PetscCall(PetscFEGetQuadrature(fe, &q));
  PetscCall(DMSetField(dm, 0, NULL, (PetscObject)fe));
  PetscCall(PetscFEDestroy(&fe));

  PetscCall(PetscFECreateByCell(PETSC_COMM_SELF, dim, 1, ct, "pres_", -1, &fe));
  PetscCall(PetscObjectSetName((PetscObject)fe, "pressure"));
  PetscCall(PetscFESetQuadrature(fe, q));
  PetscCall(DMSetField(dm, 1, NULL, (PetscObject)fe));
  PetscCall(PetscFEDestroy(&fe));

  PetscCall(DMCreateDS(dm));
  PetscCall(SetupEqn(dm));
  while (cdm) {
    PetscCall(DMCopyDisc(dm, cdm));
    PetscCall(DMGetCoarseDM(cdm, &cdm));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  SNES snes;
  DM   dm;
  Vec  u;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(CreateMesh(PETSC_COMM_WORLD, &dm));
  PetscCall(SNESCreate(PetscObjectComm((PetscObject)dm), &snes));
  PetscCall(SNESSetDM(snes, dm));

  PetscCall(SetupProblem(dm));
  PetscCall(DMPlexCreateBasisSpherical(dm));
  PetscCall(DMPlexCreateClosureIndex(dm, NULL));

  PetscCall(DMCreateGlobalVector(dm, &u));
  PetscCall(PetscObjectSetName((PetscObject)u, "Solution"));
  PetscCall(DMPlexSetSNESLocalFEM(dm, PETSC_FALSE, NULL));
  PetscCall(SNESSetFromOptions(snes));
  PetscCall(DMSNESCheckFromOptions(snes, u));
  PetscCall(SNESSolve(snes, NULL, u));

  PetscCall(VecDestroy(&u));
  PetscCall(SNESDestroy(&snes));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 2d_p2_p1_check
    requires: triangle
    args: -dm_plex_shape annulus -dm_plex_simplex 0 -dm_plex_separate_marker \
            -dm_plex_box_faces 6,3 -dm_plex_box_lower 0.,1. -dm_plex_box_upper 1.,4.\
          -vel_petscspace_degree 2 -pres_petscspace_degree 1 \
          -dmsnes_check 0.0001

TEST*/
