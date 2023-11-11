static char help[] = "Poisson Problem d finite elements.\n\
We solve the Poisson problem discretized with finite elements\n\
using a hierarchy of parallel unstructured meshes (DMPLEX).\n\
This example supports automatic convergence estimation.\n\n\n";

/*
  This example can be run using the test system:

    make -f ./gmakefile test search="snes_tutorials-plexGMG_*"

  You can see the help using

    make -f ./gmakefile test search="snes_tutorials-plexGMG_*" EXTRA_OPTIONS="-help"

  The mesh can be viewed using

    make -f ./gmakefile test search="snes_tutorials-plexGMG_*" EXTRA_OPTIONS="-dm_view"

  and many formats are supported such as VTK, HDF5, and drawing,

    make -f ./gmakefile test search="snes_tutorials-plexGMG_*" EXTRA_OPTIONS="-dm_view draw -draw_pause -1"

  We can visualize the solution using HDF5 and XDMF

    make -f ./gmakefile test search="snes_tutorials-plexGMG_*" EXTRA_OPTIONS="-dm_view hdf5:poisson.h5 -potential_view hdf5:poisson.h5::append"
    ${PETSC_DIR}/lib/petsc/bin/petsc_gen_xdmf.py poisson.h5

  We can visualize the error using X-windows (for example)

    make -f ./gmakefile test search="snes_tutorials-plexGMG_*" EXTRA_OPTIONS="-error_view -error_vec_view draw -draw_pause -1"

  We can check the consistency of the finite element using

    make -f ./gmakefile test search="snes_tutorials-plexGMG_*" EXTRA_OPTIONS="-snes_convergence_estimate -convest_num_refine 3 -convest_monitor"

  We can check that the GMG takes a constant number of iterates as the problem size is increased

    make -f ./gmakefile test search="snes_tutorials-plexGMG_*" EXTRA_OPTIONS="-snes_monitor -ksp_converged_reason -snes_convergence_estimate -convest_num_refine 3 -convest_monitor"
*/

#include <petscdmplex.h>
#include <petscds.h>
#include <petscsnes.h>

typedef struct {
  PetscBool viewError; // Output the solution error
  PetscBool userOp;    // Use a user-defined operator
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscFunctionBeginUser;
  options->viewError = PETSC_FALSE;

  PetscOptionsBegin(comm, "", "Poisson Problem Options", "DMPLEX");
  PetscCall(PetscOptionsBool("-error_view", "Output the solution error", "plexGMG.c", options->viewError, &options->viewError, NULL));
  PetscCall(PetscOptionsBool("-user_operator", "Construct GMG with a user-defined operator", "plexGMG.c", options->userOp, &options->userOp, NULL));
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscFunctionBeginUser;
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMSetApplicationContext(*dm, user));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode trig_inhomogeneous_u(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  PetscInt d;
  *u = 0.0;
  for (d = 0; d < dim; ++d) *u += PetscSinReal(2.0 * PETSC_PI * x[d]);
  return PETSC_SUCCESS;
}

static void f0_trig_inhomogeneous_u(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) f0[0] += -4.0 * PetscSqr(PETSC_PI) * PetscSinReal(2.0 * PETSC_PI * x[d]);
}

static void f1_u(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) f1[d] = u_x[d];
}

static void g3_uu(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) g3[d * dim + d] = 1.0;
}

static PetscErrorCode SetupPrimalProblem(DM dm, AppCtx *user)
{
  PetscDS              ds;
  DMLabel              label;
  const PetscInt       id = 1;
  PetscPointFunc       f0 = f0_trig_inhomogeneous_u;
  PetscSimplePointFunc ex = trig_inhomogeneous_u;

  PetscFunctionBeginUser;
  PetscCall(DMGetDS(dm, &ds));
  PetscCall(PetscDSSetResidual(ds, 0, f0, f1_u));
  PetscCall(PetscDSSetJacobian(ds, 0, 0, NULL, NULL, NULL, g3_uu));
  PetscCall(PetscDSSetExactSolution(ds, 0, ex, user));
  PetscCall(DMGetLabel(dm, "marker", &label));
  if (label) PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", label, 1, &id, 0, 0, NULL, (void (*)(void))ex, NULL, user, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetupDiscretization(DM dm, const char name[], PetscErrorCode (*setupProblem)(DM, AppCtx *), AppCtx *user)
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
  PetscCall(PetscSNPrintf(prefix, PETSC_MAX_PATH_LEN, "%s_", name));
  PetscCall(PetscFECreateByCell(PETSC_COMM_SELF, dim, 1, ct, name ? prefix : NULL, -1, &fe));
  PetscCall(PetscObjectSetName((PetscObject)fe, name));
  PetscCall(DMSetField(dm, 0, NULL, (PetscObject)fe));
  PetscCall(DMCreateDS(dm));
  PetscCall((*setupProblem)(dm, user));
  while (cdm) {
    PetscCall(DMCopyDisc(dm, cdm));
    PetscCall(DMGetCoarseDM(cdm, &cdm));
  }
  PetscCall(PetscFEDestroy(&fe));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ErrorView(Vec u, AppCtx *user)
{
  DM                   dm, edm;
  PetscDS              ds;
  PetscFE              fe;
  Vec                  error;
  PetscSimplePointFunc sol;
  void                *ctx;
  DMPolytopeType       ct;
  PetscInt             dim, cStart;

  PetscFunctionBegin;
  if (!user->viewError) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(VecGetDM(u, &dm));
  PetscCall(DMGetDS(dm, &ds));
  PetscCall(PetscDSGetExactSolution(ds, 0, &sol, &ctx));

  PetscCall(DMClone(dm, &edm));
  PetscCall(DMGetDimension(edm, &dim));
  PetscCall(DMPlexGetHeightStratum(edm, 0, &cStart, NULL));
  PetscCall(DMPlexGetCellType(edm, cStart, &ct));
  PetscCall(PetscFECreateLagrangeByCell(PETSC_COMM_SELF, dim, 1, ct, 0, -1, &fe));
  PetscCall(PetscObjectSetName((PetscObject)fe, "Error"));
  PetscCall(DMSetField(edm, 0, NULL, (PetscObject)fe));
  PetscCall(PetscFEDestroy(&fe));
  PetscCall(DMCreateDS(edm));
  PetscCall(DMCreateGlobalVector(edm, &error));
  PetscCall(DMDestroy(&edm));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)error, "error_"));

  PetscCall(DMPlexComputeL2DiffVec(dm, 0.0, &sol, &ctx, u, error));
  PetscCall(VecViewFromOptions(error, NULL, "-vec_view"));
  PetscCall(VecDestroy(&error));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM     dm;   // DMPLEX mesh
  Vec    u;    // The solution vector
  SNES   snes; // The solver
  AppCtx user; // User-defined work context

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &user));
  PetscCall(CreateMesh(PETSC_COMM_WORLD, &user, &dm));
  PetscCall(SetupDiscretization(dm, "potential", SetupPrimalProblem, &user));
  PetscCall(DMCreateGlobalVector(dm, &u));
  PetscCall(PetscObjectSetName((PetscObject)u, "Potential"));
  PetscCall(VecSet(u, 0.0));

  PetscCall(SNESCreate(PETSC_COMM_WORLD, &snes));
  PetscCall(SNESSetDM(snes, dm));
  PetscCall(SNESSetFromOptions(snes));
  PetscCall(DMPlexSetSNESLocalFEM(dm, &user, &user, &user));
  if (user.userOp) {
    DM cdm;
    KSP ksp;
    PC pc;
    PetscInt Nl = 0, l;

    PetscCall(SNESSetUp(snes));
    PetscCall(SNESGetKSP(snes, &ksp));
    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(PCSetType(pc, PCMG));
    cdm = dm;
    while (cdm) {
      ++Nl;
      PetscCall(DMGetCoarseDM(cdm, &cdm));
    }
    PetscCall(PCMGSetLevels(pc, Nl, NULL));
    // Set operator on each level
    cdm = dm;
    l   = Nl - 1;
    while (cdm) {
      DM  odm = cdm;
      Mat A;

      PetscCall(DMCreateMatrix(cdm, &A));
      PetscCall(DMPlexSNESComputeJacobianFEM(cdm, u, A, A, &user));
      PetscCall(PCMGSetOperators(pc, l, A, A));
      PetscCall(MatDestroy(&A));
      PetscCall(DMGetCoarseDM(cdm, &cdm));
      if (cdm) {
        KSP smoother;
        Mat Interp;
        Vec scale;

        PetscCall(DMCreateInterpolation(cdm, odm, &Interp, &scale));
        PetscCall(PCMGSetInterpolation(pc, l, Interp));
        PetscCall(PCMGSetRScale(pc, l, scale));
        PetscCall(MatDestroy(&Interp));
        PetscCall(VecDestroy(&scale));

        PetscCall(PCMGGetSmoother(pc, l - 1, &smoother));
        PetscCall(KSPSetDM(smoother, cdm));
        PetscCall(KSPSetDMActive(smoother, PETSC_FALSE));
      }
      --l;
    }
    PetscCall(PCSetFromOptions(pc));
  }
  PetscCall(SNESSolve(snes, NULL, u));
  PetscCall(SNESGetSolution(snes, &u));

  PetscCall(VecViewFromOptions(u, NULL, "-potential_view"));
  PetscCall(ErrorView(u, &user));
  PetscCall(SNESDestroy(&snes));
  PetscCall(VecDestroy(&u));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 2d_p1_gmg_vcycle
    requires: triangle
    args: -potential_petscspace_degree 1 -dm_plex_box_faces 2,2 -dm_refine_hierarchy 3 \
          -ksp_rtol 1e-10 -pc_type mg \
            -mg_levels_ksp_max_it 1 \
            -mg_levels_esteig_ksp_type cg \
            -mg_levels_esteig_ksp_max_it 10 \
            -mg_levels_ksp_chebyshev_esteig 0,0.1,0,1.1 \
            -mg_levels_pc_type jacobi

  test:
    suffix: 2d_p1_gmg_vcycle_op
    requires: triangle
    args: -potential_petscspace_degree 1 -dm_plex_box_faces 2,2 -dm_refine_hierarchy 3 \
          -ksp_rtol 1e-10 -user_operator \
            -mg_levels_ksp_max_it 1 \
            -mg_levels_esteig_ksp_type cg \
            -mg_levels_esteig_ksp_max_it 10 \
            -mg_levels_ksp_chebyshev_esteig 0,0.1,0,1.1 \
            -mg_levels_pc_type jacobi

TEST*/
