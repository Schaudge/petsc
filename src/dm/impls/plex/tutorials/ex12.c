static char help[] = "Test metric utils in the uniform, isotropic case.\n\n";

#include <petscdmplex.h>

static PetscErrorCode bowl(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  PetscInt d;

  *u = 0.0;
  for (d = 0; d < dim; d++) *u += 0.5 * (x[d] - 0.5) * (x[d] - 0.5);

  return PETSC_SUCCESS;
}

static PetscErrorCode CreateIndicator(DM dm, Vec *indicator, DM *dmIndi)
{
  MPI_Comm comm;
  PetscFE  fe;
  PetscInt dim;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  PetscCall(DMClone(dm, dmIndi));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(PetscFECreateLagrange(comm, dim, 1, PETSC_TRUE, 1, PETSC_DETERMINE, &fe));
  PetscCall(DMSetField(*dmIndi, 0, NULL, (PetscObject)fe));
  PetscCall(DMCreateDS(*dmIndi));
  PetscCall(PetscFEDestroy(&fe));
  PetscCall(DMCreateLocalVector(*dmIndi, indicator));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM        dm, dmAdapt;
  DMLabel   bdLabel = NULL, rgLabel = NULL;
  MPI_Comm  comm;
  PetscBool uniform = PETSC_FALSE, isotropic = PETSC_FALSE;
  PetscInt  dim;
  PetscReal scaling = 1.0;
  Vec       metric;

  /* Set up */
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;

  /* Create box mesh */
  PetscCall(DMCreate(comm, &dm));
  PetscCall(DMSetType(dm, DMPLEX));
  PetscCall(DMSetOptionsPrefix(dm, "init_"));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(PetscObjectSetName((PetscObject)dm, "DM_init"));
  PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));
  PetscCall(DMGetDimension(dm, &dim));

  /* Construct metric */
  PetscCall(DMPlexMetricSetFromOptions(dm));
  PetscCall(DMPlexMetricIsUniform(dm, &uniform));
  PetscCall(DMPlexMetricIsIsotropic(dm, &isotropic));
  if (uniform) {
    PetscCall(DMPlexMetricCreateUniform(dm, 0, scaling, &metric));
  } else {
    DM  dmIndi;
    Vec indicator;

    /* Construct "error indicator" */
    PetscCall(CreateIndicator(dm, &indicator, &dmIndi));
    if (isotropic) {
      /* Isotropic case: just specify unity */
      PetscCall(VecSet(indicator, scaling));
      PetscCall(DMPlexMetricCreateIsotropic(dm, 0, indicator, &metric));
    } else {
      PetscFE fe;

      /* 'Anisotropic' case: approximate the identity by recovering the Hessian of a parabola */
      DM dmGrad;
      PetscErrorCode (*funcs[1])(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar *, void *) = {bowl};
      Vec gradient;

      /* Project the parabola into P1 space */
      PetscCall(DMProjectFunctionLocal(dmIndi, 0.0, funcs, NULL, INSERT_ALL_VALUES, indicator));

      /* Approximate the gradient */
      PetscCall(DMClone(dmIndi, &dmGrad));
      PetscCall(PetscFECreateLagrange(comm, dim, dim, PETSC_TRUE, 1, PETSC_DETERMINE, &fe));
      PetscCall(DMSetField(dmGrad, 0, NULL, (PetscObject)fe));
      PetscCall(DMCreateDS(dmGrad));
      PetscCall(PetscFEDestroy(&fe));
      PetscCall(DMCreateLocalVector(dmGrad, &gradient));
      PetscCall(DMPlexComputeGradientClementInterpolant(dmIndi, indicator, gradient));
      PetscCall(VecViewFromOptions(gradient, NULL, "-adapt_gradient_view"));

      /* Approximate the Hessian */
      PetscCall(DMPlexMetricCreate(dm, 0, &metric));
      PetscCall(DMPlexComputeGradientClementInterpolant(dmGrad, gradient, metric));
      PetscCall(VecViewFromOptions(metric, NULL, "-adapt_hessian_view"));
      PetscCall(VecDestroy(&gradient));
      PetscCall(DMDestroy(&dmGrad));
    }
    PetscCall(VecDestroy(&indicator));
    PetscCall(DMDestroy(&dmIndi));
  }
  {
    DM  dmDet;
    Vec determinant, metric1;

    PetscCall(VecDuplicate(metric, &metric1));
    PetscCall(DMPlexMetricDeterminantCreate(dm, 0, &determinant, &dmDet));
    PetscCall(DMPlexMetricEnforceSPD(dm, metric, PETSC_TRUE, PETSC_TRUE, metric1, determinant));
    PetscCall(VecCopy(metric1, metric));
    PetscCall(VecDestroy(&metric1));
    PetscCall(VecDestroy(&determinant));
    PetscCall(DMDestroy(&dmDet));
    PetscCall(VecViewFromOptions(metric, NULL, "-adapt_metric_view"));
  }

  /* Adapt the mesh */
  PetscCall(DMAdaptMetric(dm, metric, bdLabel, rgLabel, &dmAdapt));
  PetscCall(VecDestroy(&metric));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscObjectSetName((PetscObject)dmAdapt, "DM_adapted"));
  PetscCall(DMSetOptionsPrefix(dmAdapt, "adapt_"));
  PetscCall(DMViewFromOptions(dmAdapt, NULL, "-dm_view"));

  /* Clean up */
  PetscCall(DMDestroy(&dmAdapt));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    requires: mmg
    args: -init_dm_adaptor mmg -init_dm_plex_filename LocalMeshUTM15.dat -dm_plex_metric_hausdorff_number 1e-10 \
          -init_dm_plex_scale 0.00002 -dm_plex_metric_verbosity 10

    test:
      suffix: uniform_2d_mmg
      args: -dm_plex_metric_uniform -dm_plex_metric_h_max 0.01
    test:
      suffix: iso_2d_mmg
      args: -dm_plex_metric_isotropic
    test:
      suffix: hessian_2d_mmg

TEST*/
