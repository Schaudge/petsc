const char help[] = "Test Berend's example";

#include <petscdmplex.h>
#include <petscdmforest.h>

int main(int argc, char **argv)
{
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  MPI_Comm comm = PETSC_COMM_WORLD;
  PetscInt dim = 3;
  PetscInt cells_per_dir[] = {3, 3, 3};
  PetscReal dir_min[] = {0.0, 0.0, 0.0};
  PetscReal dir_max[] = {1.0, 1.0, 1.0};
  DMBoundaryType bcs[] = {DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE};

  DM forest;
  PetscCall(DMCreate(comm, &forest));
  PetscCall(DMSetType(forest, DMP8EST));
  {
    DM dm_base;
    PetscCall(DMPlexCreateBoxMesh(comm
          , dim
          , PETSC_FALSE /* simplex */
          , cells_per_dir
          , dir_min
          , dir_max
          , bcs
          , PETSC_TRUE /* interpolate */
          , &dm_base));

    PetscCall(DMViewFromOptions(dm_base, NULL, "-dm_base_view"));
    PetscCall(DMCopyFields(dm_base, forest));
    PetscCall(DMForestSetBaseDM(forest, dm_base));
    PetscCall(DMDestroy(&dm_base));
  }
  PetscCall(DMSetFromOptions(forest));
  PetscCall(DMSetUp(forest));

  PetscCall(DMViewFromOptions(forest, NULL, "-dm_forest_view"));

  DM plex;

  PetscCall(DMConvert(forest, DMPLEX, &plex));
  
  PetscInt numFields = 4;
  PetscInt numComp[4] = {1, 1, 1, 1};
  PetscInt numDof[16] = {0};
  for (PetscInt i = 0; i < 4; i++) numDof[i * (dim + 1) + dim] = 1;

  PetscCall(DMSetNumFields(plex, numFields));

  PetscSection section;
  PetscCall(DMPlexCreateSection(plex, NULL, numComp, numDof, 0, NULL, NULL, NULL, NULL, &section));

  const char *names[] = {"field 0", "field 1", "field 2", "field 3"};
  for (PetscInt i = 0; i < 4; i++) PetscCall(PetscSectionSetFieldName(section, i, names[i]));

  PetscCall(DMSetLocalSection(plex, section));
  PetscCall(PetscSectionDestroy(&section));

  PetscFE fe;
  PetscCall(PetscFECreateDefault(comm, 3, 1, PETSC_FALSE, NULL, PETSC_DEFAULT, &fe));
  for (PetscInt i = 0; i < 4; i++) {
    PetscCall(DMSetField(plex, i, NULL, (PetscObject)fe));
    PetscCall(DMSetField(forest, i, NULL, (PetscObject)fe));
  }
  PetscCall(PetscFEDestroy(&fe));

  PetscCall(DMCreateDS(plex));
  PetscCall(DMCreateDS(forest));

  Vec g_vec, l_vec;
  PetscCall(DMCreateGlobalVector(plex, &g_vec));
  PetscCall(VecSet(g_vec,1.0));
  PetscCall(DMCreateLocalVector(plex, &l_vec));
  PetscCall(DMGlobalToLocal(plex, g_vec, INSERT_VALUES, l_vec));
  PetscCall(VecViewFromOptions(l_vec, NULL, "-local_vec_view"));
  PetscCall(VecDestroy(&l_vec));
  PetscCall(VecDestroy(&g_vec));

  PetscCall(DMViewFromOptions(forest, NULL, "-dm_plex_view"));

  PetscCall(DMDestroy(&plex));
  PetscCall(DMDestroy(&forest));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 0
    args: -dm_forest_initial_refinement 1 -dm_forest_maximum_refinement 4 -dm_p4est_refine_pattern hash -dm_plex_view -local_vec_view

TEST*/
