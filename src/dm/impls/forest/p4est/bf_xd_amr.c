#if defined(PETSC_HAVE_P4EST)

typedef struct _p_AmrCtx {
  int minLevel;
  int maxLevel;
} AmrCtx;

static int _p_coarsen_uniformly(p4est_t * p4est, p4est_topidx_t which_tree, p4est_quadrant_t *quadrants[])
{
  AmrCtx    *amrCtx = p4est->user_pointer;
  const int minLevel = amrCtx->minLevel;

  //SC_CHECK_ABORT (p4est_quadrant_is_familypv (quadrants), "Coarsen invocation");
  return (0 <= minLevel && minLevel < quadrants[0]->level);
}

static int _p_refine_uniformly(p4est_t * p4est, p4est_topidx_t which_tree, p4est_quadrant_t *quadrant)
{
  AmrCtx    *amrCtx = p4est->user_pointer;
  const int maxLevel = amrCtx->maxLevel;

  return (0 <= maxLevel && quadrant->level < maxLevel);
}

PetscErrorCode DMBF_XD_AmrCoarsenUniformly(p4est_t *p4est, PetscInt minLevel)
{
  void           *user_pointer = p4est->user_pointer;
  AmrCtx         amrCtx;

  PetscFunctionBegin;
  /* set AMR context */
  amrCtx.minLevel = (int) minLevel;
  amrCtx.maxLevel = -1;
  p4est->user_pointer = (void*) &amrCtx;
  /* run AMR */
  PetscStackCallP4est(p4est_coarsen,(p4est,0/*!recursively*/,_p_coarsen_uniformly,NULL/*init_fn*/));
  p4est->user_pointer = user_pointer;
  /* balance and partition */
  PetscStackCallP4est(p4est_balance,(p4est,P4EST_CONNECT_FULL,NULL/*init_fn*/));
  PetscStackCallP4est(p4est_partition_ext,(p4est,1/*partition_for_coarsening*/,NULL/*weight_fn*/));
  PetscFunctionReturn(0);
}

PetscErrorCode DMBF_XD_AmrRefineUniformly(p4est_t *p4est, PetscInt maxLevel)
{
  void           *user_pointer = p4est->user_pointer;
  AmrCtx         amrCtx;

  PetscFunctionBegin;
  /* set AMR context */
  amrCtx.minLevel = -1;
  amrCtx.maxLevel = (int) maxLevel;
  p4est->user_pointer = (void*) &amrCtx;
  /* run AMR */
  PetscStackCallP4est(p4est_refine,(p4est,0/*!recursively*/,_p_refine_uniformly,NULL/*init_fn*/));
  p4est->user_pointer = user_pointer;
  /* balance and partition */
  PetscStackCallP4est(p4est_balance,(p4est,P4EST_CONNECT_FULL,NULL/*init_fn*/));
  PetscStackCallP4est(p4est_partition_ext,(p4est,1/*partition_for_coarsening*/,NULL/*weight_fn*/));
  PetscFunctionReturn(0);
}

#endif /* defined(PETSC_HAVE_P4EST) */
