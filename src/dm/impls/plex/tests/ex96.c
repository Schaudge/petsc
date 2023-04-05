static char help[] = "Geometric multigrid Poisson solver on Tokamak domain with semi-coarsening grid hierarchy\n";

#include <petscdmplex.h>
#include <petscsnes.h>
#include <petscds.h>

typedef struct {
  PetscInt dim;
  PetscBool view_coarse;
  /* torus geometry  */
  PetscReal  R;
  PetscReal  r;
  PetscReal  r_inflate;
  PetscInt   np_phi;
  PetscInt   np_radius;
  PetscInt   np_theta;
  /* solver */
  PetscInt   nlevels;
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *ctx)
{
  PetscBool phiFlag,radFlag,thetaFlag;
  PetscFunctionBeginUser;
  ctx->dim = 2;
  /* mesh */
  ctx->R = 6.2;
  ctx->r = 2.0;
  ctx->r_inflate = 1;
  ctx->np_phi  = 1;
  ctx->np_radius = 1;
  ctx->np_theta  = 1;
  ctx->nlevels = 2;

  PetscOptionsBegin(comm, "tor_", "Tokamak solver", "DMPLEX");
  PetscCall(PetscOptionsInt("-dim", "parameter", "ex96.c", ctx->dim, &ctx->dim, NULL));
  PetscCheck(ctx->dim==2 || ctx->dim==3, comm,PETSC_ERR_ARG_WRONG,"dim (%d) != 2 or 3",(int)ctx->dim);
  if (ctx->dim==3) {
    ctx->np_phi = 4;
    PetscCall(PetscOptionsInt("-np_phi", "Number of planes for mesh", "ex96.c", ctx->np_phi, &ctx->np_phi, &phiFlag));
  }
  else { ctx->np_phi = 1; phiFlag = PETSC_TRUE;} // == 1
  PetscCall(PetscOptionsInt("-np_radius", "Number of radial cells for particle mesh", "ex96.c", ctx->np_radius, &ctx->np_radius, &radFlag));
  PetscCall(PetscOptionsInt("-np_theta", "Number of theta cells for particle mesh", "ex96.c", ctx->np_theta, &ctx->np_theta, &thetaFlag));
  PetscCall(PetscOptionsInt("-num_levels", "Number of multigrid levels (refinement-1)", "ex96.c", ctx->nlevels, &ctx->nlevels, NULL));
  /* Domain and mesh definition */
  PetscCall(PetscOptionsReal("-radius_minor", "Minor radius of torus", "ex96.c", ctx->r, &ctx->r, NULL));
  PetscCall(PetscOptionsReal("-radius_major", "Major radius of torus", "ex96.c", ctx->R, &ctx->R, NULL));
  PetscCall(PetscOptionsReal("-radius_inflation", "inflate domain factor from minor radius", "ex96.c", ctx->r_inflate, &ctx->r_inflate, NULL));

  PetscOptionsEnd();
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscFunctionBeginUser;
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMSetFromOptions(*dm)); // seems to create a 2x2 mesh by default'
  PetscCall(DMSetApplicationContext(*dm, user));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Simple shift to origin */
static PetscErrorCode OriginShift2D(MPI_Comm comm, DM dm, AppCtx *ctx)
{
  Vec             coordinates;
  PetscScalar    *coords;
  PetscInt N;

  PetscFunctionBeginUser;
  PetscCall(DMGetCoordinatesLocal(dm, &coordinates));
  PetscCall(VecGetSize(coordinates, &N));
  PetscCall(VecGetArrayWrite(coordinates, &coords));
  // shift coordinates to center on (R,0). Assume the domain is (0,1)^2
  for (int ii=0;ii<N;ii+=2) {
    PetscScalar *v = &coords[ii];
    v[0] *= 2*ctx->r*ctx->r_inflate;
    v[1] *= 2*ctx->r*ctx->r_inflate;
    v[0] -= ctx->r*ctx->r_inflate;
    v[1] -= ctx->r*ctx->r_inflate;
  }
  PetscCall(VecRestoreArrayWrite(coordinates, &coords));
  PetscCall(DMSetCoordinatesLocal(dm, coordinates));
  PetscFunctionReturn(0);
}

// store Cartesian (X,Y,Z) for plotting 3D, (X,Y) for 2D
// (psi,theta,phi) --> (X,Y,Z)
#define cylToCart( __R_0, __psi,  __theta,  __phi, __cart)       \
  { PetscReal __R = (__R_0) + (__psi)*PetscCosReal(__theta);            \
    __cart[0] = __R*PetscCosReal(__phi);                                \
    __cart[1] = __psi*PetscSinReal(__theta);				\
    __cart[2] = -__R*PetscSinReal(__phi);       \
  }

/* coordinate transformation - simple radial coordinates. Not really cylindrical as r_Minor is radius from plane axis */
#define XYToPsiTheta(__x,__y,__psi,__theta) {                           \
    __psi = PetscSqrtReal((__x)*(__x) + (__y)*(__y));                   \
    if (PetscAbsReal(__psi) < PETSC_SQRT_MACHINE_EPSILON) __theta = 0.; \
    else {                                                              \
      __theta = (__y) > 0. ? PetscAsinReal((__y)/__psi) : -PetscAsinReal(-(__y)/__psi); \
      if ((__x) < 0) __theta = PETSC_PI - __theta;                      \
      else if (__theta < 0.) __theta = __theta + 2.*PETSC_PI;           \
    }                                                                   \
  }

#define CartTocyl2D(__R_0, __R, __cart, __psi,  __theta) {              \
    __R = __cart[0];                                                    \
    XYToPsiTheta(__R - __R_0, __cart[1], __psi, __theta);               \
  }

/* Extrude 2D Plex to 3D Plex */
static PetscErrorCode ExtrudeTorus(MPI_Comm comm, DM *dm, AppCtx *ctx)
{
  DM dmtorus;
  PetscReal L;
  Vec             coordinates;
  PetscScalar    *coords, R_0 = ctx->R;
  PetscInt N,dim;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(*dm, &dim)); // probably 2
  PetscCheck(dim==2, PetscObjectComm((PetscObject)*dm), PETSC_ERR_ARG_WRONG,"DM dim (%d) != 2 for extruding",(int)dim);
  PetscCall(DMGetCoordinates(*dm, &coordinates));
  PetscCall(VecGetSize(coordinates, &N));
  PetscCall(VecGetArrayWrite(coordinates, &coords));
  // shift coordinates to center on (R,0). Assume the domain is (0,1)^2
  for (int ii=0;ii<N;ii+=2) {
    PetscScalar *v = &coords[ii];
    v[0] *= 2*ctx->r*ctx->r_inflate;
    v[1] *= 2*ctx->r*ctx->r_inflate;
    v[0] += R_0 - ctx->r*ctx->r_inflate;
    v[1] +=     - ctx->r*ctx->r_inflate;
  }
  PetscCall(VecRestoreArrayWrite(coordinates, &coords));
  //
  L = 2*PETSC_PI*ctx->R;
  // we could create a box mesh here but Plex starts with a 2x2 so we can just dm_refine from there, for now
  PetscCall(DMPlexExtrude(*dm, ctx->np_phi, L, PETSC_FALSE, PETSC_FALSE, NULL, NULL, &dmtorus));
  PetscCall(DMDestroy(dm));
  *dm = dmtorus;
  PetscCall(DMGetDimension(*dm, &dim));
  PetscCheck(dim==3, PetscObjectComm((PetscObject)*dm), PETSC_ERR_ARG_WRONG,"DM dim (%d) != 3 after extruding",(int)dim);
  PetscCall(DMViewFromOptions(*dm, NULL, "-tor_dm_view_extruded"));
  // wrap around torus axis
  PetscCall(DMGetCoordinates(*dm, &coordinates));
  PetscCall(VecGetSize(coordinates, &N));
  PetscCall(VecGetArrayWrite(coordinates, &coords));
  // shift coordinates to center on (R,0). Assume the domain is (0,1)^2
  for (int ii=0;ii<N;ii+=3) {
    PetscScalar *v = &coords[ii], theta, psi, R;
    CartTocyl2D(R_0, R, v, psi, theta);
    PetscReal Z = v[2], phi = Z/R_0;
    //PetscCall(PetscPrintf(PetscObjectComm((PetscObject)*dm), "\t\t[%d] ExtrudeTorus %d) psi=%12.4e theta=%12.4e phi=%12.4e. Cart=%12.4e,%12.4e,%12.4e", 0, ii/3,  psi, theta, phi, v[0], v[1], v[2]));
    cylToCart( R_0, psi, theta, phi, v);
    //PetscCall(PetscPrintf(PetscObjectComm((PetscObject)*dm), "--> X = %12.4e,%12.4e,%12.4e\n", v[0], v[1], v[2]));
  }
  PetscCall(VecRestoreArrayWrite(coordinates, &coords));
  // set periodic - TODO
  PetscFunctionReturn(0);
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

static void f0_one(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] = 1;
}

static PetscErrorCode u_zero(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  u[0] = 0.0;
  return PETSC_SUCCESS;
}

static PetscErrorCode SetupPrimalProblem(DM dm,  AppCtx *ctx)
{
  PetscDS        ds;
  DMLabel        label;
  const PetscInt id = 1;

  PetscFunctionBeginUser;
  PetscCall(DMGetDS(dm, &ds));
  PetscCall(DMGetLabel(dm, "marker", &label));
  PetscCall(PetscDSSetResidual(ds, 0, f0_one, f1_u));
  PetscCall(PetscDSSetJacobian(ds, 0, 0, NULL, NULL, NULL, g3_uu));
  PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", label, 1, &id, 0, 0, NULL, (void (*)(void))u_zero, NULL, ctx, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetupDiscretization(DM dm, const char name[], PetscErrorCode (*setup)(DM, AppCtx *), AppCtx *ctx)
{
  PetscFE        fe;
  DMPolytopeType ct;
  PetscBool      simplex;
  PetscInt       dim, cStart;
  char           prefix[PETSC_MAX_PATH_LEN];

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, NULL));
  PetscCall(DMPlexGetCellType(dm, cStart, &ct));
  PetscCall(DMPlexIsSimplex(dm, &simplex));
  PetscCall(PetscSNPrintf(prefix, PETSC_MAX_PATH_LEN, "%s_", name));
  PetscCall(PetscFECreateDefault(PetscObjectComm((PetscObject)dm), dim, 1, simplex, name ? prefix : NULL, -1, &fe));
  PetscCall(PetscObjectSetName((PetscObject)fe, name));
  PetscCall(PetscFEViewFromOptions(fe, NULL, "-fe_view"));
  PetscCall(DMSetField(dm, 0, NULL, (PetscObject)fe));
  PetscCall(DMCreateDS(dm));
  PetscCall((*setup)(dm, ctx));

  PetscCall(PetscFEDestroy(&fe));
  PetscFunctionReturn(PETSC_SUCCESS);
}
#include <petsc/private/dmpleximpl.h>  /*I      "petscdmplex.h"   I*/

static PetscErrorCode refineAndSetupSolver(DM *a_dm, SNES snes, AppCtx *ctx)
{
  DM          dmref, *dmhierarchy = NULL;
  KSP         ksp;
  PC          pc;
  MPI_Comm           comm = PetscObjectComm((PetscObject)*a_dm);
  PetscInt dim, r;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(*a_dm, &dim)); // probably 2
  /* multigrid hierarchy */
  if (ctx->nlevels > 1) {
    // coarsest grid = 0
    // finest grid = nlevels - 1
    PetscCall(PetscMalloc(sizeof(DM) * ctx->nlevels, &dmhierarchy));
    for ( r = 1; r < ctx->nlevels; r++) dmhierarchy[r] = NULL;
    dmhierarchy[0] = *a_dm;
    /* refine levels */
    PetscCall(DMRefineHierarchy(*a_dm, ctx->nlevels - 1, &dmhierarchy[1]));
  }
  /* fine grid */
  if (ctx->dim > dim) {
    PetscCall(ExtrudeTorus(comm, a_dm, ctx)); // 3D extrude, changes DM
    PetscCall(PetscObjectSetName((PetscObject)*a_dm, "tokamak"));
  } else {
    PetscCall(OriginShift2D(comm, *a_dm, ctx)); // shift to center
    PetscCall(PetscObjectSetName((PetscObject)*a_dm, "plane"));
  }
  /* Primal system */
  PetscCall(SetupDiscretization(*a_dm, "potential", SetupPrimalProblem, ctx));
  if (ctx->nlevels == 1) {
    dmref = *a_dm;
  } else { // coarse grids
    PetscBool isUniform;
    PetscCall(DMPlexGetRefinementUniform(*a_dm, &isUniform));
    PetscCheck(isUniform, comm, PETSC_ERR_ARG_WRONG,"Not isUniform");
    dmhierarchy[0] = *a_dm; // could have changed in torus
    if (ctx->dim > dim) { // 3D torus
      DM cdm = *a_dm, dm = cdm;
      for (PetscInt r = 1; r < ctx->nlevels ; ++r) {
        DM              codm, rcodm;
        PetscCall(ExtrudeTorus(comm, &dmhierarchy[r], ctx)); // 3D extrude (DMPlexTransformApply)
        PetscCall(PetscObjectSetName((PetscObject)dmhierarchy[r], "tokamak-mg"));
        PetscCall(DMSetCoarsenLevel(dmhierarchy[r], cdm->leveldown));
        PetscCall(DMSetRefineLevel(dmhierarchy[r], cdm->levelup + 1));
        PetscCall(DMCopyDisc(cdm, dmhierarchy[r]));
        PetscCall(DMGetCoordinateDM(dm, &codm));
        PetscCall(DMGetCoordinateDM(dmhierarchy[r], &rcodm));
        PetscCall(DMCopyDisc(codm, rcodm));
        //PetscCall(DMPlexTransformCreateDiscLabels(tr, dmhierarchy[r]));
        PetscCall(DMSetCoarseDM(dmhierarchy[r], cdm));
        PetscCall(DMPlexSetRegularRefinement(dmhierarchy[r], PETSC_TRUE));
        if (dmhierarchy[r]) {
          ((DM_Plex *)(dmhierarchy[r])->data)->printFEM = ((DM_Plex *)dm->data)->printFEM;
          ((DM_Plex *)(dmhierarchy[r])->data)->printL2  = ((DM_Plex *)dm->data)->printL2;
        }
        cdm = dmhierarchy[r];
      }
    } else {
      for (PetscInt r = 1; r < ctx->nlevels; ++r) {
        PetscCall(OriginShift2D(comm, dmhierarchy[r], ctx)); // shift to center
        PetscCall(PetscObjectSetName((PetscObject)dmhierarchy[r], "plane-mg"));
      }
    }
    /* multigrid solver */
    PetscCall(SNESGetKSP(snes, &ksp));
    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(PCSetType(pc, PCMG));
    PetscCall(PCMGSetLevels(pc, ctx->nlevels, NULL));
    PetscCall(PCMGSetType(pc, PC_MG_MULTIPLICATIVE));
    PetscCall(PCMGSetGalerkin(pc, PC_MG_GALERKIN_BOTH));
    PetscCall(PCMGSetCycleType(pc, PC_MG_CYCLE_V));
    PetscCall(PCMGSetNumberSmooth(pc, 2));
    for ( r = 1; r < ctx->nlevels; r++) {
      char str[] = "-tor_coarse_dm_view_0";
      Mat         R;
      DM dm = dmhierarchy[r], cdm = dmhierarchy[r - 1];
      str[20] += r - 1;
      PetscCall(PetscPrintf(PetscObjectComm((PetscObject)cdm), "%d) DMViewFromOptions: %s\n", (int)r, str));
      /* set MG */
      PetscCall(DMCreateInterpolation(cdm, dm, &R, NULL));
      PetscCall(PCMGSetInterpolation(pc, r, R));
      PetscCall(MatDestroy(&R));
      /* view */
      PetscCall(DMViewFromOptions(cdm, NULL, str));
    }
    for ( r = 0;  r < ctx->nlevels - 1 ; r++) PetscCall(DMDestroy(&dmhierarchy[r])); // destroy input & save fine grid
    dmref = dmhierarchy[ctx->nlevels-1];
    PetscCall(PetscFree(dmhierarchy));
  }
  PetscCall(PetscObjectReference((PetscObject)dmref));
  PetscCall(SNESSetFromOptions(snes));
  PetscCall(SNESSetDM(snes, dmref));
  *a_dm = dmref;
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  AppCtx     actx,*ctx=&actx; /* work context */
  MPI_Comm           comm;
  DM                 dm;
  PetscInt dim;
  SNES   snes; /* Nonlinear solver */
  Vec    u;    /* Solutions */

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;
  PetscCall(ProcessOptions(comm, ctx));// ctx->dim probably 3 or 2 for debugging
  /* Create Plex */
  PetscCall(CreateMesh(comm, ctx, &dm));
  PetscCall(DMGetDimension(dm, &dim)); // probably 2
  PetscCheck(dim <= ctx->dim && dim > 1, comm,PETSC_ERR_ARG_WRONG,"DM dim (%d) > -dim %d",(int)dim,ctx->dim);
  /* solver and refinement */
  PetscCall(SNESCreate(comm, &snes));
  PetscCall(refineAndSetupSolver(&dm, snes, ctx));
  PetscCall(DMViewFromOptions(dm, NULL, "-tor_dm_view"));
  PetscCall(DMCreateGlobalVector(dm, &u));
  PetscCall(VecSet(u, 0.0));
  PetscCall(PetscObjectSetName((PetscObject)u, "u"));
  PetscCall(DMPlexSetSNESLocalFEM(dm, ctx, ctx, ctx));
  /* view */
  PetscCall(DMSetOutputSequenceNumber(dm, 0, 0.0));
  PetscCall(VecViewFromOptions(u, NULL, "-tor_vec_view"));
  /* solve */
  PetscCall(SNESSolve(snes, NULL, u));
  PetscCall(DMSetOutputSequenceNumber(dm, 1, 0.0));
  /* view */
  PetscCall(VecViewFromOptions(u, NULL, "-tor_vec_view"));
  PetscCall(SNESGetSolution(snes, &u));
  // cleanup
  PetscCall(DMDestroy(&dm));
  PetscCall(VecDestroy(&u));
  PetscCall(SNESDestroy(&snes));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   testset:
     args: -tor_dim 3 -dm_plex_simplex 0 -dm_plex_box_faces 2,2 -dm_plex_dim 2 -tor_num_levels 2 -tor_np_phi 4 -potential_petscspace_degree 2 -snes_type ksponly -ksp_type cg -ksp_monitor 
     requires: !complex hdf5

     test:
       suffix: 2D
       args: -dim 2

     test:
       suffix: 3D
       nsize: 4
       args: -dim 3

TEST*/
