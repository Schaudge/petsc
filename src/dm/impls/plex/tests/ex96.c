static char help[] = "Geometric multigrid Poisson solver on Tokamak domain with semi-coarsening grid hierarchy\n";

#include <petscdmplex.h>
#include <petscsnes.h>
#include <petscds.h>

typedef struct {
  PetscInt dim;
  PetscBool view_coarse;
  char      filename[PETSC_MAX_PATH_LEN];
  /* torus geometry  */
  PetscReal  R;
  PetscReal  r;
  PetscReal  r_inflate;
  PetscInt   np_phi;
  PetscInt   np_radius; // not used
  PetscInt   np_theta; // not used
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
  PetscCheck(ctx->nlevels < 12 &&  ctx->nlevels > 0, comm,PETSC_ERR_ARG_WRONG,"nlevels (%d)",(int)ctx->nlevels);
  /* Domain and mesh definition */
  PetscCall(PetscOptionsReal("-radius_minor", "Minor radius of torus", "ex96.c", ctx->r, &ctx->r, NULL));
  PetscCall(PetscOptionsReal("-radius_major", "Major radius of torus", "ex96.c", ctx->R, &ctx->R, NULL));
  PetscCall(PetscOptionsReal("-radius_inflation", "inflate domain factor from minor radius", "ex96.c", ctx->r_inflate, &ctx->r_inflate, NULL));
  ctx->filename[0] = '\0';
  PetscCall(PetscOptionsString("-file", "The mesh file", "ex96.c", ctx->filename, ctx->filename, sizeof(ctx->filename), NULL));
  PetscOptionsEnd();
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  size_t      len;
  DM dmDist = NULL;

  PetscFunctionBegin;
  PetscCall(PetscStrlen(user->filename, &len));
  if (len) {
    PetscCall(PetscPrintf(comm, "CreateMesh: with file %s\n", user->filename));
    PetscCall(DMPlexCreateFromFile(comm, user->filename, "torus_plex", PETSC_FALSE, dm));
  } else {
    PetscCall(DMCreate(comm, dm));
    PetscCall(DMSetType(*dm, DMPLEX)); // this reads from options without SetFromOptions???
  }
  PetscCall(DMPlexDistribute(*dm, 0, NULL, &dmDist));
  if (dmDist) {
    PetscCall(DMDestroy(dm));
    *dm = dmDist;
  }
  PetscCall(PetscObjectSetName((PetscObject)*dm, "Coarse Mesh"));
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
  PetscCall(DMSetCoordinates(dm, coordinates));
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
  Vec             coordinates, coordinates2;
  PetscScalar    *coords, *coords2, R_0 = ctx->R;
  PetscInt N,dim;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(*dm, &dim)); // probably 2
  PetscCheck(dim==2, PetscObjectComm((PetscObject)*dm), PETSC_ERR_ARG_WRONG,"DM dim (%d) != 2 for extruding",(int)dim);
  PetscCall(DMGetCoordinatesLocal(*dm, &coordinates));
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
  PetscCall(DMViewFromOptions(*dm, NULL, "-tor_dm_view_orig"));
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
  PetscCall(DMGetCoordinatesLocal(*dm, &coordinates));
  PetscCall(DMGetCoordinates(*dm, &coordinates2));
  PetscCall(VecGetSize(coordinates, &N));
  PetscCall(VecGetArrayWrite(coordinates, &coords));
  PetscCall(VecGetArrayWrite(coordinates2, &coords2));
  // shift coordinates to center on (R,0). Assume the domain is (0,1)^2
  for (int ii=0;ii<N;ii+=3) {
    PetscScalar *v = &coords[ii], *v2 = &coords2[ii], theta, psi, R;
    CartTocyl2D(R_0, R, v, psi, theta);
    PetscReal Z = v[2], phi = Z/R_0;
    //PetscCall(PetscPrintf(PetscObjectComm((PetscObject)*dm), "\t\t[%d] ExtrudeTorus %d) psi=%12.4e theta=%12.4e phi=%12.4e. Cart=%12.4e,%12.4e,%12.4e", 0, ii/3,  psi, theta, phi, v[0], v[1], v[2]));
    cylToCart( R_0, psi, theta, phi, v);
    //PetscCall(PetscPrintf(PetscObjectComm((PetscObject)*dm), "--> X = %12.4e,%12.4e,%12.4e\n", v[0], v[1], v[2]));
    v2[0] = v[0]; v2[1] = v[1]; v2[2] = v[2];
  }
  PetscCall(VecRestoreArrayWrite(coordinates, &coords));
  PetscCall(VecRestoreArrayWrite(coordinates2, &coords2));
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
  DM          dmref, dmhierarchy[12];
  KSP         ksp;
  PC          pc;
  MPI_Comm           comm = PetscObjectComm((PetscObject)*a_dm);
  PetscInt dim;
  PetscBool isUniform;

  PetscFunctionBeginUser;
  /* multigrid KSP create */
  PetscCall(SNESGetKSP(snes, &ksp));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCMG));
  PetscCall(PCMGSetLevels(pc, ctx->nlevels, NULL));
  PetscCall(PCMGSetType(pc, PC_MG_MULTIPLICATIVE));
  PetscCall(PCMGSetGalerkin(pc, PC_MG_GALERKIN_BOTH));
  PetscCall(PCMGSetCycleType(pc, PC_MG_CYCLE_V));
  PetscCall(PCMGSetNumberSmooth(pc, 2));
  /* multigrid hierarchy with 2D mesh */
  PetscCall(DMGetDimension(*a_dm, &dim)); // probably 2
  dmhierarchy[0] = *a_dm;
  for (PetscInt r = 1; r < ctx->nlevels ; ++r) {
    PetscCall(DMRefine(dmhierarchy[r - 1], MPI_COMM_NULL, &dmhierarchy[r]));
  }
  /* extrude */
  for (PetscInt r = 0; r < ctx->nlevels; r++) {
    char str[] = "-tor_dm_view_0";
    str[13] += r;
    PetscCall(PetscPrintf(comm, "%d) DMViewFromOptions: %s\n", (int)r, str));
    if (ctx->dim > dim) {
      PetscCall(ExtrudeTorus(comm, &dmhierarchy[r], ctx)); // 3D extrude (DMPlexTransformApply)
      PetscCall(PetscObjectSetName((PetscObject)dmhierarchy[r], "tokamak"));
    } else {
      PetscCall(OriginShift2D(comm, dmhierarchy[r], ctx)); // shift to center
      PetscCall(PetscObjectSetName((PetscObject)dmhierarchy[r], "plane"));
    }
    /* view */
    PetscCall(DMViewFromOptions(dmhierarchy[r], NULL, str));
    /* P */
    if (r > 0 && 0) { // TODO
      Mat         R;
      /* set MG */
      PetscCall(DMCreateInterpolation(dmhierarchy[r-1], dmhierarchy[r], &R, NULL));
      PetscCall(PCMGSetInterpolation(pc, r, R));
      PetscCall(MatDestroy(&R));
    }
    /* Primal system */
    PetscCall(SetupDiscretization(dmhierarchy[r], "potential", SetupPrimalProblem, ctx));
    PetscCall(DMPlexGetRefinementUniform(dmhierarchy[r], &isUniform));
    PetscCheck(isUniform, comm, PETSC_ERR_ARG_WRONG,"Not isUniform");
  }
  /* destroy coarse grids & save fine grid */
  for (PetscInt r = 0;  r < ctx->nlevels - 1 ; r++) PetscCall(DMDestroy(&dmhierarchy[r]));
  dmref = dmhierarchy[ctx->nlevels-1];
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
  /* view */
  PetscCall(DMSetOutputSequenceNumber(dm, 1, 1.0));
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
     args: -dm_plex_simplex 0 -dm_plex_box_faces 2,2 -dm_plex_dim 2 -tor_num_levels 2 -potential_petscspace_degree 2 -snes_type ksponly -ksp_type cg -ksp_monitor -pc_type ilu -tor_dm_view
     requires: !complex hdf5

     test:
       suffix: 2d
       args: -tor_dim 2

     test:
       suffix: 3d
       nsize: 1
       args: -tor_dim 3 -tor_np_phi 4

TEST*/
