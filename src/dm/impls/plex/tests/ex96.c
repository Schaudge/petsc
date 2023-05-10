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
  /* torus topology */
  PetscInt   coarse_toroidal_faces;
  PetscInt   toroidal_refine;
  PetscInt   poloidal_refine;
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *ctx)
{
  PetscBool phiFlag;
  PetscFunctionBeginUser;
  ctx->dim = 3; // 2 is for plane solve (debugging)
  /* mesh */
  ctx->R = 6.2;
  ctx->r = 2.0;
  ctx->r_inflate = 1;
  ctx->coarse_toroidal_faces  = 1;
  ctx->toroidal_refine = 0; // 4 for SC model
  ctx->poloidal_refine = 1; // 6 for SC model

  PetscOptionsBegin(comm, "tor_", "Tokamak solver", "DMPLEX");
  PetscCall(PetscOptionsInt("-dim", "The dimension of problem (2 is for debugging)", "ex96.c", ctx->dim, &ctx->dim, NULL));
  PetscCheck(ctx->dim==2 || ctx->dim==3, comm,PETSC_ERR_ARG_WRONG,"dim (%d) != 2 or 3",(int)ctx->dim);
  if (ctx->dim==3) {
    ctx->coarse_toroidal_faces = 4;  // 2 for SC model
    PetscCall(PetscOptionsInt("-coarse_toroidal_faces", "Number of planes for mesh", "ex96.c", ctx->coarse_toroidal_faces, &ctx->coarse_toroidal_faces, &phiFlag));
  }
  else { ctx->coarse_toroidal_faces = 1; phiFlag = PETSC_TRUE;} // == 1
  PetscCall(PetscOptionsInt("-toroidal_refine", "Number of refinement steps in toroidal direction", "ex96.c", ctx->toroidal_refine, &ctx->toroidal_refine, NULL));
  PetscCall(PetscOptionsInt("-poloidal_refine", "Number of refinement steps in poloidal plane", "ex96.c", ctx->poloidal_refine, &ctx->poloidal_refine, NULL));
  /* Domain and mesh definition */
  PetscCall(PetscOptionsReal("-radius_minor", "Minor radius of torus", "ex96.c", ctx->r, &ctx->r, NULL));
  PetscCall(PetscOptionsReal("-radius_major", "Major radius of torus", "ex96.c", ctx->R, &ctx->R, NULL));
  PetscCall(PetscOptionsReal("-radius_inflation", "inflate domain factor from minor radius", "ex96.c", ctx->r_inflate, &ctx->r_inflate, NULL));
  ctx->filename[0] = '\0';
  PetscCall(PetscOptionsString("-file", "2D mesh file on [0, 1]^2 and scaled by 2 * r_minor", "ex96.c", ctx->filename, ctx->filename, sizeof(ctx->filename), NULL));
  PetscOptionsEnd();
  PetscFunctionReturn(0);
}

/* Simple shift to origin */
static PetscErrorCode OriginShift2D(DM dm, AppCtx *ctx)
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
static PetscErrorCode ExtrudeTorus(DM dm, AppCtx *ctx, DM *new_dm)
{
  DM dmtorus;
  PetscReal L;
  Vec             coordinates, coordinates2;
  PetscScalar    *coords, *coords2, R_0 = ctx->R;
  PetscInt N,dim=2;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCheck(dim==2, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG,"DM dim (%d) != 2 for extruding",(int)dim);
  PetscCall(DMGetCoordinatesLocal(dm, &coordinates));
  PetscCall(VecGetSize(coordinates, &N));
  PetscCall(VecGetArrayWrite(coordinates, &coords));
  // shift coordinates to center on (R,0). Assume the domain is (0,1)^2
  for (int ii=0;ii<N;ii+=2) {
    PetscScalar *v = &coords[ii];
    v[0] *= 2*ctx->r*ctx->r_inflate;
    v[1] *= 2*ctx->r*ctx->r_inflate;
    v[0] += R_0 - ctx->r*ctx->r_inflate; // move to torus section (move back for 2D)
    v[1] +=     - ctx->r*ctx->r_inflate;
  }
  PetscCall(VecRestoreArrayWrite(coordinates, &coords));
  PetscCall(DMViewFromOptions(dm, NULL, "-dm_view_orig"));
  //
  L = 2*PETSC_PI*ctx->R;
  // Coarse grids with a few elements (2-4)
  PetscCall(DMPlexExtrude(dm, ctx->coarse_toroidal_faces, L, PETSC_FALSE, PETSC_FALSE, NULL, NULL, &dmtorus)); // need to make pencils - TODO
  PetscCall(DMGetDimension(dmtorus, &dim));
  PetscCheck(dim==3, PetscObjectComm((PetscObject)dmtorus), PETSC_ERR_ARG_WRONG,"DM dim (%d) != 3 after extruding",(int)dim);
  PetscCall(DMViewFromOptions(dmtorus, NULL, "-dm_view_extruded"));
  // wrap around torus axis
  PetscCall(DMGetCoordinatesLocal(dmtorus, &coordinates));
  PetscCall(DMGetCoordinates(dmtorus, &coordinates2));
  PetscCall(VecGetSize(coordinates, &N));
  PetscCall(VecGetArrayWrite(coordinates, &coords));
  PetscCall(VecGetArrayWrite(coordinates2, &coords2));
  // shift coordinates to center on (R,0). Assume the domain is (0,1)^2
  for (int ii=0;ii<N;ii+=3) {
    PetscScalar *v = &coords[ii], *v2 = &coords2[ii], theta, psi, R;
    CartTocyl2D(R_0, R, v, psi, theta);
    PetscReal Z = v[2], phi = Z/R_0;
    //PetscCall(PetscPrintf(PetscObjectComm((PetscObject)dmtorus), "\t\t[%d] ExtrudeTorus %d) psi=%12.4e theta=%12.4e phi=%12.4e. Cart=%12.4e,%12.4e,%12.4e", 0, ii/3,  psi, theta, phi, v[0], v[1], v[2]));
    cylToCart( R_0, psi, theta, phi, v);
    //PetscCall(PetscPrintf(PetscObjectComm((PetscObject)dmtorus), "--> X = %12.4e,%12.4e,%12.4e\n", v[0], v[1], v[2]));
    v2[0] = v[0]; v2[1] = v[1]; v2[2] = v[2];
  }
  PetscCall(VecRestoreArrayWrite(coordinates, &coords));
  PetscCall(VecRestoreArrayWrite(coordinates2, &coords2));
  // set for periodic
  PetscCall(DMLocalizeCoordinates(dmtorus));
  *new_dm = dmtorus;
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
  for (d = 0; d < dim; ++d) g3[d * dim + d] = -1.0;
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

// create coarse grid DM. comm is SELF or WORLD. If SELF all ranks create it. TBD if used on "idle" ranks
static PetscErrorCode CreateCoarseMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  size_t      len;
  DM   pdm;

  PetscFunctionBegin;
  PetscCall(PetscStrlen(user->filename, &len));
  if (len) {
    PetscInt dim;
    PetscCall(PetscPrintf(comm, "CreateCoarseMesh: with file %s (fine grid or coarsened....)\n", user->filename));
    PetscCall(DMPlexCreateFromFile(comm, user->filename, "torus_plex", PETSC_FALSE, dm));
    PetscCall(DMGetDimension(*dm, &dim)); // probably 2
    PetscCheck(dim==2, PetscObjectComm((PetscObject)*dm), PETSC_ERR_ARG_WRONG,"Initial DM dim (%d) != 2",(int)dim);
  } else {
    /* PetscCall(DMCreate(comm, dm)); */
    /* PetscCall(DMSetType(*dm, DMPLEX)); // this reads from options without SetFromOptions??? */
    /* PetscCall(DMSetDimension(*dm, 2)); */
    //PetscCall(DMPlexCreateBoxMesh(comm, 2, PETSC_FALSE, NULL, NULL, NULL, NULL, PETSC_TRUE, dm));
    PetscCall(DMCreate(comm, dm));
    PetscCall(DMSetType(*dm, DMPLEX));
    PetscCall(DMPlexDistributeSetDefault(*dm, PETSC_FALSE));
    PetscCall(DMSetFromOptions(*dm)); // gets size of init grid (1x1)
    PetscCall(DMLocalizeCoordinates(*dm)); // not needed
    PetscCall(DMViewFromOptions(*dm, NULL, "-init_dm_view"));
  }
  // distribute coarse 2D (nop in scale runs)
  PetscCall(DMPlexDistribute(*dm, 0, NULL, &pdm));
  if (pdm) {
    PetscCall(DMDestroy(dm));
    *dm = pdm;
  }
  PetscCall(PetscObjectSetName((PetscObject)*dm, "Coarse Mesh"));
  PetscCall(DMSetApplicationContext(*dm, user));
  PetscCall(DMSetUp(*dm));
  PetscCall(DMViewFromOptions(*dm, NULL, "-base_dm_view"));

  PetscFunctionReturn(PETSC_SUCCESS);
}

#include <petsc/private/dmpleximpl.h>  /*I      "petscdmplex.h"   I*/

static PetscErrorCode refineAndSetupSolver(DM *a_dm, SNES snes, AppCtx *ctx)
{
  DM          dmhierarchy[16], pdm, rdm;
  KSP         ksp;
  PC          pc;
  MPI_Comm    crs_comm = MPI_COMM_NULL, fine_comm;
  PetscInt    dim, nlevels = ctx->toroidal_refine + ctx->poloidal_refine + 1, comm_sizes[16];
  PetscMPIInt world_rank, world_size, next_size, locrank, locsize, size_in; // size_in == 1
  int range[1][3];
  MPI_Group world_group, g1;
  char str[] = "-tor_dm_view_0";

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)*a_dm), &size_in));
  PetscCheck(size_in == 1, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG," On level %d, number of processors %d not 1",0, (int)size_in);
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &world_rank)); // hardwire for world
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &world_size));
  /* multigrid KSP create */
  PetscCall(SNESGetKSP(snes, &ksp));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCMG));
  PetscCall(PCMGSetLevels(pc, nlevels, NULL));
  PetscCheck(nlevels <= 16, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,"Too many levels %d > %d", (int)nlevels, 16);
  PetscCall(PCMGSetType(pc, PC_MG_MULTIPLICATIVE));
  PetscCall(PCMGSetGalerkin(pc, PC_MG_GALERKIN_BOTH));
  PetscCall(PCMGSetCycleType(pc, PC_MG_CYCLE_V));
  PetscCall(PCMGSetNumberSmooth(pc, 2));
  /* multigrid hierarchy with 2D mesh */
  PetscCall(DMGetDimension(*a_dm, &dim)); // 2
  if (world_rank < size_in) dmhierarchy[0] = *a_dm;
  else dmhierarchy[0] = NULL;
  comm_sizes[0] = 1;
  for (PetscInt r = 0; r < ctx->poloidal_refine + 1 ; r++) {
    str[13] = '0' + r;
    if (r < ctx->poloidal_refine) { // make a fine grid dmhierarchy[r+1]
      if (r==0) next_size = 4;
      else {
        PetscCallMPI(MPI_Comm_rank(crs_comm, &locrank));
        PetscCallMPI(MPI_Comm_size(crs_comm, &locsize)); // 1 of size
        next_size = 4*locsize; // 4:1 refinement in poloidal coarsening
        PetscCheck(world_size%locsize == 0, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,"On level %d, number of processors %d not even NP = %d",(int)r, (int)locsize,(int)world_size);
      }
      if (next_size > world_size) next_size = world_size; // stop refining tree
      PetscCheck(world_size%next_size == 0, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,"On level %d, number of processors %d not even NP = %d",(int)r, (int)next_size,(int)world_size);
      comm_sizes[r+1] = next_size;
      // make fine_comm, can be MPI_COMM_NULL
      PetscCallMPI(MPI_Comm_group(MPI_COMM_WORLD, &world_group));
      range[0][0] = 0;
      range[0][1] = next_size - 1; // world_size // for "spread"
      range[0][2] = 1; // world_size/next_size
      PetscCallMPI(MPI_Group_range_incl(world_group, 1, range, &g1 ));
      PetscCallMPI(MPI_Comm_create_group(MPI_COMM_WORLD, g1, 0, &fine_comm));
      PetscCallMPI(MPI_Group_free(&world_group));
      PetscCallMPI(MPI_Group_free(&g1));
      // PetscCallMPI(MPI_Comm_free(&fine_comm)); TODO
      if (fine_comm !=  MPI_COMM_NULL) {
        PetscCallMPI(MPI_Comm_rank(fine_comm, &locrank));
        PetscCallMPI(MPI_Comm_size(fine_comm, &locsize)); // 1 of size
        PetscCall(PetscPrintf(PETSC_COMM_SELF, "[%d] ***************** %d) New rank %d new size = %d\n", (int)world_rank, (int)r, (int)locrank, (int)locsize));
        // refine coarse with distribute, then refine w/o distribute
        PetscCall(DMRefine(dmhierarchy[r], fine_comm, &dmhierarchy[r+1])); // one cell per proc, fine grid started, done for this loop
        PetscCall(DMSetApplicationContext(dmhierarchy[r+1], ctx));
        // increment loop
        crs_comm = fine_comm;
      } else dmhierarchy[r+1] = NULL;
    }
    if (dmhierarchy[r] !=  NULL) { // coarse grid is active
      /* extrude coarse */
      if (ctx->dim > dim) {
        DM ext_dm;
        PetscCall(ExtrudeTorus(dmhierarchy[r], ctx, &ext_dm));
        PetscCall(DMDestroy(&dmhierarchy[r]));
        dmhierarchy[r] = ext_dm;
      } else {
        PetscCall(OriginShift2D(dmhierarchy[r], ctx)); // shift to center
      }
      // uniform refinement coarse grid in plane to make pencils. Hardwired for one level (generalize, todo)
      PetscCall(DMRefine(dmhierarchy[r], MPI_COMM_NULL, &rdm));
      PetscCheck(rdm, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,"post refinement failed");
      PetscCall(DMDestroy(&dmhierarchy[r]));
      dmhierarchy[r] = rdm;
      if (ctx->dim > dim) PetscCall(PetscObjectSetName((PetscObject)dmhierarchy[r], "tokamak"));
      else PetscCall(PetscObjectSetName((PetscObject)dmhierarchy[r], "plane"));
      /* Primal system -- needed to get FE interpolation */
      PetscCall(SetupDiscretization(dmhierarchy[r], "potential", SetupPrimalProblem, ctx));
      /* view - coarse grid r is done */
      PetscCall(DMViewFromOptions(dmhierarchy[r], NULL, str));
      PetscCall(DMViewFromOptions(dmhierarchy[r], NULL, "-ref_dm_view"));
      /* make a find grid comm, and P */
      if (r > 0) {
        Mat         P;
        PetscCall(DMSetCoarseDM(dmhierarchy[r], dmhierarchy[r-1]));
        /* set MG */
        PetscCall(DMCreateInterpolation(dmhierarchy[r-1], dmhierarchy[r], &P, NULL));
        PetscCall(PCMGSetInterpolation(pc, r, P));
        PetscCall(MatViewFromOptions(R, NULL, "-r_mat_view"));
        PetscCall(MatDestroy(&P));
      }
    }
  }
  if (ctx->dim > dim) {
    for (PetscInt ri = 0, r = ctx->poloidal_refine ; ri < ctx->toroidal_refine; ri++, r++) {
      Mat         P;
      str[13] = '0' + r;
      PetscCheck(PETSC_FALSE, crs_comm, PETSC_ERR_ARG_WRONG,"Toroidal refinement not done !!! %d", (int)r);      
      PetscMPIInt next_size = 2 * comm_sizes[r];
      if (next_size > world_size) next_size = world_size; // stop refining tree
      comm_sizes[r+1] = next_size;      
      if (next_size == world_size) fine_comm = PETSC_COMM_WORLD;
      else {
        range[0][0] = 0;
        range[0][1] = next_size - 1; // world_size // for "spread"
        range[0][2] = 1; // world_size/next_size
        PetscCallMPI(MPI_Group_range_incl(world_group, 1, range, &g1 ));
        PetscCallMPI(MPI_Comm_create_group(MPI_COMM_WORLD, g1, 0, &fine_comm));
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%d) refineAndSetupSolver: %s. size = %d\n", (int)r, str, (int)next_size));
        PetscCallMPI(MPI_Group_free(&world_group));
        PetscCallMPI(MPI_Group_free(&g1));
      }
      if (fine_comm !=  MPI_COMM_NULL) {
        // uniform refinement in toroidal direction - TODO 

        
        /* view - coarse grid r is done */
        PetscCall(DMViewFromOptions(dmhierarchy[r], NULL, str));
        PetscCall(DMViewFromOptions(dmhierarchy[r], NULL, "-ref_dm_view"));
      } else dmhierarchy[r+1] = NULL;
      /* make a find grid comm, and P */
      PetscCall(DMSetCoarseDM(dmhierarchy[r], dmhierarchy[r-1]));
      /* set MG */
      PetscCall(DMCreateInterpolation(dmhierarchy[r-1], dmhierarchy[r], &P, NULL));
      PetscCall(PCMGSetInterpolation(pc, r, P));
      PetscCall(MatDestroy(&P));
    } 
  }
  /* destroy coarse grids & save fine grid */
  for (PetscInt r = 0;  r < ctx->poloidal_refine + ctx->toroidal_refine ; r++) PetscCall(DMDestroy(&dmhierarchy[r]));
  pdm = dmhierarchy[ctx->poloidal_refine + ctx->toroidal_refine];
  PetscCall(SNESSetDM(snes, pdm));
  *a_dm = pdm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  AppCtx     actx,*ctx=&actx; /* work context */
  DM                 dm;
  PetscInt dim;
  SNES   snes; /* Nonlinear solver */
  Vec    u;    /* Solutions */
  Mat J;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, ctx));// ctx->dim probably 3 or 2 for debugging
  /* Create Plex - serial DM for serial coarse grid */
  PetscCall(CreateCoarseMesh(PETSC_COMM_SELF, ctx, &dm));
  PetscCall(DMGetDimension(dm, &dim)); // probably 2
  PetscCheck(dim == 2, PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"DM dim (%d) != 2",(int)dim);
  /* solver and refinement */
  PetscCall(SNESCreate(PETSC_COMM_WORLD, &snes));
  PetscCall(refineAndSetupSolver(&dm, snes, ctx));
  PetscCall(DMViewFromOptions(dm, NULL, "-tor_dm_view"));
  PetscCall(DMCreateGlobalVector(dm, &u));
  PetscCall(VecSet(u, 0.0));
  PetscCall(PetscObjectSetName((PetscObject)u, "u"));
  PetscCall(DMPlexSetSNESLocalFEM(dm, ctx, ctx, ctx));
  PetscCall(DMCreateMatrix(dm, &J));
  PetscCall(SNESSetJacobian(snes, J, J, NULL, NULL));
  PetscCall(SNESSetFromOptions(snes));
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
  PetscCall(MatDestroy(&J));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   testset:
     args: -dm_plex_dim 2 -dm_plex_simplex 0 -dm_plex_box_faces 1,1 -dm_plex_hash_location -tor_toroidal_refine 0 -tor_poloidal_refine 3 -tor_coarse_toroidal_faces 4 -potential_petscspace_degree 2 -snes_type ksponly -ksp_type cg -ksp_monitor -mg_levels_esteig_ksp_type cg -mg_levels_pc_type jacobi -tor_dm_view -ksp_view -options_left
     requires: !complex hdf5
     nsize: 1
     test:
       suffix: 2d
       args: -tor_dim 2 -tor_coarse_toroidal_faces 1

     test:
       suffix: 3d
       args: -tor_dim 3

TEST*/
