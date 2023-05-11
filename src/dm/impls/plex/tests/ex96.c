static char help[] = "Geometric multigrid Poisson solver on Tokamak domain with semi-coarsening grid hierarchy\n";

#include <petscdmplex.h>
#include <petscts.h>
#include <petscds.h>

typedef struct {
  PetscInt dim;
  PetscBool use_360_domains;
  char      filename[PETSC_MAX_PATH_LEN];
  /* torus geometry  */
  PetscReal  R;
  PetscReal  r;
  PetscReal  r_inflate;
  /* torus topology */
  PetscInt   coarse_toroidal_faces;
  PetscInt   toroidal_refine;
  PetscInt   poloidal_refine;
  PetscInt nlevels;
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *ctx)
{
  PetscMPIInt size;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_size(comm, &size));
  ctx->dim = 3; // 2 is for plane solve (debugging)
  /* mesh */
  ctx->R = 6.2;
  ctx->r = 2.0;
  ctx->r_inflate = 1;
  ctx->coarse_toroidal_faces  = 1;
  ctx->toroidal_refine = 0; // 4 for SC model
  ctx->poloidal_refine = 1; // 6 for SC model
  ctx->use_360_domains = PETSC_TRUE;
  PetscOptionsBegin(comm, "tor_", "Tokamak solver", "DMPLEX");
  PetscCall(PetscOptionsInt("-dim", "The dimension of problem (2 is for debugging)", "ex96.c", ctx->dim, &ctx->dim, NULL));
  PetscCheck(ctx->dim==2 || ctx->dim==3, comm,PETSC_ERR_ARG_WRONG,"dim (%d) != 2 or 3",(int)ctx->dim);
  if (ctx->dim==3) {
    ctx->coarse_toroidal_faces = 4;  // 2 for SC model
    PetscCall(PetscOptionsInt("-coarse_toroidal_faces", "Number of planes for mesh", "ex96.c", ctx->coarse_toroidal_faces, &ctx->coarse_toroidal_faces, NULL));
    PetscCheck(ctx->coarse_toroidal_faces%size == 0 || size%ctx->coarse_toroidal_faces%size == 0, comm, PETSC_ERR_ARG_WRONG,"Number of coarse face (%d) mod num procs %d != 0",(int)ctx->coarse_toroidal_faces,(int)size);
  }
  else ctx->coarse_toroidal_faces = 1;
  PetscCall(PetscOptionsInt("-toroidal_refine", "Number of refinement steps in toroidal direction", "ex96.c", ctx->toroidal_refine, &ctx->toroidal_refine, NULL));
  PetscCall(PetscOptionsInt("-poloidal_refine", "Number of refinement steps in poloidal plane", "ex96.c", ctx->poloidal_refine, &ctx->poloidal_refine, NULL));
  PetscCall(PetscOptionsBool("-use_360_domains", "inflate domain factor from minor radius", "ex96.c", ctx->use_360_domains, &ctx->use_360_domains, NULL));
  if (size>1) {
    int nn = (int)(PetscPowInt(4, ctx->poloidal_refine) * PetscPowInt(2, ctx->toroidal_refine));
    PetscCheck(size == nn || ctx->use_360_domains, comm, PETSC_ERR_ARG_WRONG,"Number of procs (%d) != 4^N_pol (%d) * 2^N_tor (%d) = %d (not 360 domains)",(int)size,(int)ctx->poloidal_refine,(int)ctx->toroidal_refine,nn);
    nn = (int)PetscPowInt(4, ctx->poloidal_refine);
    PetscCheck(size == nn || !ctx->use_360_domains, comm, PETSC_ERR_ARG_WRONG,"Number of procs (%d) != 4^N_pol (%d) = %d (360 domains)",(int)size,(int)ctx->poloidal_refine,nn);
  }
  ctx->nlevels = ctx->poloidal_refine + ctx->toroidal_refine + 1;
  /* Domain and mesh definition */
  PetscCall(PetscOptionsReal("-radius_minor", "Minor radius of torus", "ex96.c", ctx->r, &ctx->r, NULL));
  PetscCall(PetscOptionsReal("-radius_major", "Major radius of torus", "ex96.c", ctx->R, &ctx->R, NULL));
  PetscCall(PetscOptionsReal("-radius_inflation", "inflate domain factor from minor radius", "ex96.c", ctx->r_inflate, &ctx->r_inflate, NULL));
  ctx->filename[0] = '\0';
  PetscCall(PetscOptionsString("-file", "2D mesh file on [0, 1]^2 and scaled by 2 * r_minor", "ex96.c", ctx->filename, ctx->filename, sizeof(ctx->filename), NULL));
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscFunctionReturn(PETSC_SUCCESS);
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
static PetscErrorCode ExtrudeTorus(DM dm, PetscInt n_extrude, AppCtx *ctx, DM *new_dm)
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
  PetscCall(DMPlexExtrude(dm, n_extrude, L, PETSC_FALSE, PETSC_FALSE, NULL, NULL, &dmtorus)); // need to make pencils - TODO
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
  PetscCall(PetscInfo(dm, "N = %d (final domain radius = %d)\n", (int)N, (int)(PetscPowInt(4, ctx->poloidal_refine) * PetscPowInt(2, ctx->toroidal_refine))));
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
  // PetscCall(DMLocalizeCoordinates(dmtorus));
  *new_dm = dmtorus;
  PetscFunctionReturn(PETSC_SUCCESS);
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

static PetscErrorCode SetupProblem(DM dm,  AppCtx *ctx)
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

static PetscErrorCode SetupDiscretization(DM dm, const char name[], AppCtx *ctx)
{
  PetscFE        fe;
  PetscBool      simplex = PETSC_FALSE;
  PetscInt       dim, cStart,cEnd;
  char           prefix[PETSC_MAX_PATH_LEN];
  DM        cdm = dm;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCheck(dim==ctx->dim, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG,"Initial DM dim (%d) != ctx (%d)",(int)dim,(int)ctx->dim);
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  //if (cEnd - cStart) {
    PetscCall(DMPlexIsSimplex(dm, &simplex));
    PetscCall(PetscSNPrintf(prefix, PETSC_MAX_PATH_LEN, "%s_", name));
    PetscCall(PetscFECreateDefault(PetscObjectComm((PetscObject)dm), dim, 1, simplex, name ? prefix : NULL, -1, &fe));
    PetscCall(PetscObjectSetName((PetscObject)fe, name));
    PetscCall(PetscFEViewFromOptions(fe, NULL, "-fe_view"));
    PetscCall(DMSetField(dm, 0, NULL, (PetscObject)fe));
    PetscCall(DMCreateDS(dm));
    PetscCall(SetupProblem(dm, ctx));
    while (cdm) {
      PetscCall(PetscInfo(dm, "Setup level\n"));
      PetscCall(DMCopyDisc(dm, cdm));
      PetscCall(DMGetCoarseDM(cdm, &cdm));
    }
    PetscCall(PetscFEDestroy(&fe));
    //}
  PetscFunctionReturn(PETSC_SUCCESS);
}

// create coarse grid DM. one cell for now. Read file
static PetscErrorCode CreateCoarseMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  size_t      len;

  PetscFunctionBegin;
  PetscCall(PetscStrlen(user->filename, &len));
  if (len) {
    PetscInt dim;
    PetscCall(DMPlexCreateFromFile(comm, user->filename, "torus_plex", PETSC_FALSE, dm));
    PetscCall(DMGetDimension(*dm, &dim));
    PetscCheck(dim==2, comm, PETSC_ERR_ARG_WRONG,"Initial DM dim (%d) != 2",(int)dim);
  } else {
    PetscCall(DMCreate(comm, dm));
    PetscCall(DMSetType(*dm, DMPLEX));
    PetscCall(DMPlexDistributeSetDefault(*dm, PETSC_FALSE));
    PetscCall(DMSetFromOptions(*dm)); // gets size of init grid (1x1)
  }
  PetscCall(PetscObjectSetName((PetscObject)*dm, "Coarse Mesh"));
  PetscCall(DMSetApplicationContext(*dm, user));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_2d_view"));

  PetscFunctionReturn(PETSC_SUCCESS);
}

#include <petsc/private/dmpleximpl.h>  /*I      "petscdmplex.h"   I*/

static PetscErrorCode CreateHierarchy(DM dm, AppCtx *ctx, DM *a_dmhierarchy[])
{
  DM          *dmhierarchy, pdm;
  MPI_Comm    comm = PetscObjectComm((PetscObject)dm);
  PetscBool isUniform;
  //PetscMPIInt rank, size;
  char fin_str[] = "-tor_dm_view_0";

  PetscFunctionBeginUser;
  PetscCall(PetscMalloc(sizeof(DM) * ctx->nlevels, &dmhierarchy));
  *a_dmhierarchy = dmhierarchy;
  for (PetscInt r = 0; r < ctx->nlevels; r++) dmhierarchy[r] = NULL;
  PetscCall(DMPlexGetRefinementUniform(dm, &isUniform));
  PetscCheck(isUniform, comm, PETSC_ERR_ARG_WRONG,"not isUniform");
  // make 2D refined grid nref_pol + nref_tor
  dmhierarchy[0] = dm;
  for (PetscInt r = 1; r < ctx->poloidal_refine + 1 ; r++) {
    PetscCall(DMRefine(dmhierarchy[r - 1], MPI_COMM_NULL, &dmhierarchy[r]));
    if (dmhierarchy[r]) {
      ((DM_Plex *)(dmhierarchy[r])->data)->printFEM = ((DM_Plex *)dm->data)->printFEM;
      ((DM_Plex *)(dmhierarchy[r])->data)->printL2  = ((DM_Plex *)dm->data)->printL2;
    }
    PetscCall(PetscObjectSetName((PetscObject)dmhierarchy[r], "plane"));
    PetscCall(DMViewFromOptions(dmhierarchy[r], NULL, "-dm_2d_view"));
  }
  // duplicate the 2D plane for toroidal refine levels
  for (PetscInt r =  ctx->poloidal_refine + 1 ; r < ctx->poloidal_refine + ctx->toroidal_refine + 1 ; r++) {
    PetscCall(DMClone(dmhierarchy[r - 1], &dmhierarchy[r]));
    if (dmhierarchy[r]) {
      ((DM_Plex *)(dmhierarchy[r])->data)->printFEM = ((DM_Plex *)dm->data)->printFEM;
      ((DM_Plex *)(dmhierarchy[r])->data)->printL2  = ((DM_Plex *)dm->data)->printL2;
    }
    PetscCall(PetscObjectSetName((PetscObject)dmhierarchy[r], "plane"));
    PetscCall(DMViewFromOptions(dmhierarchy[r], NULL, "-dm_2d_view"));
  }
  // refine plane extrude each grid out to populate the coarse grid torus
  for (PetscInt r = 0; r < ctx->poloidal_refine + 1 ; r++) {
    char torusstr[] = "torus-x-pol";
    torusstr[6] = '0' + r;
    PetscCall(PetscInfo(dmhierarchy[r], "%d) extrude (polodal) %s\n", (int)r, torusstr));
    PetscCheck(dmhierarchy[r], comm, PETSC_ERR_ARG_WRONG, "refinement failed");
    /* extrude coarse */
    if (ctx->dim > 2) {
      DM ext_dm;
      PetscCall(ExtrudeTorus(dmhierarchy[r], ctx->coarse_toroidal_faces, ctx, &ext_dm));
      PetscCall(DMDestroy(&dmhierarchy[r]));
      dmhierarchy[r] = ext_dm;
      //PetscCall(DMLocalizeCoordinates(ext_dm)); // periodic
    } else {
      PetscCall(OriginShift2D(dmhierarchy[r], ctx)); // shift to center
    }
    PetscCall(DMPlexDistribute(dmhierarchy[r], 0, NULL, &pdm));
    if (pdm) {
      //PetscBool localized;
      //PetscCall(DMGetCoordinatesLocalized(dmhierarchy[r], &localized));
      //if (ctx->dim > 2) PetscCheck(localized, comm, PETSC_ERR_ARG_WRONG,"not localized");
      PetscCall(DMDestroy(&dmhierarchy[r]));
      dmhierarchy[r] = pdm;
      // PetscCall(DMGetCoordinatesLocalized(pdm, &localized));
      //if (ctx->dim > 2) PetscCheck(localized, comm, PETSC_ERR_ARG_WRONG,"not localized");
    }
    /* view */
    if (ctx->dim > 2) PetscCall(PetscObjectSetName((PetscObject)dmhierarchy[r], torusstr));
    else PetscCall(PetscObjectSetName((PetscObject)dmhierarchy[r], "plane-single-pol"));
    PetscCall(DMViewFromOptions(dmhierarchy[r], NULL, "-ref1_dm_view"));
  }
  // extrude the toroidal
  if (ctx->dim > 2) {
    for (PetscInt ri = 0, r = ctx->poloidal_refine + 1, nref = 2 * ctx->coarse_toroidal_faces ; ri < ctx->toroidal_refine; ri++, r++, nref *= 2) {
      char torusstr[] = "torus-x-tor";
      torusstr[6] = '0' + r;
      PetscCall(PetscInfo(dmhierarchy[r], "%d) extrude (toroidal) %s\n", (int)r, torusstr));
      // uniform refinement in toroidal direction - TODO
      /* extrude coarse */
      if (ctx->dim > 2) {
        DM ext_dm;
        PetscCall(ExtrudeTorus(dmhierarchy[r], nref, ctx, &ext_dm));
        PetscCall(DMDestroy(&dmhierarchy[r]));
        dmhierarchy[r] = ext_dm;
        //PetscCall(DMLocalizeCoordinates(ext_dm)); // periodic
      } else {
        PetscCall(OriginShift2D(dmhierarchy[r], ctx)); // shift to center
      }
      PetscCall(DMPlexDistribute(dmhierarchy[r], 0, NULL, &pdm));
      if (pdm) {
        //PetscBool localized;
        //PetscCall(DMGetCoordinatesLocalized(dmhierarchy[r], &localized));
        //if (ctx->dim > 2) PetscCheck(localized, comm, PETSC_ERR_ARG_WRONG,"not localized");
        PetscCall(DMDestroy(&dmhierarchy[r]));
        dmhierarchy[r] = pdm;
        //PetscCall(DMGetCoordinatesLocalized(pdm, &localized));
        //if (ctx->dim > 2) PetscCheck(localized, comm, PETSC_ERR_ARG_WRONG,"not localized");
      }
      /* view */
      if (ctx->dim > 2) PetscCall(PetscObjectSetName((PetscObject)dmhierarchy[r], torusstr));
      else PetscCall(PetscObjectSetName((PetscObject)dmhierarchy[r], "plane-single-tor"));
      PetscCall(DMViewFromOptions(dmhierarchy[r], NULL, "-ref1_dm_view"));
    }
  }
  // uniform refinement
  for (PetscInt r = 0; r < ctx->nlevels ; r++) {
    DM ext_dm;
    char tokstr[] = "tokamak-x";
    fin_str[13] = '0' + r;
    tokstr[8] = '0' + r;
    PetscCall(DMRefine(dmhierarchy[r], MPI_COMM_NULL, &ext_dm));
    PetscCall(DMDestroy(&dmhierarchy[r]));
    dmhierarchy[r] = ext_dm;
    if (r>0) PetscCall(DMSetCoarseDM(dmhierarchy[r], dmhierarchy[r - 1]));
    /* view - coarse grid r is done */
    if (ctx->dim > 2) PetscCall(PetscObjectSetName((PetscObject)dmhierarchy[r], tokstr));
    else PetscCall(PetscObjectSetName((PetscObject)dmhierarchy[r], "plane"));
    PetscCall(DMViewFromOptions(dmhierarchy[r], NULL, fin_str));
    PetscCall(DMViewFromOptions(dmhierarchy[r], NULL, "-ref_dm_view"));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode proc_func(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf_dummy, PetscScalar *u, void *actx)
{
  PetscFunctionBegin;
  u[0]  = time;
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  AppCtx     actx,*ctx=&actx; /* work context */
  DM         dm_crs, dm, *dmhierarchy;
  SNES snes;
  PC pc;
  TS ts;
  KSP ksp;
  PetscBool same = PETSC_FALSE;
  Vec         u;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, ctx));
  /* Create Plex - serial DM for serial coarse grid */
  PetscCall(CreateCoarseMesh(PETSC_COMM_WORLD, ctx, &dm_crs));
  PetscCall(CreateHierarchy(dm_crs, ctx, &dmhierarchy));
  dm = dmhierarchy[ctx->nlevels-1]; // fine grid
  /* Setup problem */
  PetscCall(SetupDiscretization(dm, "potential", ctx));
  PetscCall(DMPlexCreateClosureIndex(dm, NULL)); // performance
  /* solver */
  PetscCall(TSCreate(PETSC_COMM_WORLD, &ts));
  PetscCall(TSSetDM(ts, dm));
  PetscCall(TSSetType(ts, TSBEULER));
  PetscCall(TSSetMaxTime(ts, 1.0));
  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_STEPOVER));
  PetscCall(TSSetTimeStep(ts, 0.01));
  PetscCall(TSSetMaxSteps(ts, 1));
  PetscCall(DMTSSetBoundaryLocal(dm, DMPlexTSComputeBoundary, &ctx));
  PetscCall(DMTSSetIFunctionLocal(dm, DMPlexTSComputeIFunctionFEM, &ctx));
  PetscCall(DMTSSetIJacobianLocal(dm, DMPlexTSComputeIJacobianFEM, &ctx));
  // setup solver - default MG
  PetscCall(TSGetSNES(ts, &snes));
  PetscCall(SNESGetKSP(snes, &ksp));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCMG));
  PetscCall(PCMGSetLevels(pc, ctx->nlevels, NULL));
  PetscCall(PCMGSetType(pc, PC_MG_MULTIPLICATIVE));
  PetscCall(PCMGSetGalerkin(pc, PC_MG_GALERKIN_BOTH));
  PetscCall(PCMGSetCycleType(pc, PC_MG_CYCLE_V));
  PetscCall(PCMGSetNumberSmooth(pc, 2));
  PetscCall(TSSetFromOptions(ts));
  // MG setup
  PetscCall(PetscObjectTypeCompare((PetscObject)pc, PCMG, &same));
  if (same) { //PetscCall(PCMGSetupViaCoarsen(pc, da_Stokes));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Set Up MG\n"));
    for (PetscInt k = 1; k < ctx->nlevels ; k++) {
      Mat R;
      PetscInt M,N;
      PetscCall(DMCreateInterpolation(dmhierarchy[k - 1], dmhierarchy[k], &R, NULL));
      PetscCall(PCMGSetInterpolation(pc, k, R));
      PetscCall(MatGetSize(R, &M, &N));
      PetscCall(PetscInfo(dm, "%d) R is %d x %d\n", (int)k, (int)M, (int)N));
      PetscCall(MatDestroy(&R));
    }
  }
  PetscCall(DMViewFromOptions(dm, NULL, "-tor_dm_view2"));
  PetscCall(DMViewFromOptions(dm, NULL, "-tor_dm_view"));
  if (1) {
    DM celldm;
    PetscFE        fe;
    Vec uu;
    PetscCall(DMClone(dm, &celldm));
    PetscCall(PetscFECreateDefault(PETSC_COMM_WORLD, ctx->dim, 1, PETSC_FALSE,"dummy", -1, &fe));
    PetscCall(PetscObjectSetName((PetscObject)fe, "cell"));
    PetscCall(PetscFEViewFromOptions(fe, NULL, "-fe_cell_view"));
    PetscCall(DMSetField(celldm, 0, NULL, (PetscObject)fe));
    PetscCall(DMCreateDS(celldm));
    PetscCall(DMCreateGlobalVector(celldm, &uu));
    PetscCall(PetscObjectSetName((PetscObject)uu, "uu"));
    PetscErrorCode (*initu[1])(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar[], void *);
    initu[0]        = proc_func;
    PetscMPIInt rank;
    PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
    PetscCall(DMProjectFunction(celldm, (PetscReal)rank, initu, NULL, INSERT_ALL_VALUES, uu));
    PetscCall(DMViewFromOptions(celldm, NULL, "-cell_dm_view"));
    PetscCall(VecViewFromOptions(uu, NULL, "-cell_vec_view"));
    PetscCall(PetscFEDestroy(&fe));
    PetscCall(DMDestroy(&celldm));
    PetscCall(VecDestroy(&uu));
  }
  PetscCall(DMCreateGlobalVector(dm, &u));
  PetscCall(TSSetSolution(ts, u));
  PetscCall(VecSet(u, 0.0));
  PetscCall(PetscObjectSetName((PetscObject)u, "u"));
  /* view */
  PetscCall(DMSetOutputSequenceNumber(dm, 0, 0.0));
  PetscCall(VecViewFromOptions(u, NULL, "-tor_vec_view"));
  /* solve */
  PetscCall(TSSolve(ts, u));
  /* view */
  PetscCall(DMSetOutputSequenceNumber(dm, 1, 1.0));
  PetscCall(VecViewFromOptions(u, NULL, "-tor_vec_view"));
  // cleanup
  for (PetscInt r = 0;  r < ctx->nlevels ; r++) PetscCall(DMDestroy(&dmhierarchy[r]));
  PetscCall(PetscFree(dmhierarchy));
  PetscCall(VecDestroy(&u));
  PetscCall(TSDestroy(&ts));

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
