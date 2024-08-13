static char help[] = "Fermions on a hypercubic lattice.\n\n";

#include <petscdmplex.h>
#include <petscsnes.h>
#include <petscgrid.h>
#ifdef PETSC_HAVE_SLEPC
  #include <slepc.h>
#endif

#define MAX_D5 100
/* Common operations:

 - View the input \psi as ASCII in lexicographic order: -psi_view
*/
// Wilson quark mass
const PetscReal M = -0.92;
//const PetscReal M = 0.1;

static inline void TwoSpinAccumulate(PetscInt, PetscBool, PetscInt, const PetscScalar*, PetscScalar*);
static inline void TwoSpinProject(PetscInt , PetscBool , PetscInt,const PetscScalar*,PetscScalar*);
typedef struct {
  PetscBool     usePV; /* Use Pauli-Villars preconditioning */
  PetscBool     useEPS;
  PetscBool     normalEq;
  PetscBool     coarsen;
  PetscInt      domainWall;
  char          gridFile[PETSC_MAX_PATH_LEN];
  GRID_LOAD_TYPE gauge_type;
  PetscBool     load_sol;
  PetscBool     write_sol;
  PetscBool     solveFineAndCoarse;
  EPS           eps;
  PetscBool     write_eig;
  PetscInt      numEigenVectors;
  PetscBool     testCoarsening;
  PetscReal     shift;
  PetscBool     load_ic;
  PetscBool     write_ic;
  PetscReal     mass;
  PetscBool     DWsqr;
  Mat           pv;
} AppCtx;

PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscFunctionBegin;
  options->usePV      = PETSC_FALSE;
  options->useEPS     = PETSC_FALSE;
  options->normalEq   = PETSC_FALSE;
  options->domainWall = 0;
  options->coarsen    = PETSC_FALSE;
  options->load_sol   = PETSC_FALSE;
  options->write_sol  = PETSC_FALSE;
  options->solveFineAndCoarse = PETSC_FALSE;
  options->write_eig  = PETSC_FALSE;
  options->numEigenVectors = 4096;
  options->testCoarsening = PETSC_FALSE;
  options->shift          = 0.01;
  options->write_ic       = PETSC_TRUE;
  options->load_ic        = PETSC_FALSE;
  options->mass           = -0.92;
  options->DWsqr          = PETSC_FALSE;

  PetscOptionsBegin(comm, "", "Meshing Problem Options", "DMPLEX");
  PetscCall(PetscOptionsBool("-coarsen", "flag to set up coarse grid", "ex7.c", options->coarsen, &options->coarsen, NULL));
  PetscCall(PetscOptionsBool("-use_pv", "Use Pauli-Villars preconditioning", "ex7.c", options->usePV, &options->usePV, NULL));
  PetscCall(PetscOptionsBool("-use_eps_eigenvalues", "Solve for eigenvalues and use with the Chebyshev solver", "ex7.c", options->useEPS, &options->useEPS, NULL));
  PetscCall(PetscOptionsBool("-use_normal_equation", "Solve for eigenvalues and use with the Chebyshev solver", "ex7.c", options->normalEq, &options->normalEq, NULL));
  PetscCall(PetscOptionsBool("-use_fine_and_coarse", "Solve each level individually to get the projected error of the coarse basis", "ex7.c", options->solveFineAndCoarse, &options->solveFineAndCoarse, NULL));
  PetscCall(PetscOptionsBool("-save_eigenbasis", "Save each eigenvector from the EPS", "ex7.c", options->write_eig, &options->write_eig, NULL));
  PetscCall(PetscOptionsBool("-test_coarsening", "Save each eigenvector from the EPS", "ex7.c", options->testCoarsening, &options->testCoarsening, NULL));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-grid_file", options->gridFile, sizeof(options->gridFile), NULL));
  PetscCall(PetscOptionsInt("-run_domain_wall", "Number of 5D slices", "ex7.c", options->domainWall, &options->domainWall, NULL));
  PetscCall(PetscOptionsBool("-load_sol", "Flag to load a solution from a previous solve.", "ex7.c", options->load_sol, &options->load_sol, NULL));
  PetscCall(PetscOptionsBool("-run_squared_domain_wall", "Use squared operator", "ex7.c", options->DWsqr, &options->DWsqr, NULL));
  PetscCall(PetscOptionsBool("-write_sol", "Flag to write the solution from current solve to disc.", "ex7.c", options->write_sol, &options->write_sol, NULL));
  PetscCall(PetscOptionsBool("-write_ic", "Flag to write the solution from current solve to disc.", "ex7.c", options->write_ic, &options->write_ic, NULL));
  PetscCall(PetscOptionsBool("-load_ic", "Flag to write the solution from current solve to disc.", "ex7.c", options->load_ic, &options->load_ic, NULL));
  PetscCall(PetscOptionsReal("-temperature", "Temperature shift for initial guage field configuration", "ex7.c", options->shift, &options->shift, NULL));
  PetscCall(PetscOptionsReal("-mass", "fermion mass parameter", "ex7.c", options->mass, &options->mass, NULL));
  //PetscCall(PetscOptionsEnum("-grid_load_type", "How to initialize data from grid", NULL, GRID_LOAD_TYPE, (PetscEnum)options->gauge_type, (PetscEnum *)&options->gauge_type, NULL));
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscFunctionBegin;
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMSetApplicationContext(*dm, user));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetupDiscretization(DM dm, AppCtx *ctx)
{
  PetscSection s;
  PetscInt     vStart, vEnd, v;

  PetscFunctionBegin;
  PetscCall(PetscSectionCreate(PETSC_COMM_SELF, &s));
  PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  PetscCall(PetscSectionSetChart(s, vStart, vEnd));
  for (v = vStart; v < vEnd; ++v) {
    PetscCall(PetscSectionSetDof(s, v, 12));
    /* TODO Divide the values into fields/components */
  }
  PetscCall(PetscSectionSetUp(s));
  PetscCall(DMSetLocalSection(dm, s));
  PetscCall(PetscSectionDestroy(&s));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetupAuxDiscretization(DM dm, AppCtx *user)
{
  DM           dmAux, coordDM;
  PetscSection s;
  Vec          gauge;
  PetscInt     eStart, eEnd, e;

  PetscFunctionBegin;
  /* MUST call DMGetCoordinateDM() in order to get p4est setup if present */
  PetscCall(DMGetCoordinateDM(dm, &coordDM));
  PetscCall(DMClone(dm, &dmAux));
  PetscCall(DMSetCoordinateDM(dmAux, coordDM));
  PetscCall(PetscSectionCreate(PETSC_COMM_SELF, &s));
  PetscCall(DMPlexGetDepthStratum(dm, 1, &eStart, &eEnd));
  PetscCall(PetscSectionSetChart(s, eStart, eEnd));
  for (e = eStart; e < eEnd; ++e) {
    /* TODO Should we store the whole SU(3) matrix, or the symmetric part? */
    PetscCall(PetscSectionSetDof(s, e, 9));
  }
  PetscCall(PetscSectionSetUp(s));
  PetscCall(DMSetLocalSection(dmAux, s));
  PetscCall(PetscSectionDestroy(&s));
  PetscCall(DMCreateLocalVector(dmAux, &gauge));
  PetscCall(DMDestroy(&dmAux));
  PetscCall(DMSetAuxiliaryVec(dm, NULL, 0, 0, gauge));
  PetscCall(VecDestroy(&gauge));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PrintVertex(DM dm, PetscInt v)
{
  MPI_Comm       comm;
  PetscContainer c;
  PetscInt      *extent;
  PetscInt       dim, cStart, cEnd, sum;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  PetscCall(PetscObjectQuery((PetscObject)dm, "_extent", (PetscObject *)&c));
  PetscCall(PetscContainerGetPointer(c, (void **)&extent));
  sum = 1;
  PetscCall(PetscPrintf(comm, "Vertex %" PetscInt_FMT ":", v));
  for (PetscInt d = 0; d < dim; ++d) {
    PetscCall(PetscPrintf(comm, " %" PetscInt_FMT, (v / sum) % extent[d]));
    if (d < dim) sum *= extent[d];
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Apply \gamma_\mu
static PetscErrorCode ComputeGamma(PetscInt d, PetscInt ldx, PetscScalar f[])
{
  const PetscScalar fin[4] = {f[0 * ldx], f[1 * ldx], f[2 * ldx], f[3 * ldx]};

  PetscFunctionBeginHot;
  switch (d) {
  case 0:
    f[0 * ldx] = PETSC_i * fin[3];
    f[1 * ldx] = PETSC_i * fin[2];
    f[2 * ldx] = -PETSC_i * fin[1];
    f[3 * ldx] = -PETSC_i * fin[0];
    break;
  case 1:
    f[0 * ldx] = -fin[3];
    f[1 * ldx] = fin[2];
    f[2 * ldx] = fin[1];
    f[3 * ldx] = -fin[0];
    break;
  case 2:
    f[0 * ldx] = PETSC_i * fin[2];
    f[1 * ldx] = -PETSC_i * fin[3];
    f[2 * ldx] = -PETSC_i * fin[0];
    f[3 * ldx] = PETSC_i * fin[1];
    break;
  case 3:
    f[0 * ldx] = fin[2];
    f[1 * ldx] = fin[3];
    f[2 * ldx] = fin[0];
    f[3 * ldx] = fin[1];
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Direction for gamma %" PetscInt_FMT " not in [0, 4)", d);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Apply (1 \pm \gamma_\mu)/2
static inline PetscErrorCode ComputeGammaFactor(PetscInt d, PetscBool forward, PetscInt ldx, PetscScalar f[])
{
  const PetscReal   sign   = forward ? -1. : 1.;
  const PetscScalar fin[4] = {f[0 * ldx], f[1 * ldx], f[2 * ldx], f[3 * ldx]};

  PetscFunctionBeginHot;
  switch (d) {
  case 0:
    f[0 * ldx] += sign * PETSC_i * fin[3];
    f[1 * ldx] += sign * PETSC_i * fin[2];
    f[2 * ldx] += sign * -PETSC_i * fin[1];
    f[3 * ldx] += sign * -PETSC_i * fin[0];
    break;
  case 1:
    f[0 * ldx] += -sign * fin[3];
    f[1 * ldx] += sign * fin[2];
    f[2 * ldx] += sign * fin[1];
    f[3 * ldx] += -sign * fin[0];
    break;
  case 2:
    f[0 * ldx] += sign * PETSC_i * fin[2];
    f[1 * ldx] += sign * -PETSC_i * fin[3];
    f[2 * ldx] += sign * -PETSC_i * fin[0];
    f[3 * ldx] += sign * PETSC_i * fin[1];
    break;
  case 3:
    f[0 * ldx] += sign * fin[2];
    f[1 * ldx] += sign * fin[3];
    f[2 * ldx] += sign * fin[0];
    f[3 * ldx] += sign * fin[1];
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Direction for gamma %" PetscInt_FMT " not in [0, 4)", d);
  }
  f[0 * ldx] *= 0.5;
  f[1 * ldx] *= 0.5;
  f[2 * ldx] *= 0.5;
  f[3 * ldx] *= 0.5;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#include <petsc/private/dmpleximpl.h>

// ComputeAction() sums the action of 1/2 (1 \pm \gamma_\mu) U \psi into f
static PetscErrorCode ComputeAction(PetscInt d, PetscBool forward, PetscBool dag, const PetscScalar U[], const PetscScalar psi[], PetscScalar f[])
{
  PetscScalar tmp[12], utmp[12];
  PetscBool gforward;

  PetscFunctionBeginHot;
  if ( dag ) {
    if ( forward ) gforward = PETSC_FALSE;
    else           gforward = PETSC_TRUE;
  } else {
    gforward = forward;
  }

  for (PetscInt c = 0; c < 3; ++c) TwoSpinProject(d, forward, 3, &psi[c], &tmp[c]);
  for (PetscInt beta = 0; beta < 4; ++beta) {
    if (forward) DMPlex_Mult3D_Internal(U, 1, &tmp[beta * 3], &utmp[beta * 3]);
    else DMPlex_MultTranspose3D_Internal(U, 1, &tmp[beta * 3], &utmp[beta * 3]);
  }
  for (PetscInt c = 0; c < 3; ++c) TwoSpinAccumulate(d, gforward, 3, &utmp[c], &f[c]);
  PetscFunctionReturn(0);
}

/*
  The assembly loop runs over vertices. Each vertex has 2d edges in its support. The edges are ordered first by the dimension along which they run, and second from smaller to larger index, expect for edges which loop periodically. The vertices on edges are also ordered from smaller to larger index except for periodic edges.
*/
static PetscErrorCode ComputeResidual(Mat F, Vec u, Vec f, PetscBool dag)
{
  DM                 dm, dmAux;
  Vec                gauge;
  PetscSection       s, sGauge;
  const PetscScalar *ua;
  PetscScalar       *fa, *link;
  PetscInt           dim, vStart, vEnd;
  AppCtx            *user;

  PetscFunctionBeginUser;
  PetscCall(MatGetDM(F, &dm));
  PetscCall(DMGetApplicationContext(dm, &user));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMGetLocalSection(dm, &s));
  PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  PetscCall(VecGetArrayRead(u, &ua));
  PetscCall(VecGetArray(f, &fa));

  PetscCall(DMGetAuxiliaryVec(dm, NULL, 0, 0, &gauge));
  PetscCall(VecViewFromOptions(gauge, NULL, "-residual_gauge_view"));
  PetscCall(VecGetDM(gauge, &dmAux));
  PetscCall(DMGetLocalSection(dmAux, &sGauge));
  PetscCall(VecGetArray(gauge, &link));
  // Loop over y
  for (PetscInt v = vStart; v < vEnd; ++v) {
    const PetscInt *supp;
    PetscInt        xdof, xoff;

    PetscCall(DMPlexGetSupport(dm, v, &supp));
    PetscCall(PetscSectionGetDof(s, v, &xdof));
    PetscCall(PetscSectionGetOffset(s, v, &xoff));
    // Diagonal
    for (PetscInt i = 0; i < xdof; ++i) fa[xoff + i] += (user->mass + 4) * ua[xoff + i];
    // Loop over mu
    for (PetscInt d = 0; d < dim; ++d) {
      const PetscInt *cone;
      PetscInt        yoff, goff;

      // Left action -(1 + \gamma_\mu)/2 \otimes U^\dagger_\mu(y) \delta_{x - \mu,y} \psi(y)
      PetscCall(DMPlexGetCone(dm, supp[2 * d + 0], &cone));
      PetscCall(PetscSectionGetOffset(s, cone[0], &yoff));
      PetscCall(PetscSectionGetOffset(sGauge, supp[2 * d + 0], &goff));
      PetscCall(ComputeAction(d, PETSC_FALSE, dag, &link[goff], &ua[yoff], &fa[xoff]));
      // Right edge -(1 - \gamma_\mu)/2 \otimes U_\mu(x) \delta_{x + \mu,y} \psi(y)
      PetscCall(DMPlexGetCone(dm, supp[2 * d + 1], &cone));
      PetscCall(PetscSectionGetOffset(s, cone[1], &yoff));
      PetscCall(PetscSectionGetOffset(sGauge, supp[2 * d + 1], &goff));
      PetscCall(ComputeAction(d, PETSC_TRUE, dag, &link[goff], &ua[yoff], &fa[xoff]));
    }
  }
  PetscCall(VecRestoreArray(f, &fa));
  PetscCall(VecRestoreArray(gauge, &link));
  PetscCall(VecRestoreArrayRead(u, &ua));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeResidual_Forward(Mat F, Vec u, Vec f)
{
  PetscFunctionBeginUser;
  PetscCall(ComputeResidual(F, u, f, PETSC_FALSE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeResidual_Dagger(Mat F, Vec u, Vec f)
{
  PetscFunctionBeginUser;
  PetscCall(ComputeResidual(F, u, f, PETSC_TRUE));
  PetscFunctionReturn(PETSC_SUCCESS);
}
/*
-- From Peter Boyle at https://github.com/paboyle/PETSc-Grid/blob/main/petsc_fermion.h --
*/

// Apply (1 \pm \gamma_\mu)/2
static PetscErrorCode SpinProject(PetscInt mu, PetscBool minus, PetscInt ldx, PetscScalar f[])
{
  const PetscReal   sign   = (minus )   ? -1. : 1.;
  const PetscScalar fin[4] = {f[0 * ldx], f[1 * ldx], f[2 * ldx], f[3 * ldx]};

  PetscFunctionBeginHot;
  switch (mu) {
  case 0:
    f[0 * ldx] += sign * PETSC_i * fin[3];
    f[1 * ldx] += sign * PETSC_i * fin[2];
    f[2 * ldx] += sign * -PETSC_i * fin[1];
    f[3 * ldx] += sign * -PETSC_i * fin[0];
    break;
  case 1:
    f[0 * ldx] += -sign * fin[3];
    f[1 * ldx] += sign * fin[2];
    f[2 * ldx] += sign * fin[1];
    f[3 * ldx] += -sign * fin[0];
    break;
  case 2:
    f[0 * ldx] += sign * PETSC_i * fin[2];
    f[1 * ldx] += sign * -PETSC_i * fin[3];
    f[2 * ldx] += sign * -PETSC_i * fin[0];
    f[3 * ldx] += sign * PETSC_i * fin[1];
    break;
  case 3:
    f[0 * ldx] += sign * fin[2];
    f[1 * ldx] += sign * fin[3];
    f[2 * ldx] += sign * fin[0];
    f[3 * ldx] += sign * fin[1];
    break;
  case 4:
    f[0 * ldx] += sign * fin[0];
    f[1 * ldx] += sign * fin[1];
    f[2 * ldx] -= sign * fin[2];
    f[3 * ldx] -= sign * fin[3];
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Direction for gamma %" PetscInt_FMT " not in [0, 5)", mu);
  }
  f[0 * ldx] *= 0.5;
  f[1 * ldx] *= 0.5;
  f[2 * ldx] *= 0.5;
  f[3 * ldx] *= 0.5;
  PetscFunctionReturn(0);
}

static PetscErrorCode DdwfDhop(PetscInt d, PetscBool forward, PetscBool dag, const PetscScalar U[], const PetscScalar psi[], PetscScalar f[])
{
  PetscScalar tmp[12];

  PetscFunctionBeginHot;
  // Apply U
  for (PetscInt beta = 0; beta < 4; ++beta) {
    if (forward) DMPlex_Mult3D_Internal(U, 1, &psi[beta * 3], &tmp[beta * 3]);
    else DMPlex_MultTranspose3D_Internal(U, 1, &psi[beta * 3], &tmp[beta * 3]);
  }
  int gamma[] = {4,0,1,2,3};
  PetscBool gforward;
  if ( dag ) {
    if ( forward ) gforward = PETSC_FALSE;
    else           gforward = PETSC_TRUE;
  } else {
    gforward = forward;
  }
  // Apply (1 \pm \gamma_\mu)/2 to each color for gamma = xyzt
  //  PetscBool gforward = forward;
  for (PetscInt c = 0; c < 3; ++c) PetscCall(SpinProject(gamma[d], gforward, 3, &tmp[c]));
  // Note that we are subtracting this contribution
  for (PetscInt i = 0; i < 12; ++i) f[i] -= tmp[i];
  PetscFunctionReturn(0);
}

static PetscErrorCode DdwfInternal(Mat M, Vec u, Vec f, PetscBool dag)
{
  DM                 dm, dmAux;
  Vec                gauge;
  PetscSection       s, sGauge;
  const PetscScalar *ua;
  PetscScalar       *fa, *link, M5 =1.8;
  PetscInt           dim, vStart, vEnd;
  AppCtx            *ctx;

  PetscFunctionBeginUser;
  PetscCall(MatGetDM(M, &dm));
  PetscCall(DMGetApplicationContext(dm, &ctx));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMGetLocalSection(dm, &s));
  PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  PetscCall(VecGetArrayRead(u, &ua));
  PetscCall(VecGetArray(f, &fa));

  PetscCall(DMGetAuxiliaryVec(dm, NULL, 0, 0, &gauge));
  PetscCall(VecGetDM(gauge, &dmAux));
  PetscCall(DMGetLocalSection(dmAux, &sGauge));
  PetscCall(VecGetArray(gauge, &link));
  // Loop over y
  for (PetscInt v = vStart; v < vEnd; ++v) {
    const PetscInt *supp;
    PetscInt        xdof, xoff;

    PetscCall(DMPlexGetSupport(dm, v, &supp));
    PetscCall(PetscSectionGetDof(s, v, &xdof));
    PetscCall(PetscSectionGetOffset(s, v, &xoff));
    // Diagonal
    for (PetscInt i = 0; i < xdof; ++i) fa[xoff + i] = (5-M5 ) * ua[xoff + i];
    // Loop over mu
    for (PetscInt d = 0; d < dim; ++d) {
      const PetscInt *cone;
      PetscInt        yoff, goff;

      // Left action -(1 + \gamma_\mu)/2 \otimes U^\dagger_\mu(y) \delta_{x - \mu,y} \psi(y)
      PetscCall(DMPlexGetCone(dm, supp[2 * d + 0], &cone));
      PetscCall(PetscSectionGetOffset(s, cone[0], &yoff));
      PetscCall(PetscSectionGetOffset(sGauge, supp[2 * d + 0], &goff));
      PetscCall(DdwfDhop(d, PETSC_FALSE, dag, &link[goff], &ua[yoff], &fa[xoff]));
      // Right edge -(1 - \gamma_\mu)/2 \otimes U_\mu(x) \delta_{x + \mu,y} \psi(y)
      PetscCall(DMPlexGetCone(dm, supp[2 * d + 1], &cone));
      PetscCall(PetscSectionGetOffset(s, cone[1], &yoff));
      PetscCall(PetscSectionGetOffset(sGauge, supp[2 * d + 1], &goff));
      PetscCall(DdwfDhop(d, PETSC_TRUE, dag, &link[goff], &ua[yoff], &fa[xoff]));
    }
  }
  PetscCall(VecRestoreArray(f, &fa));
  PetscCall(VecRestoreArray(gauge, &link));
  PetscCall(VecRestoreArrayRead(u, &ua));
  PetscFunctionReturn(0);
}
static PetscErrorCode Ddwf(Mat M, Vec u, Vec f)
{
  return DdwfInternal(M,u,f,PETSC_FALSE);
}
static PetscErrorCode DdwfDag(Mat M, Vec u, Vec f)
{
  return DdwfInternal(M,u,f,PETSC_TRUE);
}
static PetscErrorCode DdwfDagDdwf(Mat M, Vec u, Vec f)
{
  Vec tmp;
  DM  dm;

  PetscFunctionBegin;
  PetscCall(MatGetDM(M, &dm));
  PetscCall(DMCreateGlobalVector(dm, &tmp));
  PetscCall(VecZeroEntries(tmp));// There is no guarantee thise vec is zerod when pulled from the dm
  DdwfInternal(M,u,tmp,PETSC_FALSE);
  DdwfInternal(M,tmp,f,PETSC_TRUE);
  PetscCall(VecDestroy(&tmp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PrecOp(Mat M, Vec u, Vec f)
{
  Vec tmp;
  DM  dm;
  AppCtx *user;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(M, &user));
  PetscCall(MatGetDM(M, &dm));
  PetscCall(DMCreateGlobalVector(dm, &tmp));
  PetscCall(VecZeroEntries(tmp));// There is no guarantee thise vec is zerod when pulled from the dm
  DdwfInternal(M,u,tmp,PETSC_FALSE);
  DdwfInternal(user->pv,tmp,f,PETSC_TRUE);
  PetscCall(VecDestroy(&tmp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PrecOpDag(Mat M, Vec u, Vec f)
{
  Vec tmp;
  DM  dm;
  AppCtx *user;

  PetscFunctionBegin;

  PetscCall(MatShellGetContext(M, &user));
  PetscCall(MatGetDM(M, &dm));
  PetscCall(DMCreateGlobalVector(dm, &tmp));
  PetscCall(VecZeroEntries(tmp));// There is no guarantee thise vec is zerod when pulled from the dm
  DdwfInternal(M,u,tmp,PETSC_TRUE);
  DdwfInternal(user->pv,tmp,f,PETSC_FALSE);
  PetscCall(VecDestroy(&tmp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
 -----------------------------------------------------------------------------------
*/

static PetscErrorCode PrintTraversal(DM dm)
{
  MPI_Comm comm;
  PetscInt vStart, vEnd;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  for (PetscInt v = vStart; v < vEnd; ++v) {
    const PetscInt *supp;
    PetscInt        Ns;

    PetscCall(DMPlexGetSupportSize(dm, v, &Ns));
    PetscCall(DMPlexGetSupport(dm, v, &supp));
    PetscCall(PrintVertex(dm, v));
    PetscCall(PetscPrintf(comm, "\n"));
    for (PetscInt s = 0; s < Ns; ++s) {
      const PetscInt *cone;

      PetscCall(DMPlexGetCone(dm, supp[s], &cone));
      PetscCall(PetscPrintf(comm, "  Edge %" PetscInt_FMT ": ", supp[s]));
      PetscCall(PrintVertex(dm, cone[0]));
      PetscCall(PetscPrintf(comm, " -- "));
      PetscCall(PrintVertex(dm, cone[1]));
      PetscCall(PetscPrintf(comm, "\n"));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeFFT(Mat FT, PetscInt Nc, Vec x, Vec p)
{
  Vec     *xComp, *pComp;
  PetscInt n, N;

  PetscFunctionBeginUser;
  PetscCall(PetscMalloc2(Nc, &xComp, Nc, &pComp));
  PetscCall(VecGetLocalSize(x, &n));
  PetscCall(VecGetSize(x, &N));
  for (PetscInt i = 0; i < Nc; ++i) {
    const char *vtype;

    // HACK: Make these from another DM up front
    PetscCall(VecCreate(PetscObjectComm((PetscObject)x), &xComp[i]));
    PetscCall(VecGetType(x, &vtype));
    PetscCall(VecSetType(xComp[i], vtype));
    PetscCall(VecSetSizes(xComp[i], n / Nc, N / Nc));
    PetscCall(VecDuplicate(xComp[i], &pComp[i]));
  }
  PetscCall(VecStrideGatherAll(x, xComp, INSERT_VALUES));
  for (PetscInt i = 0; i < Nc; ++i) PetscCall(MatMult(FT, xComp[i], pComp[i]));
  PetscCall(VecStrideScatterAll(pComp, p, INSERT_VALUES));
  for (PetscInt i = 0; i < Nc; ++i) {
    PetscCall(VecDestroy(&xComp[i]));
    PetscCall(VecDestroy(&pComp[i]));
  }
  PetscCall(PetscFree2(xComp, pComp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Sets each link to be the identity for the free field test
static PetscErrorCode SetGauge_Identity(DM dm)
{
  DM           auxDM;
  Vec          auxVec;
  PetscSection s;
  PetscScalar  id[9] = {1., 0., 0., 0., 1., 0., 0., 0., 1.};
  PetscInt     eStart, eEnd;

  PetscFunctionBegin;
  PetscCall(DMGetAuxiliaryVec(dm, NULL, 0, 0, &auxVec));
  PetscCall(VecGetDM(auxVec, &auxDM));
  PetscCall(DMGetLocalSection(auxDM, &s));
  PetscCall(DMPlexGetDepthStratum(dm, 1, &eStart, &eEnd));
  for (PetscInt i = eStart; i < eEnd; ++i) { PetscCall(VecSetValuesSection(auxVec, s, i, id, INSERT_VALUES)); }
  PetscCall(VecViewFromOptions(auxVec, NULL, "-gauge_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline void TwoSpinProject(PetscInt mu, PetscBool minus, PetscInt ldx,const PetscScalar f[],PetscScalar o[])
{
  const PetscReal   sign   = (minus )   ? -1. : 1.;
  const PetscScalar fin[4] = {f[0 * ldx], f[1 * ldx], f[2 * ldx], f[3 * ldx]};

  //  PetscFunctionBeginHot;
  switch (mu) {
  case 0:
    o[0*ldx] = fin[0] + sign * PETSC_i * fin[3];
    o[1*ldx] = fin[1] + sign * PETSC_i * fin[2];
    break;
  case 1:
    o[0*ldx] = fin[0] - sign * fin[3];
    o[1*ldx] = fin[1] + sign * fin[2];
    break;
  case 2:
    o[0*ldx] = fin[0] + sign * PETSC_i * fin[2];
    o[1*ldx] = fin[1] - sign * PETSC_i * fin[3];
    break;
  case 3:
    o[0*ldx] = fin[0] + sign * fin[2];
    o[1*ldx] = fin[1] + sign * fin[3];
    break;
  case 4:
    if ( sign==1 ) {
      o[0*ldx] = fin[0];
      o[1*ldx] = fin[1];
    } else {
      o[0*ldx] = fin[2];
      o[1*ldx] = fin[3];
    }
    break;
  default:
    //    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Direction for gamma %" PetscInt_FMT " not in [0, 5)", mu);
    break;
  }
  o[0 * ldx] *= 0.5;
  o[1 * ldx] *= 0.5;
  //  PetscFunctionReturn(0);
}
// Apply (1 \pm \gamma_\mu)/2
static inline void TwoSpinAccumulate(PetscInt mu, PetscBool minus, PetscInt ldx, const PetscScalar f[], PetscScalar p[])
{
  const PetscReal   sign   = (minus )   ? -1. : 1.;
  const PetscScalar fin[4] = {f[0 * ldx], f[1 * ldx], f[2 * ldx], f[3 * ldx]};

  //  PetscFunctionBeginHot;
  switch (mu) {
  case 0:
    p[0 * ldx] -= fin[0];
    p[1 * ldx] -= fin[1];
    p[2 * ldx] += sign * PETSC_i * fin[1];
    p[3 * ldx] += sign * PETSC_i * fin[0];
    break;
  case 1:
    p[0 * ldx] -= fin[0];
    p[1 * ldx] -= fin[1];
    p[2 * ldx] -= sign * fin[1];
    p[3 * ldx] += sign * fin[0];
    break;
  case 2:
    p[0 * ldx] -= fin[0];
    p[1 * ldx] -= fin[1];
    p[2 * ldx] += sign * PETSC_i * fin[0];
    p[3 * ldx] -= sign * PETSC_i * fin[1];
    break;
  case 3:
    p[0 * ldx] -= fin[0];
    p[1 * ldx] -= fin[1];
    p[2 * ldx] -= sign * fin[0];
    p[3 * ldx] -= sign * fin[1];
    break;
  case 4:
    if ( sign==1 ) {
      p[0 * ldx] += 2*fin[0]; // Not sure of this 2x
      p[1 * ldx] += 2*fin[1];
    } else {
      p[2 * ldx] += 2*fin[0];
      p[3 * ldx] += 2*fin[1];
    }
    break;
  default:
    //SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Direction for gamma %" PetscInt_FMT " not in [0, 5)", mu);
    break;
  }
  //  PetscFunctionReturn(0);
}

/*
  Test the action of the Wilson operator in the free field case U = I,

    \eta(x) = D_W(x - y) \psi(y)

  The Wilson operator is a convolution for the free field, so we can check that by the convolution theorem

    \hat\eta(x) = \mathcal{F}(D_W(x - y) \psi(y))
                = \hat D_W(p) \mathcal{F}\psi(p)

  The Fourier series for the Wilson operator is

    M + \sum_\mu 2 \sin^2(p_\mu / 2) + i \gamma_\mu \sin(p_\mu)
*/
#if 0
static PetscErrorCode TestFreeField(DM dm)
{
  PetscSection       s;
  Mat                FT;
  Vec                psi, psiHat;
  Vec                eta, etaHat;
  Vec                DHat; // The product \hat D_w \hat psi
  PetscRandom        r;
  const PetscScalar *psih;
  PetscScalar       *dh;
  PetscReal         *coef, nrm;
  const PetscInt    *extent, Nc = 12;
  PetscInt           dim, V     = 1, vStart, vEnd;
  PetscContainer     c;
  PetscBool          constRhs = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-const_rhs", &constRhs, NULL));

  PetscCall(SetGauge_Identity(dm));
  PetscCall(DMGetLocalSection(dm, &s));
  PetscCall(DMGetGlobalVector(dm, &psi));
  PetscCall(PetscObjectSetName((PetscObject)psi, "psi"));
  PetscCall(DMGetGlobalVector(dm, &psiHat));
  PetscCall(PetscObjectSetName((PetscObject)psiHat, "psihat"));
  PetscCall(DMGetGlobalVector(dm, &eta));
  PetscCall(PetscObjectSetName((PetscObject)eta, "eta"));
  PetscCall(DMGetGlobalVector(dm, &etaHat));
  PetscCall(PetscObjectSetName((PetscObject)etaHat, "etahat"));
  PetscCall(DMGetGlobalVector(dm, &DHat));
  PetscCall(PetscObjectSetName((PetscObject)DHat, "Dhat"));
  PetscCall(PetscRandomCreate(PETSC_COMM_SELF, &r));
  PetscCall(PetscRandomSetType(r, PETSCRAND48));
  if (constRhs) PetscCall(VecSet(psi, 1.));
  else PetscCall(VecSetRandom(psi, r));
  PetscCall(PetscRandomDestroy(&r));

  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  PetscCall(PetscObjectQuery((PetscObject)dm, "_extent", (PetscObject *)&c));
  PetscCall(PetscContainerGetPointer(c, (void **)&extent));
  PetscCall(MatCreateFFT(PetscObjectComm((PetscObject)dm), dim, extent, MATFFTW, &FT));

  PetscCall(PetscMalloc1(dim, &coef));
  for (PetscInt d = 0; d < dim; ++d) {
    coef[d] = 2. * PETSC_PI / extent[d];
    V *= extent[d];
  }
  PetscCall(ComputeResidual(dm, psi, eta));
  PetscCall(VecViewFromOptions(eta, NULL, "-psi_view"));
  PetscCall(VecViewFromOptions(eta, NULL, "-eta_view"));
  PetscCall(ComputeFFT(FT, Nc, psi, psiHat));
  PetscCall(VecScale(psiHat, 1. / V));
  PetscCall(ComputeFFT(FT, Nc, eta, etaHat));
  PetscCall(VecScale(etaHat, 1. / V));
  PetscCall(VecGetArrayRead(psiHat, &psih));
  PetscCall(VecGetArray(DHat, &dh));
  for (PetscInt v = vStart; v < vEnd; ++v) {
    PetscScalar tmp[12], tmp1 = 0.;
    PetscInt    dof, off;

    PetscCall(PetscSectionGetDof(s, v, &dof));
    PetscCall(PetscSectionGetOffset(s, v, &off));
    for (PetscInt d = 0, prod = 1; d < dim; prod *= extent[d], ++d) {
      const PetscInt idx = (v / prod) % extent[d];

      tmp1 += 2. * PetscSqr(PetscSinReal(0.5 * coef[d] * idx));
      for (PetscInt i = 0; i < dof; ++i) tmp[i] = PETSC_i * PetscSinReal(coef[d] * idx) * psih[off + i];
      for (PetscInt c = 0; c < 3; ++c) PetscCall(ComputeGamma(d, 3, &tmp[c]));
      for (PetscInt i = 0; i < dof; ++i) dh[off + i] += tmp[i];
    }
    for (PetscInt i = 0; i < dof; ++i) dh[off + i] += (M + tmp1) * psih[off + i];
  }
  PetscCall(VecRestoreArrayRead(psiHat, &psih));
  PetscCall(VecRestoreArray(DHat, &dh));

  {
    Vec     *etaComp, *DComp;
    PetscInt n, N;

    PetscCall(PetscMalloc2(Nc, &etaComp, Nc, &DComp));
    PetscCall(VecGetLocalSize(etaHat, &n));
    PetscCall(VecGetSize(etaHat, &N));
    for (PetscInt i = 0; i < Nc; ++i) {
      const char *vtype;

      // HACK: Make these from another DM up front
      PetscCall(VecCreate(PetscObjectComm((PetscObject)etaHat), &etaComp[i]));
      PetscCall(VecGetType(etaHat, &vtype));
      PetscCall(VecSetType(etaComp[i], vtype));
      PetscCall(VecSetSizes(etaComp[i], n / Nc, N / Nc));
      PetscCall(VecDuplicate(etaComp[i], &DComp[i]));
    }
    PetscCall(VecStrideGatherAll(etaHat, etaComp, INSERT_VALUES));
    PetscCall(VecStrideGatherAll(DHat, DComp, INSERT_VALUES));
    for (PetscInt i = 0; i < Nc; ++i) {
      if (!i) {
        PetscCall(VecViewFromOptions(etaComp[i], NULL, "-etahat_view"));
        PetscCall(VecViewFromOptions(DComp[i], NULL, "-dhat_view"));
      }
      PetscCall(VecAXPY(etaComp[i], -1., DComp[i]));
      PetscCall(VecNorm(etaComp[i], NORM_INFINITY, &nrm));
      PetscCall(PetscPrintf(PETSC_COMM_SELF, "  Slice %" PetscInt_FMT ": %g\n", i, (double)nrm));
    }
    PetscCall(VecStrideScatterAll(etaComp, etaHat, INSERT_VALUES));
    for (PetscInt i = 0; i < Nc; ++i) {
      PetscCall(VecDestroy(&etaComp[i]));
      PetscCall(VecDestroy(&DComp[i]));
    }
    PetscCall(PetscFree2(etaComp, DComp));
    PetscCall(VecNorm(etaHat, NORM_INFINITY, &nrm));
    PetscCheck(nrm < PETSC_SMALL, PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Free field test failed: %g", (double)nrm);
  }

  PetscCall(PetscFree(coef));
  PetscCall(MatDestroy(&FT));
  PetscCall(DMRestoreGlobalVector(dm, &psi));
  PetscCall(DMRestoreGlobalVector(dm, &psiHat));
  PetscCall(DMRestoreGlobalVector(dm, &eta));
  PetscCall(DMRestoreGlobalVector(dm, &etaHat));
  PetscCall(DMRestoreGlobalVector(dm, &DHat));
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

static PetscErrorCode SetUpDW(DM dm, Mat *DWOperator, PetscBool isPV, PetscInt Ls, AppCtx *user, int argc, char **argv) {
  Vec      u;
  PetscInt locSize;

  PetscFunctionBegin;
  PetscCall(PetscSetGauge_Grid5D(dm, GRID_LATTICE_FILE, isPV, Ls, argc, argv, user->gridFile));
  PetscCall(DMCreateLocalVector(dm, &u));
  PetscCall(VecGetLocalSize(u, &locSize));
  PetscCall(MatCreateShell(PETSC_COMM_WORLD, locSize, locSize, PETSC_DECIDE, PETSC_DECIDE, user, DWOperator));
  PetscCall(MatShellSetOperation(*DWOperator, MATOP_MULT, (void(*)(void))Ddwf));
  PetscCall(MatShellSetOperation(*DWOperator, MATOP_MULT_TRANSPOSE, (void(*)(void))DdwfDag));
  PetscCall(VecDestroy(&u));
  PetscFunctionReturn(PETSC_SUCCESS);
}
// Gauge should be set and these functions should operator only on configured operators with their associated DMs

static PetscErrorCode SetUpPreconditionedOperator(DM dm, Mat *Prec, AppCtx *user) {
  Vec      u;
  PetscInt locSize;

  PetscFunctionBegin;
  PetscCall(DMCreateLocalVector(dm, &u));
  PetscCall(VecGetLocalSize(u, &locSize));
  PetscCall(MatCreateShell(PETSC_COMM_WORLD, locSize, locSize, PETSC_DECIDE, PETSC_DECIDE, user, Prec));
  PetscCall(MatShellSetOperation(*Prec, MATOP_MULT, (void(*)(void))PrecOp));
  PetscCall(MatShellSetOperation(*Prec, MATOP_MULT_TRANSPOSE, (void(*)(void))PrecOpDag));
  PetscCall(VecDestroy(&u));
  PetscFunctionReturn(PETSC_SUCCESS);

}

static PetscErrorCode SetUpSqrDW(DM dm, Mat *DWOperator, PetscInt Ls, AppCtx *user, int argc, char **argv) {
  Vec      u;
  PetscInt locSize;

  PetscFunctionBegin;
  PetscCall(PetscSetGauge_Grid5D(dm, GRID_LATTICE_FILE, PETSC_FALSE, Ls, argc, argv, user->gridFile));
  PetscCall(DMCreateLocalVector(dm, &u));
  PetscCall(VecGetLocalSize(u, &locSize));
  PetscCall(MatCreateShell(PETSC_COMM_WORLD, locSize, locSize, PETSC_DECIDE, PETSC_DECIDE, user, DWOperator));
  PetscCall(MatShellSetOperation(*DWOperator, MATOP_MULT, (void(*)(void))DdwfDagDdwf));
  PetscCall(MatShellSetOperation(*DWOperator, MATOP_MULT_TRANSPOSE, (void(*)(void))DdwfDagDdwf));
  PetscCall(VecDestroy(&u));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetupWilson(DM dm, PetscBool setGauge, Mat *WilsonOperator, AppCtx *user, int argc, char **argv) {
    Vec      u;
    PetscInt locSize;

    PetscFunctionBegin;
    // Grid load type support still needed with error checking for dimensionality on the lattice files
    if (setGauge) {
      PetscCall(PetscSetGauge_Grid(dm, user->shift, GRID_LATTICE_TEPID, argc, argv, "ckpoint_lat.4000"));
      //PetscCall(SetGauge_Identity(dm));
    }
    PetscCall(DMCreateLocalVector(dm, &u));
    // Configure the matshell to represent the operator, this probably doesn't change much
    PetscCall(VecGetLocalSize(u, &locSize));
    PetscCall(DMCreateLocalVector(dm, &u));
    PetscCall(MatCreateShell(PETSC_COMM_WORLD, locSize, locSize, PETSC_DECIDE, PETSC_DECIDE, user, WilsonOperator));
    PetscCall(MatShellSetOperation(*WilsonOperator, MATOP_MULT, (void(*)(void))ComputeResidual_Forward));
    PetscCall(MatShellSetOperation(*WilsonOperator, MATOP_MULT_TRANSPOSE, (void(*)(void))ComputeResidual_Dagger));
    PetscCall(VecDestroy(&u));
    PetscFunctionReturn(PETSC_SUCCESS);
}

#ifdef PETSC_HAVE_SLEPC
  static PetscErrorCode GetSpectralBounds(Mat M, PetscScalar *min, PetscScalar *max) {
    EPS        eps_max, eps_min;
    Mat         M_min, M_max;
    PetscScalar max_eigenvalue, min_eigenvalue;
    PetscInt    nMin, nMax;

    PetscFunctionBegin;
    /* ----- min eigenvalues ------*/
    PetscCall(MatDuplicate(M, MAT_DO_NOT_COPY_VALUES, &M_min));
    PetscCall(EPSCreate(PETSC_COMM_WORLD, &eps_min));
    PetscCall(EPSSetOperators(eps_min, M, NULL));
    PetscCall(EPSSetOptionsPrefix(eps_min, "eigmin_"));
    PetscCall(EPSSetFromOptions(eps_min));
    PetscCall(EPSSolve(eps_min));
    PetscCall(EPSGetConverged(eps_min, &nMin));
    PetscCall(EPSGetEigenpair(eps_min, nMin - 1, &min_eigenvalue, NULL, NULL, NULL));
    *min = min_eigenvalue;
    /* ----- max eigenvalues ------*/
    PetscCall(MatDuplicate(M, MAT_DO_NOT_COPY_VALUES, &M_max));
    PetscCall(EPSCreate(PETSC_COMM_WORLD, &eps_max));
    PetscCall(EPSSetOperators(eps_max, M, NULL));
    PetscCall(EPSSetOptionsPrefix(eps_max, "eigmax_"));
    PetscCall(EPSSetFromOptions(eps_max));
    PetscCall(EPSSolve(eps_max));
    PetscCall(EPSGetConverged(eps_max, &nMax));
    PetscCall(EPSGetEigenpair(eps_max, nMax - 1, &max_eigenvalue, NULL, NULL, NULL));
    *max = max_eigenvalue;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
#endif

/*
  Wrapping SLEPc calls in ifdefs makes this a little gross,
  they can be tossed if SLEPc is not being used at all
  as long as everything else is still protected or removed.
*/
static PetscErrorCode SolveSystem(DM dm, Mat M, PetscBool useSpectrum, AppCtx *user) {
  PetscScalar min = -1, max = -1; // min and max eigenvalues from slepc, if present
  KSP         ksp;// Linear solver
  Mat         NE; // matrix for the normal equations
  Vec         u, f;
  PetscRandom r;

  PetscFunctionBegin;
  // Protect any slepc operations so we can pull them out easily if needed for MR.
  #ifdef PETSC_HAVE_SLEPC
    if (useSpectrum) PetscCall(GetSpectralBounds(M, &min, &max));
    else if (user->normalEq) {
      EPS epsNE;
      PetscPrintf(PETSC_COMM_WORLD, "Using normal equations with ksp operator.\n");
      PetscCall(MatCreateNormal(M, &NE));
      PetscCall(EPSCreate(PETSC_COMM_WORLD, &epsNE));
      PetscCall(EPSSetOperators(epsNE, NE, NULL));
      PetscCall(EPSSetOptionsPrefix(epsNE, "ne_"));
      PetscCall(EPSSetFromOptions(epsNE));
    }
  #else
    if (useSpectrum) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Use of spectral bounds specified but SLEPc not installed or not linked correctly");
  #endif
  PetscCall(DMGetLocalVector(dm, &u));
  PetscCall(DMGetLocalVector(dm, &f));
  // Configure the input vectors, this is not a real fermion field
  PetscCall(PetscRandomCreate(PETSC_COMM_SELF, &r));
  PetscCall(PetscRandomSetType(r, PETSCRAND48));
  PetscCall(VecSetRandom(u, r));
  PetscCall(PetscRandomDestroy(&r));
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, M, M));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSolve(ksp, u, f));
  PetscCall(DMRestoreLocalVector(dm, &u));
  PetscCall(DMRestoreLocalVector(dm, &f));
  PetscCall(KSPDestroy(&ksp));
  PetscFunctionReturn(PETSC_SUCCESS);
}
// hacky, get and configure from the fine DM upfront
// but calling sequence gets long
static PetscErrorCode SetupCoarseSpace(MPI_Comm comm, DM *cdm, AppCtx *user){
  PetscFunctionBegin;
  PetscCall(DMCreate(comm, cdm));
  PetscCall(DMSetType(*cdm, DMPLEX));
  PetscCall(DMSetOptionsPrefix(*cdm, "coarse_"));
  PetscCall(DMSetFromOptions(*cdm));
  PetscCall(DMSetApplicationContext(*cdm, user));
  PetscCall(DMViewFromOptions(*cdm, NULL, "-coarse_dm_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Restriction operator for the gauge fields. Perform link
// concatenation on the fine space to produce a DM representing
// the coarse space.
static PetscErrorCode RestrictGaugeField(DM fdm, DM cdm)
{
  DM                 fdmAux, cdmAux;
  Vec                fgauge, cgauge;
  PetscSection       fs, fsGauge, cs, csGauge;
  const PetscScalar *ua;
  PetscScalar       *flinks, *clinks;
  PetscInt           dim, vStart, vEnd, shift = 2, cvStart, cvEnd;
  PetscInt           faces[4], nf=4, nt, nz, ny, nx;
  PetscBool          flg;
  AppCtx            *ctx;

  PetscFunctionBeginUser;
  PetscCall(PetscOptionsGetIntArray(NULL, NULL, "-dm_plex_box_faces", faces, &nf, &flg));
  nx=faces[0];ny=faces[1];nz=faces[2];nt=faces[3];//TODO: Get from DM for generalization and >2 levels
  PetscCall(DMGetApplicationContext(fdm, &ctx));
  PetscCall(DMGetDimension(fdm, &dim));
  //PetscCall(DMGetLocalSection(fdm, &s));
  PetscCall(DMPlexGetDepthStratum(fdm, 0, &vStart, &vEnd));
  PetscCall(DMPlexGetDepthStratum(cdm, 0, &cvStart, &cvEnd));

  PetscCall(DMGetAuxiliaryVec(fdm, NULL, 0, 0, &fgauge));
  PetscCall(VecGetDM(fgauge, &fdmAux));
  PetscCall(DMGetAuxiliaryVec(cdm, NULL, 0, 0, &cgauge));
  PetscCall(VecGetDM(cgauge, &cdmAux));
  PetscCall(DMGetLocalSection(fdmAux, &fsGauge));
  PetscCall(DMGetLocalSection(cdmAux, &csGauge));
  // get the links
  PetscCall(VecGetArray(fgauge, &flinks));
  PetscCall(VecGetArray(cgauge, &clinks));
  // Start at vStart, loop over the vertices and get the right edges
  // of the neighboring vertices, concatinate. Doing right edges in
  // all dimensions avoids double counting vertices. Offset into
  // depth computed from dimensions t, z, y, x
  //
  // TODO: Fix for parallel
  // TODO: Rewrite to leverage ordering of the supports to make it simpler
  PetscInt cv=cvStart;
  for (PetscInt t = 0; t < nt; t+=2){
    for (PetscInt z = 0; z < nz; z+=2){
      for (PetscInt y = 0; y < ny; y+=2){
        for (PetscInt x = 0; x < nx; x+=2, cv++){
          const PetscInt *supp, *cSupp;
          PetscInt        xdof, xoff;
          PetscInt        toff, zoff, yoff, v;

          toff = t*nx*ny*nz;
          zoff = z*nx*ny;
          yoff = y*nx;
          v = vStart + toff + zoff + yoff + x;//gross
          PetscCall(DMPlexGetSupport(fdm, v, &supp));
          PetscCall(DMPlexGetSupport(cdm, cv, &cSupp));
          // Loop over each dimension for right edges
          for (PetscInt d = 0; d < dim; ++d) {
            const PetscInt *cone, *cone2, *suppsupp, *cSupp;
            PetscInt        yoff, rgoff, rrgoff, gdoff, coff;

            // Right edges
            PetscCall(PetscSectionGetOffset(fsGauge, supp[2 * d + 1], &rgoff));
            // get the support of the support for the second edge in d dimension
            // this may need to come from cone
            PetscCall(DMPlexGetCone(fdm, supp[2*d+1], &cone2));
            PetscCall(DMPlexGetSupport(fdm, cone2[1], &suppsupp));

            PetscCall(DMPlexGetSupport(cdm, cv, &cSupp));
            PetscCall(PetscSectionGetOffset(csGauge, cSupp[2*d+1], &coff));

            PetscCall(PetscSectionGetOffset(fsGauge, suppsupp[2*d + 1], &rrgoff));
            PetscCall(PetscSectionGetDof(fsGauge, suppsupp[2*d + 1], &gdoff));
            for (PetscInt dof = 0; dof < gdoff; dof++){
              // dim 3 stride 1
              DMPlex_MatMult3D_Internal(&flinks[rgoff], 3, 3, &flinks[rrgoff], &clinks[coff]);
              //clinks[coff+dof] = (0.5)*(flinks[rgoff+dof] * flinks[rrgoff+dof]);// update to product
            }
          }
        }
      }
    }
  }
  PetscCall(VecRestoreArray(fgauge, &flinks));
  PetscCall(VecRestoreArray(cgauge, &clinks));
  PetscCall(VecViewFromOptions(cgauge, NULL, "-coarse_gauge_view"));
  PetscFunctionReturn(0);
}

// Matrix free multiplication routine to give to MatShell representing
// Injection or Full weight restriction. The fine solution is passed in
// as u to output v as the coarse solution. M must have the fine (non-
// auxiliar) DM
static PetscErrorCode RestrictSolution_FullWeight(Mat M, Vec u, Vec c){
  DM                 fdm, cdm;
  const PetscScalar *ua;
  PetscScalar       *ca;
  PetscInt           faces[4], nf=4, nt, nz, ny, nx;
  PetscInt           cvStart, cvEnd, vStart, vEnd, v, cv;
  PetscInt           dim;
  PetscBool          flg;
  PetscSection       s, cs;
  AppCtx            *user;

  PetscFunctionBegin;
  PetscCall(PetscOptionsGetIntArray(NULL, NULL, "-dm_plex_box_faces", faces, &nf, &flg));
  nx=faces[0];ny=faces[1];nz=faces[2];nt=faces[3];//TODO: Get from DM extent for generalization and >2 levels
  PetscCall(MatGetDM(M, &fdm));
  //PetscCall(VecGetDM(c, &cdm));
  PetscCall(DMGetCoarseDM(fdm, &cdm));
  // make sure its empty
  PetscCall(VecZeroEntries(c));
  PetscCall(DMGetDimension(fdm, &dim));
  PetscCall(DMGetLocalSection(fdm, &s));
  PetscCall(DMGetLocalSection(cdm, &cs));
  PetscCall(DMPlexGetDepthStratum(fdm, 0, &vStart, &vEnd));
  PetscCall(DMPlexGetDepthStratum(cdm, 0, &cvStart, &cvEnd));
  cv = cvStart;
  PetscCall(VecGetArrayRead(u, &ua));
  PetscCall(VecGetArray(c, &ca));
  // loop over vertices on the fine grid and get their associated
  // fermion field values
  for (PetscInt t = 0; t < nt; t+=2){
    for (PetscInt z = 0; z < nz; z+=2){
      for (PetscInt y = 0; y < ny; y+=2){
        for (PetscInt x = 0; x < nx; x+=2, cv++){
          const PetscInt *cone;
          const PetscInt *supp;
          PetscInt        uoff, goff, xdof, toff, zoff, yoff, coff;

          toff = t*nx*ny*nz;
          zoff = z*nx*ny;
          yoff = y*nx;
          v = vStart + toff + zoff + yoff + x;
          PetscCall(DMPlexGetSupport(fdm, v, &supp));
          PetscCall(PetscSectionGetDof(s, v, &xdof));
          PetscCall(PetscSectionGetOffset(s, v, &uoff));
          PetscCall(PetscSectionGetOffset(cs, cv, &coff));
          for (PetscInt dof = 0; dof < xdof; ++dof) ca[coff+dof] += 1./2.*ua[uoff+dof];
          for (PetscInt d = 0; d < dim; ++d) {

            // Left vertex
            PetscCall(DMPlexGetCone(fdm, supp[2 * d + 0], &cone));
            PetscCall(PetscSectionGetOffset(s, cone[0], &uoff));
            for (PetscInt dof = 0; dof < xdof; ++dof) ca[coff+dof] += 1./16.*(ua[uoff+dof]);//not one half scaling
            // Right vertex
            PetscCall(DMPlexGetCone(fdm, supp[2 * d + 1], &cone));
            PetscCall(PetscSectionGetOffset(s, cone[1], &uoff));
            for (PetscInt dof = 0; dof < xdof; ++dof) ca[coff+dof] += 1./16.*(ua[uoff+dof]);
          }
        }
      }
    }
  }
  PetscCall(VecRestoreArray(c, &ca));
  PetscCall(VecRestoreArrayRead(u, &ua));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Matrix free multiplication routine to give to MatShell representing
// prolongation. The fine solution is passed in
// as u to output v as the coarse solution. M must have the fine (non-
// auxiliar) DM
static PetscErrorCode Interpolation_FullWeight(Mat M, Vec c, Vec u){
  DM                 fdm, cdm;
  const PetscScalar *ca;
  PetscScalar       *ua;
  PetscInt           faces[4], nf=4, nt, nz, ny, nx;
  PetscInt           cvStart, cvEnd, vStart, vEnd, v, cv;
  PetscInt           dim;
  PetscBool          flg;
  PetscSection       s, cs;
  AppCtx            *user;

  PetscFunctionBegin;
  PetscCall(PetscOptionsGetIntArray(NULL, NULL, "-dm_plex_box_faces", faces, &nf, &flg));
  nx=faces[0];ny=faces[1];nz=faces[2];nt=faces[3];//TODO: Get from DM for generalization and >2 levels
  PetscCall(MatGetDM(M, &fdm));
  //PetscCall(VecGetDM(c, &cdm));
  PetscCall(DMGetCoarseDM(fdm, &cdm));
  // make sure its empty
  PetscCall(VecZeroEntries(u));
  PetscCall(DMGetDimension(fdm, &dim));
  PetscCall(DMGetLocalSection(fdm, &s));
  PetscCall(DMGetLocalSection(cdm, &cs));
  PetscCall(DMPlexGetDepthStratum(fdm, 0, &vStart, &vEnd));
  PetscCall(DMPlexGetDepthStratum(cdm, 0, &cvStart, &cvEnd));
  cv = cvStart;
  PetscCall(VecZeroEntries(u));
  PetscCall(VecGetArray(u, &ua));
  PetscCall(VecGetArrayRead(c, &ca));
  // loop over vertices on the fine grid and get their associated
  // fermion field values
  for (PetscInt t = 0; t < nt; t+=2){
    for (PetscInt z = 0; z < nz; z+=2){
      for (PetscInt y = 0; y < ny; y+=2){
        for (PetscInt x = 0; x < nx; x+=2, cv++){
          const PetscInt *cone;
          const PetscInt *supp;
          PetscInt        uoff, goff, xdof, toff, zoff, yoff, coff;

          toff = t*nx*ny*nz;
          zoff = z*nx*ny;
          yoff = y*nx;
          v = vStart + toff + zoff + yoff + x;
          PetscCall(DMPlexGetSupport(fdm, v, &supp));
          PetscCall(PetscSectionGetDof(s, v, &xdof));
          PetscCall(PetscSectionGetOffset(s, v, &uoff));
          PetscCall(PetscSectionGetOffset(cs, cv, &coff));
          for (PetscInt dof = 0; dof < xdof; ++dof) ua[uoff+dof] = 1. * ca[coff+dof];
          for (PetscInt d = 0; d < dim; ++d) {
            // Left vertex
            PetscCall(DMPlexGetCone(fdm, supp[2 * d + 0], &cone));
            PetscCall(PetscSectionGetOffset(s, cone[0], &uoff));
            for (PetscInt dof = 0; dof < xdof; ++dof) ua[uoff+dof] += 1/2. * ca[coff+dof];// One half, because these only get contributions from 1/4*dim of the neighbors because of the star stencil
            #if 1

            for (PetscInt e = 0; e < dim; ++e) {
              const PetscInt* supp2, *cone2;

              PetscCall(DMPlexGetSupport(fdm, cone[0], &supp2));
              if (d == e) continue;

              // Left vertex
              PetscCall(DMPlexGetCone(fdm, supp2[2 * e + 0], &cone2));
              PetscCall(PetscSectionGetOffset(s, cone2[0], &uoff));
              for (PetscInt dof = 0; dof < xdof; ++dof) ua[uoff+dof] += 1/8. * ca[coff+dof];// 1/16 because of the double counting of the two paths to the vertex from the original coarse vertex

              for (PetscInt f = 0; f < dim; ++f) {
                const PetscInt* supp3, *cone3;
                if (f == d || f == e) continue;

                PetscCall(DMPlexGetSupport(fdm, cone2[0], &supp3));

                PetscCall(DMPlexGetCone(fdm, supp3[2 * f + 0], &cone3));
                PetscCall(PetscSectionGetOffset(s, cone3[0], &uoff));
                for (PetscInt dof = 0; dof < xdof; ++dof) ua[uoff+dof] += 1/48. * ca[coff+dof];

                for (PetscInt g = 0; g < dim; ++g){
                  const PetscInt* supp4, *cone4;
                  if (g == d || g == e || g == f) continue;
                  PetscCall(DMPlexGetSupport(fdm, cone3[0], &supp4));

                  PetscCall(DMPlexGetCone(fdm, supp4[2 * g + 0], &cone4));
                  PetscCall(PetscSectionGetOffset(s, cone4[0], &uoff));
                  for (PetscInt dof = 0; dof < xdof; ++dof) ua[uoff+dof] += 1/384. * ca[coff+dof];

                  PetscCall(DMPlexGetCone(fdm, supp4[2 * g + 1], &cone4));
                  PetscCall(PetscSectionGetOffset(s, cone4[1], &uoff));
                  for (PetscInt dof = 0; dof < xdof; ++dof) ua[uoff+dof] += 1/384. * ca[coff+dof];

                }


                PetscCall(DMPlexGetCone(fdm, supp3[2 * f + 1], &cone3));
                PetscCall(PetscSectionGetOffset(s, cone3[1], &uoff));
                for (PetscInt dof = 0; dof < xdof; ++dof) ua[uoff+dof] += 1/48. * ca[coff+dof];

                for (PetscInt g = 0; g < dim; ++g){
                  const PetscInt* supp4, *cone4;
                  if (g == d || g == e || g == f) continue;
                  PetscCall(DMPlexGetSupport(fdm, cone3[1], &supp4));

                  PetscCall(DMPlexGetCone(fdm, supp4[2 * g + 0], &cone4));
                  PetscCall(PetscSectionGetOffset(s, cone4[0], &uoff));
                  for (PetscInt dof = 0; dof < xdof; ++dof) ua[uoff+dof] += 1/384. * ca[coff+dof];

                  PetscCall(DMPlexGetCone(fdm, supp4[2 * g + 1], &cone4));
                  PetscCall(PetscSectionGetOffset(s, cone4[1], &uoff));
                  for (PetscInt dof = 0; dof < xdof; ++dof) ua[uoff+dof] += 1/384. * ca[coff+dof];

                }

              }

              // Right vertex
              PetscCall(DMPlexGetCone(fdm, supp2[2 * e + 1], &cone2));
              PetscCall(PetscSectionGetOffset(s, cone2[1], &uoff));
              for (PetscInt dof = 0; dof < xdof; ++dof) ua[uoff+dof] += 1/8. * ca[coff+dof];
              for (PetscInt f = 0; f < dim; ++f) {
                const PetscInt* supp3, *cone3;
                if (f == d || f == e) continue;

                PetscCall(DMPlexGetSupport(fdm, cone2[1], &supp3));

                PetscCall(DMPlexGetCone(fdm, supp3[2 * f + 0], &cone3));
                PetscCall(PetscSectionGetOffset(s, cone3[0], &uoff));
                for (PetscInt dof = 0; dof < xdof; ++dof) ua[uoff+dof] += 1/48. * ca[coff+dof];

                for (PetscInt g = 0; g < dim; ++g){
                  const PetscInt* supp4, *cone4;
                  if (g == d || g == e || g == f) continue;
                  PetscCall(DMPlexGetSupport(fdm, cone3[0], &supp4));

                  PetscCall(DMPlexGetCone(fdm, supp4[2 * g + 0], &cone4));
                  PetscCall(PetscSectionGetOffset(s, cone4[0], &uoff));
                  for (PetscInt dof = 0; dof < xdof; ++dof) ua[uoff+dof] += 1/384. * ca[coff+dof];

                  PetscCall(DMPlexGetCone(fdm, supp4[2 * g + 1], &cone4));
                  PetscCall(PetscSectionGetOffset(s, cone4[1], &uoff));
                  for (PetscInt dof = 0; dof < xdof; ++dof) ua[uoff+dof] += 1/384. * ca[coff+dof];

                }


                PetscCall(DMPlexGetCone(fdm, supp3[2 * f + 1], &cone3));
                PetscCall(PetscSectionGetOffset(s, cone3[1], &uoff));
                for (PetscInt dof = 0; dof < xdof; ++dof) ua[uoff+dof] += 1/48. * ca[coff+dof];

                for (PetscInt g = 0; g < dim; ++g){
                  const PetscInt* supp4, *cone4;
                  if (g == d || g == e || g == f) continue;
                  PetscCall(DMPlexGetSupport(fdm, cone3[1], &supp4));

                  PetscCall(DMPlexGetCone(fdm, supp4[2 * g + 0], &cone4));
                  PetscCall(PetscSectionGetOffset(s, cone4[0], &uoff));
                  for (PetscInt dof = 0; dof < xdof; ++dof) ua[uoff+dof] += 1/384. * ca[coff+dof];

                  PetscCall(DMPlexGetCone(fdm, supp4[2 * g + 1], &cone4));
                  PetscCall(PetscSectionGetOffset(s, cone4[1], &uoff));
                  for (PetscInt dof = 0; dof < xdof; ++dof) ua[uoff+dof] += 1/384. * ca[coff+dof];

                }

              }

            }
            #endif
            // Right vertex
            PetscCall(DMPlexGetCone(fdm, supp[2 * d + 1], &cone));
            PetscCall(PetscSectionGetOffset(s, cone[1], &uoff));
            for (PetscInt dof = 0; dof < xdof; ++dof) ua[uoff+dof] += 1/2. * ca[coff+dof];
            #if 1
            for (PetscInt e = 0; e < dim; ++e) {
              const PetscInt* supp2, *cone2;

              PetscCall(DMPlexGetSupport(fdm, cone[1], &supp2));
              if (d == e) continue;

              // Left vertex
              PetscCall(DMPlexGetCone(fdm, supp2[2 * e + 0], &cone2));
              PetscCall(PetscSectionGetOffset(s, cone2[0], &uoff));
              for (PetscInt dof = 0; dof < xdof; ++dof) ua[uoff+dof] += 1/8. * ca[coff+dof];// 1/16 because of the double counting of the two paths to the vertex from the original coarse vertex

              for (PetscInt f = 0; f < dim; ++f) {
                const PetscInt* supp3, *cone3;
                if (f == d || f == e) continue;

                PetscCall(DMPlexGetSupport(fdm, cone2[0], &supp3));

                PetscCall(DMPlexGetCone(fdm, supp3[2 * f + 0], &cone3));
                PetscCall(PetscSectionGetOffset(s, cone3[0], &uoff));
                for (PetscInt dof = 0; dof < xdof; ++dof) ua[uoff+dof] += 1/48. * ca[coff+dof];

                for (PetscInt g = 0; g < dim; ++g){
                  const PetscInt* supp4, *cone4;
                  if (g == d || g == e || g == f) continue;
                  PetscCall(DMPlexGetSupport(fdm, cone3[0], &supp4));

                  PetscCall(DMPlexGetCone(fdm, supp4[2 * g + 0], &cone4));
                  PetscCall(PetscSectionGetOffset(s, cone4[0], &uoff));
                  for (PetscInt dof = 0; dof < xdof; ++dof) ua[uoff+dof] += 1/384. * ca[coff+dof];

                  PetscCall(DMPlexGetCone(fdm, supp4[2 * g + 1], &cone4));
                  PetscCall(PetscSectionGetOffset(s, cone4[1], &uoff));
                  for (PetscInt dof = 0; dof < xdof; ++dof) ua[uoff+dof] += 1/384. * ca[coff+dof];

                }


                PetscCall(DMPlexGetCone(fdm, supp3[2 * f + 1], &cone3));
                PetscCall(PetscSectionGetOffset(s, cone3[1], &uoff));
                for (PetscInt dof = 0; dof < xdof; ++dof) ua[uoff+dof] += 1/48. * ca[coff+dof];

                for (PetscInt g = 0; g < dim; ++g){
                  const PetscInt* supp4, *cone4;
                  if (g == d || g == e || g == f) continue;
                  PetscCall(DMPlexGetSupport(fdm, cone3[1], &supp4));

                  PetscCall(DMPlexGetCone(fdm, supp4[2 * g + 0], &cone4));
                  PetscCall(PetscSectionGetOffset(s, cone4[0], &uoff));
                  for (PetscInt dof = 0; dof < xdof; ++dof) ua[uoff+dof] += 1/384. * ca[coff+dof];

                  PetscCall(DMPlexGetCone(fdm, supp4[2 * g + 1], &cone4));
                  PetscCall(PetscSectionGetOffset(s, cone4[1], &uoff));
                  for (PetscInt dof = 0; dof < xdof; ++dof) ua[uoff+dof] += 1/384. * ca[coff+dof];

                }

              }

              // Right vertex
              PetscCall(DMPlexGetCone(fdm, supp2[2 * e + 1], &cone2));
              PetscCall(PetscSectionGetOffset(s, cone2[1], &uoff));
              for (PetscInt dof = 0; dof < xdof; ++dof) ua[uoff+dof] += 1/8. * ca[coff+dof];
              for (PetscInt f = 0; f < dim; ++f) {
                const PetscInt* supp3, *cone3;
                if (f == d || f == e) continue;

                PetscCall(DMPlexGetSupport(fdm, cone2[1], &supp3));

                PetscCall(DMPlexGetCone(fdm, supp3[2 * f + 0], &cone3));
                PetscCall(PetscSectionGetOffset(s, cone3[0], &uoff));
                for (PetscInt dof = 0; dof < xdof; ++dof) ua[uoff+dof] += 1/48. * ca[coff+dof];

                for (PetscInt g = 0; g < dim; ++g){
                  const PetscInt* supp4, *cone4;
                  if (g == d || g == e || g == f) continue;
                  PetscCall(DMPlexGetSupport(fdm, cone3[0], &supp4));

                  PetscCall(DMPlexGetCone(fdm, supp4[2 * g + 0], &cone4));
                  PetscCall(PetscSectionGetOffset(s, cone4[0], &uoff));
                  for (PetscInt dof = 0; dof < xdof; ++dof) ua[uoff+dof] += 1/384. * ca[coff+dof];

                  PetscCall(DMPlexGetCone(fdm, supp4[2 * g + 1], &cone4));
                  PetscCall(PetscSectionGetOffset(s, cone4[1], &uoff));
                  for (PetscInt dof = 0; dof < xdof; ++dof) ua[uoff+dof] += 1/384. * ca[coff+dof];

                }


                PetscCall(DMPlexGetCone(fdm, supp3[2 * f + 1], &cone3));
                PetscCall(PetscSectionGetOffset(s, cone3[1], &uoff));
                for (PetscInt dof = 0; dof < xdof; ++dof) ua[uoff+dof] += 1/48. * ca[coff+dof];

                for (PetscInt g = 0; g < dim; ++g){
                  const PetscInt* supp4, *cone4;
                  if (g == d || g == e || g == f) continue;
                  PetscCall(DMPlexGetSupport(fdm, cone3[1], &supp4));

                  PetscCall(DMPlexGetCone(fdm, supp4[2 * g + 0], &cone4));
                  PetscCall(PetscSectionGetOffset(s, cone4[0], &uoff));
                  for (PetscInt dof = 0; dof < xdof; ++dof) ua[uoff+dof] += 1/384. * ca[coff+dof];

                  PetscCall(DMPlexGetCone(fdm, supp4[2 * g + 1], &cone4));
                  PetscCall(PetscSectionGetOffset(s, cone4[1], &uoff));
                  for (PetscInt dof = 0; dof < xdof; ++dof) ua[uoff+dof] += 1/384. * ca[coff+dof];

                }

              }

            }
            #endif
          }
        }
      }
    }
  }
  PetscCall(VecRestoreArrayRead(c, &ca));
  PetscCall(VecRestoreArray(u, &ua));
  PetscFunctionReturn(PETSC_SUCCESS);
}



// Matrix free multiplication routine to give to MatShell representing
// Injection or Full weight restriction. The fine solution is passed in
// as u to output v as the coarse solution. M must have the fine (non-
// auxiliar) DM
static PetscErrorCode RestrictSolution_Injection(Mat M, Vec u, Vec v){
  PetscFunctionBegin;
  //TODO
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetupRestriction(DM dm, Mat *RestrictionOperator, AppCtx *user) {
    Vec      u;
    PetscInt locSize;

    PetscFunctionBegin;
    PetscCall(DMCreateLocalVector(dm, &u));
    // Configure the matshell to represent the operator, this probably doesn't change much
    PetscCall(VecGetLocalSize(u, &locSize));//reduction of size is 1/2^dim
    PetscCall(MatCreateShell(PETSC_COMM_WORLD, locSize/16, locSize, PETSC_DECIDE, PETSC_DECIDE, user, RestrictionOperator));
    PetscCall(MatShellSetOperation(*RestrictionOperator, MATOP_MULT, (void(*)(void))RestrictSolution_FullWeight));
    PetscCall(MatShellSetOperation(*RestrictionOperator, MATOP_MULT_TRANSPOSE, (void(*)(void))Interpolation_FullWeight));
    PetscCall(MatSetDM(*RestrictionOperator, dm));
    PetscCall(VecDestroy(&u));
    PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MonitorProjectError(KSP ksp, PetscInt it, PetscReal rnorm, void *ctx)
{
  Vec u, r, error, uhat;// current solution, residual, error, solution of the old solve
  Mat M;
  DM dm;// the plex
  PetscInt ncv;
  AppCtx *user = (AppCtx*)ctx;
  PetscFunctionBeginUser;

  if ((it%10 == 0)){
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Stage: %"PetscInt_FMT"\n", it/10));
    PetscCall(KSPBuildSolution(ksp, NULL, &u));
    PetscCall(KSPGetApplicationContext(ksp, &user));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Solution built.\n"));

    // load the solution to compute the error
    {
      PetscViewer viewer;

      PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, "solution.dat", FILE_MODE_READ, &viewer));
      PetscCall(VecCreate(PETSC_COMM_WORLD, &uhat));
      PetscCall(VecLoad(uhat, viewer));
      PetscCall(PetscViewerDestroy(&viewer));
      PetscCall(VecDuplicate(uhat, &error));
      PetscCall(VecWAXPY(error, -1., u, uhat));
    }
    VecViewFromOptions(uhat, NULL, "-uhat_view");
    VecViewFromOptions(u, NULL, "-u_view");
    VecViewFromOptions(error, NULL, "-error_view");


    for (PetscInt i = 0; i < 100; ++i) {
      PetscViewer viewer;
      char eigenVector[15];
      Vec eigenVectorR;
      PetscScalar proj;

      PetscCall(PetscSNPrintf(eigenVector, 15, "eig%"PetscInt_FMT".dat", i));
      PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, eigenVector, FILE_MODE_READ, &viewer));
      PetscCall(VecCreate(PETSC_COMM_WORLD, &eigenVectorR));
      PetscCall(VecLoad(eigenVectorR, viewer));
      PetscCall(PetscViewerDestroy(&viewer));
      PetscCall(VecViewFromOptions(eigenVectorR, NULL, "-eig_vec_view"));

      PetscCall(VecDot(eigenVectorR, error, &proj));
      PetscReal normEig, normErr, normSol;
      PetscCall(VecNorm(eigenVectorR, NORM_2, &normEig));
      PetscCall(VecNorm(error, NORM_2, &normErr));
      PetscCall(VecNorm(uhat, NORM_2, &normSol));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Error: %f %f Mod Coef.: %f normEig: %f normErr: %f normUHat %f\n", PetscRealPart(proj), PetscImaginaryPart(proj), PetscSqrtReal(PetscSqr(PetscRealPart(proj))+PetscSqr(PetscImaginaryPart(proj))), normEig, normErr, normSol));
      PetscCall(VecDestroy(&eigenVectorR));
    }
    #if 0
    for (PetscInt i = 0; i < ncv; ++i){
      PetscScalar proj;
      // TODO: THIS IS BACKWARDS, FIX VECTOR NAMES
      PetscCall(MatCreateVecs(M, NULL, &eigenVectorI));// Imaginary part (will not be zeroed)
      PetscCall(MatCreateVecs(M, NULL, &eigenVectorR));// this will be zeroed
      PetscCall(EPSGetEigenvector(user->eps, i, eigenVectorR, eigenVectorI));
      PetscCall(VecViewFromOptions(eigenVectorI, NULL, "-eigen_vector_imaginary_view"));
      PetscCall(VecViewFromOptions(eigenVectorR, NULL, "-eigen_vector_real_view"));
      PetscCall(VecDot(eigenVectorR, error, &proj));
      /*
        before loop, create vec of the size of number of eigenvectors (num rows), get out array, put proj in the array value for i
        put in print statement, or just do vecview which defaults to a histogram for draw, get histogram of coefficients.
      */
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Iterate %"PetscInt_FMT": %f %f\n", it, PetscRealPart(proj), PetscImaginaryPart(proj)));
    }
    #endif
    //PetscCall(VecDestroy(&error));
    //PetscCall(VecDestroy(&u));
    //PetscCall(VecDestroy(&uhat));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SolveSystemError(MPI_Comm comm, DM dm, Mat M, Mat MC, Mat R, AppCtx *user){
  PetscScalar min, max;
  KSP         ksp;// Linear solver
  PC          pc;
  Mat         NE; // matrix for the normal equations
  Vec         u, f, uhat;
  PetscRandom r;

  PetscFunctionBegin;
  // Protect any slepc operations so we can pull them out easily if needed for MR.
  #ifdef PETSC_HAVE_SLEPC
    if (0) PetscCall(GetSpectralBounds(M, &min, &max));
    else if (user->normalEq) {
      EPS epsNE;

      PetscPrintf(PETSC_COMM_WORLD, "Using normal equations with ksp operator.\n");
      PetscCall(MatCreateNormal(M, &NE));
      PetscCall(EPSCreate(PETSC_COMM_WORLD, &epsNE));
      PetscCall(EPSSetOperators(epsNE, NE, NULL));
      PetscCall(EPSSetOptionsPrefix(epsNE, "ne_"));
      PetscCall(EPSSetFromOptions(epsNE));
    }
  #else
    if (0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Use of spectral bounds specified but SLEPc not installed or not linked correctly");
  #endif

  PetscCall(DMGetLocalVector(dm, &f));
  // Configure the solution vectors. We may need to store it if we wish to use it to
  // compute the error at a later solve. We need to ensure it is the same input vector
  if (user->load_sol){
    PetscViewer viewer;

    PetscCall(PetscViewerBinaryOpen(comm, "ic.dat", FILE_MODE_READ, &viewer));
    PetscCall(VecCreate(comm, &u));
    PetscCall(VecLoad(u, viewer));
    PetscCall(PetscViewerDestroy(&viewer));
  }
  else{
    PetscCall(DMGetLocalVector(dm, &u));
    // Configure the input vectors, this is not a real fermion field
    PetscCall(PetscRandomCreate(PETSC_COMM_SELF, &r));
    PetscCall(PetscRandomSetType(r, PETSCRAND48));
    PetscCall(VecSetRandom(u, r));
    PetscCall(PetscRandomDestroy(&r));
    // write the initial vector
    if (user->write_sol) {
      PetscViewer viewer;
      PetscCall(PetscViewerBinaryOpen(comm, "ic.dat", FILE_MODE_WRITE, &viewer));
      PetscCall(VecView(u, viewer));
      PetscCall(PetscViewerDestroy(&viewer));
    }
  }

  VecViewFromOptions(u, NULL, "-ic_view");
  EPS      eig;
  KSP      smoother_c, smoother_f;
  PetscInt ncv;

  // Setup SLEPc to get the eigenvectors. We can use this to evaluate
  // the behavior of the eigenvalues in regards to the error at each
  // iteration of the solve, and verify what the smoother is doing to
  // our solution (ie. whether or not multigrid is actually working at all
  // or where it is not working more specifically)
  #if 0
  PetscCall(EPSCreate(comm, &eig));
  PetscCall(EPSSetOperators(eig, M, NULL));
  PetscCall(EPSSetOptionsPrefix(eig, "eig_"));
  PetscCall(EPSSetFromOptions(eig));

  if (user->load_sol) PetscCall(EPSSolve(eig));
  user->eps = eig;// we will need this later
  #endif
  // Configure the solver to use multigrid on the operator. We start at the fine grid
  // and coarsen. Currently only support 2 levels for verification. This will generalize
  // The first pass, absolutely kill it with a ridiculous tolerance on iterations. Save the solution
  // to compute error terms
  PetscCall(KSPCreate(comm, &ksp));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCMG));
  PetscCall(PCMGSetLevels(pc, 2, NULL));
  PetscCall(PCMGSetOperators(pc, 0, MC, MC));
  PetscCall(PCMGSetOperators(pc, 1, M, M));
  PetscCall(PCMGSetRestriction(pc, 1, R));
  PetscCall(PCSetFromOptions(pc));
  PetscCall(KSPSetOperators(ksp, M, M));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSetApplicationContext(ksp, user));
  PetscCall(KSPMonitorSet(ksp, MonitorProjectError, user, NULL));
  PetscCall(KSPSolve(ksp, u, f));

  // Cleanup
  PetscCall(DMRestoreLocalVector(dm, &u));
  PetscCall(DMRestoreLocalVector(dm, &f));
  PetscCall(KSPDestroy(&ksp));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SolveDW_Fine(MPI_Comm comm, DM dm, Mat M, AppCtx *user){
  KSP         kspFine;
  EPS         eig;
  Vec         u, f;
  PetscRandom r;
  PetscInt    ncv;

  PetscFunctionBegin;
  PetscCall(DMGetGlobalVector(dm, &u));
  PetscCall(DMGetGlobalVector(dm, &f));
  // Configure the input vectors, this is not a real fermion field
  PetscCall(PetscRandomCreate(comm, &r));
  PetscCall(PetscRandomSetType(r, PETSCRAND48));
  PetscCall(VecSetRandom(u, r));
  PetscCall(PetscRandomDestroy(&r));
  PetscCall(KSPCreate(comm, &kspFine));
  PetscCall(KSPSetOperators(kspFine, M, M));
  PetscCall(KSPSetOptionsPrefix(kspFine, "ksp_fine_"));
  PetscCall(KSPSetFromOptions(kspFine));
  PetscCall(KSPSetApplicationContext(kspFine, user));
  PetscCall(KSPSolve(kspFine, u, f));
  if (user->write_sol) {
    PetscViewer viewer;

    PetscCall(PetscViewerBinaryOpen(comm, "dw_fine_solution.dat", FILE_MODE_WRITE, &viewer));
    PetscCall(VecView(f, viewer));
    PetscCall(PetscViewerDestroy(&viewer));
  }
  PetscCall(KSPDestroy(&kspFine));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SolveDW_Fine_Eig(MPI_Comm comm, DM dm, Mat M, AppCtx *user){
  KSP         kspFine;
  EPS         eig;
  Vec         u, f;
  PetscRandom r;
  PetscInt    ncv;

  PetscFunctionBegin;

  PetscCall(EPSCreate(comm, &eig));
  PetscCall(EPSSetOperators(eig, M, NULL));
  PetscCall(EPSSetOptionsPrefix(eig, "eig_"));
  PetscCall(EPSSetFromOptions(eig));
  PetscCall(EPSSolve(eig));
  PetscCall(EPSGetConverged(eig, &ncv));
  PetscCall(EPSGetOperators(eig, &M, NULL));
  for (PetscInt i = 0; i < ncv; ++i){
    PetscScalar eigr, eigi;

    PetscCall(EPSGetEigenvalue(eig, i, &eigr, &eigi));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%"PetscInt_FMT": %g %g\n", i, PetscRealPart(eigr), PetscImaginaryPart(eigr)));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SolveFineAndCoarse(MPI_Comm comm, DM dm, Mat M, Mat MC, Mat R, AppCtx *user){
  PetscScalar min, max;
  KSP         ksp;// Linear solver
  PC          pc;
  Mat         NE; // matrix for the normal equations
  Vec         u, f, uhat, error;
  PetscRandom r;

  PetscFunctionBegin;

  PetscCall(DMGetLocalVector(dm, &u));
  // Configure the input vectors, this is not a real fermion field
if (user->load_ic){
    PetscViewer viewer;

    PetscCall(PetscViewerBinaryOpen(comm, "ic.dat", FILE_MODE_READ, &viewer));
    PetscCall(VecCreate(comm, &u));
    PetscCall(VecLoad(u, viewer));
    PetscCall(PetscViewerDestroy(&viewer));
  }
  else{
    PetscCall(DMGetLocalVector(dm, &u));
    // Configure the input vectors, this is not a real fermion field
    PetscCall(PetscRandomCreate(PETSC_COMM_SELF, &r));
    PetscCall(PetscRandomSetType(r, PETSCRAND48));
    PetscCall(VecSetRandom(u, r));
    PetscCall(PetscRandomDestroy(&r));
    // write the initial vector
    if (user->write_ic) {
      PetscViewer viewer;
      PetscCall(PetscViewerBinaryOpen(comm, "ic.dat", FILE_MODE_WRITE, &viewer));
      PetscCall(VecView(u, viewer));
      PetscCall(PetscViewerDestroy(&viewer));
    }
  }
  PetscCall(DMGetLocalVector(dm, &f));

  VecViewFromOptions(u, NULL, "-ic_view");

  // solve the fine system alone all the way, down to machine precision if possible (most likely a few thousand iterates if you're patient enough)
  // then use that to compute the error
  {
    KSP kspFine;

    PetscCall(KSPCreate(comm, &kspFine));
    PetscCall(KSPSetOperators(kspFine, M, M));
    PetscCall(KSPSetOptionsPrefix(kspFine, "ksp_fine_"));
    PetscCall(KSPSetFromOptions(kspFine));
    PetscCall(KSPSetApplicationContext(kspFine, user));
    PetscCall(KSPSolve(kspFine, u, f));
    if (user->write_sol) {
        PetscViewer viewer;

        PetscCall(PetscViewerBinaryOpen(comm, "solution.dat", FILE_MODE_WRITE, &viewer));
        PetscCall(VecView(f, viewer));
        PetscCall(PetscViewerDestroy(&viewer));
      }
    PetscCall(KSPDestroy(&kspFine));
  }

  // solve the coarse system alone all the way down now,
  // project back to the fine space, compare the error.
  Vec fCProj;
  PetscCall(DMGetLocalVector(dm, &fCProj));
  {
    KSP kspCoarse;
    DM  cdm;
    Vec phiCoarse, fCoarse;

    PetscCall(DMGetCoarseDM(dm, &cdm));
    PetscCall(DMGetLocalVector(cdm, &phiCoarse));//Get vectors conforming to coarse space dofs
    PetscCall(DMGetLocalVector(cdm, &fCoarse));
    PetscCall(KSPCreate(comm, &kspCoarse));
    PetscCall(KSPSetOperators(kspCoarse, MC, MC));
    PetscCall(KSPSetOptionsPrefix(kspCoarse, "ksp_coarse_"));
    PetscCall(KSPSetFromOptions(kspCoarse));
    PetscCall(KSPSetApplicationContext(kspCoarse, user));
    PetscCall(MatMult(R, u, phiCoarse));
    PetscCall(VecViewFromOptions(phiCoarse, NULL, "-coarse_basis_vec_view"));
    PetscCall(KSPSolve(kspCoarse, phiCoarse, fCoarse));// solve all the way down
    // project back to the fine space w/ R^T
    PetscCall(MatMultTranspose(R, fCoarse, fCProj));// run fine solver
  }

  // Subtract this from the "exact" solution in the coarse space and plot the errors
  // that still remain.
  PetscCall(VecDuplicate(fCProj, &error));
  PetscCall(VecWAXPY(error, -1., f, fCProj));
  PetscCall(VecViewFromOptions(error, NULL, "-error_vec_view"));
  EPS      eig;
  KSP      smoother_c, smoother_f;
  PetscInt ncv;
  PetscReal projectedNorm;
  PetscCall(VecNorm(fCProj, NORM_2, &projectedNorm));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Projected Norm: %f\n", projectedNorm));

  // Setup SLEPc to get the eigenvectors. We can use this to evaluate
  // the behavior of the eigenvalues in regards to the error at each
  // iteration of the solve, and verify what the smoother is doing to
  // our solution (ie. whether or not multigrid is actually working at all
  // or where it is not working more specifically)
  // experimenting with writing out the eigenvectors because this is terrible
  // to rerun after every little thing
  if (user->write_eig) {
    PetscCall(PetscPrintf(comm, "creating eps\n"));
    PetscCall(EPSCreate(comm, &eig));
    PetscCall(EPSSetOperators(eig, M, NULL));
    PetscCall(EPSSetOptionsPrefix(eig, "eig_"));
    PetscCall(EPSSetFromOptions(eig));
    // Do the eigensolve on the fine operator
    PetscCall(PetscPrintf(comm, "solving\n"));
    PetscCall(EPSSolve(eig));

    PetscCall(EPSGetConverged(eig, &ncv));
    PetscCall(EPSGetOperators(eig, &M, NULL));

    // Compute and ouptut |fine solution - projected coarse solution| \cdot eigenvector
    Vec eigenVectorI, eigenVectorR;
    for (PetscInt i = 0; i < 100; ++i){
      PetscScalar proj;
      PetscViewer viewer;
      char eigenVector[15];

      PetscCall(MatCreateVecs(M, NULL, &eigenVectorI));// Imaginary part (will not be zeroed)
      PetscCall(MatCreateVecs(M, NULL, &eigenVectorR));// this will be zeroed
      PetscCall(EPSGetEigenvector(eig, i, eigenVectorR, eigenVectorI));
      // write the eigenvectors out;
      PetscCall(PetscSNPrintf(eigenVector, 15, "eig%"PetscInt_FMT".dat", i));
      PetscCall(PetscViewerBinaryOpen(comm, eigenVector, FILE_MODE_WRITE, &viewer));
      PetscCall(VecView(eigenVectorR, viewer));
      PetscCall(PetscViewerFlush(viewer));
      PetscCall(PetscViewerDestroy(&viewer));
      PetscCall(VecDot(eigenVectorR, error, &proj));
      PetscReal norm;
      PetscCall(VecNorm(eigenVectorR, NORM_2, &norm));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Error: %f %f Mod Coef.: %f Norm: %f\n", PetscRealPart(proj), PetscImaginaryPart(proj), PetscSqrtReal(PetscSqr(PetscRealPart(proj))+PetscSqr(PetscImaginaryPart(proj))), norm));
      PetscCall(PetscPrintf(comm, "cleaning up\n"));
      PetscCall(VecDestroy(&eigenVectorI));
      PetscCall(VecDestroy(&eigenVectorR));
    }
  }
  else {
    // get the number of eigenvectors from the user, default is 10
    for (PetscInt i = 0; i < 100; ++i) {
      PetscViewer viewer;
      char eigenVector[15];
      Vec eigenVectorR;
      PetscScalar proj;

      PetscCall(PetscSNPrintf(eigenVector, 15, "eig%"PetscInt_FMT".dat", i));
      PetscCall(PetscViewerBinaryOpen(comm, eigenVector, FILE_MODE_READ, &viewer));
      PetscCall(VecCreate(comm, &eigenVectorR));
      PetscCall(VecLoad(eigenVectorR, viewer));
      PetscCall(PetscViewerDestroy(&viewer));
      PetscCall(VecViewFromOptions(eigenVectorR, NULL, "-eig_vec_view"));

      PetscCall(VecDot(eigenVectorR, error, &proj));
      PetscReal norm;
      PetscCall(VecNorm(eigenVectorR, NORM_2, &norm));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Error: %f %f Mod Coef.: %f Norm: %f\n", PetscRealPart(proj), PetscImaginaryPart(proj), PetscSqrtReal(PetscSqr(PetscRealPart(proj))+PetscSqr(PetscImaginaryPart(proj))), norm));
      PetscCall(VecDestroy(&eigenVectorR));
    }
  }
  // Compute the error of the smoother, we need to solve for the eigenbasis first so do this after the eigensolve.
  {
    KSP kspFineSmoother;

    PetscCall(KSPCreate(comm, &kspFineSmoother));
    PetscCall(KSPSetOperators(kspFineSmoother, M, M));
    PetscCall(KSPSetOptionsPrefix(kspFineSmoother, "ksp_smoother_"));
    PetscCall(KSPSetFromOptions(kspFineSmoother));
    PetscCall(KSPSetApplicationContext(kspFineSmoother, user));
    PetscCall(KSPMonitorSet(kspFineSmoother, MonitorProjectError, user, NULL));
    PetscCall(KSPSetInitialGuessNonzero(kspFineSmoother, PETSC_TRUE));
    PetscCall(KSPSolve(kspFineSmoother, u, fCProj));
  }
  PetscCall(EPSDestroy(&eig));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestCoarsening(MPI_Comm comm, DM dm, Mat M, Mat MC, Mat R, AppCtx *user){
  PetscScalar min, max;
  KSP         ksp;// Linear solver
  PC          pc;
  Mat         NE; // matrix for the normal equations
  Vec         u, f, uhat, error;
  PetscRandom r;
  DM          cdm;
  PetscFunctionBegin;
  PetscCall(DMGetLocalVector(dm, &u));
  PetscCall(DMGetLocalVector(dm, &f));
  PetscCall(DMGetLocalVector(dm, &error));
  // Configure the input vectors, this is not a real fermion field
  PetscCall(PetscRandomCreate(PETSC_COMM_SELF, &r));
  PetscCall(PetscRandomSetType(r, PETSCRAND48));
  //PetscCall(VecSetRandom(u, r));
  PetscCall(VecSet(u, 1.));
  PetscCall(PetscRandomDestroy(&r));
  PetscReal normF, normC, normProjC;
  PetscCall(VecNorm(u, NORM_2, &normF));

  PetscCall(DMGetCoarseDM(dm, &cdm));

  Vec fine, coarse;
  PetscCall(DMGetAuxiliaryVec(dm, NULL, 0, 0,&fine));
  PetscCall(DMGetAuxiliaryVec(dm, NULL, 0, 0,&coarse));
  PetscCall(VecViewFromOptions(fine, NULL, "-fine_field_view"));
  PetscCall(VecViewFromOptions(coarse, NULL, "-coarse_field_view"));

  PetscCall(DMGetLocalVector(cdm, &uhat));

  PetscCall(MatMult(R, u, uhat));
  PetscCall(VecViewFromOptions(uhat, NULL, "-restricted_vec_view"));
  PetscCall(VecNorm(uhat, NORM_2, &normC));
  PetscCall(MatMultTranspose(R, uhat, f));

  VecViewFromOptions(f, NULL, "-interpolated_vec_view");
  PetscCall(VecNorm(f, NORM_2, &normProjC));

  PetscCall(VecWAXPY(error, -1., u, f));
  PetscCall(VecViewFromOptions(error, NULL, "-coarsening_error_vec_view"));
  PetscPrintf(comm, "Fine Norm: %f Coarse Norm: %f Projected Coarse Norm: %f\n", PetscRealPart(normF), PetscRealPart(normC), PetscRealPart(normProjC));

  PetscFunctionReturn(PETSC_SUCCESS);
}
// PV test
// ./ex7 -dm_plex_dim 5 -dm_plex_shape hypercubic -dm_plex_box_faces 8,8,8,8,8 -grid_load_type 3 -grid_file ${GRID_LATTICE_FILE} --grid 8.8.8.8 -use_pv -eig_eps_monitor -eig_eps_nev 200 -eig_eps_smallest_real
int main(int argc, char **argv)
{
  DM     dm, cdm;
  Vec    u, f;
  Mat    M, R;
  PetscRandom r;
  PetscInt locSize;
  AppCtx user;
  MPI_Comm comm;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  #ifdef PETSC_HAVE_SLEPC
  PetscCall(SlepcInitialize(&argc, &argv, NULL, help));
  #endif
  comm = PETSC_COMM_WORLD;
  PetscCall(ProcessOptions(comm, &user));
  PetscCall(CreateMesh(comm, &user, &dm));
  PetscCall(DMSetApplicationContext(dm, &user));
  PetscCall(SetupDiscretization(dm, &user));
  PetscCall(SetupAuxDiscretization(dm, &user));

  PetscCall(DMCreateGlobalVector(dm, &u));
  PetscCall(DMCreateGlobalVector(dm, &f));

  PetscCall(PetscRandomCreate(PETSC_COMM_SELF, &r));
  PetscCall(PetscRandomSetType(r, PETSCRAND48));
  PetscCall(VecSetRandom(u, r));
  PetscCall(PetscRandomDestroy(&r));
  PetscPrintf(comm, "%s\n", user.gridFile);
  if (user.DWsqr){
    PetscCall(PetscPrintf(comm, "Running squared domain wall operator.\n"));
    PetscCall(SetUpSqrDW(dm, &M, 8, &user, argc, argv));
    PetscCall(MatSetDM(M, dm));
    PetscCall(SolveDW_Fine(comm, dm, M, &user));
  }
  else if (user.domainWall) {
    PetscCall(PetscPrintf(comm, "Running standard domain wall operator.\n"));
    PetscCall(SetUpDW(dm, &M, PETSC_FALSE, 16, &user, argc, argv));
    PetscCall(MatSetDM(M, dm));
    //PetscCall(SolveDW_Fine_Eig(comm, dm, M, &user));
    PetscCheckDwfWithGrid(dm, M, u,f);
  }
  else if (user.usePV) {
    DM pvdm;
    Mat P, precOp;
    // Just represent the matrix as a unit mass on the diagonal?
    // currently m is stored in the gauge links, so we may need to
    // break that out of SetGauge5D in plexGrid.cxx
    PetscCall(CreateMesh(comm, &user, &pvdm));
    PetscCall(DMSetApplicationContext(pvdm, &user));
    PetscCall(SetupDiscretization(pvdm, &user));
    PetscCall(SetupAuxDiscretization(pvdm, &user));
    PetscCall(SetUpDW(dm, &M, PETSC_FALSE, 16, &user, argc, argv));
    PetscCall(SetUpDW(pvdm, &P, PETSC_TRUE, 16, &user, argc, argv));
    PetscCall(MatSetDM(P, pvdm));
    user.pv = P;
    PetscCall(MatShellSetContext(M, (void*)&user));
    PetscCall(SetUpPreconditionedOperator(dm, &precOp, &user));
    PetscCall(MatSetDM(precOp, dm));
    PetscCall(SolveDW_Fine(comm, dm, precOp, &user));
  }
  else {
    PetscCall(SetupWilson(dm, PETSC_TRUE, &M, &user, argc, argv));
    PetscCall(MatSetDM(M, dm));
    if (user.coarsen) {
      Vec ff, cf;
      Mat MC;

      PetscCall(SetupCoarseSpace(comm, &cdm, &user));
      PetscCall(SetupDiscretization(cdm, &user));
      PetscCall(SetupAuxDiscretization(cdm, &user));
      PetscCall(RestrictGaugeField(dm, cdm));
      PetscCall(SetupWilson(cdm, PETSC_FALSE, &MC, &user, argc, argv));
      PetscCall(DMSetCoarseDM(dm, cdm));
      PetscCall(MatSetDM(MC, cdm));
      PetscCall(SetupRestriction(dm, &R, &user));
      if (user.testCoarsening) PetscCall(TestCoarsening(comm, dm, M, MC, R, &user));
      else{
        if (user.solveFineAndCoarse) PetscCall(SolveFineAndCoarse(comm, dm, M, MC, R, &user));
        //PetscCall(SolveSystemError(comm, dm, M, MC, R, &user));
      }
      PetscCall(MatDestroy(&R));
      PetscCall(DMDestroy(&cdm));
    }
  }
  PetscCall(MatDestroy(&M));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}
/*
./ex7 -dm_plex_dim 4 -dm_plex_shape hypercubic -dm_plex_box_faces 8,8,8,8 -dm_view -coarsen -coarse_dm_plex_dim 4 -coarse_dm_plex_shape hypercubic -coarse_dm_plex_box_faces 4,4,4,4 -grid_file ckpoint_lat.4000 -grid_load_type 2 -eig_eps_type krylovschur -eig_eps_monitor -eig_eps_view -eig_eps_nev 100 -eig_eps_non_hermitian -eig_eps_max_it 10000 --grid 8.8.8.8 -ksp_fine_ksp_type bcgs -ksp_fine_ksp_rtol 1e-12 -ksp_fine_pc_type none -ksp_fine_ksp_max_it 10000 -ksp_coarse_ksp_type bcgs -ksp_coarse_ksp_pc_type none -ksp_coarse_ksp_rtol 1e-12 -ksp_coarse_ksp_max_it 10000 -ksp_fine_ksp_monitor -ksp_coarse_ksp_monitor  -ksp_gmres_breakdown_tolerance 1000 -ksp_monitor -ksp_gmres_restart 1000 -use_fine_and_coarse -ksp_smoother_ksp_initial_guess_nonzero -ksp_smoother_ksp_type bcgs -ksp_smoother_ksp_rtol 1e-12 -ksp_smoother_pc_type none -ksp_smoother_ksp_max_its 1000 -ksp_smoother_ksp_monitor -write_sol -save_eigenbasis -temperature 0.012 >> test_t0012.out

./ex7 -dm_plex_dim 5 -dm_plex_shape hypercubic -dm_plex_box_faces 8,8,8,8,8 -grid_load_type 3 -grid_file ${GRID_LATTICE_FILE} --grid 8.8.8.8 -log_view -run_squared_domain_wall 1 -eig_eps_nev 100 -eig_eps_hermitian -eig_eps_tol 1e-16 -eig_eps_monitor -log_view -eig_eps_smallest_real  > output/square_dw_smallest_log_8x8.out

for DW run that converges, use cgne w/
./ex7 -dm_plex_dim 5 -dm_plex_shape hypercubic -dm_plex_box_faces 16,16,16,16,32 -dm_view -sol_view -ksp_fine_ksp_type cgne -ksp_fine_ksp_monitor -ksp_fine_ksp_view -ksp_fine_ksp_rtol 1e-12 -ksp_fine_ksp_atol 1e-50 -ksp_fine_ksp_max_it 1000 -ksp_fine_pc_type none -grid_load_type 3 -grid_file ${GRID_LATTICE_FILE} --grid 16.16.16.32 -log_view -run_domain_wall 1
*/
/*TEST
  build:
    requires: complex
  testset:
    requires: fftw
    suffix: pv_
    args: -dm_plex_dim 5 -dm_plex_shape hypercubic -dm_plex_box_faces 8,8,8,8,8 -grid_load_type 3 -grid_file ${GRID_LATTICE_FILE}\
          --grid 8.8.8.8 -use_pv -log_view
    test:
      suffix: cg
      args: -ksp_fine_ksp_type cg -ksp_fine_ksp_max_it 50000 -ksp_fine_ksp_rtol 1e-14
    test:
      suffix: cgne
      args: -ksp_fine_ksp_type cgne -ksp_fine_ksp_max_it 50000 -ksp_fine_ksp_rtol 1e-14
    test:
      suffix: bicg
      args: -ksp_fine_ksp_type bicg -ksp_fine_ksp_max_it 50000 -ksp_fine_ksp_rtol 1e-14
    test:
      suffix: tfqmr
      args: -ksp_fine_ksp_type tfqmr -ksp_fine_ksp_max_it 50000 -ksp_fine_ksp_rtol 1e-14
    test:
      suffix: lsqr
      args: -ksp_fine_ksp_type lsqr -pc_type none -ksp_fine_ksp_max_it 50000 -ksp_fine_ksp_rtol 1e-14
    test:
      suffix: cgr
      args: -ksp_fine_ksp_type cgr -ksp_fine_ksp_max_it 50000 -ksp_fine_ksp_rtol 1e-14
    test:
      suffix: cgs
      args: -ksp_fine_ksp_type cgs -ksp_fine_ksp_max_it 50000 -ksp_fine_ksp_rtol 1e-14
    test:
      suffix: tcqmr
      args: -ksp_fine_ksp_type tcqmr -ksp_fine_ksp_max_it 50000 -ksp_fine_ksp_rtol 1e-14
  test:
    requires: fftw
    suffix: wilson_find_mass
    args: -dm_plex_dim 4 -dm_plex_shape hypercubic -dm_view\
          -dm_plex_box_faces 8,8,8,8\
          -coarsen -coarse_dm_plex_dim 4 -coarse_dm_plex_shape hypercubic\
          -coarse_dm_plex_box_faces 4,4,4,4\
          -grid_file ckpoint_lat.4000 -grid_load_type 2\
          -eig_eps_type krylovschur -eig_eps_monitor -eig_eps_view -eig_eps_nev 100 -eig_eps_non_hermitian -eig_eps_max_it 10000\
          -ksp_fine_ksp_type bcgs -ksp_fine_ksp_rtol 1e-12 -ksp_fine_pc_type none -ksp_fine_ksp_max_it 10000\
          -ksp_coarse_ksp_type bcgs -ksp_coarse_ksp_pc_type none -ksp_coarse_ksp_rtol 1e-12 -ksp_coarse_ksp_max_it 10000\
          -ksp_fine_ksp_monitor -ksp_coarse_ksp_monitor  -ksp_gmres_breakdown_tolerance 1000 -ksp_monitor -ksp_gmres_restart 1000\
          -use_fine_and_coarse\
          -ksp_smoother_ksp_initial_guess_nonzero -ksp_smoother_ksp_type bcgs -ksp_smoother_ksp_rtol 1e-12\
          -ksp_smoother_pc_type none -ksp_smoother_ksp_max_its 1000 -ksp_smoother_ksp_monitor\
          -write_sol -save_eigenbasis -load_ic\
          -temperature {{0.01 0.001 0.0001 0.00001}} -mass {{0.001 0.01 0.1 0.11 0.12 0.13 0.2 0.3 0.4 0.5 0.6 0.7}}\
          --grid 8.8.8.8
  test:
    requires: fftw
    suffix: wilson_restriction
    args: -dm_plex_dim 4 -dm_plex_shape hypercubic -dm_plex_box_faces 8,8,8,8 -dm_view -sol_view -coarsen -coarse_dm_plex_dim 4 -coarse_dm_plex_shape hypercubic -coarse_dm_plex_box_faces 4,4,4,4 -coarse_dm_view -grid_file ckpoint_lat.4000 -grid_load_type 2 --grid 8.8.8.8 -ksp_rtol 1e-16 -ksp_max_it 1000 -ksp_converged_reason -pc_type mg -mg_levels_ksp_max_it 10 -mg_levels_esteig_ksp_type cgne -mg_levels_esteig_ksp_max_it 10 -mg_levels_pc_type none -ksp_view_pre -mg_coarse_pc_type none -eig_eps_type krylovschur -eig_eps_monitor -eig_eps_view -eig_eps_nev 4096 -eig_eps_non_hermitian -load_sol -ksp_monitor -ksp_gmres_restart 100
  test:
    requires: fftw
    suffix: dirac_free_field
    args: -dm_plex_dim 4 -dm_plex_shape hypercubic -dm_plex_box_faces 4,4,4,4 -dm_view -sol_view \
          -eigmax_eps_largest_magnitude -eigmin_eps_smallest_magnitude\
          -ksp_type chebyshev\
          -grid_file ckpoint_lat.4000\
          -grid_load_type 3\
          --grid 4.4.4.4
  test:
    requires: fftw
    suffix: domain_wall_5d_plex_pc_none_cg
    args: -dm_plex_dim 5 -dm_plex_shape hypercubic -dm_plex_box_faces 16,16,16,16,32 -dm_view -sol_view \
          -ksp_type cg -ksp_monitor -ksp_view -ksp_rtol 1e-12 -ksp_atol 1e-50 -ksp_max_it 1000\
          -pc_type none\
          -grid_load_type 3\
          -grid_file ${GRID_LATTICE_FILE}\
          --grid 16.16.16.32\
          -log_view\
          -run_domain_wall 1
  test:
    requires: fftw
    nsize: 1
    suffix: domain_wall_5d_plex_pc_none_cgne
    args: -dm_plex_dim 5 -dm_plex_shape hypercubic -dm_plex_box_faces 16,16,16,16,32 -dm_view -sol_view \
          -ksp_type cgne -ksp_monitor -ksp_view -ksp_rtol 1e-12 -ksp_atol 1e-50 -ksp_max_it 1000\
          -pc_type none\
          -grid_load_type 3\
          -grid_file ${GRID_LATTICE_FILE}\
          --grid 16.16.16.32\
          -log_view\
          -run_domain_wall 1
  test:
    requires: fftw
    suffix: domain_wall_5d_plex_pc_none_bicg
    args: -dm_plex_dim 5 -dm_plex_shape hypercubic -dm_plex_box_faces 16,16,16,16,32 -dm_view -sol_view \
          -ksp_type bicg -ksp_monitor -ksp_view -ksp_rtol 1e-12 -ksp_atol 1e-50 -ksp_max_it 1000\
          -pc_type none\
          -grid_load_type 3\
          -grid_file ${GRID_LATTICE_FILE}\
          --grid 16.16.16.32\
          -log_view\
          -run_domain_wall 1
  test:
    requires: fftw
    suffix: domain_wall_5d_plex_pc_none_bicgstab
    args: -dm_plex_dim 5 -dm_plex_shape hypercubic -dm_plex_box_faces 16,16,16,16,32 -dm_view -sol_view \
          -ksp_type bcgs -ksp_monitor -ksp_view -ksp_rtol 1e-12 -ksp_atol 1e-50 -ksp_max_it 1000\
          -pc_type none\
          -grid_load_type 3\
          -grid_file ${GRID_LATTICE_FILE}\
          --grid 16.16.16.32\
          -log_view\
          -run_domain_wall 1
  test:
    requires: fftw
    suffix: domain_wall_5d_plex_pc_none_gmres
    args: -dm_plex_dim 5 -dm_plex_shape hypercubic -dm_plex_box_faces 16,16,16,16,32 -dm_view -sol_view \
          -ksp_type gmres -ksp_monitor -ksp_view -ksp_rtol 1e-12 -ksp_atol 1e-50 -ksp_max_it 1000\
          -pc_type none\
          -grid_load_type 3\
          -grid_file ${GRID_LATTICE_FILE}\
          --grid 16.16.16.32\
          -log_view\
          -run_domain_wall 1
  test:
    requires: fftw
    suffix: domain_wall_5d_plex_pc_none_tfqmr
    args: -dm_plex_dim 5 -dm_plex_shape hypercubic -dm_plex_box_faces 16,16,16,16,32 -dm_view -sol_view \
          -ksp_type tfqmr -ksp_monitor -ksp_view -ksp_rtol 1e-12 -ksp_atol 1e-50 -ksp_max_it 1000\
          -pc_type none\
          -grid_load_type 3\
          -grid_file ${GRID_LATTICE_FILE}\
          --grid 16.16.16.32\
          -log_view\
          -run_domain_wall 1
  test:
    requires: fftw
    suffix: domain_wall_5d_plex_pc_none_qmrcgs
    args: -dm_plex_dim 5 -dm_plex_shape hypercubic -dm_plex_box_faces 16,16,16,16,32 -dm_view -sol_view \
          -ksp_type qmrcgs -ksp_monitor -ksp_view -ksp_rtol 1e-12 -ksp_atol 1e-50 -ksp_max_it 1000\
          -pc_type none\
          -grid_load_type 3\
          -grid_file ${GRID_LATTICE_FILE}\
          --grid 16.16.16.32\
          -log_view\
          -run_domain_wall 1
  test:
    requires: fftw
    suffix: domain_wall_5d_plex_pc_none_cgs
    args: -dm_plex_dim 5 -dm_plex_shape hypercubic -dm_plex_box_faces 16,16,16,16,32 -dm_view -sol_view \
          -ksp_type cgs -ksp_monitor -ksp_view -ksp_rtol 1e-12 -ksp_atol 1e-50 -ksp_max_it 1000\
          -pc_type none\
          -grid_load_type 3\
          -grid_file ${GRID_LATTICE_FILE}\
          --grid 16.16.16.32\
          -log_view\
          -run_domain_wall 1
  test:
    requires: fftw
    suffix: domain_wall_5d_plex_pc_none_gcr
    args: -dm_plex_dim 5 -dm_plex_shape hypercubic -dm_plex_box_faces 16,16,16,16,32 -dm_view -sol_view \
          -ksp_type gcr -ksp_monitor -ksp_view -ksp_rtol 1e-12 -ksp_atol 1e-50 -ksp_max_it 1000\
          -pc_type none\
          -grid_load_type 3\
          -grid_file ${GRID_LATTICE_FILE}\
          --grid 16.16.16.32\
          -log_view\
          -run_domain_wall 1
  test:
    requires: fftw
    suffix: domain_wall_5d_plex_pc_none_cgls
    args: -dm_plex_dim 5 -dm_plex_shape hypercubic -dm_plex_box_faces 16,16,16,16,32 -dm_view -sol_view \
          -ksp_type cgls -ksp_monitor -ksp_view -ksp_rtol 1e-12 -ksp_atol 1e-50 -ksp_max_it 1000\
          -pc_type none\
          -grid_load_type 3\
          -grid_file ${GRID_LATTICE_FILE}\
          --grid 16.16.16.32\
          -log_view\
          -run_domain_wall 1
  test:
    requires: fftw
    suffix: domain_wall_5d_plex_pc_none_lcd
    args: -dm_plex_dim 5 -dm_plex_shape hypercubic -dm_plex_box_faces 16,16,16,16,32 -dm_view -sol_view \
          -ksp_type lcd -ksp_monitor -ksp_view -ksp_rtol 1e-12 -ksp_atol 1e-50 -ksp_max_it 1000\
          -pc_type none\
          -grid_load_type 3\
          -grid_file ${GRID_LATTICE_FILE}\
          --grid 16.16.16.32\
          -log_view\
          -run_domain_wall 1
  testset:
    args: -dm_plex_dim 5 -dm_plex_shape hypercubic -dm_plex_box_faces 16,16,16,16,32 -dm_view -sol_view \
          -ksp_monitor -ksp_view -ksp_rtol 1e-12 -ksp_atol 1e-50 -ksp_max_it 1000\
          -pc_type none\
          -grid_load_type 3\
          -grid_file ${GRID_LATTICE_FILE}\
          --grid 16.16.16.32\
          -log_view\
          -run_domain_wall 1
    test:
      suffix: domain_wall_5d_plex_pc_none_tcqmr
      args: -ksp_type tcqmr
    test:
      suffix: domain_wall_5d_plex_pc_none_lsqr
      args: -ksp_type lsqr
    test:
      suffix: domain_wall_5d_plex_pc_none_cr
      args: -ksp_type cr
TEST*/
