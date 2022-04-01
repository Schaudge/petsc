#include <petsc.h>
#include <petscdmbf.h>
#include <petscdmforest.h>

/* A Poisson problem with Dirichlet boundary conditions on the square [-1,1]^2.
   We add a slit on the strip \Gamma = {(x,y) : y = 0, -1 <= x <= 0} and treat \Gamma as part of the boundary.
   The exact solution is r^{1/2}*Sin(\theta/2) in polar coordinates, and has a singularity at the origin which we resolve using AMR.

  Three steps to applying the matrix-free operator:
    1. Transcribe unknowns from Vec into cell data.
    2. Set guard layer by iterating over faces, interpolating where necessary.
    3. Apply operator by looping over cells, using guard layer for fluxes.
 */

// JR need a chart how cell data is organized

static char help[] = "";

/*
stencils of convergence order O(h^p), we refer to p as the order of the stencil
*/

#define ORDER_2_STENCIL {-1.0,2.0,-1.0}
#define ORDER_4_STENCIL {1.0/12.0,-4.0/3.0,5.0/2.0,-4.0/3.0,1.0/12.0}
#define ORDER_6_STENCIL {-1.0/90.0,3.0/20.0,-3.0/2.0,49.0/18.0,-3.0/2.0,3.0/20.0,-1.0/90.0}
// JR these stencils are not used

typedef enum {GUARD_W, GUARD_E, GUARD_S, GUARD_N, GUARD_C} guardloc;
// JR the name GUARD_C is misleading since it is not actually a guard entry, but the interior of the cell

typedef PetscScalar (*BndryConditionFn)(PetscReal,PetscReal);
typedef PetscScalar (*SourceFn)(PetscReal,PetscReal);
// JR these two function declarations follow the same pattern.  only one is needed actually

typedef struct {
  BndryConditionFn g;
  SourceFn         f;
} AppCtx;

typedef struct {
  //nothing for now
} cellData_t;
// PetscErrorCode get(Boundary)FaceMidpoint()

// JR clearing something usually has am association of destroying/deleting; but in `clear_cell_data` the data seems to be initialized/zeroed
PetscErrorCode clear_cell_data(DM dm, DM_BF_Cell *cell, void *ctx) {
  PetscFunctionBegin;
  cell->data[0][0]=0.0;
  PetscFunctionReturn(0);
}
PetscErrorCode set_up_boundary_condition(DM dm, DM_BF_Face *face, void *ctx) {
  AppCtx          *user      = ctx;
  DM_BF_Cell      *cell      = (face->cellL[0] ? face->cellL[0] : face->cellR[0]);
  PetscBool       isBoundary = (DM_BF_FACEBOUNDARY_NONE != face->boundary);
  PetscScalar     h, bndryFaceMidpoint_x, bndryFaceMidpoint_y;

  PetscFunctionBegin;

  if (isBoundary) {
    switch(face->dir) {
      case DM_BF_FACEDIR_XNEG:
        h = cell->sidelength[0];
        bndryFaceMidpoint_x = cell->corner[0];
        bndryFaceMidpoint_y = cell->corner[1] + .5*h;
        break;
      case DM_BF_FACEDIR_XPOS:
        h = cell->sidelength[0];
        bndryFaceMidpoint_x = cell->corner[0] + h;
        bndryFaceMidpoint_y = cell->corner[1] + .5*h;
        break;
      case DM_BF_FACEDIR_YNEG:
        h = cell->sidelength[1];
        bndryFaceMidpoint_x = cell->corner[0] + .5*h;
        bndryFaceMidpoint_y = cell->corner[1];
        break;
      case DM_BF_FACEDIR_YPOS:
        h = cell->sidelength[1];
        bndryFaceMidpoint_x = cell->corner[0] + .5*h;
        bndryFaceMidpoint_y = cell->corner[1] + h;
        break;
      default:
        SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Unknown or incorrect face direction %i",face->dir);
    }
    // JR looking at these many cases, it seems it would be better to store the coordinates of cell centers and face centers in the DMBF cell data

    cell->data[0][0] += -2*user->g(bndryFaceMidpoint_x,bndryFaceMidpoint_y) / (h*h);
    // JR here cell data is accessed with a slightly different code than in, eg, line 120
    //    i think it's important to unify this so a new user can quickly understand our design and the intended use of DMBF
  }

  PetscFunctionReturn(0);
}

PetscErrorCode interpolate_source(DM dm, DM_BF_Cell *cell, void *ctx) {
  AppCtx    *user      = ctx;
  PetscReal *cell_val  = cell->vecViewReadWrite[0];
  PetscReal x          = cell->corner[0]+.5*cell->sidelength[0];
  PetscReal y          = cell->corner[1]+.5*cell->sidelength[1];

  PetscFunctionBegin;
  *cell_val = user->f(x,y) + cell->data[0][0];
  PetscFunctionReturn(0);
}

PetscErrorCode interpolate_exact(DM dm, DM_BF_Cell *cell, void *ctx) {
  AppCtx    *user      = ctx;
  PetscReal *cell_val  = cell->vecViewReadWrite[0];
  PetscReal x          = cell->corner[0]+.5*cell->sidelength[0];
  PetscReal y          = cell->corner[1]+.5*cell->sidelength[1];

  PetscFunctionBegin;
  *cell_val = user->g(x,y);
  PetscFunctionReturn(0);
}

static PetscErrorCode _p_dmbf_poisson_set_unk_cellfn(DM dm, DM_BF_Cell *cell, void *ctx) {
  const PetscScalar *vec_unk   = cell->vecViewRead[0];
  PetscReal         *cell_unk  = cell->data[0];

  PetscFunctionBegin;
  cell_unk[GUARD_C] = *vec_unk;
  PetscFunctionReturn(0);
}

static PetscErrorCode _p_dmbf_poisson_set_guards_facefn(DM dm, DM_BF_Face *face, void *ctx) {
  const PetscInt  nCellsL    = face->nCellsL;
  const PetscInt  nCellsR    = face->nCellsR;
  const PetscBool isBoundary = (!nCellsL || !nCellsR);//(DM_BF_FACEBOUNDARY_NONE != face->boundary);
  const PetscBool isHangingL = (1 < nCellsL);
  const PetscBool isHangingR = (1 < nCellsR);
  const PetscBool X_DIR = (face->dir == DM_BF_FACEDIR_XNEG || face->dir == DM_BF_FACEDIR_XPOS);

  PetscFunctionBegin;

  if (isBoundary) {
    DM_BF_Cell *cell = face->cellL[0] ? face->cellL[0] : face->cellR[0];
    cell->data[0][face->dir] = -cell->data[0][GUARD_C];
  } else if (!X_DIR && face->cellL[0]->corner[1] +.5*face->cellL[0]->sidelength[1] < 0 && face->cellR[0]->corner[1]+.5*face->cellR[0]->sidelength[1] > 0
                  && face->cellL[0]->corner[0] + .5*face->cellL[0]->sidelength[0] < 0) { /* the strip {(x,y) : y = 0, -1 <= x <= 0} is part of the boundary */
                  // JR what are all these conditions?
    DM_BF_Cell *cell;
    for(PetscInt i=0;i<nCellsL;i++) {
      cell = face->cellL[i];
      cell->data[0][GUARD_N] = -cell->data[0][GUARD_C];
    }
    for(PetscInt i=0;i<nCellsR;i++) {
      cell = face->cellR[i];
      cell->data[0][GUARD_S] = -cell->data[0][GUARD_C];
    }
  } else {
    if (isHangingL) {
      DM_BF_Cell **cellL = face->cellL;
      DM_BF_Cell *cellR  = face->cellR[0];
      cellL[0]->data[0][X_DIR ? GUARD_E : GUARD_N] = (2./3.)*cellR->data[0][GUARD_C]    + (2./3.)*cellL[0]->data[0][GUARD_C] - (1./3.)*cellL[1]->data[0][GUARD_C];
      cellL[1]->data[0][X_DIR ? GUARD_E : GUARD_N] = (2./3.)*cellR->data[0][GUARD_C]    + (2./3.)*cellL[1]->data[0][GUARD_C] - (1./3.)*cellL[0]->data[0][GUARD_C];
      cellR->data[0][X_DIR ? GUARD_W : GUARD_S]    = (2./3.)*cellL[0]->data[0][GUARD_C] + (2./3.)*cellL[1]->data[0][GUARD_C] - (1./3.)*cellR->data[0][GUARD_C];
    } else if (isHangingR) {
      DM_BF_Cell **cellR  = face->cellR;
      DM_BF_Cell *cellL = face->cellL[0];
      cellR[0]->data[0][X_DIR ? GUARD_W : GUARD_S] = (2./3.)*cellL->data[0][GUARD_C]    + (2./3.)*cellR[0]->data[0][GUARD_C] - (1./3.)*cellR[1]->data[0][GUARD_C];
      cellR[1]->data[0][X_DIR ? GUARD_W : GUARD_S] = (2./3.)*cellL->data[0][GUARD_C]    + (2./3.)*cellR[1]->data[0][GUARD_C] - (1./3.)*cellR[0]->data[0][GUARD_C];
      cellL->data[0][X_DIR ? GUARD_E : GUARD_N]    = (2./3.)*cellR[0]->data[0][GUARD_C] + (2./3.)*cellR[1]->data[0][GUARD_C] - (1./3.)*cellL->data[0][GUARD_C];
    } else {
      DM_BF_Cell *cellL = face->cellL[0];
      DM_BF_Cell *cellR = face->cellR[0];
      cellL->data[0][X_DIR ? GUARD_E : GUARD_N]    = cellR->data[0][GUARD_C];
      cellR->data[0][X_DIR ? GUARD_W : GUARD_S]    = cellL->data[0][GUARD_C];
    }
  }

  PetscFunctionReturn(0);
}

static PetscErrorCode _p_dmbf_poisson_apply_operator_cellfn(DM dm, DM_BF_Cell *cell, void *ctx) {
  PetscReal *vec_out = cell->vecViewReadWrite[0];
  PetscReal hx       = cell->sidelength[0];
  PetscReal hy       = cell->sidelength[1];

  PetscFunctionBegin;

  *vec_out = (cell->data[0][GUARD_W] - 2*cell->data[0][GUARD_C] + cell->data[0][GUARD_E])/(hx*hx)
           + (cell->data[0][GUARD_N] - 2*cell->data[0][GUARD_C] + cell->data[0][GUARD_S])/(hy*hy);

  PetscFunctionReturn(0);
}

PetscErrorCode apply_operator_dm(DM dm, Vec in, Vec out)
{
  AppCtx          *user;
  PetscErrorCode  ierr;

  PetscFunctionBegin;

  ierr = DMGetApplicationContext(dm,&user);CHKERRQ(ierr);
  ierr = DMBFIterateOverCellsVectors(dm,_p_dmbf_poisson_set_unk_cellfn,user,&in,1,PETSC_NULL,0);CHKERRQ(ierr);
  ierr = DMBFCommunicateGhostCells(dm);CHKERRQ(ierr);
  ierr = DMBFIterateOverFaces(dm,_p_dmbf_poisson_set_guards_facefn,user);CHKERRQ(ierr);
  ierr = DMBFIterateOverCellsVectors(dm,_p_dmbf_poisson_apply_operator_cellfn,user,PETSC_NULL,0,&out,1);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode apply_operator_mf(Mat K, Vec in, Vec out)
{
  DM              dm;
  PetscErrorCode  ierr;

  PetscFunctionBegin;

  ierr = MatGetDM(K,&dm);CHKERRQ(ierr);
  ierr = apply_operator_dm(dm,in,out);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscScalar g(PetscReal x, PetscReal y) {
  PetscReal theta = PetscAtan2Real(y,x) + PETSC_PI;
  PetscReal rsqr  = PetscSqr(x) + PetscSqr(y);

  return PetscPowReal(rsqr,.25)*PetscSinReal(.5*theta);
}

PETSC_STATIC_INLINE PetscScalar f(PetscReal x, PetscReal y) {
  return 0.0;
}

PetscErrorCode amr_refine_center(DM dm, DM_BF_Cell *cell, void *ctx) {
  PetscReal   x = cell->corner[0] + .5*cell->sidelength[0];
  PetscReal   y = cell->corner[1] + .5*cell->sidelength[1];
  PetscScalar r = PetscSqrtReal(PetscSqr(x) + PetscSqr(y));

  PetscFunctionBegin;

  if (r < 5e-2) {
    cell->adaptFlag=DM_ADAPT_REFINE;
  } else {
    cell->adaptFlag=DM_ADAPT_KEEP;
  }

  // JR i think refining along the slit is also necessary

  PetscFunctionReturn(0);
}

PetscErrorCode proj2coarse(DM dm, DM_BF_Cell **cellsFine, PetscInt i, DM_BF_Cell **cellsCoarse, PetscInt j, void *ctx) {
  PetscFunctionReturn(0);
}

PetscErrorCode apply_mass_matrix(DM dm, DM_BF_Cell *cell, void *ctx) {
  PetscReal *cell_val  = cell->vecViewReadWrite[0];
  PetscReal h          = cell->sidelength[0];

  PetscFunctionBegin;
  *cell_val *= h; // JR should this be h^2
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  const char      funcname[] = "DMBF-Poisson-2D";
  DM              dm;
  Vec             sol,rhs,exact,error;
  Mat             A;
  KSP             ksp;
  PetscInt        blockSize[2]; /* basic three point Laplacian */ // JR blockSize is never actually used (aside from reading it)
  PetscInt        dataShape[2] = { 5}; // JR why is there only one value?
  PetscReal       inf_norm,l2_norm;
  AppCtx          ctx;
  PetscViewer     viewer;
  PetscErrorCode  ierr;

  // initialize Petsc
  ierr = PetscInitialize(&argc,&argv,(char*)0,help);
  if (ierr) return ierr;

  PetscPrintf(PETSC_COMM_WORLD,"[%s] Begin\n",funcname);

  // create DM
  PetscPrintf(PETSC_COMM_WORLD,"[%s] Create DM\n",funcname);
  ierr = DMCreate(PETSC_COMM_WORLD,&dm);CHKERRQ(ierr);
  ierr = DMSetType(dm,"bf");CHKERRQ(ierr);

  // set DM options
  ierr = DMSetDimension(dm,2);CHKERRQ(ierr);
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
  ierr = DMBFGetBlockSize(dm,blockSize);CHKERRQ(ierr);
  // set cell data shapes
  ierr = DMBFSetCellDataShape(dm,dataShape,1,1);CHKERRQ(ierr);
  //ierr = DMBFSetCellDataVSize(dm,sizeof(cellData_t));CHKERRQ(ierr);

  // set application-specific data
  ierr = DMSetApplicationContext(dm,&ctx);CHKERRQ(ierr);

  // setup DM
  ierr = DMSetUp(dm);CHKERRQ(ierr);

  {
    DM_BF_AmrOps    amrOps;
    PetscInt        maxRefinement,initRefinement;
    DM              adapt;

    amrOps.setAmrFlag      = amr_refine_center;
    amrOps.setAmrFlagCtx   = &ctx;
    amrOps.projectToCoarse = proj2coarse;
    amrOps.projectToFine   = proj2coarse;
    amrOps.projectToFineCtx= PETSC_NULL;
    amrOps.projectToCoarseCtx= PETSC_NULL;

    ierr = DMBFAMRSetOperators(dm,&amrOps);CHKERRQ(ierr);

    ierr = DMForestGetInitialRefinement(dm,&initRefinement);CHKERRQ(ierr);
    ierr = DMForestGetMaximumRefinement(dm,&maxRefinement);CHKERRQ(ierr);

    for(PetscInt l=initRefinement;l<maxRefinement;l++) {
      ierr = DMBFAMRFlag(dm);CHKERRQ(ierr);
      ierr = DMBFAMRAdapt(dm,&adapt);CHKERRQ(ierr);
      ierr = DMDestroy(&dm);CHKERRQ(ierr);
      dm=adapt;
    }
  }

  ctx.f = f;
  ctx.g = g;

  // create vectors
  ierr = DMCreateGlobalVector(dm,&sol);CHKERRQ(ierr);
  ierr = VecDuplicate(sol,&rhs);CHKERRQ(ierr);
  ierr = VecDuplicate(sol,&exact);CHKERRQ(ierr);

  ierr = PetscObjectSetName((PetscObject)sol,"sol");CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)rhs,"rhs");CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)exact,"exact");CHKERRQ(ierr);

  ierr = DMBFIterateOverCells(dm,clear_cell_data,PETSC_NULL);CHKERRQ(ierr);
  ierr = DMBFIterateOverFaces(dm,set_up_boundary_condition,&ctx);CHKERRQ(ierr);
  ierr = DMBFIterateOverCellsVectors(dm,interpolate_source,&ctx,PETSC_NULL,0,&rhs,1);CHKERRQ(ierr);

  ierr = DMSetMatType(dm,MATSHELL);CHKERRQ(ierr);
  ierr = DMCreateMatrix(dm,&A);CHKERRQ(ierr);
  ierr = MatSetOperation(A,MATOP_MULT,(void(*)(void))apply_operator_mf);CHKERRQ(ierr);
  ierr = apply_operator_dm(dm,rhs,sol);CHKERRQ(ierr); //JR why computing sol = A*rhs
  ierr = PetscViewerCreate(PETSC_COMM_WORLD,&viewer);CHKERRQ(ierr);
  ierr = PetscViewerSetType(viewer,PETSCVIEWERVTK);CHKERRQ(ierr);
  ierr = PetscViewerFileSetMode(viewer,FILE_MODE_WRITE);CHKERRQ(ierr);
  ierr = PetscViewerFileSetName(viewer,"ex1_solution.vtu");CHKERRQ(ierr);
  ierr = VecView(rhs,viewer);CHKERRQ(ierr);

  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
  // JR since it is a poissson problem, we need a preconditioner, at least a diag matrix
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPSolve(ksp,rhs,sol);CHKERRQ(ierr);

  ierr = VecView(sol,viewer);CHKERRQ(ierr);
  ierr = DMBFIterateOverCellsVectors(dm,interpolate_exact,&ctx,PETSC_NULL,0,&exact,1);CHKERRQ(ierr);
  ierr = VecView(exact,viewer);CHKERRQ(ierr);
  ierr = VecAXPY(exact,-1,sol);CHKERRQ(ierr);
  error = exact;
  ierr = PetscObjectSetName((PetscObject)error,"error");CHKERRQ(ierr);
  ierr = VecView(error,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  ierr = VecNorm(error,NORM_INFINITY,&inf_norm);CHKERRQ(ierr);
  ierr = DMBFIterateOverCellsVectors(dm,apply_mass_matrix,&ctx,PETSC_NULL,0,&error,1);CHKERRQ(ierr);
  ierr = VecNorm(error,NORM_2,&l2_norm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"[%s] Linf error: %1.15f\n",funcname,inf_norm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"[%s] L2 error:   %1.15f\n",funcname,l2_norm);CHKERRQ(ierr);

  // destroy Petsc objects
  ierr = VecDestroy(&sol);CHKERRQ(ierr);
  ierr = VecDestroy(&rhs);CHKERRQ(ierr);
  ierr = VecDestroy(&exact);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);

  PetscPrintf(PETSC_COMM_WORLD,"[%s] End\n",funcname);

  // finalize Petsc
  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
./ex1 -dm_forest_topology brick \
      -dm_p4est_brick_size 2,2  \
      -dm_p4est_brick_bounds -1.0,1.0,-1.0,1.0 \
      -dm_p4est_brick_periodicity 0,0 \
      -dm_forest_initial_refinement 5 \
      -dm_forest_maximum_refinement 9 \
      -ksp_type gmres -ksp_max_it 10000 -ksp_atol 1e-10 -ksp_rtol 1e-11 \
      -ksp_monitor -ksp_view -ksp_converged_reason

JR this problem has way too many krylov iterations
   why is the krylov method GMRES, because the poissson problem should be symmetric, CG is more natural

TODO check if all objects are destroyed
*/
