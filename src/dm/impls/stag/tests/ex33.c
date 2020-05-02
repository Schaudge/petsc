static char help[] = "Test PCFieldSplit in Distributed Gauss-Seidel mode, with DMStag\n\n";

#include <petscdmstag.h>
#include <petscksp.h>

static PetscErrorCode CreateSystem2d(DM,Mat*,Vec*,PetscBool);

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  DM             dm;
  Mat            A;
  Vec            x,y;
  PC             pc;
  PetscInt       dim;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  dim = 2;
  ierr = PetscOptionsGetInt(NULL,NULL,"-dim",&dim,NULL);CHKERRQ(ierr);
  switch (dim) {
    case 2:
      ierr = DMStagCreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,2,3,PETSC_DECIDE,PETSC_DECIDE,0,1,1,DMSTAG_STENCIL_BOX,1,NULL,NULL,&dm);CHKERRQ(ierr);
      break;
    default:SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Not Implemented!");
  }
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
  ierr = DMSetUp(dm);CHKERRQ(ierr);
  ierr = DMStagSetUniformCoordinatesProduct(dm,0.0,2.0,0.0,3.0,0.0,1.0);CHKERRQ(ierr);

  switch (dim) {
    case 2:
      ierr = CreateSystem2d(dm,&A,NULL,PETSC_FALSE);CHKERRQ(ierr);
      break;
    default:SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Not Implemented!");
  }

  ierr = PCCreate(PetscObjectComm((PetscObject)dm),&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCFIELDSPLIT);CHKERRQ(ierr);
  ierr = PCFieldSplitSetType(pc,PC_COMPOSITE_DGS);CHKERRQ(ierr);
  ierr = PCSetOperators(pc,A,A);CHKERRQ(ierr); // Not using a separate Pmat in this test (though that would be better)
  ierr = PCSetDM(pc,dm);CHKERRQ(ierr);

  ierr = PCSetFromOptions(pc);CHKERRQ(ierr);
  ierr = PCSetUp(pc);CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(dm,&x);CHKERRQ(ierr);
  ierr = VecSet(x,1.0);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&y);CHKERRQ(ierr);

  ierr = PCApply(pc,x,y);CHKERRQ(ierr);

  ierr = VecView(y,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = PCDestroy(&pc);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/* A test system, copied from a tutorial without much thought */
static PetscScalar uxRef(PetscScalar x,PetscScalar y ){return 0.0*x + y*y - 2.0*y*y*y + y*y*y*y;}    /* no x-dependence  */
static PetscScalar uyRef(PetscScalar x,PetscScalar y) {return x*x - 2.0*x*x*x + x*x*x*x + 0.0*y;}    /* no y-dependence  */
static PetscScalar pRef (PetscScalar x,PetscScalar y) {return -1.0*(x-0.5) + -3.0/2.0*y*y + 0.5;}    /* zero integral    */
static PetscScalar fx   (PetscScalar x,PetscScalar y) {return 0.0*x + 2.0 -12.0*y + 12.0*y*y + 1.0;} /* no x-dependence  */
static PetscScalar fy   (PetscScalar x,PetscScalar y) {return 2.0 -12.0*x + 12.0*x*x + 3.0*y;}
static PetscScalar g    (PetscScalar x,PetscScalar y) {return 0.0*x*y;}                              /* identically zero */

static PetscErrorCode CreateSystem2d(DM dmSol,Mat *pA,Vec *pRhs, PetscBool pinPressure)
{
  PetscErrorCode ierr;
  PetscInt       N[2];
  PetscInt       ex,ey,startx,starty,nx,ny;
  PetscInt       iprev,icenter,inext;
  Mat            A;
  Vec            rhs;
  PetscReal      hx,hy,dv;
  PetscScalar    **cArrX,**cArrY;
  PetscBool      build_rhs;

  /* Here, we showcase two different methods for manipulating local vector entries.
     One can use DMStagStencil objects with DMStagVecSetValuesStencil(),
     making sure to call VecAssemble[Begin/End]() after all values are set.
     Alternately, one can use DMStagVecGetArray[Read]() and DMStagVecRestoreArray[Read]().
     The first approach is used to build the rhs, and the second is used to
     obtain coordinate values. Working with the array is almost certainly more efficient,
     but only allows setting local entries, requires understanding which "slot" to use,
     and doesn't correspond as precisely to the matrix assembly process using DMStagStencil objects */

  PetscFunctionBeginUser;
  ierr = DMCreateMatrix(dmSol,pA);CHKERRQ(ierr);
  A = *pA;
  build_rhs = pRhs != NULL;
  if (build_rhs) {
    ierr = DMCreateGlobalVector(dmSol,pRhs);CHKERRQ(ierr);
    rhs = *pRhs;
  } else {
    rhs = NULL;
  }
  ierr = DMStagGetCorners(dmSol,&startx,&starty,NULL,&nx,&ny,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = DMStagGetGlobalSizes(dmSol,&N[0],&N[1],NULL);CHKERRQ(ierr);
  hx = 1.0/N[0]; hy = 1.0/N[1];
  dv = 1.0; /* No scaling, for ease of debugging */
  ierr = DMStagGetProductCoordinateArraysRead(dmSol,&cArrX,&cArrY,NULL);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dmSol,DMSTAG_ELEMENT,&icenter);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dmSol,DMSTAG_LEFT,&iprev);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dmSol,DMSTAG_RIGHT,&inext);CHKERRQ(ierr);

  /* Loop over all local elements. Note that it may be more efficient in real
     applications to loop over each boundary separately */
  for (ey = starty; ey<starty+ny; ++ey) { /* With DMStag, always iterate x fastest, y second fastest, z slowest */
    for (ex = startx; ex<startx+nx; ++ex) {

      if (ex == N[0]-1) {
        /* Right Boundary velocity Dirichlet */
        DMStagStencil     row;
        PetscScalar       valRhs;
        const PetscScalar valA = 1.0;
        row.i = ex; row.j = ey; row.loc = DMSTAG_RIGHT; row.c = 0;
        ierr = DMStagMatSetValuesStencil(dmSol,A,1,&row,1,&row,&valA,INSERT_VALUES);CHKERRQ(ierr);
        if (build_rhs) {
          valRhs = uxRef(cArrX[ex][inext],cArrY[ey][icenter]);
          ierr = DMStagVecSetValuesStencil(dmSol,rhs,1,&row,&valRhs,INSERT_VALUES);CHKERRQ(ierr);
        }
      }
      if (ey == N[1]-1) {
        /* Top boundary velocity Dirichlet */
        DMStagStencil     row;
        PetscScalar       valRhs;
        const PetscScalar valA = 1.0;
        row.i = ex; row.j = ey; row.loc = DMSTAG_UP; row.c = 0;
        ierr = DMStagMatSetValuesStencil(dmSol,A,1,&row,1,&row,&valA,INSERT_VALUES);CHKERRQ(ierr);
        if (build_rhs) {
          valRhs = uyRef(cArrX[ex][icenter],cArrY[ey][inext]);
          ierr = DMStagVecSetValuesStencil(dmSol,rhs,1,&row,&valRhs,INSERT_VALUES);CHKERRQ(ierr);
        }
      }

      if (ey == 0) {
        /* Bottom boundary velocity Dirichlet */
        DMStagStencil     row;
        PetscScalar       valRhs;
        const PetscScalar valA = 1.0;
        row.i = ex; row.j = ey; row.loc = DMSTAG_DOWN; row.c = 0;
        ierr = DMStagMatSetValuesStencil(dmSol,A,1,&row,1,&row,&valA,INSERT_VALUES);CHKERRQ(ierr);
        if (build_rhs) {
          valRhs = uyRef(cArrX[ex][icenter],cArrY[ey][iprev]);
          ierr = DMStagVecSetValuesStencil(dmSol,rhs,1,&row,&valRhs,INSERT_VALUES);CHKERRQ(ierr);
        }
      } else {
        /* Y-momentum equation : (u_xx + u_yy) - p_y = f^y */
        DMStagStencil row,col[7];
        PetscScalar   valA[7],valRhs;
        PetscInt      nEntries;

        row.i    = ex  ; row.j    = ey  ; row.loc    = DMSTAG_DOWN;    row.c     = 0;
        if (ex == 0) {
          nEntries = 6;
          col[0].i = ex  ; col[0].j = ey  ; col[0].loc = DMSTAG_DOWN;    col[0].c  = 0; valA[0] = -dv*1.0 / (hx*hx) -dv*2.0 / (hy*hy);
          col[1].i = ex  ; col[1].j = ey-1; col[1].loc = DMSTAG_DOWN;    col[1].c  = 0; valA[1] =  dv*1.0 / (hy*hy);
          col[2].i = ex  ; col[2].j = ey+1; col[2].loc = DMSTAG_DOWN;    col[2].c  = 0; valA[2] =  dv*1.0 / (hy*hy);
          /* Missing left element */
          col[3].i = ex+1; col[3].j = ey  ; col[3].loc = DMSTAG_DOWN;    col[3].c  = 0; valA[3] =  dv*1.0 / (hx*hx);
          col[4].i = ex  ; col[4].j = ey-1; col[4].loc = DMSTAG_ELEMENT; col[4].c  = 0; valA[4] =  dv*1.0 / hy;
          col[5].i = ex  ; col[5].j = ey  ; col[5].loc = DMSTAG_ELEMENT; col[5].c  = 0; valA[5] = -dv*1.0 / hy;
        } else if (ex == N[0]-1) {
          /* Right boundary y velocity stencil */
          nEntries = 6;
          col[0].i = ex  ; col[0].j = ey  ; col[0].loc = DMSTAG_DOWN;    col[0].c  = 0; valA[0] = -dv*1.0 / (hx*hx) -dv*2.0 / (hy*hy);
          col[1].i = ex  ; col[1].j = ey-1; col[1].loc = DMSTAG_DOWN;    col[1].c  = 0; valA[1] =  dv*1.0 / (hy*hy);
          col[2].i = ex  ; col[2].j = ey+1; col[2].loc = DMSTAG_DOWN;    col[2].c  = 0; valA[2] =  dv*1.0 / (hy*hy);
          col[3].i = ex-1; col[3].j = ey  ; col[3].loc = DMSTAG_DOWN;    col[3].c  = 0; valA[3] =  dv*1.0 / (hx*hx);
          /* Missing right element */
          col[4].i = ex  ; col[4].j = ey-1; col[4].loc = DMSTAG_ELEMENT; col[4].c  = 0; valA[4] =  dv*1.0 / hy;
          col[5].i = ex  ; col[5].j = ey  ; col[5].loc = DMSTAG_ELEMENT; col[5].c  = 0; valA[5] = -dv*1.0 / hy;
        } else {
          nEntries = 7;
          col[0].i = ex  ; col[0].j = ey  ; col[0].loc = DMSTAG_DOWN;    col[0].c  = 0; valA[0] = -dv*2.0 / (hx*hx) -dv*2.0 / (hy*hy);
          col[1].i = ex  ; col[1].j = ey-1; col[1].loc = DMSTAG_DOWN;    col[1].c  = 0; valA[1] =  dv*1.0 / (hy*hy);
          col[2].i = ex  ; col[2].j = ey+1; col[2].loc = DMSTAG_DOWN;    col[2].c  = 0; valA[2] =  dv*1.0 / (hy*hy);
          col[3].i = ex-1; col[3].j = ey  ; col[3].loc = DMSTAG_DOWN;    col[3].c  = 0; valA[3] =  dv*1.0 / (hx*hx);
          col[4].i = ex+1; col[4].j = ey  ; col[4].loc = DMSTAG_DOWN;    col[4].c  = 0; valA[4] =  dv*1.0 / (hx*hx);
          col[5].i = ex  ; col[5].j = ey-1; col[5].loc = DMSTAG_ELEMENT; col[5].c  = 0; valA[5] =  dv*1.0 / hy;
          col[6].i = ex  ; col[6].j = ey  ; col[6].loc = DMSTAG_ELEMENT; col[6].c  = 0; valA[6] = -dv*1.0 / hy;
        }
        ierr = DMStagMatSetValuesStencil(dmSol,A,1,&row,nEntries,col,valA,INSERT_VALUES);CHKERRQ(ierr);
        if (build_rhs) {
          valRhs = dv*fy(cArrX[ex][icenter],cArrY[ey][iprev]);
          ierr = DMStagVecSetValuesStencil(dmSol,rhs,1,&row,&valRhs,INSERT_VALUES);CHKERRQ(ierr);
        }
      }

      if (ex == 0) {
        /* Left velocity Dirichlet */
        DMStagStencil row;
        PetscScalar   valRhs;
        const PetscScalar valA = 1.0;
        row.i = ex; row.j = ey; row.loc = DMSTAG_LEFT; row.c = 0;
        ierr = DMStagMatSetValuesStencil(dmSol,A,1,&row,1,&row,&valA,INSERT_VALUES);CHKERRQ(ierr);
        if (build_rhs) {
          valRhs = uxRef(cArrX[ex][iprev],cArrY[ey][icenter]);
          ierr = DMStagVecSetValuesStencil(dmSol,rhs,1,&row,&valRhs,INSERT_VALUES);CHKERRQ(ierr);
        }
      } else {
        /* X-momentum equation : (u_xx + u_yy) - p_x = f^x */
        DMStagStencil row,col[7];
        PetscScalar   valA[7],valRhs;
        PetscInt      nEntries;
        row.i    = ex  ; row.j    = ey  ; row.loc    = DMSTAG_LEFT;    row.c     = 0;

        if (ey == 0) {
          nEntries = 6;
          col[0].i = ex  ; col[0].j = ey  ; col[0].loc = DMSTAG_LEFT;    col[0].c  = 0; valA[0] = -dv*2.0 /(hx*hx) -dv*1.0 /(hy*hy);
          /* missing term from element below */
          col[1].i = ex  ; col[1].j = ey+1; col[1].loc = DMSTAG_LEFT;    col[1].c  = 0; valA[1] =  dv*1.0 / (hy*hy);
          col[2].i = ex-1; col[2].j = ey  ; col[2].loc = DMSTAG_LEFT;    col[2].c  = 0; valA[2] =  dv*1.0 / (hx*hx);
          col[3].i = ex+1; col[3].j = ey  ; col[3].loc = DMSTAG_LEFT;    col[3].c  = 0; valA[3] =  dv*1.0 / (hx*hx);
          col[4].i = ex-1; col[4].j = ey  ; col[4].loc = DMSTAG_ELEMENT; col[4].c  = 0; valA[4] =  dv*1.0 / hx;
          col[5].i = ex  ; col[5].j = ey  ; col[5].loc = DMSTAG_ELEMENT; col[5].c  = 0; valA[5] = -dv*1.0 / hx;
        } else if (ey == N[1]-1) {
          /* Top boundary x velocity stencil */
          nEntries = 6;
          row.i    = ex  ; row.j    = ey  ; row.loc    = DMSTAG_LEFT;    row.c     = 0;
          col[0].i = ex  ; col[0].j = ey  ; col[0].loc = DMSTAG_LEFT;    col[0].c  = 0; valA[0] = -dv*2.0 / (hx*hx) -dv*1.0 / (hy*hy);
          col[1].i = ex  ; col[1].j = ey-1; col[1].loc = DMSTAG_LEFT;    col[1].c  = 0; valA[1] =  dv*1.0 / (hy*hy);
          /* Missing element above term */
          col[2].i = ex-1; col[2].j = ey  ; col[2].loc = DMSTAG_LEFT;    col[2].c  = 0; valA[2] =  dv*1.0 / (hx*hx);
          col[3].i = ex+1; col[3].j = ey  ; col[3].loc = DMSTAG_LEFT;    col[3].c  = 0; valA[3] =  dv*1.0 / (hx*hx);
          col[4].i = ex-1; col[4].j = ey  ; col[4].loc = DMSTAG_ELEMENT; col[4].c  = 0; valA[4] =  dv*1.0 / hx;
          col[5].i = ex  ; col[5].j = ey  ; col[5].loc = DMSTAG_ELEMENT; col[5].c  = 0; valA[5] = -dv*1.0 / hx;
        } else {
          /* Note how this is identical to the stencil for U_y, with "DMSTAG_DOWN" replaced by "DMSTAG_LEFT" and the pressure derivative in the other direction */
          nEntries = 7;
          col[0].i = ex  ; col[0].j = ey  ; col[0].loc = DMSTAG_LEFT;    col[0].c  = 0; valA[0] = -dv*2.0 / (hx*hx) + -dv*2.0 / (hy*hy);
          col[1].i = ex  ; col[1].j = ey-1; col[1].loc = DMSTAG_LEFT;    col[1].c  = 0; valA[1] =  dv*1.0 / (hy*hy);
          col[2].i = ex  ; col[2].j = ey+1; col[2].loc = DMSTAG_LEFT;    col[2].c  = 0; valA[2] =  dv*1.0 / (hy*hy);
          col[3].i = ex-1; col[3].j = ey  ; col[3].loc = DMSTAG_LEFT;    col[3].c  = 0; valA[3] =  dv*1.0 / (hx*hx);
          col[4].i = ex+1; col[4].j = ey  ; col[4].loc = DMSTAG_LEFT;    col[4].c  = 0; valA[4] =  dv*1.0 / (hx*hx);
          col[5].i = ex-1; col[5].j = ey  ; col[5].loc = DMSTAG_ELEMENT; col[5].c  = 0; valA[5] =  dv*1.0 / hx;
          col[6].i = ex  ; col[6].j = ey  ; col[6].loc = DMSTAG_ELEMENT; col[6].c  = 0; valA[6] = -dv*1.0 / hx;

        }
        ierr = DMStagMatSetValuesStencil(dmSol,A,1,&row,nEntries,col,valA,INSERT_VALUES);CHKERRQ(ierr);
        if (build_rhs) {
          valRhs = dv*fx(cArrX[ex][iprev],cArrY[ey][icenter]);
          ierr = DMStagVecSetValuesStencil(dmSol,rhs,1,&row,&valRhs,INSERT_VALUES);CHKERRQ(ierr);
        }
      }

      /* P equation : u_x + v_y = g
         Note that this includes an explicit zero on the diagonal. This is only needed for
         direct solvers (not required if using an iterative solver and setting the constant-pressure nullspace) */
      /* Note: the pressure scaling by dv here is not chosen in a principled way. It could perhaps be improved */
      if (pinPressure && ex == 0 && ey == 0) { /* Pin the first pressure node, if requested */
        DMStagStencil row;
        PetscScalar valA,valRhs;
        row.i = ex; row.j = ey; row.loc  = DMSTAG_ELEMENT; row.c = 0;
        valA = 1.0;
        ierr = DMStagMatSetValuesStencil(dmSol,A,1,&row,1,&row,&valA,INSERT_VALUES);CHKERRQ(ierr);
        if (build_rhs) {
          valRhs = pRef(cArrX[ex][icenter],cArrY[ey][icenter]);
          ierr = DMStagVecSetValuesStencil(dmSol,rhs,1,&row,&valRhs,INSERT_VALUES);CHKERRQ(ierr);
        }
      } else {
        DMStagStencil row,col[5];
        PetscScalar   valA[5],valRhs;

        row.i    = ex; row.j    = ey; row.loc    = DMSTAG_ELEMENT; row.c    = 0;
        col[0].i = ex; col[0].j = ey; col[0].loc = DMSTAG_LEFT;    col[0].c = 0; valA[0] = -dv*1.0 / hx;
        col[1].i = ex; col[1].j = ey; col[1].loc = DMSTAG_RIGHT;   col[1].c = 0; valA[1] =  dv*1.0 / hx;
        col[2].i = ex; col[2].j = ey; col[2].loc = DMSTAG_DOWN;    col[2].c = 0; valA[2] = -dv*1.0 / hy;
        col[3].i = ex; col[3].j = ey; col[3].loc = DMSTAG_UP;      col[3].c = 0; valA[3] =  dv*1.0 / hy;
        col[4] = row;                                                     valA[4] = dv*0.0;
        ierr = DMStagMatSetValuesStencil(dmSol,A,1,&row,5,col,valA,INSERT_VALUES);CHKERRQ(ierr);
        if (build_rhs) {
          valRhs = dv*g(cArrX[ex][icenter],cArrY[ey][icenter]);
          ierr = DMStagVecSetValuesStencil(dmSol,rhs,1,&row,&valRhs,INSERT_VALUES);CHKERRQ(ierr);
        }
      }
    }
  }
  ierr = DMStagRestoreProductCoordinateArraysRead(dmSol,&cArrX,&cArrY,NULL);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (build_rhs) {
    ierr = VecAssemblyBegin(rhs);CHKERRQ(ierr);
  }
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (build_rhs) {
    ierr = VecAssemblyEnd(rhs);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

// TODO tests, e.g.
/*
  test:
    suffix: 1

  test:
    suffix: 2
    args: -fieldsplit_dgs_aux_pc_type jacobi -fieldsplit_face_pc_type jacobi -fieldsplit_dgs_aux_ksp_type preonly

*/
