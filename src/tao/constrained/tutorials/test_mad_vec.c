/* Program usage: mpiexec -n 2 ./test_mad_vec [-help] [all TAO options] */

#include <petsc.h>
#include <../src/tao/constrained/impls/mad/mad.h> /*I "petsctao.h" I*/ /*I "petscvecF->h" I*/

static  char help[]= "Tests the special vector spaces for the MAD algorithm\n\n";

const char *const Flabels[] = {"X","Sc","Scl","Scu","Sxl","Sxu","Yi","Ye","Vl","Vu","Zl","Zu"};
const char *const Rlabels[] = {"X","Yi","Ye"};
const char *const Plabels[] = {"X","Sc","Scl","Scu","Sxl","Sxu"};
const char *const Slabels[] = {"Sc","Scl","Scu","Sxl","Sxu"};
const char *const Ylabels[] = {"Yi","Ye","Vl","Vu","Zl","Zu"};

PetscErrorCode main(int argc,char **argv)
{ 
  FullSpaceVec      *vecF;
  ReducedSpaceVec   *vecR;
  Vec               *vb, tmp;
  const PetscScalar *vv;
  PetscReal         selfdot, norm2, norm2manual;
  PetscInt          vn, n, i, j;
  PetscMPIInt       size;
  PetscErrorCode    ierr;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  if (size>1){
    SETERRQ(PETSC_COMM_SELF,1,"Parallel test not supported.\n");
  }
  ierr = PetscNew(&vecF);CHKERRQ(ierr);
  vecF->nF=12; vecF->nR=3; vecF->nP=6; vecF->nS=5; vecF->nY=6;
  ierr = VecCreateSeq(PETSC_COMM_SELF, 3, &vecF->X);CHKERRQ(ierr); 
  ierr = VecCreateSeq(PETSC_COMM_SELF, 2, &vecF->Sc);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF, 2, &vecF->Scl);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF, 2, &vecF->Scu);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF, 3, &vecF->Sxl);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF, 3, &vecF->Sxu);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF, 2, &vecF->Yi);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF, 1, &vecF->Ye);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF, 2, &vecF->Vl);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF, 2, &vecF->Vu);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF, 3, &vecF->Zl);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF, 3, &vecF->Zu);CHKERRQ(ierr);
  ierr = FullSpaceVecCreate(vecF);CHKERRQ(ierr);

  /* ----------------------------------------------------------------------------------------- */

  ierr = VecNestGetSubVecs(vecF->F, &vn, &vb);CHKERRQ(ierr);
  ierr = VecSet(vecF->F, 0.0);CHKERRQ(ierr);

  PetscPrintf(PETSC_COMM_SELF, "Testing FullSpaceVec->F VECNEST...\n");
  for (j=0; j<vn; j++) {
    if (j == 0)  tmp = vecF->X;
    if (j == 1)  tmp = vecF->Sc;
    if (j == 2)  tmp = vecF->Scl;
    if (j == 3)  tmp = vecF->Scu;
    if (j == 4)  tmp = vecF->Sxl;
    if (j == 5)  tmp = vecF->Sxu;
    if (j == 6)  tmp = vecF->Yi;
    if (j == 7)  tmp = vecF->Ye;
    if (j == 8)  tmp = vecF->Vl;
    if (j == 9)  tmp = vecF->Vu;
    if (j == 10) tmp = vecF->Zl;
    if (j == 11) tmp = vecF->Zu;
    ierr = VecSet(tmp, (PetscReal)(j+1));CHKERRQ(ierr);
    ierr = VecGetArrayRead(vb[j], &vv);CHKERRQ(ierr);
    ierr = VecGetSize(vb[j], &n);CHKERRQ(ierr);
    for (i=0; i<n; i++) {
      PetscPrintf(PETSC_COMM_SELF, "  %s[%i] = %.2f | ", Flabels[j], i, vv[i]);
      if (vv[i] == (PetscReal)(j+1)) {
        PetscPrintf(PETSC_COMM_SELF, "correct\n");
      } else {
        PetscPrintf(PETSC_COMM_SELF, "wrong\n");
      }
    }
    ierr = VecRestoreArrayRead(vb[j], &vv);CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_SELF, "\n");
  }

  ierr = VecDot(vecF->F, vecF->F, &selfdot);CHKERRQ(ierr);
  norm2manual = PetscAbsScalar(PetscSqrtScalar(selfdot));CHKERRQ(ierr);
  ierr = VecNorm(vecF->F, NORM_2, &norm2);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_SELF, "  ||F|| = %.3e | ", norm2manual);
  if (norm2 == norm2manual) {
    PetscPrintf(PETSC_COMM_SELF, "correct\n");
  } else {
    PetscPrintf(PETSC_COMM_SELF, "wrong (%.3e)\n", norm2);
  }
  PetscPrintf(PETSC_COMM_SELF, "\n");

  /* ----------------------------------------------------------------------------------------- */

  ierr = VecNestGetSubVecs(vecF->R, &vn, &vb);CHKERRQ(ierr);
  ierr = VecSet(vecF->R, 0.0);CHKERRQ(ierr);

  PetscPrintf(PETSC_COMM_SELF, "Testing FullSpaceVec->R VECNEST...\n");
  for (j=0; j<vn; j++) {
    if (j == 0)  tmp = vecF->X;
    if (j == 1)  tmp = vecF->Yi;
    if (j == 2)  tmp = vecF->Ye;
    ierr = VecSet(tmp, (PetscReal)(j+1));CHKERRQ(ierr);
    ierr = VecGetArrayRead(vb[j], &vv);CHKERRQ(ierr);
    ierr = VecGetSize(vb[j], &n);CHKERRQ(ierr);
    for (i=0; i<n; i++) {
      PetscPrintf(PETSC_COMM_SELF, "  %s[%i] = %.2f | ", Rlabels[j], i, vv[i]);
      if (vv[i] == (PetscReal)(j+1)) {
        PetscPrintf(PETSC_COMM_SELF, "correct\n");
      } else {
        PetscPrintf(PETSC_COMM_SELF, "wrong\n");
      }
    }
    ierr = VecRestoreArrayRead(vb[j], &vv);CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_SELF, "\n");
  }

  ierr = VecDot(vecF->R, vecF->R, &selfdot);CHKERRQ(ierr);
  norm2manual = PetscAbsScalar(PetscSqrtScalar(selfdot));CHKERRQ(ierr);
  ierr = VecNorm(vecF->R, NORM_2, &norm2);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_SELF, "  ||Fr|| = %.3e | ", norm2manual);
  if (norm2 == norm2manual) {
    PetscPrintf(PETSC_COMM_SELF, "correct\n");
  } else {
    PetscPrintf(PETSC_COMM_SELF, "wrong (%.3e)\n", norm2);
  }
  PetscPrintf(PETSC_COMM_SELF, "\n");

  /* ----------------------------------------------------------------------------------------- */

  ierr = VecNestGetSubVecs(vecF->P, &vn, &vb);CHKERRQ(ierr);
  ierr = VecSet(vecF->R, 0.0);CHKERRQ(ierr);

  PetscPrintf(PETSC_COMM_SELF, "Testing FullSpaceVec->P VECNEST...\n");
  for (j=0; j<vn; j++) {
    if (j == 0)  tmp = vecF->X;
    if (j == 1)  tmp = vecF->Sc;
    if (j == 2)  tmp = vecF->Scl;
    if (j == 3)  tmp = vecF->Scu;
    if (j == 4)  tmp = vecF->Sxl;
    if (j == 5)  tmp = vecF->Sxu;
    ierr = VecSet(tmp, (PetscReal)(j+1));CHKERRQ(ierr);
    ierr = VecGetArrayRead(vb[j], &vv);CHKERRQ(ierr);
    ierr = VecGetSize(vb[j], &n);CHKERRQ(ierr);
    for (i=0; i<n; i++) {
      PetscPrintf(PETSC_COMM_SELF, "  %s[%i] = %.2f | ", Plabels[j], i, vv[i]);
      if (vv[i] == (PetscReal)(j+1)) {
        PetscPrintf(PETSC_COMM_SELF, "correct\n");
      } else {
        PetscPrintf(PETSC_COMM_SELF, "wrong\n");
      }
    }
    ierr = VecRestoreArrayRead(vb[j], &vv);CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_SELF, "\n");
  }

  ierr = VecDot(vecF->P, vecF->P, &selfdot);CHKERRQ(ierr);
  norm2manual = PetscAbsScalar(PetscSqrtScalar(selfdot));CHKERRQ(ierr);
  ierr = VecNorm(vecF->P, NORM_2, &norm2);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_SELF, "  ||Fp|| = %.3e | ", norm2manual);
  if (norm2 == norm2manual) {
    PetscPrintf(PETSC_COMM_SELF, "correct\n");
  } else {
    PetscPrintf(PETSC_COMM_SELF, "wrong (%.3e)\n", norm2);
  }
  PetscPrintf(PETSC_COMM_SELF, "\n");

  /* ----------------------------------------------------------------------------------------- */

  ierr = VecNestGetSubVecs(vecF->S, &vn, &vb);CHKERRQ(ierr);
  ierr = VecSet(vecF->R, 0.0);CHKERRQ(ierr);

  PetscPrintf(PETSC_COMM_SELF, "Testing FullSpaceVec->S VECNEST...\n");
  for (j=0; j<vn; j++) {
    if (j == 0)  tmp = vecF->Sc;
    if (j == 1)  tmp = vecF->Scl;
    if (j == 2)  tmp = vecF->Scu;
    if (j == 3)  tmp = vecF->Sxl;
    if (j == 4)  tmp = vecF->Sxu;
    ierr = VecSet(tmp, (PetscReal)(j+1));CHKERRQ(ierr);
    ierr = VecGetArrayRead(vb[j], &vv);CHKERRQ(ierr);
    ierr = VecGetSize(vb[j], &n);CHKERRQ(ierr);
    for (i=0; i<n; i++) {
      PetscPrintf(PETSC_COMM_SELF, "  %s[%i] = %.2f | ", Slabels[j], i, vv[i]);
      if (vv[i] == (PetscReal)(j+1)) {
        PetscPrintf(PETSC_COMM_SELF, "correct\n");
      } else {
        PetscPrintf(PETSC_COMM_SELF, "wrong\n");
      }
    }
    ierr = VecRestoreArrayRead(vb[j], &vv);CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_SELF, "\n");
  }

  ierr = VecDot(vecF->S, vecF->S, &selfdot);CHKERRQ(ierr);
  norm2manual = PetscAbsScalar(PetscSqrtScalar(selfdot));CHKERRQ(ierr);
  ierr = VecNorm(vecF->S, NORM_2, &norm2);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_SELF, "  ||Fs|| = %.3e | ", norm2manual);
  if (norm2 == norm2manual) {
    PetscPrintf(PETSC_COMM_SELF, "correct\n");
  } else {
    PetscPrintf(PETSC_COMM_SELF, "wrong (%.3e)\n", norm2);
  }
  PetscPrintf(PETSC_COMM_SELF, "\n");

  /* ----------------------------------------------------------------------------------------- */

  ierr = VecNestGetSubVecs(vecF->Y, &vn, &vb);CHKERRQ(ierr);
  ierr = VecSet(vecF->R, 0.0);CHKERRQ(ierr);

  PetscPrintf(PETSC_COMM_SELF, "Testing FullSpaceVec->Y VECNEST...\n");
  for (j=0; j<vn; j++) {
    if (j == 0)  tmp = vecF->Yi;
    if (j == 1)  tmp = vecF->Ye;
    if (j == 2)  tmp = vecF->Vl;
    if (j == 3)  tmp = vecF->Vu;
    if (j == 4)  tmp = vecF->Zl;
    if (j == 5)  tmp = vecF->Zu;
    ierr = VecSet(tmp, (PetscReal)(j+1));CHKERRQ(ierr);
    ierr = VecGetArrayRead(vb[j], &vv);CHKERRQ(ierr);
    ierr = VecGetSize(vb[j], &n);CHKERRQ(ierr);
    for (i=0; i<n; i++) {
      PetscPrintf(PETSC_COMM_SELF, "  %s[%i] = %.2f | ", Ylabels[j], i, vv[i]);
      if (vv[i] == (PetscReal)(j+1)) {
        PetscPrintf(PETSC_COMM_SELF, "correct\n");
      } else {
        PetscPrintf(PETSC_COMM_SELF, "wrong\n");
      }
    }
    ierr = VecRestoreArrayRead(vb[j], &vv);CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_SELF, "\n");
  }

  ierr = VecDot(vecF->Y, vecF->Y, &selfdot);CHKERRQ(ierr);
  norm2manual = PetscAbsScalar(PetscSqrtScalar(selfdot));CHKERRQ(ierr);
  ierr = VecNorm(vecF->Y, NORM_2, &norm2);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_SELF, "  ||Fy|| = %.3e | ", norm2manual);
  if (norm2 == norm2manual) {
    PetscPrintf(PETSC_COMM_SELF, "correct\n");
  } else {
    PetscPrintf(PETSC_COMM_SELF, "wrong (%.3e)\n", norm2);
  }
  PetscPrintf(PETSC_COMM_SELF, "\n");

  /* ----------------------------------------------------------------------------------------- */

  ierr = PetscNew(&vecR);CHKERRQ(ierr);
  vecR->nR = 3;
  ierr = VecDuplicate(vecF->X, &vecR->X);CHKERRQ(ierr);
  ierr = VecDuplicate(vecF->Yi, &vecR->Yi);CHKERRQ(ierr);
  ierr = VecDuplicate(vecF->Ye, &vecR->Ye);CHKERRQ(ierr);
  ierr = ReducedSpaceVecCreate(vecR);CHKERRQ(ierr);

  /* ----------------------------------------------------------------------------------------- */

  ierr = VecNestGetSubVecs(vecR->R, &vn, &vb);CHKERRQ(ierr);
  ierr = VecSet(vecR->R, 0.0);CHKERRQ(ierr);

  PetscPrintf(PETSC_COMM_SELF, "Testing ReducedSpaceVec->R VECNEST...\n");
  for (j=0; j<vn; j++) {
    if (j == 0)  tmp = vecR->X;
    if (j == 1)  tmp = vecR->Yi;
    if (j == 2)  tmp = vecR->Ye;
    ierr = VecSet(tmp, (PetscReal)(j+1));CHKERRQ(ierr);
    ierr = VecGetArrayRead(vb[j], &vv);CHKERRQ(ierr);
    ierr = VecGetSize(vb[j], &n);CHKERRQ(ierr);
    for (i=0; i<n; i++) {
      PetscPrintf(PETSC_COMM_SELF, "  %s[%i] = %.2f | ", Rlabels[j], i, vv[i]);
      if (vv[i] == (PetscReal)(j+1)) {
        PetscPrintf(PETSC_COMM_SELF, "correct\n");
      } else {
        PetscPrintf(PETSC_COMM_SELF, "wrong\n");
      }
    }
    ierr = VecRestoreArrayRead(vb[j], &vv);CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_SELF, "\n");
  }

  ierr = VecDot(vecR->R, vecR->R, &selfdot);CHKERRQ(ierr);
  norm2manual = PetscAbsScalar(PetscSqrtScalar(selfdot));CHKERRQ(ierr);
  ierr = VecNorm(vecR->R, NORM_2, &norm2);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_SELF, "  ||R|| = %.3e | ", norm2manual);
  if (norm2 == norm2manual) {
    PetscPrintf(PETSC_COMM_SELF, "correct\n");
  } else {
    PetscPrintf(PETSC_COMM_SELF, "wrong (%.3e)\n", norm2);
  }
  PetscPrintf(PETSC_COMM_SELF, "\n");

  /* ----------------------------------------------------------------------------------------- */

  ierr = FullSpaceVecDestroy(vecF);CHKERRQ(ierr);
  ierr = ReducedSpaceVecDestroy(vecR);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   build:
      requires: !complex

   test:
      suffix: 1
      args:

TEST*/
